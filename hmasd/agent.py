import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.optim import Adam
from torch.distributions import Categorical
import time
import os
from collections import deque
from torch.utils.tensorboard import SummaryWriter

from logger import main_logger
from hmasd.networks import SkillCoordinator, SkillDiscoverer, TeamDiscriminator, IndividualDiscriminator
from hmasd.utils import ReplayBuffer, StateSkillDataset, compute_gae, compute_ppo_loss, one_hot

def huber_loss(input, target, delta=1.0, reduction='mean'):
    """
    计算Huber Loss（也称为Smooth L1 Loss）
    
    参数:
        input: 预测值 [batch_size, ...]
        target: 目标值 [batch_size, ...]
        delta: Huber Loss的delta参数，控制L1/L2切换点
        reduction: 'mean', 'sum', 'none'
        
    返回:
        loss: Huber损失值
    """
    residual = torch.abs(input - target)
    condition = residual < delta
    
    # 当|residual| < delta时使用L2损失：0.5 * residual^2 / delta
    # 当|residual| >= delta时使用L1损失：residual - 0.5 * delta
    loss = torch.where(
        condition,
        0.5 * residual.pow(2) / delta,
        residual - 0.5 * delta
    )
    
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:
        return loss

class HMASDAgent:
    """
    层次化多智能体技能发现（HMASD）代理
    """
    def __init__(self, config, log_dir='logs', device=None, debug=False):
        """
        初始化HMASD代理
        
        参数:
            config: 配置对象，包含所有超参数
            log_dir: TensorBoard日志目录
            device: 计算设备，如果为None则自动检测
            debug: 是否启用自动求导异常检测
        """
        # 启用异常检测以帮助调试
        if debug:
            torch.autograd.set_detect_anomaly(True)
            main_logger.info("已启用自动求导异常检测")
            
        self.config = config
        self.device = device if device is not None else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        main_logger.info(f"使用设备: {self.device}")
        
        # 确保环境维度已设置
        assert config.state_dim is not None, "必须先设置state_dim"
        assert config.obs_dim is not None, "必须先设置obs_dim"
        
        # 【修复E1】设置logger属性以支持在多个方法中使用
        self.logger = main_logger
        
        # 初始化TensorBoard
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir)
        main_logger.debug(f"HMASDAgent.__init__: SummaryWriter created: {self.writer}")
        self.global_step = 0
        
        # 训练模式控制
        self.rollout_based_training = getattr(config, 'rollout_based_training', True)
        self.episode_based_training = getattr(config, 'episode_based_training', False)
        self.sync_training_mode = getattr(config, 'sync_training_mode', False)
        
        # 确保只有一种训练模式被启用
        active_modes = sum([self.rollout_based_training, self.episode_based_training, self.sync_training_mode])
        if active_modes > 1:
            main_logger.warning("检测到多个训练模式被启用，将使用rollout_based_training作为默认模式")
            self.rollout_based_training = True
            self.episode_based_training = False
            self.sync_training_mode = False
        elif active_modes == 0:
            main_logger.warning("没有训练模式被启用，将使用rollout_based_training作为默认模式")
            self.rollout_based_training = True
        
        if self.rollout_based_training:
            # Rollout-based训练状态管理
            self.rollout_length = config.rollout_length
            self.num_parallel_envs = config.num_parallel_envs
            self.ppo_epochs = config.ppo_epochs
            self.num_mini_batch = config.num_mini_batch
            self.steps_collected = 0               # 当前rollout收集的步数
            self.rollout_count = 0                 # rollout计数器
            self.total_steps_collected = 0         # 总收集步数
            main_logger.info(f"Rollout-based训练模式已启用，rollout长度: {self.rollout_length}, 并行环境: {self.num_parallel_envs}")
            
        elif self.episode_based_training:
            # Episode-based训练状态管理
            self.episodes_collected = 0
            self.update_frequency = config.update_frequency
            self.min_episodes_for_update = config.min_episodes_for_update
            self.max_episodes_per_update = config.max_episodes_per_update
            self.min_high_level_samples = config.min_high_level_samples
            self.min_low_level_samples = config.min_low_level_samples
            self.last_update_episode = 0
            main_logger.info(f"Episode-based训练模式已启用，更新频率: {self.update_frequency} episodes")
            
        else:  # sync_training_mode
            # 同步训练控制机制（兼容性保留）
            self.sync_mode = True                     # 启用同步训练模式
            self.collection_enabled = True            # 数据收集开关
            self.policy_version = 0                   # 策略版本号
            self.sync_batch_size = config.batch_size  # 同步batch大小（从配置获取）
            self.samples_collected_this_round = 0     # 本轮收集的样本数
            self.last_sync_step = 0                   # 上次同步更新的步数
            main_logger.info(f"同步训练模式已启用，同步batch大小: {self.sync_batch_size}")
        
        # 确保所有训练模式都有必要的属性（向后兼容）
        if not hasattr(self, 'sync_mode'):
            self.sync_mode = False  # 默认关闭同步模式
        if not hasattr(self, 'collection_enabled'):
            self.collection_enabled = True  # 默认启用数据收集
        
        # 创建网络
        self.skill_coordinator = SkillCoordinator(config).to(self.device)
        self.skill_discoverer = SkillDiscoverer(config, logger=main_logger).to(self.device) # Pass logger
        self.team_discriminator = TeamDiscriminator(config).to(self.device)
        self.individual_discriminator = IndividualDiscriminator(config).to(self.device)
        
        # 创建优化器
        self.coordinator_optimizer = Adam(
            self.skill_coordinator.parameters(),
            lr=config.lr_coordinator
        )
        self.discoverer_optimizer = Adam(
            self.skill_discoverer.parameters(),
            lr=config.lr_discoverer
        )
        self.discriminator_optimizer = Adam(
            list(self.team_discriminator.parameters()) + 
            list(self.individual_discriminator.parameters()),
            lr=config.lr_discriminator
        )
        
        # 创建经验回放缓冲区
        self.high_level_buffer = ReplayBuffer(config.buffer_size)
        self.high_level_buffer_with_logprobs = []  # 新增：高层经验缓冲区（带log probabilities）
        self.low_level_buffer = ReplayBuffer(config.buffer_size)
        self.state_skill_dataset = StateSkillDataset(config.buffer_size)
        
        # 其他初始化
        self.current_team_skill = None  # 当前团队技能 (保留用于单环境兼容性)
        self.current_agent_skills = None  # 当前个体技能列表 (保留用于单环境兼容性)
        self.skill_change_timer = 0  # 技能更换计时器 (保留用于单环境兼容性)
        self.current_high_level_reward_sum = 0.0 # 当前技能周期的累积奖励
        self.env_reward_sums = {}  # 用于存储每个环境ID的累积奖励，用于并行训练
        self.env_timers = {}  # 用于存储每个环境ID的技能计时器，用于并行训练
        
        # 新增：环境特定的状态跟踪
        self.env_team_skills = {}  # 各环境的当前团队技能
        self.env_agent_skills = {}  # 各环境的当前个体技能列表
        self.env_log_probs = {}  # 各环境的log probabilities
        self.env_hidden_states = {}  # 各环境的GRU隐藏状态
        
        # 预初始化32个并行环境的奖励累积和技能计时器(与config.num_envs=32对应)
        for i in range(32):
            self.env_reward_sums[i] = 0.0
            self.env_timers[i] = 0
            self.env_team_skills[i] = None
            self.env_agent_skills[i] = None
            self.env_log_probs[i] = None
            self.env_hidden_states[i] = None
        self.accumulated_rewards = 0.0  # 用于测试的累积奖励属性
        self.episode_rewards = []  # 记录每个完整episode的奖励

        # 用于记录整个episode的技能使用计数
        self.episode_team_skill_counts = {}
        # 将在第一次分配技能时根据实际智能体数量初始化
        self.episode_agent_skill_counts = [] 
        
        # 训练指标
        self.training_info = {
            'high_level_loss': [],
            'low_level_loss': [],
            'discriminator_loss': [],
            'team_skill_entropy': [],
            'agent_skill_entropy': [],
            'action_entropy': [],
            'episode_rewards': [],
            # 新增用于记录内在奖励组件和价值估计的列表
            'intrinsic_reward_env_component': [],
            'intrinsic_reward_team_disc_component': [],
            'intrinsic_reward_ind_disc_component': [],
            'intrinsic_reward_low_level_average': [], # 用于记录批次平均内在奖励
            'coordinator_state_value_mean': [],
            'coordinator_agent_value_mean': [],
            'discoverer_value_mean': []
        }
        
        # 用于减少高层缓冲区警告日志的计数器
        self.high_level_buffer_warning_counter = 0
        self.last_high_level_buffer_size = 0
        
        # 高层经验统计
        self.high_level_samples_total = 0        # 总收集高层样本数
        self.high_level_samples_by_env = {}      # 各环境贡献的样本数
        self.high_level_samples_by_reason = {'技能周期结束': 0, '环境终止': 0, '周期完成检测': 0}  # 收集原因统计
        
        # 高层经验收集增强
        self.env_last_contribution = {}          # 跟踪每个环境上次贡献高层样本的时间步
        self.force_high_level_collection = {}    # 强制采集标志，用于确保所有环境都能贡献样本
        self.env_reward_thresholds = {}          # 环境特定的奖励阈值
        
        # 记录内在奖励组成部分的累积值，用于统计分析
        self.cumulative_env_reward = 0.0
        self.cumulative_team_disc_reward = 0.0
        self.cumulative_ind_disc_reward = 0.0
        self.reward_component_counts = 0
        
        # Huber Loss自适应delta参数
        self.adaptive_coordinator_delta = getattr(config, 'huber_coordinator_delta', 1.0)
        self.adaptive_discoverer_delta = getattr(config, 'huber_discoverer_delta', 1.0)
        self.delta_update_count = 0
    
    def should_sync_update(self):
        """检查是否应该进行同步更新"""
        if not self.sync_mode:
            return False
        
        return self.samples_collected_this_round >= self.sync_batch_size
    
    def enable_data_collection(self):
        """启用数据收集"""
        self.collection_enabled = True
        main_logger.debug(f"数据收集已启用，策略版本: {self.policy_version}")
    
    def disable_data_collection(self):
        """禁用数据收集"""
        self.collection_enabled = False
        main_logger.debug(f"数据收集已禁用，策略版本: {self.policy_version}")
    
    def force_collect_pending_high_level_experiences(self):
        """
        在同步更新前强制收集所有未完成技能周期的高层经验
        确保高层缓冲区有足够的样本进行更新
        """
        pending_collections = 0
        
        for env_id in range(32):  # 假设最多32个并行环境
            timer = self.env_timers.get(env_id, 0)
            reward_sum = self.env_reward_sums.get(env_id, 0.0)
            
            # 如果该环境有未完成的技能周期且累积奖励不为0，强制收集
            if timer > 0 and timer < self.config.k - 1:
                main_logger.info(f"强制收集环境{env_id}的高层经验: timer={timer}, 累积奖励={reward_sum:.4f}")
                self.force_high_level_collection[env_id] = True
                pending_collections += 1
        
        if pending_collections > 0:
            main_logger.info(f"同步更新前强制收集了 {pending_collections} 个环境的pending高层经验")
        
        return pending_collections

    def sync_update(self):
        """
        改进的同步更新机制 - 基于HMASD论文算法
        确保高层经验收集不受低层缓冲区清空影响
        
        返回:
            update_info: 更新信息字典
        """
        # 1. 停止数据收集
        self.disable_data_collection()
        
        # 2. 记录同步更新信息
        samples_count = self.samples_collected_this_round
        main_logger.info(f"同步更新开始 - 收集了 {samples_count} 个样本，策略版本: {self.policy_version}")
        
        # 3. 【新增】强制收集所有pending的高层经验
        pending_count = self.force_collect_pending_high_level_experiences()
        
        # 4. 记录缓冲区状态
        high_level_buffer_size_before = len(self.high_level_buffer)
        low_level_buffer_size_before = len(self.low_level_buffer)
        main_logger.info(f"同步更新前缓冲区状态 - 高层: {high_level_buffer_size_before}, 低层: {low_level_buffer_size_before}")
        
        # 5. 【修改顺序】先更新高层策略（使用现有的高层经验）
        coordinator_loss, coordinator_policy_loss, coordinator_value_loss, team_skill_entropy, agent_skill_entropy, \
        mean_coord_state_val, mean_coord_agent_val, mean_high_level_reward = self.update_coordinator()
        
        # 6. 再更新低层策略（会清空低层缓冲区）
        discoverer_loss, discoverer_policy_loss, discoverer_value_loss, action_entropy, \
        avg_intrinsic_reward, avg_env_comp, avg_team_disc_comp, avg_ind_disc_comp, \
        avg_discoverer_val = self.update_discoverer()
        
        # 7. 最后更新判别器
        discriminator_loss = self.update_discriminators()
        
        # 8. 记录缓冲区状态变化
        high_level_buffer_size_after = len(self.high_level_buffer)
        low_level_buffer_size_after = len(self.low_level_buffer)
        main_logger.info(f"同步更新后缓冲区状态 - 高层: {high_level_buffer_size_after}, 低层: {low_level_buffer_size_after}")
        
        # 9. 重置同步状态
        self.policy_version += 1
        self.samples_collected_this_round = 0
        self.last_sync_step = self.global_step
        
        # 10. 重新启用数据收集
        self.enable_data_collection()
        
        # 11. 记录同步更新完成
        main_logger.info(f"同步更新完成 - 策略版本更新到: {self.policy_version}, 已重置样本计数")
        
        # 12. 构建更新信息
        update_info = {
            'sync_samples_collected': samples_count,
            'policy_version': self.policy_version,
            'is_sync_update': True,
            'pending_high_level_forced': pending_count,
            'discriminator_loss': discriminator_loss,
            'coordinator_loss': coordinator_loss,
            'coordinator_policy_loss': coordinator_policy_loss,
            'coordinator_value_loss': coordinator_value_loss,
            'discoverer_loss': discoverer_loss,
            'discoverer_policy_loss': discoverer_policy_loss,
            'discoverer_value_loss': discoverer_value_loss,
            'team_skill_entropy': team_skill_entropy,
            'agent_skill_entropy': agent_skill_entropy,
            'action_entropy': action_entropy,
            'avg_intrinsic_reward': avg_intrinsic_reward,
            'avg_env_comp': avg_env_comp,
            'avg_team_disc_comp': avg_team_disc_comp,
            'avg_ind_disc_comp': avg_ind_disc_comp,
            'mean_coord_state_val': mean_coord_state_val,
            'mean_coord_agent_val': mean_coord_agent_val,
            'avg_discoverer_val': avg_discoverer_val,
            'mean_high_level_reward': mean_high_level_reward
        }
        
        return update_info
    
    def reset_buffers(self):
        """重置所有经验缓冲区"""
        main_logger.info("重置所有经验缓冲区")
        self.high_level_buffer.clear()
        self.high_level_buffer_with_logprobs = []
        self.low_level_buffer.clear()
        self.state_skill_dataset.clear()
        
        # 重置计数器和累积值
        self.current_high_level_reward_sum = 0.0
        self.accumulated_rewards = 0.0
        self.skill_change_timer = 0
        self.high_level_buffer_warning_counter = 0
        self.last_high_level_buffer_size = 0
        
        # 重置环境特定的奖励累积字典和计时器字典
        self.env_reward_sums = {}
        self.env_timers = {}
        
        # 重置奖励组成部分的累积值
        self.cumulative_env_reward = 0.0
        self.cumulative_team_disc_reward = 0.0
        self.cumulative_ind_disc_reward = 0.0
        self.reward_component_counts = 0
        
        # 重置技能使用计数
        self.episode_team_skill_counts = {}
        self.episode_agent_skill_counts = []
    
    def select_action(self, observations, agent_skills=None, deterministic=False, env_id=0):
        """
        为所有智能体选择动作
        
        参数:
            observations: 所有智能体的观测 [n_agents, obs_dim]
            agent_skills: 所有智能体的技能 [n_agents]，如果为None则使用当前技能
            deterministic: 是否使用确定性策略
            env_id: 环境ID，用于多环境并行训练
            
        返回:
            actions: 所有智能体的动作 [n_agents, action_dim]
            action_logprobs: 所有智能体的动作对数概率 [n_agents]
        """
        if agent_skills is None:
            agent_skills = self.env_agent_skills.get(env_id, self.current_agent_skills)
            
        n_agents = observations.shape[0]
        actions = torch.zeros((n_agents, self.config.action_dim), device=self.device)
        action_logprobs = torch.zeros(n_agents, device=self.device)
        
        # 初始化或获取环境特定的GRU隐藏状态
        if env_id not in self.env_hidden_states or self.env_hidden_states[env_id] is None:
            self.skill_discoverer.init_hidden(batch_size=1)
            self.env_hidden_states[env_id] = self.skill_discoverer.actor_hidden
        else:
            self.skill_discoverer.actor_hidden = self.env_hidden_states[env_id]
        
        with torch.no_grad():
            for i in range(n_agents):
                obs = torch.FloatTensor(observations[i]).unsqueeze(0).to(self.device)
                skill = torch.tensor(agent_skills[i], device=self.device)
                
                action, action_logprob, _ = self.skill_discoverer(obs, skill, deterministic)
                
                actions[i] = action.squeeze(0)
                action_logprobs[i] = action_logprob.squeeze(0)
        
        # 保存更新后的GRU隐藏状态
        self.env_hidden_states[env_id] = self.skill_discoverer.actor_hidden
        
        return actions.cpu().numpy(), action_logprobs.cpu().numpy()
    
    def assign_skills(self, state, observations, deterministic=False):
        """
        为所有智能体分配技能
        
        参数:
            state: 全局状态 [state_dim]
            observations: 所有智能体的观测 [n_agents, obs_dim]
            deterministic: 是否使用确定性策略
            
        返回:
            team_skill: 团队技能索引
            agent_skills: 个体技能索引列表 [n_agents]
            log_probs: 包含团队技能和个体技能log probabilities的字典
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        obs_tensor = torch.FloatTensor(observations).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            team_skill, agent_skills, Z_logits, z_logits = self.skill_coordinator(
                state_tensor, obs_tensor, deterministic
            )
            
            # 计算log probabilities
            Z_dist = torch.distributions.Categorical(logits=Z_logits)
            Z_log_prob = Z_dist.log_prob(team_skill)
            
            z_log_probs = []
            n_agents_actual = agent_skills.size(1)
            for i in range(n_agents_actual):
                zi_dist = torch.distributions.Categorical(logits=z_logits[i])
                zi_log_prob = zi_dist.log_prob(agent_skills[0, i])
                z_log_probs.append(zi_log_prob.item())
            
            log_probs = {
                'team_log_prob': Z_log_prob.item(),
                'agent_log_probs': z_log_probs
            }
        
        return team_skill.item(), agent_skills.squeeze(0).cpu().numpy(), log_probs
    
    def step(self, state, observations, ep_t, deterministic=False, env_id=0):
        """
        执行一个环境步骤 - 修复版本
        
        关键修复：
        1. 统一技能周期判断逻辑
        2. 确保高层经验在skill完成时正确收集
        3. 避免重复收集和遗漏
        
        参数:
            state: 全局状态 [state_dim]
            observations: 所有智能体的观测 [n_agents, obs_dim]
            ep_t: 当前episode中的时间步
            deterministic: 是否使用确定性策略（用于评估）
            env_id: 环境ID，用于多环境并行训练
            
        返回:
            actions: 所有智能体的动作 [n_agents, action_dim]
            info: 额外信息，如当前技能
        """
        # 获取或初始化环境特定的状态
        current_team_skill = self.env_team_skills.get(env_id, self.current_team_skill)
        current_agent_skills = self.env_agent_skills.get(env_id, self.current_agent_skills)
        env_timer = self.env_timers.get(env_id, 0)
        
        # 【关键修复】统一技能周期判断逻辑
        # 使用环境特定的timer而不是全局ep_t来判断技能周期
        skill_cycle_completed = env_timer >= self.config.k
        need_skill_reassignment = skill_cycle_completed or current_team_skill is None
        
        main_logger.debug(f"[SKILL_CYCLE_DEBUG] 环境{env_id} 技能周期检查: "
                          f"ep_t={ep_t}, env_timer={env_timer}, k={self.config.k}, "
                          f"cycle_completed={skill_cycle_completed}, "
                          f"need_reassignment={need_skill_reassignment}, "
                          f"current_team_skill={current_team_skill}")
        
        if need_skill_reassignment:
            # 【修复】在重新分配技能之前，先收集上一周期的高层经验（如果存在）
            if skill_cycle_completed and current_team_skill is not None:
                old_reward_sum = self.env_reward_sums.get(env_id, 0.0)
                if old_reward_sum != 0.0:  # 只有当累积奖励不为0时才收集
                    observations_tensor = torch.FloatTensor(observations).to(self.device)
                    state_tensor = torch.FloatTensor(state).to(self.device)
                    
                    success = self._collect_high_level_experience(
                        env_id, state_tensor, observations_tensor,
                        current_team_skill, current_agent_skills, 
                        reason="技能周期完成"
                    )
                    
                    if success:
                        main_logger.debug(f"[STEP_FIX] 环境{env_id}在技能重分配前成功收集高层经验: "
                                        f"累积奖励={old_reward_sum:.4f}")
            
            # 分配新技能
            team_skill, agent_skills, log_probs = self.assign_skills(state, observations, deterministic)
            
            # 更新环境特定的状态
            self.env_team_skills[env_id] = team_skill
            self.env_agent_skills[env_id] = agent_skills
            self.env_log_probs[env_id] = log_probs
            self.env_timers[env_id] = 0  # 重置计时器
            
            # 重置环境特定的累积奖励
            self.env_reward_sums[env_id] = 0.0
            
            # 同时更新全局状态（用于兼容性）
            if env_id == 0:  # 只有环境0更新全局状态
                self.current_team_skill = team_skill
                self.current_agent_skills = agent_skills
                self.current_log_probs = log_probs
                self.skill_change_timer = 0
                self.current_high_level_reward_sum = 0.0
                self.accumulated_rewards = 0.0
            
            skill_changed = True
            
            # 【调试日志】记录技能分配结果
            main_logger.debug(f"[SKILL_ASSIGN_DEBUG] 环境{env_id} 技能已重新分配: "
                              f"team_skill={team_skill}, agent_skills={agent_skills}, "
                              f"timer重置: {env_timer}→0, "
                              f"奖励累积重置")

            # 更新技能使用计数（只有环境0更新全局计数）
            if env_id == 0:
                # 初始化 agent skill counts 列表（如果尚未初始化或智能体数量已更改）
                if not self.episode_agent_skill_counts or len(self.episode_agent_skill_counts) != len(agent_skills):
                    self.episode_agent_skill_counts = [{} for _ in range(len(agent_skills))]

                # 记录团队技能
                self.episode_team_skill_counts[team_skill] = self.episode_team_skill_counts.get(team_skill, 0) + 1
                # 记录个体技能
                for i, agent_skill in enumerate(agent_skills):
                    self.episode_agent_skill_counts[i][agent_skill] = self.episode_agent_skill_counts[i].get(agent_skill, 0) + 1
        else:
            self.env_timers[env_id] += 1
            # 同时更新全局计时器（用于兼容性）
            if env_id == 0:
                self.skill_change_timer += 1
            skill_changed = False
            main_logger.debug(f"环境{env_id}技能未更新: timer增加到{self.env_timers[env_id]}")
            
        # 选择动作，使用环境特定的技能
        actions, action_logprobs = self.select_action(observations, self.env_agent_skills[env_id], deterministic, env_id)
        
        info = {
            'team_skill': self.env_team_skills[env_id],
            'agent_skills': self.env_agent_skills[env_id],
            'action_logprobs': action_logprobs,
            'skill_changed': skill_changed,
            'skill_timer': self.env_timers[env_id],
            'log_probs': self.env_log_probs[env_id],
            'env_id': env_id
        }
        
        return actions, info
    
    def collect_episode(self, env, max_steps=1000):
        """
        收集完整episode的数据（episode-based训练模式）
        
        参数:
            env: 环境实例
            max_steps: 最大步数限制
            
        返回:
            episode_info: episode信息字典
        """
        if not self.episode_based_training:
            raise ValueError("collect_episode只能在episode_based_training模式下使用")
        
        episode_reward = 0.0
        episode_steps = 0
        episode_start_time = time.time()
        
        # 重置环境
        state, observations = env.reset()
        done = False
        ep_t = 0
        
        main_logger.info(f"开始收集Episode {self.episodes_collected + 1}")
        
        while not done and ep_t < max_steps:
            # 执行step，收集数据但不触发更新
            actions, info = self.step(state, observations, ep_t, deterministic=False, env_id=0)
            
            # 环境交互
            next_state, next_observations, rewards, dones = env.step(actions)
            
            # 存储经验（仅收集，不触发更新）
            success = self.store_transition(
                state, next_state, observations, next_observations,
                actions, rewards, dones, info['team_skill'], info['agent_skills'],
                info['action_logprobs'], info['log_probs'], info['skill_timer'], env_id=0
            )
            
            if not success:
                main_logger.warning(f"Episode {self.episodes_collected + 1}: 步骤 {ep_t} 存储经验失败")
            
            # 更新状态
            episode_reward += rewards if isinstance(rewards, (int, float)) else rewards.item()
            state, observations = next_state, next_observations
            done = dones if isinstance(dones, bool) else dones.any()
            ep_t += 1
            episode_steps += 1
            
            # 每100步记录一次进度
            if ep_t % 100 == 0:
                main_logger.debug(f"Episode {self.episodes_collected + 1}: 步骤 {ep_t}, 累积奖励: {episode_reward:.4f}")
        
        # Episode结束
        episode_duration = time.time() - episode_start_time
        self.episodes_collected += 1
        
        # 记录episode信息
        episode_info = {
            'episode_id': self.episodes_collected,
            'episode_reward': episode_reward,
            'episode_steps': episode_steps,
            'episode_duration': episode_duration,
            'termination_reason': 'done' if done else 'max_steps',
            'high_level_buffer_size': len(self.high_level_buffer),
            'low_level_buffer_size': len(self.low_level_buffer)
        }
        
        # 记录到训练信息
        self.training_info['episode_rewards'].append(episode_reward)
        
        # 记录到TensorBoard
        if hasattr(self, 'writer'):
            self.writer.add_scalar('Episodes/Reward', episode_reward, self.episodes_collected)
            self.writer.add_scalar('Episodes/Steps', episode_steps, self.episodes_collected)
            self.writer.add_scalar('Episodes/Duration', episode_duration, self.episodes_collected)
            self.writer.add_scalar('Episodes/BufferSizes/HighLevel', len(self.high_level_buffer), self.episodes_collected)
            self.writer.add_scalar('Episodes/BufferSizes/LowLevel', len(self.low_level_buffer), self.episodes_collected)
        
        main_logger.info(f"Episode {self.episodes_collected} 完成: "
                        f"奖励={episode_reward:.4f}, 步数={episode_steps}, "
                        f"耗时={episode_duration:.2f}s, 原因={episode_info['termination_reason']}")
        
        return episode_info
    
    def collect_rollout_step(self, envs, env_states, env_observations, env_dones=None):
        """
        收集单步rollout数据（rollout-based训练模式）
        
        参数:
            envs: 并行环境（SubprocVecEnv对象）
            env_states: 各环境的当前状态 [num_envs, state_dim]
            env_observations: 各环境的当前观测 [num_envs, n_agents, obs_dim]
            env_dones: 各环境的终止状态 [num_envs] (可选)
            
        返回:
            rollout_data: 单步rollout数据
        """
        if not self.rollout_based_training:
            raise ValueError("collect_rollout_step只能在rollout_based_training模式下使用")
        
        # 修复：使用 SubprocVecEnv 的 num_envs 属性而不是 len()
        num_envs = envs.num_envs
        actions_all = []
        infos_all = []
        
        # 为每个环境执行step
        for env_id in range(num_envs):
            actions, info = self.step(
                env_states[env_id], 
                env_observations[env_id], 
                self.steps_collected,  # 使用全局步数计数器
                deterministic=False, 
                env_id=env_id
            )
            actions_all.append(actions)
            infos_all.append(info)
        
        return {
            'actions': actions_all,
            'infos': infos_all,
            'step_count': self.steps_collected
        }
    
    def should_rollout_update(self):
        """
        判断是否应该进行rollout更新 - 修复版本
        
        返回:
            bool: 是否应该更新
        """
        if not self.rollout_based_training:
            return False
        
        # 【修复C1】使用正确的目标步数计算：rollout_length × num_parallel_envs
        target_steps = self.rollout_length * self.num_parallel_envs
        
        # 【修复C2】每100步记录一次进度，避免日志过多
        if self.steps_collected % 100 == 0 or not hasattr(self, '_last_debug_step'):
            progress_percent = (self.steps_collected / target_steps) * 100
            main_logger.info(f"[ROLLOUT_UPDATE_CHECK] 当前进度: {self.steps_collected}/{target_steps} "
                           f"({progress_percent:.1f}%) - rollout_length={self.rollout_length}, "
                           f"num_parallel_envs={self.num_parallel_envs}")
            self._last_debug_step = self.steps_collected
        
        should_update = self.steps_collected >= target_steps
        
        # 【修复C3】记录更新决策的详细信息
        if should_update:
            main_logger.info(f"🔄 满足rollout更新条件: 收集步数={self.steps_collected}, "
                           f"目标步数={target_steps}, 超出={self.steps_collected - target_steps}")
        
        # 【新增C4】如果接近目标但还没达到，记录详细状态
        elif self.steps_collected >= target_steps * 0.9:  # 90%以上时记录
            remaining = target_steps - self.steps_collected
            main_logger.info(f"⏳ 接近更新条件: 还需{remaining}步 "
                           f"({self.steps_collected}/{target_steps})")
        
        return should_update
    
    def rollout_update(self):
        """
        执行rollout-based批量更新（严格按照论文Algorithm 1实现）
        
        训练流程：
        1. 并行收集: 32环境 × 128步 = 4096样本 → B_h, B_l, D
        2. 训练阶段: PPO用全部数据训练15轮，判别器从D采样训练
        3. 清空缓冲区: B_h和B_l清空，D保留
        4. 重复循环
        
        返回:
            update_info: 更新信息字典
        """
        if not self.rollout_based_training:
            raise ValueError("rollout_update只能在rollout_based_training模式下使用")
        
        update_start_time = time.time()
        steps_for_update = self.steps_collected
        target_samples = self.rollout_length * self.num_parallel_envs
        
        main_logger.info(f"🔄 开始Rollout更新 #{self.rollout_count + 1}")
        main_logger.info(f"📊 数据统计: 收集步数={steps_for_update}, 目标样本={target_samples}, "
                        f"并行环境={self.num_parallel_envs}")
        
        # 【关键修复】在训练前强制收集所有pending的高层经验
        #main_logger.info("🔍 Rollout结束，强制收集所有pending高层经验...")
        #pending_collections = self._force_collect_all_pending_high_level_experiences()
        #main_logger.info(f"✅ 强制收集完成，新增 {pending_collections} 个高层经验")
        
        # 记录更新前的缓冲区状态
        high_level_size_before = len(self.high_level_buffer)
        low_level_size_before = len(self.low_level_buffer)
        state_skill_size_before = len(self.state_skill_dataset)
        
        # 【调试日志】详细记录缓冲区状态和高层经验收集情况
        main_logger.warning(f"[ROLLOUT_BUFFER_DEBUG] 更新前缓冲区详细状态:")
        main_logger.warning(f"   - B_h (高层): {high_level_size_before} (目标: {self.config.high_level_batch_size})")
        main_logger.warning(f"   - B_l (低层): {low_level_size_before} (目标: {self.config.batch_size})")
        main_logger.warning(f"   - D (判别器): {state_skill_size_before}")
        main_logger.warning(f"   - 高层样本统计: 总计={self.high_level_samples_total}, "
                           f"环境贡献={self.high_level_samples_by_env}, "
                           f"原因统计={self.high_level_samples_by_reason}")
        
        # 【调试日志】检查各环境的技能计时器状态
        env_timer_status = {}
        for env_id in range(self.num_parallel_envs):
            timer = self.env_timers.get(env_id, 0)
            reward_sum = self.env_reward_sums.get(env_id, 0.0)
            env_timer_status[env_id] = {'timer': timer, 'reward_sum': reward_sum}
        
        main_logger.warning(f"[ROLLOUT_TIMER_DEBUG] 各环境技能计时器状态: {env_timer_status}")
        
        # 验证数据收集的完整性
        if steps_for_update != self.rollout_length:
            main_logger.warning(f"⚠️ 收集步数({steps_for_update})与目标({self.rollout_length})不匹配")
        
        # 执行15轮PPO训练（严格按照论文设置）
        main_logger.info(f"🎯 开始{self.ppo_epochs}轮PPO训练（使用全部数据）")
        
        coordinator_losses = []
        discoverer_losses = []
        discriminator_losses = []
        
        for epoch in range(self.ppo_epochs):
            epoch_start_time = time.time()
            main_logger.debug(f"   轮次 {epoch + 1}/{self.ppo_epochs}")
            
            # 1. 更新高层策略（PPO，使用B_h全部数据）
            coordinator_info = self._rollout_update_coordinator()
            coordinator_losses.append(coordinator_info)
            
            # 2. 更新低层策略（PPO，使用B_l全部数据）
            discoverer_info = self._rollout_update_discoverer()
            discoverer_losses.append(discoverer_info)
            
            # 3. 更新判别器（监督学习，从D采样）
            discriminator_loss = self.update_discriminators()
            discriminator_losses.append(discriminator_loss)
            
            epoch_time = time.time() - epoch_start_time
            
            if epoch % 5 == 0 or epoch == self.ppo_epochs - 1:
                main_logger.debug(f"   轮次 {epoch + 1} 完成，耗时: {epoch_time:.3f}s")
        
        # 【关键】严格按照论文要求清空PPO缓冲区
        main_logger.info("🧹 清空PPO缓冲区（保持on-policy特性）")
        
        # 记录清空前的统计信息
        total_high_level_samples_used = high_level_size_before
        total_low_level_samples_used = low_level_size_before
        
        # 清空B_h和B_l（PPO要求）
        self.high_level_buffer.clear()
        self.low_level_buffer.clear()
        
        # D（判别器数据集）保留（监督学习可重复使用）
        # self.state_skill_dataset 不清空
        
        # 验证清空是否成功
        high_level_size_after = len(self.high_level_buffer)
        low_level_size_after = len(self.low_level_buffer)
        state_skill_size_after = len(self.state_skill_dataset)
        
        if high_level_size_after != 0 or low_level_size_after != 0:
            main_logger.error(f"❌ 缓冲区清空失败！B_h={high_level_size_after}, B_l={low_level_size_after}")
        else:
            main_logger.info(f"✅ PPO缓冲区清空成功")
        
        main_logger.info(f"📦 更新后缓冲区状态:")
        main_logger.info(f"   - B_h (高层): {high_level_size_before} → {high_level_size_after}")
        main_logger.info(f"   - B_l (低层): {low_level_size_before} → {low_level_size_after}")
        main_logger.info(f"   - D (判别器): {state_skill_size_before} → {state_skill_size_after} (保留)")
        
        # 【修复D1】重置rollout状态 - 确保完整重置
        steps_before_reset = self.steps_collected
        self.steps_collected = 0
        self.rollout_count += 1
        self.total_steps_collected += steps_for_update
        update_duration = time.time() - update_start_time
        
        # 【修复D2】验证步数重置是否成功
        if self.steps_collected != 0:
            main_logger.error(f"❌ 步数重置失败！steps_collected={self.steps_collected}")
        else:
            main_logger.debug(f"✅ 步数重置成功: {steps_before_reset} → {self.steps_collected}")
        
        # 【修复D3】重置环境相关的计数器和状态
        # 重置所有环境的技能计时器和奖励累积
        for env_id in range(self.num_parallel_envs):
            if env_id in self.env_timers:
                self.env_timers[env_id] = 0
            if env_id in self.env_reward_sums:
                self.env_reward_sums[env_id] = 0.0
        
        main_logger.debug(f"🔄 所有环境状态已重置: timers和reward_sums已清零")
        
        # 计算平均损失
        avg_coordinator_info = self._average_update_info(coordinator_losses)
        avg_discoverer_info = self._average_update_info(discoverer_losses)
        avg_discriminator_loss = np.mean(discriminator_losses) if discriminator_losses else 0.0
        
        # 计算样本使用效率
        samples_per_second = target_samples / update_duration if update_duration > 0 else 0
        
        main_logger.info(f"🎉 Rollout更新 #{self.rollout_count} 完成")
        main_logger.info(f"⏱️ 耗时: {update_duration:.2f}s, 效率: {samples_per_second:.0f} 样本/秒")
        main_logger.info(f"📈 累计: rollouts={self.rollout_count}, 总步数={self.total_steps_collected:,}")
        
        # 构建详细的更新信息
        update_info = {
            'update_type': 'rollout_batch',
            'rollout_count': self.rollout_count,
            'steps_used': steps_for_update,
            'target_samples': target_samples,
            'total_steps': self.total_steps_collected,
            'ppo_epochs': self.ppo_epochs,
            'num_parallel_envs': self.num_parallel_envs,
            'update_duration': update_duration,
            'samples_per_second': samples_per_second,
            'buffer_changes': {
                'high_level': (high_level_size_before, high_level_size_after),
                'low_level': (low_level_size_before, low_level_size_after),
                'state_skill': (state_skill_size_before, state_skill_size_after)
            },
            'samples_used': {
                'high_level': total_high_level_samples_used,
                'low_level': total_low_level_samples_used,
                'discriminator': state_skill_size_before
            },
            'coordinator': avg_coordinator_info,
            'discoverer': avg_discoverer_info,
            'discriminator': {'discriminator_loss': avg_discriminator_loss},
            'buffer_cleared': high_level_size_after == 0 and low_level_size_after == 0,
            'algorithm_compliance': {
                'ppo_epochs_executed': self.ppo_epochs,
                'buffers_cleared': high_level_size_after == 0 and low_level_size_after == 0,
                'discriminator_preserved': state_skill_size_after > 0
            }
        }
        
        # 记录到TensorBoard
        if hasattr(self, 'writer'):
            # 基本指标
            self.writer.add_scalar('Rollout/UpdateDuration', update_duration, self.rollout_count)
            self.writer.add_scalar('Rollout/StepsUsed', steps_for_update, self.rollout_count)
            self.writer.add_scalar('Rollout/TotalSteps', self.total_steps_collected, self.rollout_count)
            self.writer.add_scalar('Rollout/SamplesPerSecond', samples_per_second, self.rollout_count)
            
            # 缓冲区状态
            self.writer.add_scalar('Rollout/BufferSizeBefore/HighLevel', high_level_size_before, self.rollout_count)
            self.writer.add_scalar('Rollout/BufferSizeBefore/LowLevel', low_level_size_before, self.rollout_count)
            self.writer.add_scalar('Rollout/BufferSizeAfter/HighLevel', high_level_size_after, self.rollout_count)
            self.writer.add_scalar('Rollout/BufferSizeAfter/LowLevel', low_level_size_after, self.rollout_count)
            
            # 算法合规性
            self.writer.add_scalar('Rollout/Algorithm/BuffersCleared', 
                                  1.0 if update_info['buffer_cleared'] else 0.0, self.rollout_count)
            self.writer.add_scalar('Rollout/Algorithm/PPOEpochs', self.ppo_epochs, self.rollout_count)
            
            # 损失记录
            if avg_coordinator_info:
                self.writer.add_scalar('Rollout/AvgCoordinatorLoss', 
                                      avg_coordinator_info.get('coordinator_loss', 0), self.rollout_count)
                self.writer.add_scalar('Rollout/AvgCoordinatorPolicyLoss', 
                                      avg_coordinator_info.get('coordinator_policy_loss', 0), self.rollout_count)
                self.writer.add_scalar('Rollout/AvgCoordinatorValueLoss', 
                                      avg_coordinator_info.get('coordinator_value_loss', 0), self.rollout_count)
            
            if avg_discoverer_info:
                self.writer.add_scalar('Rollout/AvgDiscovererLoss', 
                                      avg_discoverer_info.get('discoverer_loss', 0), self.rollout_count)
                self.writer.add_scalar('Rollout/AvgDiscovererPolicyLoss', 
                                      avg_discoverer_info.get('discoverer_policy_loss', 0), self.rollout_count)
                self.writer.add_scalar('Rollout/AvgDiscovererValueLoss', 
                                      avg_discoverer_info.get('discoverer_value_loss', 0), self.rollout_count)
            
            self.writer.add_scalar('Rollout/AvgDiscriminatorLoss', avg_discriminator_loss, self.rollout_count)
        
        return update_info
    
    def _force_collect_all_pending_high_level_experiences(self):
        """
        强制收集所有环境中pending的高层经验
        解决rollout结束时部分环境技能周期未完成的问题
        
        返回:
            int: 新收集的高层经验数量
        """
        pending_collections = 0
        
        for env_id in range(self.num_parallel_envs):
            timer = self.env_timers.get(env_id, 0)
            reward_sum = self.env_reward_sums.get(env_id, 0.0)
            
            # 如果该环境有未完成的技能周期（timer > 0且还没到k-1），强制收集
            if timer > 0:
                main_logger.info(f"🔧 强制收集环境{env_id}的pending高层经验: "
                               f"timer={timer}/{self.config.k-1}, 累积奖励={reward_sum:.4f}")
                
                # 创建一个虚拟的高层经验（使用当前累积的奖励）
                # 注意：这里我们需要模拟store_transition中的高层经验收集逻辑
                if reward_sum != 0.0 or timer >= self.config.k // 2:  # 只收集有意义的经验
                    # 获取环境的当前技能状态
                    team_skill = self.env_team_skills.get(env_id, 0)
                    agent_skills = self.env_agent_skills.get(env_id, [0] * self.config.n_agents)
                    
                    # 创建虚拟的状态和观测（使用零向量作为占位符）
                    state_tensor = torch.zeros(self.config.state_dim, device=self.device)
                    team_skill_tensor = torch.tensor(team_skill, device=self.device)
                    observations_tensor = torch.zeros(self.config.n_agents, self.config.obs_dim, device=self.device)
                    agent_skills_tensor = torch.tensor(agent_skills[:self.config.n_agents], device=self.device)
                    
                    # 创建高层经验元组
                    high_level_experience = (
                        state_tensor,                  # 全局状态s
                        team_skill_tensor,             # 团队技能Z
                        observations_tensor,           # 所有智能体观测o
                        agent_skills_tensor,           # 所有个体技能z
                        torch.tensor(reward_sum, device=self.device) # 累积奖励
                    )
                    
                    # 存储高层经验
                    self.high_level_buffer.push(high_level_experience)
                    
                    # 更新统计信息
                    self.high_level_samples_total += 1
                    self.high_level_samples_by_env[env_id] = self.high_level_samples_by_env.get(env_id, 0) + 1
                    self.high_level_samples_by_reason['Rollout结束强制收集'] = self.high_level_samples_by_reason.get('Rollout结束强制收集', 0) + 1
                    
                    pending_collections += 1
                    
                    main_logger.info(f"✅ 环境{env_id}高层经验已强制收集: "
                                   f"累积奖励={reward_sum:.4f}, 新缓冲区大小={len(self.high_level_buffer)}")
                
                # 重置该环境的状态
                self.env_reward_sums[env_id] = 0.0
                self.env_timers[env_id] = 0
        
        if pending_collections > 0:
            main_logger.info(f"🎯 Rollout结束强制收集总结: 新增 {pending_collections} 个高层经验, "
                           f"高层缓冲区: {len(self.high_level_buffer)}")
        
        return pending_collections
    
    def _rollout_update_coordinator(self):
        """rollout模式下的高层策略更新（使用全部数据，不采样）"""
        if len(self.high_level_buffer) == 0:
            return self._get_default_coordinator_info()
        
        # 使用全部数据（num_mini_batch=1的含义）
        return self._update_coordinator_with_all_buffer()
    
    def _rollout_update_discoverer(self):
        """rollout模式下的低层策略更新（使用全部数据，不采样）"""
        if len(self.low_level_buffer) == 0:
            return self._get_default_discoverer_info()
        
        # 使用全部数据进行训练
        return self._update_discoverer_with_all_buffer()
    
    def _update_coordinator_with_all_buffer(self):
        """使用缓冲区中的全部数据更新协调器"""
        # 复用现有的update_coordinator逻辑
        coordinator_loss, coordinator_policy_loss, coordinator_value_loss, team_skill_entropy, agent_skill_entropy, \
        mean_coord_state_val, mean_coord_agent_val, mean_high_level_reward = self.update_coordinator()
        
        return {
            'coordinator_loss': coordinator_loss,
            'coordinator_policy_loss': coordinator_policy_loss,
            'coordinator_value_loss': coordinator_value_loss,
            'team_skill_entropy': team_skill_entropy,
            'agent_skill_entropy': agent_skill_entropy,
            'mean_coord_state_val': mean_coord_state_val,
            'mean_coord_agent_val': mean_coord_agent_val,
            'mean_high_level_reward': mean_high_level_reward
        }
    
    def _update_discoverer_with_all_buffer(self):
        """使用缓冲区中的全部数据更新发现器"""
        # 复用现有的update_discoverer逻辑
        discoverer_loss, discoverer_policy_loss, discoverer_value_loss, action_entropy, \
        avg_intrinsic_reward, avg_env_comp, avg_team_disc_comp, avg_ind_disc_comp, \
        avg_discoverer_val = self.update_discoverer()
        
        return {
            'discoverer_loss': discoverer_loss,
            'discoverer_policy_loss': discoverer_policy_loss,
            'discoverer_value_loss': discoverer_value_loss,
            'action_entropy': action_entropy,
            'avg_intrinsic_reward': avg_intrinsic_reward,
            'avg_env_comp': avg_env_comp,
            'avg_team_disc_comp': avg_team_disc_comp,
            'avg_ind_disc_comp': avg_ind_disc_comp,
            'avg_discoverer_val': avg_discoverer_val
        }
    
    def _average_update_info(self, info_list):
        """计算多次更新信息的平均值"""
        if not info_list:
            return {}
        
        # 过滤掉空的info
        valid_infos = [info for info in info_list if info and isinstance(info, dict)]
        if not valid_infos:
            return {}
        
        avg_info = {}
        for key in valid_infos[0].keys():
            if isinstance(valid_infos[0][key], (int, float)):
                avg_info[key] = np.mean([info.get(key, 0) for info in valid_infos])
            else:
                avg_info[key] = valid_infos[0][key]  # 保留非数值类型的第一个值
        
        return avg_info
    
    def step_rollout_counter(self):
        """
        增加rollout步数计数器
        在每次环境step后调用
        """
        if self.rollout_based_training:
            self.steps_collected += 1
    
    def should_update(self):
        """
        判断是否应该进行批量训练更新
        
        返回:
            bool: 是否应该更新
        """
        if not self.episode_based_training:
            return False
        
        # 条件1：收集了足够的episodes
        episodes_since_last_update = self.episodes_collected - self.last_update_episode
        enough_episodes = episodes_since_last_update >= self.update_frequency
        
        # 条件2：达到最少episode要求
        min_episodes_met = self.episodes_collected >= self.min_episodes_for_update
        
        # 条件3：缓冲区有足够的数据
        enough_high_level_data = len(self.high_level_buffer) >= self.min_high_level_samples
        enough_low_level_data = len(self.low_level_buffer) >= self.min_low_level_samples
        
        should_update = enough_episodes and min_episodes_met and enough_high_level_data and enough_low_level_data
        
        if episodes_since_last_update > 0 and episodes_since_last_update % 5 == 0:  # 每5个episode记录一次
            main_logger.debug(f"更新检查: episodes_since_last={episodes_since_last_update}, "
                             f"enough_episodes={enough_episodes}, min_episodes_met={min_episodes_met}, "
                             f"enough_high_level={enough_high_level_data}({len(self.high_level_buffer)}/{self.min_high_level_samples}), "
                             f"enough_low_level={enough_low_level_data}({len(self.low_level_buffer)}/{self.min_low_level_samples}), "
                             f"should_update={should_update}")
        
        return should_update
    
    def batch_update(self):
        """
        批量更新所有网络（严格按照论文Algorithm 1的episode-based模式）
        
        返回:
            update_info: 更新信息字典
        """
        if not self.episode_based_training:
            raise ValueError("batch_update只能在episode_based_training模式下使用")
        
        update_start_time = time.time()
        episodes_for_update = self.episodes_collected - self.last_update_episode
        
        main_logger.info(f"开始批量更新 - Episode {self.episodes_collected}, "
                        f"使用过去 {episodes_for_update} 个episodes的数据")
        
        # 记录更新前的缓冲区状态
        high_level_size_before = len(self.high_level_buffer)
        low_level_size_before = len(self.low_level_buffer)
        state_skill_size_before = len(self.state_skill_dataset)
        
        main_logger.info(f"更新前缓冲区状态 - 高层: {high_level_size_before}, "
                        f"低层: {low_level_size_before}, 判别器: {state_skill_size_before}")
        
        # 1. 更新高层策略（PPO + 清空缓冲区）
        coordinator_info = self.update_coordinator_batch()
        
        # 2. 更新低层策略（PPO + 清空缓冲区）  
        discoverer_info = self.update_discoverer_batch()
        
        # 3. 更新判别器（监督学习，保留部分数据）
        discriminator_info = self.update_discriminators_batch()
        
        # 记录更新后的缓冲区状态
        high_level_size_after = len(self.high_level_buffer)
        low_level_size_after = len(self.low_level_buffer) 
        state_skill_size_after = len(self.state_skill_dataset)
        
        # 更新状态
        self.last_update_episode = self.episodes_collected
        update_duration = time.time() - update_start_time
        
        main_logger.info(f"批量更新完成 - 耗时: {update_duration:.2f}s, "
                        f"缓冲区变化: 高层({high_level_size_before}→{high_level_size_after}), "
                        f"低层({low_level_size_before}→{low_level_size_after}), "
                        f"判别器({state_skill_size_before}→{state_skill_size_after})")
        
        # 构建更新信息
        update_info = {
            'update_type': 'episode_batch',
            'episodes_used': episodes_for_update,
            'total_episodes': self.episodes_collected,
            'update_duration': update_duration,
            'buffer_changes': {
                'high_level': (high_level_size_before, high_level_size_after),
                'low_level': (low_level_size_before, low_level_size_after),
                'state_skill': (state_skill_size_before, state_skill_size_after)
            },
            'coordinator': coordinator_info,
            'discoverer': discoverer_info,
            'discriminator': discriminator_info
        }
        
        # 记录到TensorBoard
        if hasattr(self, 'writer'):
            self.writer.add_scalar('Updates/Duration', update_duration, self.episodes_collected)
            self.writer.add_scalar('Updates/EpisodesUsed', episodes_for_update, self.episodes_collected)
            self.writer.add_scalar('Updates/BufferSizeBefore/HighLevel', high_level_size_before, self.episodes_collected)
            self.writer.add_scalar('Updates/BufferSizeBefore/LowLevel', low_level_size_before, self.episodes_collected)
            self.writer.add_scalar('Updates/BufferSizeAfter/HighLevel', high_level_size_after, self.episodes_collected)
            self.writer.add_scalar('Updates/BufferSizeAfter/LowLevel', low_level_size_after, self.episodes_collected)
        
        return update_info
    
    def update_coordinator_batch(self):
        """批量更新高层策略（严格PPO，episode-based）"""
        if len(self.high_level_buffer) < self.min_high_level_samples:
            main_logger.warning(f"高层缓冲区数据不足，需要{self.min_high_level_samples}个样本，"
                               f"但只有{len(self.high_level_buffer)}个。跳过更新。")
            return self._get_default_coordinator_info()
        
        # 使用所有可用数据进行训练（而不是采样）
        all_data = list(self.high_level_buffer.buffer)
        actual_batch_size = len(all_data)
        
        main_logger.info(f"高层策略批量更新 - 使用全部 {actual_batch_size} 个样本")
        
        # 执行PPO更新（使用现有的update_coordinator逻辑，但使用全部数据）
        update_info = self._update_coordinator_with_all_data(all_data)
        
        # 【关键】清空缓冲区（PPO要求）
        buffer_size_before = len(self.high_level_buffer)
        self.high_level_buffer.clear()
        self.high_level_buffer_with_logprobs = []
        
        main_logger.info(f"高层策略更新完成，缓冲区已清空: {buffer_size_before}→0 (符合PPO on-policy要求)")
        
        update_info['samples_used'] = actual_batch_size
        update_info['buffer_cleared'] = True
        
        return update_info
    
    def update_discoverer_batch(self):
        """批量更新低层策略（严格PPO，episode-based）"""
        if len(self.low_level_buffer) < self.min_low_level_samples:
            main_logger.warning(f"低层缓冲区数据不足，需要{self.min_low_level_samples}个样本，"
                               f"但只有{len(self.low_level_buffer)}个。跳过更新。")
            return self._get_default_discoverer_info()
        
        # 使用所有可用数据进行训练
        actual_batch_size = len(self.low_level_buffer)
        main_logger.info(f"低层策略批量更新 - 使用全部 {actual_batch_size} 个样本")
        
        # 执行PPO更新（使用现有的update_discoverer逻辑）
        update_info = self._update_discoverer_with_all_data()
        
        # 【关键】清空缓冲区（PPO要求）- 这个已经在原有的update_discoverer中实现了
        
        update_info['samples_used'] = actual_batch_size
        update_info['buffer_cleared'] = True
        
        return update_info
    
    def update_discriminators_batch(self):
        """批量更新判别器（监督学习，可以保留部分数据）"""
        if len(self.state_skill_dataset) < self.config.batch_size:
            main_logger.warning(f"判别器数据集不足，需要{self.config.batch_size}个样本，"
                               f"但只有{len(self.state_skill_dataset)}个。跳过更新。")
            return {'discriminator_loss': 0.0, 'samples_used': 0}
        
        # 判别器使用监督学习，可以多次使用数据，因此不需要清空
        discriminator_loss = self.update_discriminators()
        
        return {
            'discriminator_loss': discriminator_loss,
            'samples_used': len(self.state_skill_dataset),
            'note': '判别器使用监督学习，数据集未清空'
        }
    
    def _update_coordinator_with_all_data(self, all_data):
        """使用所有高层数据更新协调器（内部方法）"""
        # 复用现有的update_coordinator逻辑，但传入所有数据
        # 这里暂时返回现有update_coordinator的结果
        coordinator_loss, coordinator_policy_loss, coordinator_value_loss, team_skill_entropy, agent_skill_entropy, \
        mean_coord_state_val, mean_coord_agent_val, mean_high_level_reward = self.update_coordinator()
        
        return {
            'coordinator_loss': coordinator_loss,
            'coordinator_policy_loss': coordinator_policy_loss,
            'coordinator_value_loss': coordinator_value_loss,
            'team_skill_entropy': team_skill_entropy,
            'agent_skill_entropy': agent_skill_entropy,
            'mean_coord_state_val': mean_coord_state_val,
            'mean_coord_agent_val': mean_coord_agent_val,
            'mean_high_level_reward': mean_high_level_reward
        }
    
    def _update_discoverer_with_all_data(self):
        """使用所有低层数据更新发现器（内部方法）"""
        # 复用现有的update_discoverer逻辑
        discoverer_loss, discoverer_policy_loss, discoverer_value_loss, action_entropy, \
        avg_intrinsic_reward, avg_env_comp, avg_team_disc_comp, avg_ind_disc_comp, \
        avg_discoverer_val = self.update_discoverer()
        
        return {
            'discoverer_loss': discoverer_loss,
            'discoverer_policy_loss': discoverer_policy_loss,
            'discoverer_value_loss': discoverer_value_loss,
            'action_entropy': action_entropy,
            'avg_intrinsic_reward': avg_intrinsic_reward,
            'avg_env_comp': avg_env_comp,
            'avg_team_disc_comp': avg_team_disc_comp,
            'avg_ind_disc_comp': avg_ind_disc_comp,
            'avg_discoverer_val': avg_discoverer_val
        }
    
    def _get_default_coordinator_info(self):
        """获取默认的协调器更新信息（当跳过更新时）"""
        return {
            'coordinator_loss': 0.0,
            'coordinator_policy_loss': 0.0,
            'coordinator_value_loss': 0.0,
            'team_skill_entropy': 0.0,
            'agent_skill_entropy': 0.0,
            'mean_coord_state_val': 0.0,
            'mean_coord_agent_val': 0.0,
            'mean_high_level_reward': 0.0,
            'samples_used': 0,
            'buffer_cleared': False,
            'skipped': True
        }
    
    def _get_default_discoverer_info(self):
        """获取默认的发现器更新信息（当跳过更新时）"""
        return {
            'discoverer_loss': 0.0,
            'discoverer_policy_loss': 0.0,
            'discoverer_value_loss': 0.0,
            'action_entropy': 0.0,
            'avg_intrinsic_reward': 0.0,
            'avg_env_comp': 0.0,
            'avg_team_disc_comp': 0.0,
            'avg_ind_disc_comp': 0.0,
            'avg_discoverer_val': 0.0,
            'samples_used': 0,
            'buffer_cleared': False,
            'skipped': True
        }

    def _collect_high_level_experience(self, env_id, state_tensor, observations_tensor, 
                                     team_skill, agent_skills, reason="技能周期结束"):
        """
        统一的高层经验收集入口
        
        参数:
            env_id: 环境ID
            state_tensor: 全局状态张量
            observations_tensor: 所有智能体观测张量
            team_skill: 团队技能索引
            agent_skills: 个体技能索引列表
            reason: 收集原因，用于日志记录
            
        返回:
            bool: 是否成功收集
        """
        # 获取当前环境的累积奖励
        env_accumulated_reward = self.env_reward_sums.get(env_id, 0.0)
        
        # 创建高层经验元组
        team_skill_tensor = torch.tensor(team_skill, device=self.device)
        agent_skills_tensor = torch.tensor(agent_skills, device=self.device)
        
        high_level_experience = (
            state_tensor,                                                    # 全局状态s
            team_skill_tensor,                                               # 团队技能Z
            observations_tensor,                                             # 所有智能体观测o
            agent_skills_tensor,                                             # 所有个体技能z
            torch.tensor(env_accumulated_reward, device=self.device)         # 累积奖励
        )
        
        # 存储高层经验
        self.high_level_buffer.push(high_level_experience)
        
        # 更新统计信息
        self.high_level_samples_total += 1
        self.high_level_samples_by_env[env_id] = self.high_level_samples_by_env.get(env_id, 0) + 1
        self.high_level_samples_by_reason[reason] = self.high_level_samples_by_reason.get(reason, 0) + 1
        
        # 更新环境最后贡献时间
        self.env_last_contribution[env_id] = self.global_step
        
        # 重置强制收集标志
        if env_id in self.force_high_level_collection:
            self.force_high_level_collection[env_id] = False
        
        # 记录成功收集的info级别日志
        current_buffer_size = len(self.high_level_buffer)
        main_logger.debug(f"✅ 高层经验收集成功: 环境ID={env_id}, step={self.global_step}, "
                        f"缓冲区大小: {current_buffer_size}, 累积奖励: {env_accumulated_reward:.4f}, "
                        f"原因: {reason}")
        
        # 重置该环境的奖励累积和技能计时器
        self.env_reward_sums[env_id] = 0.0
        self.env_timers[env_id] = 0
        
        return True

    def store_high_level_transition(self, state, team_skill, observations, agent_skills, 
                                   accumulated_reward, skill_log_probs=None, worker_id=0):
        """
        存储高层经验（专门用于多线程训练）
        
        参数:
            state: 全局状态 [state_dim]
            team_skill: 团队技能索引
            observations: 所有智能体的观测 [n_agents, obs_dim]  
            agent_skills: 个体技能索引列表 [n_agents]
            accumulated_reward: k步累积奖励
            skill_log_probs: 技能的log probabilities字典
            worker_id: worker ID（用作env_id）
            
        返回:
            bool: 是否成功存储
        """
        try:
            # 转换为tensor格式
            state_tensor = torch.FloatTensor(state).to(self.device)
            team_skill_tensor = torch.tensor(team_skill, device=self.device)
            observations_tensor = torch.FloatTensor(observations).to(self.device)
            agent_skills_tensor = torch.tensor(agent_skills, device=self.device)
            
            # 创建高层经验元组
            high_level_experience = (
                state_tensor,                                                    # 全局状态s
                team_skill_tensor,                                               # 团队技能Z
                observations_tensor,                                             # 所有智能体观测o
                agent_skills_tensor,                                             # 所有个体技能z
                torch.tensor(accumulated_reward, device=self.device)             # 累积奖励
            )
            
            # 存储高层经验
            self.high_level_buffer.push(high_level_experience)
            
            # 更新统计信息
            self.high_level_samples_total += 1
            self.high_level_samples_by_env[worker_id] = self.high_level_samples_by_env.get(worker_id, 0) + 1
            self.high_level_samples_by_reason['多线程存储'] = self.high_level_samples_by_reason.get('多线程存储', 0) + 1
            
            # 存储带log probabilities的经验
            if skill_log_probs is not None:
                self.high_level_buffer_with_logprobs.append({
                    'state': state_tensor.clone(),
                    'team_skill': team_skill,
                    'observations': observations_tensor.clone(),
                    'agent_skills': agent_skills_tensor.clone(),
                    'reward': accumulated_reward,
                    'team_log_prob': skill_log_probs.get('team_log_prob', 0.0),
                    'agent_log_probs': skill_log_probs.get('agent_log_probs', [0.0] * len(agent_skills))
                })
                
                # 保持缓冲区大小不超过config.buffer_size
                if len(self.high_level_buffer_with_logprobs) > self.config.buffer_size:
                    self.high_level_buffer_with_logprobs = self.high_level_buffer_with_logprobs[-self.config.buffer_size:]
            
            main_logger.debug(f"高层经验存储成功: worker_id={worker_id}, 累积奖励={accumulated_reward:.4f}, "
                            f"缓冲区大小={len(self.high_level_buffer)}")
            
            return True
            
        except Exception as e:
            main_logger.error(f"存储高层经验失败: {e}")
            return False
    
    def store_low_level_transition(self, state, next_state, observations, next_observations,
                                 actions, rewards, dones, team_skill, agent_skills, 
                                 action_logprobs, skill_log_probs=None, worker_id=0):
        """
        存储低层经验（专门用于多线程训练）
        
        参数:
            state: 全局状态 [state_dim]
            next_state: 下一全局状态 [state_dim] 
            observations: 所有智能体的观测 [n_agents, obs_dim]
            next_observations: 所有智能体的下一观测 [n_agents, obs_dim]
            actions: 所有智能体的动作 [n_agents, action_dim]
            rewards: 环境奖励
            dones: 是否结束
            team_skill: 团队技能索引
            agent_skills: 个体技能索引列表 [n_agents]
            action_logprobs: 动作对数概率 [n_agents]
            skill_log_probs: 技能的log probabilities字典
            worker_id: worker ID（用作env_id）
            
        返回:
            bool: 是否成功存储
        """
        try:
            n_agents = len(agent_skills)
            state_tensor = torch.FloatTensor(state).to(self.device)
            next_state_tensor = torch.FloatTensor(next_state).to(self.device)
            team_skill_tensor = torch.tensor(team_skill, device=self.device)
            
            # 确保rewards是数值类型
            current_reward = rewards if isinstance(rewards, (int, float)) else rewards.item()
            
            # 计算团队技能判别器输出
            with torch.no_grad():
                team_disc_logits = self.team_discriminator(next_state_tensor.unsqueeze(0))
                team_disc_log_probs = F.log_softmax(team_disc_logits, dim=-1)
                team_skill_log_prob = team_disc_log_probs[0, team_skill]
            
            # 为每个智能体存储低层经验
            for i in range(n_agents):
                obs = torch.FloatTensor(observations[i]).to(self.device)
                next_obs = torch.FloatTensor(next_observations[i]).to(self.device)
                action = torch.FloatTensor(actions[i]).to(self.device)
                done = dones if isinstance(dones, bool) else dones[i] if isinstance(dones, list) else dones
                
                # 计算个体技能判别器输出
                with torch.no_grad():
                    agent_disc_logits = self.individual_discriminator(
                        next_obs.unsqueeze(0), 
                        team_skill_tensor
                    )
                    agent_disc_log_probs = F.log_softmax(agent_disc_logits, dim=-1)
                    agent_skill_log_prob = agent_disc_log_probs[0, agent_skills[i]]
                    
                # 计算低层奖励（Eq. 4）及其组成部分
                env_reward_component = self.config.lambda_e * current_reward
                team_disc_component = self.config.lambda_D * team_skill_log_prob.item()
                ind_disc_component = self.config.lambda_d * agent_skill_log_prob.item()
                
                intrinsic_reward = env_reward_component + team_disc_component + ind_disc_component
                
                # 存储低层经验
                low_level_experience = (
                    state_tensor,                           # 全局状态s
                    team_skill_tensor,                      # 团队技能Z
                    obs,                                    # 智能体观测o_i
                    torch.tensor(agent_skills[i], device=self.device),  # 个体技能z_i
                    action,                                 # 动作a_i
                    torch.tensor(intrinsic_reward, device=self.device),  # 总内在奖励r_i
                    torch.tensor(done, dtype=torch.float, device=self.device),  # 是否结束
                    torch.tensor(action_logprobs[i], device=self.device),  # 动作对数概率
                    torch.tensor(env_reward_component, device=self.device), # 环境奖励部分
                    torch.tensor(team_disc_component, device=self.device),  # 团队判别器部分
                    torch.tensor(ind_disc_component, device=self.device)   # 个体判别器部分
                )
                self.low_level_buffer.push(low_level_experience)
            
            # 存储技能判别器训练数据
            observations_tensor = torch.FloatTensor(next_observations).to(self.device)
            agent_skills_tensor = torch.tensor(agent_skills, device=self.device)
            self.state_skill_dataset.push(
                next_state_tensor,
                team_skill_tensor,
                observations_tensor,
                agent_skills_tensor
            )
            
            main_logger.debug(f"低层经验存储成功: worker_id={worker_id}, n_agents={n_agents}, "
                            f"奖励={current_reward:.4f}, 缓冲区大小={len(self.low_level_buffer)}")
            
            return True
            
        except Exception as e:
            main_logger.error(f"存储低层经验失败: {e}")
            return False

    def store_transition(self, state, next_state, observations, next_observations,
                         actions, rewards, dones, team_skill, agent_skills, action_logprobs, log_probs=None, 
                         skill_timer_for_env=None, env_id=0):
        """
        存储环境交互经验（支持同步训练）
        
        参数:
            state: 全局状态 [state_dim]
            next_state: 下一全局状态 [state_dim]
            observations: 所有智能体的观测 [n_agents, obs_dim]
            next_observations: 所有智能体的下一观测 [n_agents, obs_dim]
            actions: 所有智能体的动作 [n_agents, action_dim]
            rewards: 环境奖励
            dones: 是否结束 [n_agents]
            team_skill: 团队技能索引
            agent_skills: 个体技能索引列表 [n_agents]
            action_logprobs: 动作对数概率 [n_agents]
            log_probs: 技能的log probabilities字典，包含'team_log_prob'和'agent_log_probs'
            skill_timer_for_env: 当前环境的技能计时器值，用于多环境并行训练
            env_id: 环境ID，用于多环境并行训练
            
        返回:
            bool: 是否成功存储（同步模式下可能拒绝存储）
        """
        # 修复：分离低层和高层经验的同步控制
        # Episode-based模式下始终允许数据收集
        # 同步模式下才受collection_enabled控制
        low_level_collection_allowed = True
        if not self.episode_based_training and hasattr(self, 'sync_mode') and self.sync_mode and not self.collection_enabled:
            low_level_collection_allowed = False
            main_logger.debug(f"同步模式：低层数据收集已禁用，环境{env_id}只收集高层经验")
        
        n_agents = len(agent_skills)
        state_tensor = torch.FloatTensor(state).to(self.device)
        next_state_tensor = torch.FloatTensor(next_state).to(self.device)
        team_skill_tensor = torch.tensor(team_skill, device=self.device)
        
        # 累加当前步的团队奖励
        # 确保rewards是数值类型
        current_reward = rewards if isinstance(rewards, (int, float)) else rewards.item()
        
        # 使用环境ID为键创建或更新环境特定的奖励累积
        if env_id not in self.env_reward_sums:
            self.env_reward_sums[env_id] = 0.0
        
        self.env_reward_sums[env_id] += current_reward
        
        # 记录高层奖励累积情况（增加total_step和skill_timer信息）
        main_logger.debug(f"store_transition: 环境ID={env_id}, step={self.global_step}, skill_timer={skill_timer_for_env}, "
                          f"当前步奖励={current_reward:.4f}, 此环境累积高层奖励={self.env_reward_sums[env_id]:.4f}")
        
        # 计算团队技能判别器输出
        with torch.no_grad():
            team_disc_logits = self.team_discriminator(next_state_tensor.unsqueeze(0))
            team_disc_log_probs = F.log_softmax(team_disc_logits, dim=-1)
            team_skill_log_prob = team_disc_log_probs[0, team_skill]
        
        # 为每个智能体存储低层经验（仅在数据收集允许时）
        if low_level_collection_allowed:
            for i in range(n_agents):
                obs = torch.FloatTensor(observations[i]).to(self.device)
                next_obs = torch.FloatTensor(next_observations[i]).to(self.device)
                action = torch.FloatTensor(actions[i]).to(self.device)
                done = dones[i] if isinstance(dones, list) else dones
                
                # 计算个体技能判别器输出
                with torch.no_grad():
                    agent_disc_logits = self.individual_discriminator(
                        next_obs.unsqueeze(0), 
                        team_skill_tensor
                    )
                    agent_disc_log_probs = F.log_softmax(agent_disc_logits, dim=-1)
                    agent_skill_log_prob = agent_disc_log_probs[0, agent_skills[i]]
                    
                # 计算低层奖励（Eq. 4）及其组成部分
                env_reward_component = self.config.lambda_e * current_reward # 使用 current_reward
                team_disc_component = self.config.lambda_D * team_skill_log_prob.item()
                ind_disc_component = self.config.lambda_d * agent_skill_log_prob.item()
                
                intrinsic_reward = env_reward_component + team_disc_component + ind_disc_component
                
                # 存储低层经验
                low_level_experience = (
                    state_tensor,                           # 全局状态s
                    team_skill_tensor,                      # 团队技能Z
                    obs,                                    # 智能体观测o_i
                    torch.tensor(agent_skills[i], device=self.device),  # 个体技能z_i
                    action,                                 # 动作a_i
                    torch.tensor(intrinsic_reward, device=self.device),  # 总内在奖励r_i
                    torch.tensor(done, dtype=torch.float, device=self.device),  # 是否结束
                    torch.tensor(action_logprobs[i], device=self.device),  # 动作对数概率
                    torch.tensor(env_reward_component, device=self.device), # 环境奖励部分
                    torch.tensor(team_disc_component, device=self.device),  # 团队判别器部分
                    torch.tensor(ind_disc_component, device=self.device)   # 个体判别器部分
                )
                self.low_level_buffer.push(low_level_experience)
                
                # 在同步模式下，增加样本计数
                if self.sync_mode:
                    self.samples_collected_this_round += 1
        else:
            main_logger.debug(f"环境{env_id}: 跳过低层经验存储（同步模式数据收集已禁用）")
            
        # 存储技能判别器训练数据
        observations_tensor = torch.FloatTensor(next_observations).to(self.device)
        agent_skills_tensor = torch.tensor(agent_skills, device=self.device)
        self.state_skill_dataset.push(
            next_state_tensor,
            team_skill_tensor,
            observations_tensor,
            agent_skills_tensor
        )
        
        # 获取或初始化当前环境的技能计时器
        if env_id not in self.env_timers:
            self.env_timers[env_id] = 0
        
        # 优先使用传入的技能计时器值，如果没有则使用环境专用计时器
        skill_timer = skill_timer_for_env if skill_timer_for_env is not None else self.env_timers[env_id]
        
        # 记录当前技能计时器状态
        main_logger.debug(f"store_transition: 环境ID={env_id}, skill_timer={skill_timer}, k={self.config.k}, 条件判断={skill_timer == self.config.k - 1}")
        
        # 获取或初始化环境的最后贡献时间
        if env_id not in self.env_last_contribution:
            self.env_last_contribution[env_id] = 0
        
        # 获取或初始化环境特定的奖励阈值
        if env_id not in self.env_reward_thresholds:
            self.env_reward_thresholds[env_id] = 0.0  # 将默认阈值设为0，确保始终能存储高层经验
        
        # 判断该环境是否需要强制收集高层样本
        force_collection = self.force_high_level_collection.get(env_id, False)
        
        # 简化逻辑：取消所有奖励阈值，确保始终收集高层样本
        self.env_reward_thresholds[env_id] = 0.0
        
        # 对长时间未贡献的环境强制收集
        steps_since_contribution = self.global_step - self.env_last_contribution.get(env_id, 0)
        if steps_since_contribution > 500:  # 降低检查间隔至500步
            self.force_high_level_collection[env_id] = True
            if steps_since_contribution % 500 == 0:  # 避免日志过多
                main_logger.info(f"环境ID={env_id}已{steps_since_contribution}步未贡献高层样本，将强制收集")
        
        # 【修复】简化高层经验收集逻辑，只使用环境特定的timer判断
        # 废弃global_step判断，解决多环境并行时的冲突问题
        timer_completed_cycle = skill_timer == self.config.k - 1
        
        # 存储高层经验的条件：
        # 1. 技能周期完成（基于环境特定timer）
        # 2. 环境终止
        # 3. 强制收集
        should_store_high_level = timer_completed_cycle or dones or force_collection
        
        # 【调试日志】记录高层经验收集的详细判断过程
        main_logger.debug(f"[HIGH_LEVEL_DEBUG] 环境{env_id} 高层存储检查: "
                          f"skill_timer={skill_timer}, k-1={self.config.k-1}, "
                          f"timer_completed_cycle={timer_completed_cycle}, "
                          f"dones={dones}, force_collection={force_collection}, "
                          f"should_store={should_store_high_level}, "
                          f"累积奖励={self.env_reward_sums.get(env_id, 0.0):.4f}, "
                          f"global_step={self.global_step}")
        
        if should_store_high_level:
            # 获取当前环境的累积奖励
            env_accumulated_reward = self.env_reward_sums.get(env_id, 0.0)
            
            # 记录高层经验存储检查信息
            reason = "未知原因"
            if timer_completed_cycle:
                reason = "周期完成检测"
                main_logger.debug(f"环境ID={env_id}技能周期完成检测: 累积奖励={env_accumulated_reward:.4f}, "
                               f"global_step={self.global_step}, k={self.config.k}")
            elif skill_timer == self.config.k - 1:
                reason = "技能周期结束"
                main_logger.debug(f"环境ID={env_id}技能周期结束: 累积奖励={env_accumulated_reward:.4f}, "
                               f"离上次贡献={steps_since_contribution}步, k={self.config.k}")
            elif dones:
                reason = "环境终止"
                # 记录episode结束信息，包含环境ID和详细信息
                episode_info = {
                    'env_id': env_id,
                    'total_reward': env_accumulated_reward,
                    'skill_timer': skill_timer,
                    'team_skill': team_skill,
                    'agent_skills': agent_skills
                }
                
                # 从环境信息中提取额外的episode统计信息
                if hasattr(next_state, '__len__') and len(next_state) > 0:
                    episode_info['final_state_norm'] = float(torch.norm(torch.tensor(next_state)).item())
                
                # 记录episode结束的详细信息
                main_logger.info(f"Episode结束 - 环境ID: {env_id}, "
                               f"总奖励: {env_accumulated_reward:.4f}, "
                               f"技能计时器: {skill_timer}, "
                               f"团队技能: {team_skill}, "
                               f"个体技能: {agent_skills}")
            elif force_collection:
                reason = "强制收集"
                main_logger.info(f"环境ID={env_id}强制收集: 累积奖励={env_accumulated_reward:.4f}, 技能计时器={skill_timer}")
            # 创建高层经验元组
            high_level_experience = (
                state_tensor,                  # 全局状态s
                team_skill_tensor,             # 团队技能Z
                observations_tensor,           # 所有智能体观测o
                agent_skills_tensor,           # 所有个体技能z
                torch.tensor(env_accumulated_reward, device=self.device) # 存储该环境的k步累积奖励
            )
            
            # 存储高层经验
            self.high_level_buffer.push(high_level_experience)
        
            # 无论buffer长度是否变化，都认为成功添加了一个样本
            # 避免在buffer满时因为长度不变而误判为未添加样本
            samples_added = 1
            self.high_level_samples_total += samples_added
            # 记录环境贡献
            self.high_level_samples_by_env[env_id] = self.high_level_samples_by_env.get(env_id, 0) + 1
            # 记录原因统计
            self.high_level_samples_by_reason[reason] = self.high_level_samples_by_reason.get(reason, 0) + 1
            
            # 更新环境最后贡献时间
            self.env_last_contribution[env_id] = self.global_step
            
            # 重置强制收集标志
            if force_collection:
                self.force_high_level_collection[env_id] = False
            
            # 每收集5个样本记录一次统计信息（从10改为5，增加反馈频率）
            if self.high_level_samples_total % 5 == 0:
                main_logger.debug(f"[HIGH_LEVEL_COLLECT] 高层经验统计 - 总样本: {self.high_level_samples_total}, 环境贡献: {self.high_level_samples_by_env}, 原因统计: {self.high_level_samples_by_reason}")
                
                # 记录到TensorBoard
                if hasattr(self, 'writer'):
                    self.writer.add_scalar('Buffer/high_level_samples_total', self.high_level_samples_total, self.global_step)
                    # 记录各环境的样本贡献比例
                    for e_id, count in self.high_level_samples_by_env.items():
                        self.writer.add_scalar(f'Buffer/env_{e_id}_contribution', count, self.global_step)
        
            # 增加日志以便跟踪高层经验添加状态
            current_buffer_size = len(self.high_level_buffer)
            main_logger.debug(f"[HIGH_LEVEL_COLLECT] ✓ 高层经验已添加: 环境ID={env_id}, step={self.global_step}, "
                           f"缓冲区大小: {current_buffer_size}/{self.config.high_level_batch_size}, "
                           f"累积奖励: {env_accumulated_reward:.4f}, 原因: {reason}")
            
            # 将带有log probabilities的经验存储到专用缓冲区
            if log_probs is not None:
                self.high_level_buffer_with_logprobs.append({
                    'state': state_tensor.clone(),
                    'team_skill': team_skill,
                    'observations': observations_tensor.clone(),
                    'agent_skills': agent_skills_tensor.clone(),
                    'reward': env_accumulated_reward,  # 使用环境特定的累积奖励
                    'team_log_prob': log_probs['team_log_prob'],
                    'agent_log_probs': log_probs['agent_log_probs']
                })
                
                # 保持缓冲区大小不超过config.buffer_size
                if len(self.high_level_buffer_with_logprobs) > self.config.buffer_size:
                    self.high_level_buffer_with_logprobs = self.high_level_buffer_with_logprobs[-self.config.buffer_size:]
            
            # 重置该环境的奖励累积
            self.env_reward_sums[env_id] = 0.0
            
            # 重置该环境的技能计时器
            self.env_timers[env_id] = 0
            
        else:
            # 如果不到技能周期结束时间，增加该环境的技能计时器，但确保不超过k-1
            if self.env_timers[env_id] < self.config.k - 1:
                self.env_timers[env_id] += 1
        
        # 返回成功存储
        return True
    
    def update_coordinator(self):
        """更新高层技能协调器网络"""
        # 记录高层缓冲区状态
        buffer_len = len(self.high_level_buffer)
        required_batch_size = self.config.high_level_batch_size
        main_logger.info(f"[BUFFER_STATUS] 高层缓冲区状态: {buffer_len}/{required_batch_size} (当前/所需)")
        
        if buffer_len < required_batch_size:
            # 如果缓冲区不足，使用计数器减少警告日志频率
            # 只有当缓冲区大小变化或者每10次更新才记录一次警告
            if buffer_len != self.last_high_level_buffer_size or self.high_level_buffer_warning_counter % 10 == 0:
                main_logger.info(f"Training: 收集中... 高层缓冲区: {buffer_len}/{required_batch_size} 样本")
            else:
                main_logger.debug(f"[BUFFER_STATUS] 高层缓冲区样本不足，需要{required_batch_size}个样本，但只有{buffer_len}个。跳过更新。")
            
            # 更新计数器和上次缓冲区大小
            self.high_level_buffer_warning_counter += 1
            self.last_high_level_buffer_size = buffer_len
            
            # 保持与函数正常返回值相同数量的元素
            return 0, 0, 0, 0, 0, 0, 0, 0
        
        # 缓冲区已满，继续更新
        main_logger.info(f"[HIGH_LEVEL_UPDATE] 高层缓冲区满足更新条件，从{buffer_len}个样本中采样{required_batch_size}个")
            
        # 从缓冲区采样数据
        batch = self.high_level_buffer.sample(self.config.high_level_batch_size)
        states, team_skills, observations, agent_skills, rewards = zip(*batch)
        
        states = torch.stack(states)
        team_skills = torch.stack(team_skills)
        observations = torch.stack(observations)
        agent_skills = torch.stack(agent_skills)
        rewards = torch.stack(rewards) # rewards现在是累积的k步奖励r_h
        
        # 记录高层奖励的统计信息
        reward_mean = rewards.mean().item()
        reward_std = rewards.std().item()
        reward_min = rewards.min().item()
        reward_max = rewards.max().item()
        main_logger.info(f"[HIGH_LEVEL_UPDATE] 高层奖励统计: 均值={reward_mean:.4f}, 标准差={reward_std:.4f}, 最小值={reward_min:.4f}, 最大值={reward_max:.4f}")
        
        # 获取当前状态价值
        state_values, agent_values = self.skill_coordinator.get_value(states, observations)
        
        # 由于我们假设每个高层经验都是一个k步序列的端点，
        # 所以我们可以假设下一状态价值为0（或者可以从新的状态计算）
        next_values = torch.zeros_like(state_values)
        
        # 在计算GAE之前详细记录奖励和价值的统计信息
        rewards_mean = rewards.mean().item()
        rewards_std = rewards.std().item()
        rewards_min = rewards.min().item()
        rewards_max = rewards.max().item()
        state_values_mean = state_values.mean().item()
        state_values_std = state_values.std().item()
        state_values_min = state_values.min().item()
        state_values_max = state_values.max().item()
        
        main_logger.debug(f"GAE输入统计:")
        main_logger.debug(f"  rewards: 均值={rewards_mean:.4f}, 标准差={rewards_std:.4f}, 最小值={rewards_min:.4f}, 最大值={rewards_max:.4f}")
        main_logger.debug(f"  state_values: 均值={state_values_mean:.4f}, 标准差={state_values_std:.4f}, 最小值={state_values_min:.4f}, 最大值={state_values_max:.4f}")
        
        # 检查是否有异常值
        rewards_has_nan = torch.isnan(rewards).any().item()
        rewards_has_inf = torch.isinf(rewards).any().item()
        values_has_nan = torch.isnan(state_values).any().item()
        values_has_inf = torch.isinf(state_values).any().item()
        
        if rewards_has_nan or rewards_has_inf:
            main_logger.error(f"奖励中存在NaN或Inf: NaN={rewards_has_nan}, Inf={rewards_has_inf}")
            # 尝试修复NaN/Inf值，以避免整个训练中断
            rewards = torch.nan_to_num(rewards, nan=0.0, posinf=10.0, neginf=-10.0)
            main_logger.info("已将奖励中的NaN/Inf值替换为有限值")
        
        if values_has_nan or values_has_inf:
            main_logger.error(f"状态价值中存在NaN或Inf: NaN={values_has_nan}, Inf={values_has_inf}")
            # 尝试修复NaN/Inf值
            state_values = torch.nan_to_num(state_values, nan=0.0, posinf=10.0, neginf=-10.0)
            main_logger.info("已将状态价值中的NaN/Inf值替换为有限值")
        
        # 计算GAE
        dones = torch.zeros_like(rewards)  # 假设高层经验不包含终止信息
        # 确保传递给compute_gae的values是1D，使用clone避免原地操作
        try:
            advantages, returns = compute_gae(rewards.clone(), state_values.squeeze(-1).clone(), 
                                            next_values.squeeze(-1).clone(), dones.clone(), 
                                            self.config.gamma, self.config.gae_lambda)
            # advantages 和 returns 都是 [batch_size]，分离计算图
            advantages = advantages.detach()
            returns = returns.detach()
            
            # 检查 advantages 和 returns 的统计信息
            adv_mean = advantages.mean().item()
            adv_std = advantages.std().item()
            adv_min = advantages.min().item()
            adv_max = advantages.max().item()
            ret_mean = returns.mean().item()
            ret_std = returns.std().item()
            ret_min = returns.min().item()
            ret_max = returns.max().item()
            
            main_logger.debug(f"GAE输出统计:")
            main_logger.debug(f"  Advantages: 均值={adv_mean:.4f}, 标准差={adv_std:.4f}, 最小值={adv_min:.4f}, 最大值={adv_max:.4f}")
            main_logger.debug(f"  Returns: 均值={ret_mean:.4f}, 标准差={ret_std:.4f}, 最小值={ret_min:.4f}, 最大值={ret_max:.4f}")
            
            # 检查GAE输出是否有异常值
            adv_has_nan = torch.isnan(advantages).any().item()
            adv_has_inf = torch.isinf(advantages).any().item()
            ret_has_nan = torch.isnan(returns).any().item()
            ret_has_inf = torch.isinf(returns).any().item()
            
            if adv_has_nan or adv_has_inf:
                main_logger.error(f"advantages中存在NaN或Inf: NaN={adv_has_nan}, Inf={adv_has_inf}")
                # 尝试修复NaN/Inf值
                advantages = torch.nan_to_num(advantages, nan=0.0, posinf=10.0, neginf=-10.0)
                main_logger.info("已将advantages中的NaN/Inf值替换为有限值")
            
            if ret_has_nan or ret_has_inf:
                main_logger.error(f"returns中存在NaN或Inf: NaN={ret_has_nan}, Inf={ret_has_inf}")
                # 尝试修复NaN/Inf值
                returns = torch.nan_to_num(returns, nan=0.0, posinf=10.0, neginf=-10.0)
                main_logger.info("已将returns中的NaN/Inf值替换为有限值")
                
            # 归一化advantages，有助于稳定训练
            if adv_std > 0:
                advantages = (advantages - adv_mean) / (adv_std + 1e-8)
                main_logger.debug("已对advantages进行归一化处理")
                
        except Exception as e:
            main_logger.error(f"计算GAE时发生错误: {e}")
            # 使用安全的默认值
            advantages = torch.zeros_like(rewards)
            returns = rewards.clone()  # 在缺乏更好选择的情况下，使用原始奖励作为返回值
            main_logger.info("由于GAE计算失败，使用安全的默认值作为替代")
        
        # 获取当前策略
        try:
            Z, z, Z_logits, z_logits = self.skill_coordinator(states, observations)
            
            # 在使用logits前检查是否有异常值
            Z_logits_has_nan = torch.isnan(Z_logits).any().item()
            Z_logits_has_inf = torch.isinf(Z_logits).any().item()
            
            if Z_logits_has_nan or Z_logits_has_inf:
                main_logger.error(f"Z_logits中存在NaN或Inf: NaN={Z_logits_has_nan}, Inf={Z_logits_has_inf}")
                # 尝试修复NaN/Inf值
                Z_logits = torch.nan_to_num(Z_logits, nan=0.0, posinf=10.0, neginf=-10.0)
                main_logger.info("已将Z_logits中的NaN/Inf值替换为有限值")
            
            # 重新计算团队技能概率分布
            team_skills_detached = team_skills.clone().detach()  # 分离计算图，防止原地操作
            Z_dist = Categorical(logits=Z_logits)
            Z_log_probs = Z_dist.log_prob(team_skills_detached)
            Z_entropy = Z_dist.entropy().mean()
            
            # 记录团队技能熵的统计信息
            main_logger.debug(f"团队技能熵: {Z_entropy.item():.4f}")
            
        except Exception as e:
            main_logger.error(f"在计算策略分布时发生错误: {e}")
            # 使用安全的默认值
            batch_size = states.size(0)
            Z = torch.zeros(batch_size, dtype=torch.long, device=self.device)
            z = torch.zeros(batch_size, self.config.n_agents, dtype=torch.long, device=self.device)
            Z_logits = torch.zeros((batch_size, self.config.n_Z), device=self.device)
            z_logits = [torch.zeros((batch_size, self.config.n_z), device=self.device) for _ in range(self.config.n_agents)]
            Z_log_probs = torch.zeros(batch_size, device=self.device)
            Z_entropy = torch.tensor(0.0, device=self.device)
            main_logger.info("由于错误，使用安全的默认值进行计算")
        
        # 检查是否有带log probabilities的高层经验
        use_stored_logprobs = len(self.high_level_buffer_with_logprobs) >= self.config.high_level_batch_size
        
        try:
            # 计算高层策略损失
            if use_stored_logprobs:
                # 使用存储的log probabilities计算更准确的PPO ratio
                
                # 从带log probabilities的缓冲区中随机选择样本
                indices = torch.randperm(len(self.high_level_buffer_with_logprobs))[:self.config.high_level_batch_size]
                old_team_log_probs = [self.high_level_buffer_with_logprobs[i]['team_log_prob'] for i in indices]
                old_team_log_probs_tensor = torch.tensor(old_team_log_probs, device=self.device).detach()  # 使用detach()防止求导错误
                
                # 检查old_team_log_probs_tensor是否有异常值
                old_log_probs_has_nan = torch.isnan(old_team_log_probs_tensor).any().item()
                old_log_probs_has_inf = torch.isinf(old_team_log_probs_tensor).any().item()
                
                if old_log_probs_has_nan or old_log_probs_has_inf:
                    main_logger.error(f"old_team_log_probs_tensor中存在NaN或Inf: NaN={old_log_probs_has_nan}, Inf={old_log_probs_has_inf}")
                    # 尝试修复NaN/Inf值
                    old_team_log_probs_tensor = torch.nan_to_num(old_team_log_probs_tensor, nan=0.0, posinf=0.0, neginf=0.0)
                    main_logger.info("已将old_team_log_probs_tensor中的NaN/Inf值替换为0")
                
                # 记录log_probs的统计信息
                main_logger.debug(f"当前log_probs统计: 均值={Z_log_probs.mean().item():.4f}, 标准差={Z_log_probs.std().item():.4f}")
                main_logger.debug(f"历史log_probs统计: 均值={old_team_log_probs_tensor.mean().item():.4f}, 标准差={old_team_log_probs_tensor.std().item():.4f}")
                
                # 安全计算PPO ratio，避免数值上溢
                log_ratio = Z_log_probs - old_team_log_probs_tensor
                # 裁剪log_ratio以避免exp操作导致数值溢出
                log_ratio = torch.clamp(log_ratio, -10.0, 10.0)
                Z_ratio = torch.exp(log_ratio)
                
                # 记录ratio的统计信息
                ratio_mean = Z_ratio.mean().item()
                ratio_std = Z_ratio.std().item()
                ratio_min = Z_ratio.min().item()
                ratio_max = Z_ratio.max().item()
                main_logger.debug(f"PPO ratio统计: 均值={ratio_mean:.4f}, 标准差={ratio_std:.4f}, 最小值={ratio_min:.4f}, 最大值={ratio_max:.4f}")
                
                # 打印debug信息
                main_logger.debug(f"使用存储的log probabilities进行PPO更新，共有{len(self.high_level_buffer_with_logprobs)}个样本")
            else:
                # 如果没有存储log probabilities，则假设old_log_probs=0
                # 同样需要裁剪以避免数值溢出
                log_ratio = torch.clamp(Z_log_probs, -10.0, 10.0)
                Z_ratio = torch.exp(log_ratio)
                main_logger.warning("未使用存储的log probabilities，假设old_log_probs=0")
            
            # 检查ratio是否有异常值
            ratio_has_nan = torch.isnan(Z_ratio).any().item()
            ratio_has_inf = torch.isinf(Z_ratio).any().item()
            
            if ratio_has_nan or ratio_has_inf:
                main_logger.error(f"Z_ratio中存在NaN或Inf: NaN={ratio_has_nan}, Inf={ratio_has_inf}")
                # 尝试修复NaN/Inf值
                Z_ratio = torch.nan_to_num(Z_ratio, nan=1.0, posinf=2.0, neginf=0.5)
                main_logger.info("已将Z_ratio中的NaN/Inf值替换为有限值")
            
            # 计算带裁剪的目标函数
            Z_surr1 = Z_ratio * advantages
            Z_surr2 = torch.clamp(Z_ratio, 1.0 - self.config.clip_epsilon, 1.0 + self.config.clip_epsilon) * advantages
            Z_policy_loss = -torch.min(Z_surr1, Z_surr2).mean()
            
            # 检查损失是否有异常值
            if torch.isnan(Z_policy_loss).any().item() or torch.isinf(Z_policy_loss).any().item():
                main_logger.error(f"Z_policy_loss包含NaN或Inf值: {Z_policy_loss.item()}")
                # 使用一个安全的默认损失值
                Z_policy_loss = torch.tensor(0.1, device=self.device, requires_grad=True)
                main_logger.info("已将Z_policy_loss替换为安全的默认值0.1")
                
        except Exception as e:
            main_logger.error(f"计算高层策略损失时发生错误: {e}")
            # 使用安全的默认损失值
            Z_policy_loss = torch.tensor(0.1, device=self.device, requires_grad=True)
            main_logger.info("由于错误，使用安全的默认值0.1作为Z_policy_loss")
        
        try:
            # 计算高层价值损失 - 使用配置化的Huber Loss提高鲁棒性
            state_values = state_values.float() # Shape [batch_size, 1]
            # returns 是 [batch_size], 需要 unsqueeze 匹配 state_values
            returns = returns.float().unsqueeze(-1) # Shape [batch_size, 1]
            
            # 根据配置选择损失函数
            if getattr(self.config, 'use_huber_loss', True):
                # 使用自适应或配置的Huber Loss
                if getattr(self.config, 'huber_adaptive_delta', False):
                    delta = self.adaptive_coordinator_delta
                    main_logger.debug(f"使用自适应Huber Loss计算协调器价值损失，delta={delta:.4f}")
                else:
                    delta = getattr(self.config, 'huber_coordinator_delta', 1.0)
                    main_logger.debug(f"使用固定Huber Loss计算协调器价值损失，delta={delta}")
                Z_value_loss = huber_loss(state_values, returns, delta=delta)
            else:
                # 使用传统的MSE Loss
                Z_value_loss = F.mse_loss(state_values, returns)
                main_logger.debug("使用MSE Loss计算协调器价值损失")
            
            # 检查价值损失是否有异常值
            if torch.isnan(Z_value_loss).any().item() or torch.isinf(Z_value_loss).any().item():
                main_logger.error(f"Z_value_loss包含NaN或Inf值: {Z_value_loss.item()}")
                # 使用安全的默认损失值
                Z_value_loss = torch.tensor(0.1, device=self.device, requires_grad=True)
                main_logger.info("已将Z_value_loss替换为安全的默认值0.1")
            
        except Exception as e:
            main_logger.error(f"计算高层价值损失时发生错误: {e}")
            # 使用安全的默认损失值
            Z_value_loss = torch.tensor(0.1, device=self.device, requires_grad=True)
            main_logger.info("由于错误，使用安全的默认值0.1作为Z_value_loss")
        
        # 初始化智能体策略损失
        z_policy_losses = []
        z_entropy_losses = []
        z_value_losses = []
        
        # 处理每个智能体的个体技能损失
        # 使用实际智能体数量，由智能体技能形状决定，而不是配置中的n_agents
        n_agents_actual = agent_skills.shape[1]  # 从采样的agent_skills中获取实际智能体数量
        for i in range(n_agents_actual):
            agent_skills_i = agent_skills[:, i].clone().detach()  # 分离计算图，防止原地操作
            zi_dist = Categorical(logits=z_logits[i])
            zi_log_probs = zi_dist.log_prob(agent_skills_i)
            zi_entropy = zi_dist.entropy().mean()
            
            if use_stored_logprobs:
                # 使用存储的agent log probabilities
                old_agent_log_probs = [self.high_level_buffer_with_logprobs[j]['agent_log_probs'][i] 
                                      for j in indices 
                                      if i < len(self.high_level_buffer_with_logprobs[j]['agent_log_probs'])]
                
                if len(old_agent_log_probs) == len(zi_log_probs):
                    old_agent_log_probs_tensor = torch.tensor(old_agent_log_probs, device=self.device).detach()  # 使用detach()防止求导错误
                    zi_ratio = torch.exp(zi_log_probs - old_agent_log_probs_tensor)
                else:
                    # 如果长度不匹配（例如智能体数量变化），则退回到假设old_log_probs=0
                    zi_ratio = torch.exp(zi_log_probs)
            else:
                # 如果没有存储的log probabilities，则假设old_log_probs=0
                zi_ratio = torch.exp(zi_log_probs)
                
            zi_surr1 = zi_ratio * advantages
            zi_surr2 = torch.clamp(zi_ratio, 1.0 - self.config.clip_epsilon, 1.0 + self.config.clip_epsilon) * advantages
            zi_policy_loss = -torch.min(zi_surr1, zi_surr2).mean()
            
            z_policy_losses.append(zi_policy_loss)
            z_entropy_losses.append(zi_entropy)
            
            if i < len(agent_values):
                # 确保数据类型匹配
                agent_value = agent_values[i].float() # Shape [128, 1]
                # returns 已经是 [128, 1]
                returns_i = returns.float() 
                
                # 使用Huber Loss (smooth_l1_loss) 替代MSE Loss
                zi_value_loss = F.smooth_l1_loss(agent_value, returns_i)
                z_value_losses.append(zi_value_loss)
        
        # 合并所有智能体的损失
        z_policy_loss = torch.stack(z_policy_losses).mean()
        z_entropy = torch.stack(z_entropy_losses).mean()
        
        if z_value_losses:
            z_value_loss = torch.stack(z_value_losses).mean()
        else:
            z_value_loss = torch.tensor(0.0, device=self.device)
        
        try:
            # 总策略损失
            policy_loss = Z_policy_loss + z_policy_loss
            
            # 总价值损失
            value_loss = Z_value_loss + z_value_loss
            
            # 总熵损失
            entropy_loss = -(Z_entropy + z_entropy) * self.config.lambda_h
            
            # 总损失
            loss = policy_loss + self.config.value_loss_coef * value_loss + entropy_loss
            
            # 检查总损失是否有异常值
            if torch.isnan(loss).any().item() or torch.isinf(loss).any().item():
                main_logger.error(f"总损失包含NaN或Inf值: {loss.item()}")
                # 分析损失组成部分
                main_logger.error(f"损失组成部分: policy_loss={policy_loss.item()}, value_loss={value_loss.item()}, entropy_loss={entropy_loss.item()}")
                
                # 尝试创建一个新的、安全的损失
                policy_loss_safe = torch.tensor(0.1, device=self.device, requires_grad=True)
                value_loss_safe = torch.tensor(0.1, device=self.device, requires_grad=True)
                entropy_loss_safe = torch.tensor(-0.1, device=self.device, requires_grad=True)
                loss = policy_loss_safe + self.config.value_loss_coef * value_loss_safe + entropy_loss_safe
                main_logger.info("已将总损失替换为安全的默认值")
            
            # 记录损失值
            main_logger.debug(f"损失统计: 总损失={loss.item():.6f}, 策略损失={policy_loss.item():.6f}, 价值损失={value_loss.item():.6f}, 熵损失={entropy_loss.item():.6f}")
            
            # 更新网络
            self.coordinator_optimizer.zero_grad()
            loss.backward()
            
        except Exception as e:
            main_logger.error(f"计算总损失时发生错误: {e}")
            # 创建一个新的、安全的损失
            loss = torch.tensor(0.3, device=self.device, requires_grad=True)
            policy_loss = torch.tensor(0.1, device=self.device)
            value_loss = torch.tensor(0.1, device=self.device)
            entropy_loss = torch.tensor(-0.1, device=self.device)
            
            main_logger.info("由于错误，使用安全的默认值作为损失")
            
            # 更新网络
            self.coordinator_optimizer.zero_grad()
            loss.backward()
        
        # 检查loss是否正确连接到计算图
        main_logger.debug(f"损失连接状态: requires_grad={loss.requires_grad}, grad_fn={loss.grad_fn}")
        
        # 检查coordinator参数是否正确设置requires_grad
        params_requiring_grad = 0
        for name, param in self.skill_coordinator.named_parameters():
            if param.requires_grad:
                params_requiring_grad += 1
                main_logger.debug(f"参数 {name} requires_grad=True")
        main_logger.debug(f"Coordinator中需要梯度的参数数量: {params_requiring_grad}")
        
        # 详细记录梯度信息
        params_with_grads = [p for p in self.skill_coordinator.parameters() if p.grad is not None]
        if params_with_grads:
            # 检查梯度是否包含NaN或Inf
            has_nan_grad = any(torch.isnan(p.grad).any().item() for p in params_with_grads)
            has_inf_grad = any(torch.isinf(p.grad).any().item() for p in params_with_grads)
            
            if has_nan_grad or has_inf_grad:
                main_logger.error(f"梯度中包含NaN或Inf值: NaN={has_nan_grad}, Inf={has_inf_grad}")
                # 尝试修复梯度中的NaN/Inf值
                for p in params_with_grads:
                    if torch.isnan(p.grad).any() or torch.isinf(p.grad).any():
                        p.grad.data = torch.nan_to_num(p.grad.data, nan=0.0, posinf=1.0, neginf=-1.0)
                main_logger.info("已将梯度中的NaN和Inf值替换为有限值")
            
            # 计算梯度的统计信息
            grad_norms = [torch.norm(p.grad.detach()).item() for p in params_with_grads]
            mean_norm = np.mean(grad_norms)
            max_norm = max(grad_norms)
            min_norm = min(grad_norms)
            std_norm = np.std(grad_norms)
            total_norm = torch.sqrt(sum(p.grad.detach().pow(2).sum() for p in params_with_grads)).item()
            
            main_logger.debug(f"梯度统计 (裁剪前): 总范数={total_norm:.6f}, 均值={mean_norm:.6f}, "
                             f"标准差={std_norm:.6f}, 最大={max_norm:.6f}, 最小={min_norm:.6f}")
            
            # 检查是否有较大梯度
            large_grad_threshold = 10.0
            large_grads = [(name, torch.norm(param.grad).item()) 
                           for name, param in self.skill_coordinator.named_parameters() 
                           if param.grad is not None and torch.norm(param.grad).item() > large_grad_threshold]
            
            if large_grads:
                main_logger.warning(f"检测到{len(large_grads)}个参数具有较大梯度 (>{large_grad_threshold}):")
                for name, norm in large_grads[:5]:  # 只显示前5个
                    main_logger.warning(f"  参数 {name}: 梯度范数 = {norm:.6f}")
                if len(large_grads) > 5:
                    main_logger.warning(f"  ... 还有{len(large_grads)-5}个参数有较大梯度")
            
            # 梯度裁剪
            try:
                torch.nn.utils.clip_grad_norm_(self.skill_coordinator.parameters(), self.config.max_grad_norm)
                
                # 记录裁剪后的梯度信息
                params_with_grads_after = [p for p in self.skill_coordinator.parameters() if p.grad is not None]
                if params_with_grads_after:
                    grad_norms_after = [torch.norm(p.grad.detach()).item() for p in params_with_grads_after]
                    mean_norm_after = np.mean(grad_norms_after)
                    max_norm_after = max(grad_norms_after)
                    min_norm_after = min(grad_norms_after)
                    std_norm_after = np.std(grad_norms_after)
                    total_norm_after = torch.sqrt(sum(p.grad.detach().pow(2).sum() for p in params_with_grads_after)).item()
                    
                    main_logger.debug(f"梯度统计 (裁剪后): 总范数={total_norm_after:.6f}, 均值={mean_norm_after:.6f}, "
                                     f"标准差={std_norm_after:.6f}, 最大={max_norm_after:.6f}, 最小={min_norm_after:.6f}")
            except Exception as e:
                main_logger.error(f"梯度裁剪失败: {e}")
                
        else:
            main_logger.warning("没有参数接收到梯度! 检查loss.backward()是否正确传播梯度。")
            
            # 详细检查每个参数的梯度状态
            grad_status = {}
            for name, param in self.skill_coordinator.named_parameters():
                if param.grad is None:
                    grad_status[name] = "None"
                else:
                    norm = torch.norm(param.grad).item()
                    has_nan = torch.isnan(param.grad).any().item()
                    has_inf = torch.isinf(param.grad).any().item()
                    grad_status[name] = f"有梯度，范数: {norm:.6f}, NaN: {has_nan}, Inf: {has_inf}"
            
            # 记录所有参数的梯度状态
            main_logger.debug("详细的参数梯度状态:")
            for name, status in grad_status.items():
                main_logger.debug(f"参数 {name} 梯度状态: {status}")
        
        # 记录参数更新前的多个网络参数样本
        sample_params = {}
        for name, param in list(self.skill_coordinator.named_parameters())[:5]:  # 只取前5个参数作为样本
            if param.requires_grad and param.numel() > 0:
                sample_params[name] = param.clone().detach()
                main_logger.debug(f"参数 {name} 更新前: 均值={param.mean().item():.6f}, 标准差={param.std().item():.6f}")
        
        try:
            self.coordinator_optimizer.step()
            
            # 记录参数更新后的变化
            for name, old_param in sample_params.items():
                for curr_name, curr_param in self.skill_coordinator.named_parameters():
                    if curr_name == name:
                        param_mean_diff = (curr_param.detach().mean() - old_param.mean()).item()
                        param_abs_diff = torch.mean(torch.abs(curr_param.detach() - old_param)).item()
                        main_logger.debug(f"参数 {name} 更新后: 均值变化={param_mean_diff:.6f}, 平均绝对变化={param_abs_diff:.6f}")
                        break
                        
        except Exception as e:
            main_logger.error(f"优化器step失败: {e}")
            # 这种情况下我们无法继续，但至少记录了错误
        
        # 计算平均价值估计
        mean_state_value = state_values.mean().item()
        mean_agent_value = 0.0
        if agent_values and len(agent_values) > 0:
            # agent_values 是一个列表的张量，每个张量是 [batch_size, 1]
            # 我们需要将它们堆叠起来，然后计算均值
            stacked_agent_values = torch.stack(agent_values, dim=0) # Shape [n_agents, batch_size, 1]
            mean_agent_value = stacked_agent_values.mean().item()
        
        # rewards 是累积的k步环境奖励 r_h
        mean_high_level_reward = rewards.mean().item()
            
        # 返回：总损失, 策略损失, 价值损失, 团队熵, 个体熵, 状态价值均值, 智能体价值均值, 高层奖励均值
        return loss.item(), policy_loss.item(), value_loss.item(), \
               Z_entropy.item(), z_entropy.item(), \
               mean_state_value, mean_agent_value, mean_high_level_reward
    
    def update_discoverer(self):
        """更新低层技能发现器网络"""
        if len(self.low_level_buffer) < self.config.batch_size:
            return 0, 0, 0, 0, 0, 0, 0, 0, 0 # 增加返回数量以匹配期望（9个值）
        
        # 从缓冲区采样数据，包含内在奖励的三个组成部分
        batch = self.low_level_buffer.sample(self.config.batch_size)
        states, team_skills, observations, agent_skills, actions, rewards, dones, old_log_probs, \
        env_rewards_comp, team_disc_rewards_comp, ind_disc_rewards_comp = zip(*batch)
        
        states = torch.stack(states)
        team_skills = torch.stack(team_skills)
        observations = torch.stack(observations)
        agent_skills = torch.stack(agent_skills)
        actions = torch.stack(actions)
        rewards = torch.stack(rewards)
        dones = torch.stack(dones)
        old_log_probs = torch.stack(old_log_probs)
        
        # 初始化GRU隐藏状态
        self.skill_discoverer.init_hidden(batch_size=self.config.batch_size)
        
        # 获取当前状态价值
        values = self.skill_discoverer.get_value(states, team_skills)
        
        # 构造下一状态的占位符
        next_values = torch.zeros_like(values)  # 实际应用中应该使用真实下一状态计算
        
        # 计算GAE
        # 确保传递给compute_gae的values是1D，使用clone避免原地操作
        advantages, returns = compute_gae(rewards.clone(), values.squeeze(-1).clone(), 
                                         next_values.squeeze(-1).clone(), dones.clone(), 
                                         self.config.gamma, self.config.gae_lambda)
        # advantages 和 returns 都是 [batch_size]，分离计算图
        advantages = advantages.detach()
        returns = returns.detach()
        
        # 重新初始化GRU隐藏状态
        self.skill_discoverer.init_hidden(batch_size=self.config.batch_size)
        
        # 获取当前策略
        _, action_log_probs, action_dist = self.skill_discoverer(observations, agent_skills)
        
        # 计算策略比率，使用detach()防止求导错误
        old_log_probs_detached = old_log_probs.clone().detach()
        ratios = torch.exp(action_log_probs - old_log_probs_detached)
        
        # 限制策略比率
        surr1 = ratios * advantages
        surr2 = torch.clamp(ratios, 1.0 - self.config.clip_epsilon, 1.0 + self.config.clip_epsilon) * advantages
        
        # 计算策略损失
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # 重新初始化GRU隐藏状态
        self.skill_discoverer.init_hidden(batch_size=self.config.batch_size)
        
        # 计算价值损失 - 使用配置化的Huber Loss提高鲁棒性
        current_values = self.skill_discoverer.get_value(states, team_skills) # Shape [128, 1]
        # 确保维度匹配并转换为float32类型
        current_values = current_values.float()
        # returns 是 [128], 需要 unsqueeze 匹配 current_values
        returns = returns.float().unsqueeze(-1) # Shape [128, 1]
        
        # 根据配置选择损失函数
        if getattr(self.config, 'use_huber_loss', True):
            # 使用自适应或配置的Huber Loss
            if getattr(self.config, 'huber_adaptive_delta', False):
                delta = self.adaptive_discoverer_delta
                main_logger.debug(f"使用自适应Huber Loss计算发现器价值损失，delta={delta:.4f}")
            else:
                delta = getattr(self.config, 'huber_discoverer_delta', 1.0)
                main_logger.debug(f"使用固定Huber Loss计算发现器价值损失，delta={delta}")
            value_loss = huber_loss(current_values, returns, delta=delta)
        else:
            # 使用传统的MSE Loss
            value_loss = F.mse_loss(current_values, returns)
            main_logger.debug("使用MSE Loss计算发现器价值损失")
        
        # 计算熵损失
        entropy_loss = -action_dist.entropy().mean() * self.config.lambda_l
        
        # 总损失
        loss = policy_loss + self.config.value_loss_coef * value_loss + entropy_loss
        
        # 更新网络
        self.discoverer_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.skill_discoverer.parameters(), self.config.max_grad_norm)
        self.discoverer_optimizer.step()
        
        # 清空低层缓冲区，确保on-policy训练
        buffer_size_before = len(self.low_level_buffer)
        self.low_level_buffer.clear()
        main_logger.info(f"底层策略更新完成，已清空low_level_buffer（之前大小: {buffer_size_before}）")
        
        # 计算内在奖励各部分的平均值
        avg_intrinsic_reward = rewards.mean().item()
        avg_env_reward_comp = torch.stack(env_rewards_comp).mean().item()
        avg_team_disc_reward_comp = torch.stack(team_disc_rewards_comp).mean().item()
        avg_ind_disc_reward_comp = torch.stack(ind_disc_rewards_comp).mean().item()
        avg_discoverer_value = current_values.mean().item() # 使用更新前的 current_values
        
        action_entropy_val = -entropy_loss.item() / self.config.lambda_l if self.config.lambda_l > 0 else 0.0

        return loss.item(), policy_loss.item(), value_loss.item(), action_entropy_val, \
               avg_intrinsic_reward, avg_env_reward_comp, avg_team_disc_reward_comp, avg_ind_disc_reward_comp, avg_discoverer_value
    
    def update_discriminators(self):
        """更新技能判别器网络"""
        if len(self.state_skill_dataset) < self.config.batch_size:
            return 0
        
        # 从数据集采样数据
        batch = self.state_skill_dataset.sample(self.config.batch_size)
        states, team_skills, observations, agent_skills = zip(*batch)
        
        states = torch.stack(states)
        team_skills = torch.stack(team_skills)
        observations = torch.stack(observations)
        agent_skills = torch.stack(agent_skills)
        
        # 更新团队技能判别器
        team_disc_logits = self.team_discriminator(states)
        team_disc_loss = F.cross_entropy(team_disc_logits, team_skills)
        
        # 更新个体技能判别器
        batch_size, n_agents = agent_skills.shape
        
        # 扁平化处理
        observations_flat = observations.reshape(-1, observations.size(-1))
        agent_skills_flat = agent_skills.reshape(-1)
        team_skills_expanded = team_skills.unsqueeze(1).expand(-1, n_agents).reshape(-1)
        
        agent_disc_logits = self.individual_discriminator(observations_flat, team_skills_expanded)
        agent_disc_loss = F.cross_entropy(agent_disc_logits, agent_skills_flat)
        
        # 总技能判别器损失
        disc_loss = team_disc_loss + agent_disc_loss
        
        # 更新网络
        self.discriminator_optimizer.zero_grad()
        disc_loss.backward()
        self.discriminator_optimizer.step()
        
        return disc_loss.item()
    
    def update_adaptive_delta(self, coordinator_value_loss, discoverer_value_loss):
        """
        自适应调整Huber Loss的delta参数
        
        参数:
            coordinator_value_loss: 协调器价值损失
            discoverer_value_loss: 发现器价值损失
        """
        if not getattr(self.config, 'huber_adaptive_delta', False):
            return
        
        self.delta_update_count += 1
        
        # 每100次更新调整一次delta
        if self.delta_update_count % 100 == 0:
            decay_rate = getattr(self.config, 'huber_delta_decay', 0.999)
            min_delta = getattr(self.config, 'huber_min_delta', 0.1)
            
            # 根据损失大小调整delta
            if coordinator_value_loss > 1.0:  # 损失较大时增加delta
                self.adaptive_coordinator_delta = min(self.adaptive_coordinator_delta * 1.1, 2.0)
            elif coordinator_value_loss < 0.1:  # 损失较小时减少delta
                self.adaptive_coordinator_delta = max(self.adaptive_coordinator_delta * decay_rate, min_delta)
            
            if discoverer_value_loss > 1.0:
                self.adaptive_discoverer_delta = min(self.adaptive_discoverer_delta * 1.1, 2.0)
            elif discoverer_value_loss < 0.1:
                self.adaptive_discoverer_delta = max(self.adaptive_discoverer_delta * decay_rate, min_delta)
            
            main_logger.debug(f"自适应delta更新: coordinator_delta={self.adaptive_coordinator_delta:.4f}, "
                             f"discoverer_delta={self.adaptive_discoverer_delta:.4f}")
            
            # 记录到TensorBoard
            if hasattr(self, 'writer'):
                self.writer.add_scalar('HuberLoss/AdaptiveCoordinatorDelta', self.adaptive_coordinator_delta, self.global_step)
                self.writer.add_scalar('HuberLoss/AdaptiveDiscovererDelta', self.adaptive_discoverer_delta, self.global_step)

    def update(self):
        """更新所有网络"""
        # 更新全局步数
        self.global_step += 1
        main_logger.debug(f"HMASDAgent.update (step {self.global_step}): self.writer object: {self.writer}")
        
        # 更频繁地检查环境贡献情况（从1000步降至200步）
        if self.global_step % 200 == 0:
            # 获取所有环境的贡献情况
            env_contributions = {}
            for env_id in range(32):  # 假设最多32个并行环境
                env_contributions[env_id] = self.high_level_samples_by_env.get(env_id, 0)
            
            # 找出贡献较少的环境，降低贡献阈值使更多环境被标记
            low_contribution_envs = {env_id: count for env_id, count in env_contributions.items() if count < 3}
            if low_contribution_envs:
                main_logger.info(f"以下环境贡献样本较少，将强制其在下一个技能周期结束时贡献: {low_contribution_envs}")
                # 标记这些环境在下一个技能周期结束时强制贡献样本
                for env_id in low_contribution_envs:
                    self.force_high_level_collection[env_id] = True
                    # 同时将这些环境的奖励阈值重置为0
                    self.env_reward_thresholds[env_id] = 0.0
            
            # 记录高层缓冲区状态
            high_level_buffer_size = len(self.high_level_buffer)
            main_logger.debug(f"当前高层缓冲区大小: {high_level_buffer_size}/{self.config.high_level_batch_size} (当前/所需)")
            
            # 如果高层缓冲区增长过慢，强制所有环境进行贡献
            if high_level_buffer_size < self.config.high_level_batch_size * 0.5 and self.global_step > 5000:
                main_logger.warning(f"高层缓冲区增长过慢 ({high_level_buffer_size}/{self.config.high_level_batch_size})，强制所有环境贡献样本")
                for env_id in range(32):
                    self.force_high_level_collection[env_id] = True
                    self.env_reward_thresholds[env_id] = 0.0
            
            # 记录环境贡献分布到TensorBoard
            if hasattr(self, 'writer'):
                contrib_data = np.zeros(32)
                for env_id, count in env_contributions.items():
                    contrib_data[env_id] = count
                # 记录贡献标准差，衡量是否平衡
                contrib_std = np.std(contrib_data)
                self.writer.add_scalar('Buffer/contribution_stddev', contrib_std, self.global_step)
                # 记录有效贡献环境数量
                contrib_envs = np.sum(contrib_data > 0)
                self.writer.add_scalar('Buffer/contributing_envs_count', contrib_envs, self.global_step)
        
        # 更新技能判别器
        discriminator_loss = self.update_discriminators()
        
        # 更新高层技能协调器
        coordinator_loss, coordinator_policy_loss, coordinator_value_loss, team_skill_entropy, agent_skill_entropy, \
        mean_coord_state_val, mean_coord_agent_val, mean_high_level_reward = self.update_coordinator()
        
        # 更新低层技能发现器
        discoverer_loss, discoverer_policy_loss, discoverer_value_loss, action_entropy, \
        avg_intrinsic_reward, avg_env_comp, avg_team_disc_comp, avg_ind_disc_comp, \
        avg_discoverer_val = self.update_discoverer()
        
        # 更新自适应Huber Loss delta参数
        self.update_adaptive_delta(coordinator_value_loss, discoverer_value_loss)
        
        # 更新训练信息
        self.training_info['high_level_loss'].append(coordinator_loss)
        self.training_info['low_level_loss'].append(discoverer_loss)
        self.training_info['discriminator_loss'].append(discriminator_loss)
        self.training_info['team_skill_entropy'].append(team_skill_entropy) # 真正的团队技能熵
        self.training_info['agent_skill_entropy'].append(agent_skill_entropy) # 个体技能熵，不再是占位符
        self.training_info['action_entropy'].append(action_entropy)
        
        self.training_info['intrinsic_reward_low_level_average'].append(avg_intrinsic_reward)
        self.training_info['intrinsic_reward_env_component'].append(avg_env_comp)
        self.training_info['intrinsic_reward_team_disc_component'].append(avg_team_disc_comp)
        self.training_info['intrinsic_reward_ind_disc_component'].append(avg_ind_disc_comp)
        
        self.training_info['coordinator_state_value_mean'].append(mean_coord_state_val)
        self.training_info['coordinator_agent_value_mean'].append(mean_coord_agent_val)
        self.training_info['discoverer_value_mean'].append(avg_discoverer_val)

        # 记录到TensorBoard
        # 损失函数记录
        self.writer.add_scalar('Losses/Coordinator/Total', coordinator_loss, self.global_step)
        self.writer.add_scalar('Losses/Discoverer/Total', discoverer_loss, self.global_step)
        self.writer.add_scalar('Losses/Discriminator/Total', discriminator_loss, self.global_step)
        
        # 详细损失组成
        self.writer.add_scalar('Losses/Coordinator/Policy', coordinator_policy_loss, self.global_step)
        self.writer.add_scalar('Losses/Coordinator/Value', coordinator_value_loss, self.global_step)
        self.writer.add_scalar('Losses/Discoverer/Policy', discoverer_policy_loss, self.global_step)
        self.writer.add_scalar('Losses/Discoverer/Value', discoverer_value_loss, self.global_step)
        
        # 熵记录
        # 现在分别记录团队和个体技能熵，而不是平均值
        self.writer.add_scalar('Entropy/Coordinator/TeamSkill_Z', team_skill_entropy, self.global_step)
        self.writer.add_scalar('Entropy/Coordinator/AgentSkill_z_Average', agent_skill_entropy, self.global_step)
        self.writer.add_scalar('Entropy/Discoverer/Action', action_entropy, self.global_step)

        # 奖励记录
        # 新增对高层奖励的记录（k步累积环境奖励均值）
        self.writer.add_scalar('Rewards/HighLevel/K_Step_Accumulated_Mean', mean_high_level_reward, self.global_step)
        
        # 内在奖励记录
        self.writer.add_scalar('Rewards/Intrinsic/LowLevel_Average', avg_intrinsic_reward, self.global_step)
        self.writer.add_scalar('Rewards/Intrinsic/Components/Environmental_Portion_Average', avg_env_comp, self.global_step)
        self.writer.add_scalar('Rewards/Intrinsic/Components/TeamDiscriminator_Portion_Average', avg_team_disc_comp, self.global_step)
        self.writer.add_scalar('Rewards/Intrinsic/Components/IndividualDiscriminator_Portion_Average', avg_ind_disc_comp, self.global_step)

        # 价值函数估计记录
        self.writer.add_scalar('ValueEstimates/Coordinator/StateValue_Mean', mean_coord_state_val, self.global_step)
        self.writer.add_scalar('ValueEstimates/Coordinator/AgentValue_Average_Mean', mean_coord_agent_val, self.global_step)
        self.writer.add_scalar('ValueEstimates/Discoverer/Value_Mean', avg_discoverer_val, self.global_step)

        # 添加一个固定的测试值，用于调试TensorBoard显示问题
        self.writer.add_scalar('Debug/test_value', 1.0, self.global_step)
        
        # 每次更新后都刷新数据到硬盘，确保TensorBoard能尽快看到
        self.writer.flush()
        
        # 返回的字典也应包含新指标，方便外部调用者获取
        return {
            'discriminator_loss': discriminator_loss,
            'coordinator_loss': coordinator_loss,
            'coordinator_policy_loss': coordinator_policy_loss,
            'coordinator_value_loss': coordinator_value_loss,
            'discoverer_loss': discoverer_loss,
            'discoverer_policy_loss': discoverer_policy_loss,
            'discoverer_value_loss': discoverer_value_loss,
            'team_skill_entropy': team_skill_entropy, # 团队技能熵
            'agent_skill_entropy': agent_skill_entropy, # 个体技能熵
            'action_entropy': action_entropy, # 低层动作熵
            'avg_intrinsic_reward': avg_intrinsic_reward,
            'avg_env_comp': avg_env_comp,
            'avg_team_disc_comp': avg_team_disc_comp,
            'avg_ind_disc_comp': avg_ind_disc_comp,
            'mean_coord_state_val': mean_coord_state_val,
            'mean_coord_agent_val': mean_coord_agent_val,
            'avg_discoverer_val': avg_discoverer_val,
            'mean_high_level_reward': mean_high_level_reward # 高层奖励均值
        }
    
    def save_model(self, path):
        """保存模型"""
        torch.save({
            'skill_coordinator': self.skill_coordinator.state_dict(),
            'skill_discoverer': self.skill_discoverer.state_dict(),
            'team_discriminator': self.team_discriminator.state_dict(),
            'individual_discriminator': self.individual_discriminator.state_dict(),
            'config': self.config
        }, path)
        main_logger.info(f"模型已保存到 {path}")
    
    def log_skill_distribution(self, team_skill, agent_skills, episode=None):
        """记录技能分配分布到TensorBoard
        
        参数:
            team_skill: 团队技能索引
            agent_skills: 个体技能索引列表
            episode: 如果提供，将作为x轴记录点；否则使用global_step
        """
        if not hasattr(self, 'writer'):
            return
            
        step = episode if episode is not None else self.global_step
        
        # 记录当前团队技能 (瞬时)
        self.writer.add_scalar('Skills/Current/TeamSkill', team_skill, step)
        
        # 记录当前个体技能分布 (瞬时)
        for i, skill_val in enumerate(agent_skills): # Renamed skill to skill_val to avoid conflict
            self.writer.add_scalar(f'Skills/Current/Agent{i}_Skill', skill_val, step)
        
        # 计算并记录当前个体技能的多样性 (瞬时)
        if len(agent_skills) > 0:
            current_skill_counts = {}
            for skill_val in agent_skills:
                current_skill_counts[skill_val] = current_skill_counts.get(skill_val, 0) + 1
            
            n_agents_current = len(agent_skills)
            current_skill_entropy = 0
            for count in current_skill_counts.values():
                p = count / n_agents_current
                if p > 0: # Avoid log(0)
                    current_skill_entropy -= p * np.log(p)
            self.writer.add_scalar('Skills/Current/Diversity', current_skill_entropy, step)

        # 记录整个episode的技能使用计数
        if episode is not None: #只在提供了episode（通常在episode结束时）才记录和重置计数
            for skill_id, count_val in self.episode_team_skill_counts.items():
                self.writer.add_scalar(f'Skills/EpisodeCounts/TeamSkill_{skill_id}', count_val, episode)
            
            for i, agent_counts in enumerate(self.episode_agent_skill_counts):
                for skill_id, count_val in agent_counts.items():
                    self.writer.add_scalar(f'Skills/EpisodeCounts/Agent{i}_Skill_{skill_id}', count_val, episode)
            
            # 重置计数器为下一个episode做准备
            self.episode_team_skill_counts = {}
            # 根据当前智能体数量（如果有）或配置重新初始化，以防智能体数量变化
            num_current_agents = len(agent_skills) if agent_skills is not None and len(agent_skills) > 0 else self.config.n_agents
            self.episode_agent_skill_counts = [{} for _ in range(num_current_agents)]
            # 降级为DEBUG日志，避免频繁输出到控制台
            main_logger.debug(f"Episode {episode} skill counts logged and reset.")

    def load_model(self, path):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.skill_coordinator.load_state_dict(checkpoint['skill_coordinator'])
        self.skill_discoverer.load_state_dict(checkpoint['skill_discoverer'])
        self.team_discriminator.load_state_dict(checkpoint['team_discriminator'])
        self.individual_discriminator.load_state_dict(checkpoint['individual_discriminator'])
        main_logger.info(f"模型已从 {path} 加载")
