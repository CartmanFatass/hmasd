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

from hmasd.networks import SkillCoordinator, SkillDiscoverer, TeamDiscriminator, IndividualDiscriminator
from hmasd.utils import ReplayBuffer, StateSkillDataset, compute_gae, compute_ppo_loss, one_hot

class HMASDAgent:
    """
    层次化多智能体技能发现（HMASD）代理
    """
    def __init__(self, config, log_dir='logs', device=None):
        """
        初始化HMASD代理
        
        参数:
            config: 配置对象，包含所有超参数
            log_dir: TensorBoard日志目录
            device: 计算设备，如果为None则自动检测
        """
        self.config = config
        self.device = device if device is not None else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {self.device}")
        
        # 确保环境维度已设置
        assert config.state_dim is not None, "必须先设置state_dim"
        assert config.obs_dim is not None, "必须先设置obs_dim"
        
        # 初始化TensorBoard
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir)
        self.global_step = 0
        
        # 创建网络
        self.skill_coordinator = SkillCoordinator(config).to(self.device)
        self.skill_discoverer = SkillDiscoverer(config).to(self.device)
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
        self.low_level_buffer = ReplayBuffer(config.buffer_size)
        self.state_skill_dataset = StateSkillDataset(config.buffer_size)
        
        # 其他初始化
        self.current_team_skill = None  # 当前团队技能
        self.current_agent_skills = None  # 当前个体技能列表
        self.skill_change_timer = 0  # 技能更换计时器
        self.episode_rewards = []  # 记录每个完整episode的奖励
        
        # 训练指标
        self.training_info = {
            'high_level_loss': [],
            'low_level_loss': [],
            'discriminator_loss': [],
            'team_skill_entropy': [],
            'agent_skill_entropy': [],
            'action_entropy': [],
            'episode_rewards': []
        }
    
    def select_action(self, observations, agent_skills=None, deterministic=False):
        """
        为所有智能体选择动作
        
        参数:
            observations: 所有智能体的观测 [n_agents, obs_dim]
            agent_skills: 所有智能体的技能 [n_agents]，如果为None则使用当前技能
            deterministic: 是否使用确定性策略
            
        返回:
            actions: 所有智能体的动作 [n_agents, action_dim]
            action_logprobs: 所有智能体的动作对数概率 [n_agents]
        """
        if agent_skills is None:
            agent_skills = self.current_agent_skills
            
        n_agents = observations.shape[0]
        actions = torch.zeros((n_agents, self.config.action_dim), device=self.device)
        action_logprobs = torch.zeros(n_agents, device=self.device)
        
        # 初始化GRU隐藏状态
        self.skill_discoverer.init_hidden(batch_size=1)
        
        with torch.no_grad():
            for i in range(n_agents):
                obs = torch.FloatTensor(observations[i]).unsqueeze(0).to(self.device)
                skill = torch.tensor(agent_skills[i], device=self.device)
                
                action, action_logprob, _ = self.skill_discoverer(obs, skill, deterministic)
                
                actions[i] = action.squeeze(0)
                action_logprobs[i] = action_logprob.squeeze(0)
        
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
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        obs_tensor = torch.FloatTensor(observations).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            team_skill, agent_skills, _, _ = self.skill_coordinator(
                state_tensor, obs_tensor, deterministic
            )
        
        return team_skill.item(), agent_skills.squeeze(0).cpu().numpy()
    
    def step(self, state, observations, ep_t):
        """
        执行一个环境步骤
        
        参数:
            state: 全局状态 [state_dim]
            observations: 所有智能体的观测 [n_agents, obs_dim]
            ep_t: 当前episode中的时间步
            
        返回:
            actions: 所有智能体的动作 [n_agents, action_dim]
            info: 额外信息，如当前技能
        """
        # 判断是否需要重新分配技能
        if ep_t % self.config.k == 0 or self.current_team_skill is None:
            self.current_team_skill, self.current_agent_skills = self.assign_skills(state, observations)
            self.skill_change_timer = 0
            skill_changed = True
        else:
            self.skill_change_timer += 1
            skill_changed = False
            
        # 选择动作
        actions, action_logprobs = self.select_action(observations)
        
        info = {
            'team_skill': self.current_team_skill,
            'agent_skills': self.current_agent_skills,
            'action_logprobs': action_logprobs,
            'skill_changed': skill_changed,
            'skill_timer': self.skill_change_timer
        }
        
        return actions, info
    
    def store_transition(self, state, next_state, observations, next_observations, 
                         actions, rewards, dones, team_skill, agent_skills, action_logprobs):
        """
        存储环境交互经验
        
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
        """
        n_agents = len(agent_skills)
        state_tensor = torch.FloatTensor(state).to(self.device)
        next_state_tensor = torch.FloatTensor(next_state).to(self.device)
        team_skill_tensor = torch.tensor(team_skill, device=self.device)
        
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
            done = dones[i] if isinstance(dones, list) else dones
            
            # 计算个体技能判别器输出
            with torch.no_grad():
                agent_disc_logits = self.individual_discriminator(
                    next_obs.unsqueeze(0), 
                    team_skill_tensor
                )
                agent_disc_log_probs = F.log_softmax(agent_disc_logits, dim=-1)
                agent_skill_log_prob = agent_disc_log_probs[0, agent_skills[i]]
                
            # 计算低层奖励（Eq. 4）
            intrinsic_reward = (
                self.config.lambda_e * rewards + 
                self.config.lambda_D * team_skill_log_prob.item() + 
                self.config.lambda_d * agent_skill_log_prob.item()
            )
            
            # 存储低层经验
            self.low_level_buffer.push(
                state_tensor,                           # 全局状态s
                team_skill_tensor,                      # 团队技能Z
                obs,                                    # 智能体观测o_i
                torch.tensor(agent_skills[i], device=self.device),  # 个体技能z_i
                action,                                 # 动作a_i
                torch.tensor(intrinsic_reward, device=self.device),  # 奖励r_i
                torch.tensor(done, dtype=torch.float, device=self.device),  # 是否结束
                torch.tensor(action_logprobs[i], device=self.device)  # 动作对数概率
            )
            
        # 存储技能判别器训练数据
        observations_tensor = torch.FloatTensor(next_observations).to(self.device)
        agent_skills_tensor = torch.tensor(agent_skills, device=self.device)
        self.state_skill_dataset.push(
            next_state_tensor,
            team_skill_tensor,
            observations_tensor,
            agent_skills_tensor
        )
        
        # 存储高层经验（每k步一次）
        if self.skill_change_timer == self.config.k - 1:
            # 我们假设这个函数在存储完当前步之后将被调用
            # 所以这里我们获取存储的这一段k步的总奖励
            self.high_level_buffer.push(
                state_tensor,                  # 全局状态s
                team_skill_tensor,             # 团队技能Z
                observations_tensor,           # 所有智能体观测o
                agent_skills_tensor,           # 所有个体技能z
                torch.tensor(rewards, device=self.device)  # 当前步奖励（高层奖励是k步内总奖励，在更新时计算）
            )
    
    def update_coordinator(self):
        """更新高层技能协调器网络"""
        if len(self.high_level_buffer) < self.config.batch_size:
            return 0, 0, 0, 0
            
        # 从缓冲区采样数据
        batch = self.high_level_buffer.sample(self.config.batch_size)
        states, team_skills, observations, agent_skills, rewards = zip(*batch)
        
        states = torch.stack(states)
        team_skills = torch.stack(team_skills)
        observations = torch.stack(observations)
        agent_skills = torch.stack(agent_skills)
        rewards = torch.stack(rewards)
        
        # 假设这里的rewards已经是累积k步的高层奖励
        # 如果不是，我们需要计算累积奖励
        
        # 获取当前状态价值
        state_values, agent_values = self.skill_coordinator.get_value(states, observations)
        
        # 由于我们假设每个高层经验都是一个k步序列的端点，
        # 所以我们可以假设下一状态价值为0（或者可以从新的状态计算）
        next_values = torch.zeros_like(state_values)
        
        # 计算GAE
        dones = torch.zeros_like(rewards)  # 假设高层经验不包含终止信息
        advantages, returns = compute_gae(rewards, state_values, next_values, dones, 
                                         self.config.gamma, self.config.gae_lambda)
        
        # 获取当前策略
        Z, z, Z_logits, z_logits = self.skill_coordinator(states, observations)
        
        # 重新计算团队技能概率分布
        Z_dist = Categorical(logits=Z_logits)
        Z_log_probs = Z_dist.log_prob(team_skills)
        Z_entropy = Z_dist.entropy().mean()
        
        # 计算高层策略损失
        Z_ratio = torch.exp(Z_log_probs)  # 假设old_log_probs=0，可以根据需要修改
        Z_surr1 = Z_ratio * advantages
        Z_surr2 = torch.clamp(Z_ratio, 1.0 - self.config.clip_epsilon, 1.0 + self.config.clip_epsilon) * advantages
        Z_policy_loss = -torch.min(Z_surr1, Z_surr2).mean()
        
        # 计算高层价值损失
        state_values = state_values.float()
        returns = returns.float()
        # 打印当前形状，便于调试
        print(f"Debug - state_values shape: {state_values.shape}, returns shape: {returns.shape}, size: {returns.numel()}")
        
        # 如果维度不匹配，显式调整returns的维度
        if returns.dim() > state_values.dim() or returns.size(-1) != state_values.size(-1):
            try:
                # 不使用view_as，而是根据state_values的形状明确重塑
                returns = returns.reshape(state_values.shape)
            except RuntimeError as e:
                # 如果重塑失败，尝试展平两个张量
                state_values_flat = state_values.reshape(-1)
                returns_flat = returns.reshape(-1)
                # 确保长度匹配，必要时截断较长的张量
                min_len = min(state_values_flat.size(0), returns_flat.size(0))
                state_values_flat = state_values_flat[:min_len]
                returns_flat = returns_flat[:min_len]
                Z_value_loss = F.mse_loss(state_values_flat, returns_flat)
                
                # 初始化智能体策略损失
                z_policy_losses = []
                z_entropy_losses = []
                z_value_losses = []
                
                # 处理每个智能体的个体技能损失
                for i in range(self.config.n_agents):
                    zi_dist = Categorical(logits=z_logits[i])
                    zi_log_probs = zi_dist.log_prob(agent_skills[:, i])
                    zi_entropy = zi_dist.entropy().mean()
                    
                    zi_ratio = torch.exp(zi_log_probs)  # 假设old_log_probs=0
                    zi_surr1 = zi_ratio * advantages
                    zi_surr2 = torch.clamp(zi_ratio, 1.0 - self.config.clip_epsilon, 1.0 + self.config.clip_epsilon) * advantages
                    zi_policy_loss = -torch.min(zi_surr1, zi_surr2).mean()
                    
                    z_policy_losses.append(zi_policy_loss)
                    z_entropy_losses.append(zi_entropy)
                
                # 合并所有智能体的损失
                z_policy_loss = torch.stack(z_policy_losses).mean()
                z_entropy = torch.stack(z_entropy_losses).mean()
                z_value_loss = torch.tensor(0.0, device=self.device)  # 这种情况下没有智能体价值损失
                
                # 总策略损失
                policy_loss = Z_policy_loss + z_policy_loss
                
                # 总价值损失
                value_loss = Z_value_loss + z_value_loss
                
                # 总熵损失
                entropy_loss = -(Z_entropy + z_entropy) * self.config.lambda_h
                
                # 总损失
                loss = policy_loss + self.config.value_loss_coef * value_loss + entropy_loss
                
                # 更新网络
                self.coordinator_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.skill_coordinator.parameters(), self.config.max_grad_norm)
                self.coordinator_optimizer.step()
                
                return loss.item(), policy_loss.item(), value_loss.item(), (Z_entropy.item() + z_entropy.item()) / 2
                
        Z_value_loss = F.mse_loss(state_values, returns)
        
        # 初始化智能体策略损失
        z_policy_losses = []
        z_entropy_losses = []
        z_value_losses = []
        
        # 处理每个智能体的个体技能损失
        for i in range(self.config.n_agents):
            zi_dist = Categorical(logits=z_logits[i])
            zi_log_probs = zi_dist.log_prob(agent_skills[:, i])
            zi_entropy = zi_dist.entropy().mean()
            
            zi_ratio = torch.exp(zi_log_probs)  # 假设old_log_probs=0
            zi_surr1 = zi_ratio * advantages
            zi_surr2 = torch.clamp(zi_ratio, 1.0 - self.config.clip_epsilon, 1.0 + self.config.clip_epsilon) * advantages
            zi_policy_loss = -torch.min(zi_surr1, zi_surr2).mean()
            
            z_policy_losses.append(zi_policy_loss)
            z_entropy_losses.append(zi_entropy)
            
            if i < len(agent_values):
                # 确保数据类型和维度匹配
                agent_value = agent_values[i].float()
                returns_i = returns.float()
                print(f"Debug - agent_value[{i}] shape: {agent_value.shape}, returns_i shape: {returns_i.shape}")
                
                # 如果维度不匹配，显式调整returns的维度
                if returns_i.dim() > agent_value.dim() or returns_i.size(-1) != agent_value.size(-1):
                    try:
                        # 不使用view_as，而是根据agent_value的形状明确重塑
                        returns_i = returns_i.reshape(agent_value.shape)
                    except RuntimeError as e:
                        # 如果重塑失败，尝试展平两个张量
                        agent_value_flat = agent_value.reshape(-1)
                        returns_i_flat = returns_i.reshape(-1)
                        # 确保长度匹配，必要时截断较长的张量
                        min_len = min(agent_value_flat.size(0), returns_i_flat.size(0))
                        agent_value_flat = agent_value_flat[:min_len]
                        returns_i_flat = returns_i_flat[:min_len]
                        zi_value_loss = F.mse_loss(agent_value_flat, returns_i_flat)
                        z_value_losses.append(zi_value_loss)
                        continue
                
                zi_value_loss = F.mse_loss(agent_value, returns_i)
                z_value_losses.append(zi_value_loss)
        
        # 合并所有智能体的损失
        z_policy_loss = torch.stack(z_policy_losses).mean()
        z_entropy = torch.stack(z_entropy_losses).mean()
        
        if z_value_losses:
            z_value_loss = torch.stack(z_value_losses).mean()
        else:
            z_value_loss = torch.tensor(0.0, device=self.device)
        
        # 总策略损失
        policy_loss = Z_policy_loss + z_policy_loss
        
        # 总价值损失
        value_loss = Z_value_loss + z_value_loss
        
        # 总熵损失
        entropy_loss = -(Z_entropy + z_entropy) * self.config.lambda_h
        
        # 总损失
        loss = policy_loss + self.config.value_loss_coef * value_loss + entropy_loss
        
        # 更新网络
        self.coordinator_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.skill_coordinator.parameters(), self.config.max_grad_norm)
        self.coordinator_optimizer.step()
        
        return loss.item(), policy_loss.item(), value_loss.item(), (Z_entropy.item() + z_entropy.item()) / 2
    
    def update_discoverer(self):
        """更新低层技能发现器网络"""
        if len(self.low_level_buffer) < self.config.batch_size:
            return 0, 0, 0, 0
        
        # 从缓冲区采样数据
        batch = self.low_level_buffer.sample(self.config.batch_size)
        states, team_skills, observations, agent_skills, actions, rewards, dones, old_log_probs = zip(*batch)
        
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
        advantages, returns = compute_gae(rewards, values, next_values, dones, 
                                         self.config.gamma, self.config.gae_lambda)
        
        # 重新初始化GRU隐藏状态
        self.skill_discoverer.init_hidden(batch_size=self.config.batch_size)
        
        # 获取当前策略
        _, action_log_probs, action_dist = self.skill_discoverer(observations, agent_skills)
        
        # 计算策略比率
        ratios = torch.exp(action_log_probs - old_log_probs)
        
        # 限制策略比率
        surr1 = ratios * advantages
        surr2 = torch.clamp(ratios, 1.0 - self.config.clip_epsilon, 1.0 + self.config.clip_epsilon) * advantages
        
        # 计算策略损失
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # 重新初始化GRU隐藏状态
        self.skill_discoverer.init_hidden(batch_size=self.config.batch_size)
        
        # 计算价值损失
        current_values = self.skill_discoverer.get_value(states, team_skills)
        # 确保维度匹配并转换为float32类型
        current_values = current_values.float()
        returns = returns.float()
        # 打印当前形状，便于调试
        print(f"Debug - current_values shape: {current_values.shape}, returns shape: {returns.shape}, size: {returns.numel()}")
        
        # 如果维度不匹配，显式调整returns的维度
        if returns.dim() > current_values.dim() or returns.size(-1) != current_values.size(-1):
            try:
                # 不使用view_as，而是根据current_values的形状明确重塑
                returns = returns.reshape(current_values.shape)
            except RuntimeError as e:
                # 如果重塑失败，尝试展平两个张量
                current_values_flat = current_values.reshape(-1)
                returns_flat = returns.reshape(-1)
                # 确保长度匹配，必要时截断较长的张量
                min_len = min(current_values_flat.size(0), returns_flat.size(0))
                current_values_flat = current_values_flat[:min_len]
                returns_flat = returns_flat[:min_len]
                value_loss = F.mse_loss(current_values_flat, returns_flat)
                
                # 计算熵损失
                entropy_loss = -action_dist.entropy().mean() * self.config.lambda_l
                
                # 总损失
                loss = policy_loss + self.config.value_loss_coef * value_loss + entropy_loss
                
                # 更新网络
                self.discoverer_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.skill_discoverer.parameters(), self.config.max_grad_norm)
                self.discoverer_optimizer.step()
                
                return loss.item(), policy_loss.item(), value_loss.item(), -entropy_loss.item() / self.config.lambda_l
                
        value_loss = F.mse_loss(current_values, returns)
        
        # 计算熵损失
        entropy_loss = -action_dist.entropy().mean() * self.config.lambda_l
        
        # 总损失
        loss = policy_loss + self.config.value_loss_coef * value_loss + entropy_loss
        
        # 更新网络
        self.discoverer_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.skill_discoverer.parameters(), self.config.max_grad_norm)
        self.discoverer_optimizer.step()
        
        return loss.item(), policy_loss.item(), value_loss.item(), -entropy_loss.item() / self.config.lambda_l
    
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
    
    def update(self):
        """更新所有网络"""
        # 更新全局步数
        self.global_step += 1
        
        # 更新技能判别器
        discriminator_loss = self.update_discriminators()
        
        # 更新高层技能协调器
        coordinator_loss, coordinator_policy_loss, coordinator_value_loss, skill_entropy = self.update_coordinator()
        
        # 更新低层技能发现器
        discoverer_loss, discoverer_policy_loss, discoverer_value_loss, action_entropy = self.update_discoverer()
        
        # 更新训练信息
        self.training_info['high_level_loss'].append(coordinator_loss)
        self.training_info['low_level_loss'].append(discoverer_loss)
        self.training_info['discriminator_loss'].append(discriminator_loss)
        self.training_info['team_skill_entropy'].append(skill_entropy)
        self.training_info['action_entropy'].append(action_entropy)
        
        # 记录到TensorBoard
        # 损失函数记录
        self.writer.add_scalar('Loss/high_level', coordinator_loss, self.global_step)
        self.writer.add_scalar('Loss/low_level', discoverer_loss, self.global_step)
        self.writer.add_scalar('Loss/discriminator', discriminator_loss, self.global_step)
        
        # 详细损失组成
        self.writer.add_scalar('Loss/high_level_policy', coordinator_policy_loss, self.global_step)
        self.writer.add_scalar('Loss/high_level_value', coordinator_value_loss, self.global_step)
        self.writer.add_scalar('Loss/low_level_policy', discoverer_policy_loss, self.global_step)
        self.writer.add_scalar('Loss/low_level_value', discoverer_value_loss, self.global_step)
        
        # 熵记录
        self.writer.add_scalar('Entropy/team_skill', skill_entropy, self.global_step)
        self.writer.add_scalar('Entropy/action', action_entropy, self.global_step)
        
        # 定期刷新数据到硬盘
        if self.global_step % 100 == 0:
            self.writer.flush()
        
        return {
            'discriminator_loss': discriminator_loss,
            'coordinator_loss': coordinator_loss,
            'discoverer_loss': discoverer_loss,
            'skill_entropy': skill_entropy,
            'action_entropy': action_entropy
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
        print(f"模型已保存到 {path}")
    
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
        
        # 记录当前团队技能
        self.writer.add_scalar('Skills/team_skill', team_skill, step)
        
        # 记录个体技能分布
        for i, skill in enumerate(agent_skills):
            self.writer.add_scalar(f'Skills/agent{i}_skill', skill, step)
        
        # 计算技能熵（衡量技能多样性）
        if len(agent_skills) > 0:
            # 创建技能计数字典
            skill_counts = {}
            for skill in agent_skills:
                if skill not in skill_counts:
                    skill_counts[skill] = 0
                skill_counts[skill] += 1
            
            # 计算技能分布概率
            n_agents = len(agent_skills)
            skill_entropy = 0
            for skill, count in skill_counts.items():
                p = count / n_agents
                skill_entropy -= p * np.log(p)
            
            # 记录技能熵
            self.writer.add_scalar('Skills/diversity', skill_entropy, step)
    
    def load_model(self, path):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.skill_coordinator.load_state_dict(checkpoint['skill_coordinator'])
        self.skill_discoverer.load_state_dict(checkpoint['skill_discoverer'])
        self.team_discriminator.load_state_dict(checkpoint['team_discriminator'])
        self.individual_discriminator.load_state_dict(checkpoint['individual_discriminator'])
        print(f"模型已从 {path} 加载")
