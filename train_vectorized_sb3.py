#!/usr/bin/env python3
"""
基于Stable Baselines3向量化环境的HMASD训练脚本
使用SubprocVecEnv实现真正的并行环境执行，大幅提升训练效率

核心优势：
1. 32个并行进程同时执行环境
2. 批量数据收集和处理
3. 消除锁竞争和线程同步问题
4. GPU友好的批量计算
5. 简化的数据流架构
"""

import os
import sys
import time
import numpy as np
import torch
import argparse
import multiprocessing as mp
from datetime import datetime
from collections import deque
import logging

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 设置matplotlib后端（避免多进程问题）
import matplotlib
matplotlib.use('Agg')

from logger import get_logger, init_multiproc_logging, LOG_LEVELS, shutdown_logging
from config import Config
from hmasd.thread_safe_hmasd_agent import ThreadSafeHMASDAgent
from envs.pettingzoo.scenario1 import UAVBaseStationEnv
from envs.pettingzoo.scenario2 import UAVCooperativeNetworkEnv
from envs.pettingzoo.env_adapter import ParallelToArrayAdapter

# 导入 Stable Baselines3 的向量化环境
from stable_baselines3.common.vec_env import SubprocVecEnv, VecEnv
from stable_baselines3.common.vec_env.base_vec_env import VecEnvWrapper

class UAVVecEnvWrapper(VecEnvWrapper):
    """UAV环境的向量化包装器"""
    
    def __init__(self, venv):
        super().__init__(venv)
        self.n_uavs = None
        self.state_dim = None
        self.obs_dim = None
        self.action_dim = None
        
        # 从第一个环境获取维度信息
        self._get_env_dims()
    
    def _get_env_dims(self):
        """获取环境维度信息"""
        # 临时创建一个环境来获取维度
        temp_env = self.venv.envs[0].env
        if hasattr(temp_env, 'unwrapped'):
            temp_env = temp_env.unwrapped
        
        self.n_uavs = temp_env.n_uavs
        self.state_dim = temp_env.state_dim
        self.obs_dim = temp_env.obs_dim
        self.action_dim = temp_env.action_dim
    
    def reset(self):
        """重置所有环境"""
        observations = self.venv.reset()
        return observations
    
    def step_async(self, actions):
        """异步执行步骤"""
        self.venv.step_async(actions)
    
    def step_wait(self):
        """等待步骤完成"""
        return self.venv.step_wait()
    
    def get_global_states(self):
        """获取所有环境的全局状态"""
        states = []
        for env in self.venv.envs:
            if hasattr(env.env, 'get_state'):
                state = env.env.get_state()
            elif hasattr(env.env, 'unwrapped') and hasattr(env.env.unwrapped, 'get_state'):
                state = env.env.unwrapped.get_state()
            else:
                # 回退：使用零状态
                state = np.zeros(self.state_dim)
            states.append(state)
        return np.array(states)

class VectorizedHMASDTrainer:
    """基于SB3向量化环境的HMASD训练器"""
    
    def __init__(self, config, args=None):
        self.config = config
        self.args = args or argparse.Namespace()
        
        # 验证并设置训练模式
        config.rollout_based_training = True
        config.episode_based_training = False
        config.sync_training_mode = False
        
        # 向量化环境配置
        self.n_envs = getattr(args, 'n_envs', 32)
        
        # 设置设备
        self.device = self._get_device()
        
        # 创建日志目录
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_dir = f"logs/vectorized_sb3_training_{timestamp}"
        os.makedirs(self.log_dir, exist_ok=True)
        
        # 初始化日志系统
        self._init_logging()
        
        # 统计信息
        self.start_time = None
        self.total_updates = 0
        self.total_samples = 0
        self.total_episodes = 0
        
        # 性能监控
        self.performance_stats = {
            'rollout_times': deque(maxlen=100),
            'update_times': deque(maxlen=100),
            'samples_per_second': deque(maxlen=100),
            'last_log_time': time.time()
        }
        
        self.logger.info("基于SB3向量化环境的HMASD训练器初始化完成")
        self.logger.info(f"日志目录: {self.log_dir}")
        self.logger.info(f"向量化环境数量: {self.n_envs}")
        self.logger.info(config.get_rollout_summary())
    
    def _get_device(self):
        """获取计算设备"""
        device_pref = getattr(self.args, 'device', 'auto')
        if device_pref == 'auto':
            return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        elif device_pref == 'cuda':
            if torch.cuda.is_available():
                return torch.device('cuda')
            else:
                self.logger.warning("请求CUDA但未检测到GPU，使用CPU")
                return torch.device('cpu')
        else:
            return torch.device('cpu')
    
    def _init_logging(self):
        """初始化日志系统"""
        log_level = getattr(self.args, 'log_level', 'INFO')
        console_level = getattr(self.args, 'console_log_level', 'INFO')
        
        init_multiproc_logging(
            log_dir=self.log_dir,
            log_file='vectorized_sb3_training.log',
            file_level=LOG_LEVELS.get(log_level.lower(), 20),
            console_level=LOG_LEVELS.get(console_level.lower(), 20)
        )
        
        self.logger = get_logger("VectorizedHMASDTrainer")
    
    def create_env_factory(self):
        """创建环境工厂函数"""
        scenario = getattr(self.args, 'scenario', 2)
        n_uavs = getattr(self.args, 'n_uavs', 5)
        n_users = getattr(self.args, 'n_users', 50)
        user_distribution = getattr(self.args, 'user_distribution', 'uniform')
        channel_model = getattr(self.args, 'channel_model', '3gpp-36777')
        max_hops = getattr(self.args, 'max_hops', 3)
        
        def make_env():
            def _init():
                env_seed = np.random.randint(0, 10000)  # 随机种子
                if scenario == 1:
                    raw_env = UAVBaseStationEnv(
                        n_uavs=n_uavs,
                        n_users=n_users,
                        user_distribution=user_distribution,
                        channel_model=channel_model,
                        seed=env_seed
                    )
                elif scenario == 2:
                    raw_env = UAVCooperativeNetworkEnv(
                        n_uavs=n_uavs,
                        n_users=n_users,
                        max_hops=max_hops,
                        user_distribution=user_distribution,
                        channel_model=channel_model,
                        seed=env_seed
                    )
                else:
                    raise ValueError(f"未知场景: {scenario}")
                
                env = ParallelToArrayAdapter(raw_env, seed=env_seed)
                return env
            return _init
        
        return make_env
    
    def create_vectorized_env(self):
        """创建向量化环境"""
        self.logger.info(f"创建 {self.n_envs} 个向量化环境...")
        
        env_factory = self.create_env_factory()
        env_fns = [env_factory() for _ in range(self.n_envs)]
        
        # 创建SubprocVecEnv
        vec_env = SubprocVecEnv(env_fns, start_method='spawn')
        
        # 包装为UAVVecEnvWrapper
        self.vec_env = UAVVecEnvWrapper(vec_env)
        
        self.logger.info(f"向量化环境创建完成")
        self.logger.info(f"环境维度: n_uavs={self.vec_env.n_uavs}, "
                        f"state_dim={self.vec_env.state_dim}, "
                        f"obs_dim={self.vec_env.obs_dim}, "
                        f"action_dim={self.vec_env.action_dim}")
        
        return self.vec_env
    
    def initialize_agent(self):
        """初始化HMASD代理"""
        # 创建向量化环境
        vec_env = self.create_vectorized_env()
        
        # 更新配置
        self.config.update_env_dims(vec_env.state_dim, vec_env.obs_dim)
        self.config.n_agents = vec_env.n_uavs
        
        # 创建代理
        self.agent = ThreadSafeHMASDAgent(
            config=self.config,
            log_dir=self.log_dir,
            device=self.device,
            debug=getattr(self.args, 'debug', False)
        )
        
        self.logger.info("HMASD代理初始化完成")
    
    def collect_vectorized_rollout(self):
        """向量化rollout数据收集"""
        rollout_start_time = time.time()
        
        # 重置所有环境
        observations = self.vec_env.reset()  # (n_envs, n_agents, obs_dim)
        states = self.vec_env.get_global_states()  # (n_envs, state_dim)
        
        # 存储收集的经验
        collected_experiences = {
            'high_level': [],
            'low_level': [],
            'state_skill': []
        }
        
        # 技能状态跟踪
        accumulated_rewards = np.zeros(self.n_envs)
        skill_timers = np.zeros(self.n_envs)
        current_team_skills = None
        current_agent_skills = None
        
        self.logger.info(f"开始向量化rollout收集，目标步数: {self.config.rollout_length}")
        
        for step in range(self.config.rollout_length):
            step_start_time = time.time()
            
            # 批量技能分配
            if step % self.config.k == 0 or current_team_skills is None:
                team_skills, agent_skills, skill_log_probs = self._assign_skills_batch(states, observations)
                current_team_skills = team_skills
                current_agent_skills = agent_skills
                
                # 存储高层经验（如果不是第一步）
                if step > 0:
                    self._store_high_level_experiences_batch(
                        states, current_team_skills, observations, current_agent_skills,
                        accumulated_rewards, skill_log_probs, collected_experiences
                    )
                    accumulated_rewards.fill(0.0)  # 重置累积奖励
                    skill_timers.fill(0)
            
            # 批量动作选择
            actions, action_logprobs = self._select_actions_batch(observations, current_agent_skills)
            
            # 向量化环境步骤
            next_observations, rewards, dones, infos = self.vec_env.step(actions)
            next_states = self.vec_env.get_global_states()
            
            # 累积奖励
            accumulated_rewards += rewards
            skill_timers += 1
            
            # 存储低层经验
            self._store_low_level_experiences_batch(
                states, observations, actions, rewards, next_states, next_observations,
                dones, current_team_skills, current_agent_skills, action_logprobs,
                skill_log_probs, collected_experiences
            )
            
            # 存储状态技能数据
            self._store_state_skill_experiences_batch(
                next_states, current_team_skills, next_observations, current_agent_skills,
                collected_experiences
            )
            
            # 更新状态
            states = next_states
            observations = next_observations
            
            # 处理环境重置
            if np.any(dones):
                self._handle_environment_resets(dones)
            
            # 性能监控
            step_time = time.time() - step_start_time
            if step % 100 == 0:
                self.logger.debug(f"步骤 {step}/{self.config.rollout_length}, "
                                f"步骤耗时: {step_time*1000:.2f}ms")
        
        # 存储最后的高层经验
        self._store_high_level_experiences_batch(
            states, current_team_skills, observations, current_agent_skills,
            accumulated_rewards, skill_log_probs, collected_experiences
        )
        
        rollout_time = time.time() - rollout_start_time
        self.performance_stats['rollout_times'].append(rollout_time)
        
        # 统计收集的经验数量
        high_level_count = len(collected_experiences['high_level'])
        low_level_count = len(collected_experiences['low_level'])
        state_skill_count = len(collected_experiences['state_skill'])
        
        self.logger.info(f"向量化rollout收集完成，耗时: {rollout_time:.2f}s")
        self.logger.info(f"收集经验: 高层={high_level_count}, 低层={low_level_count}, 状态技能={state_skill_count}")
        
        return collected_experiences
    
    def _assign_skills_batch(self, states, observations):
        """批量技能分配"""
        try:
            # 转换为tensor
            states_tensor = torch.FloatTensor(states).to(self.device)
            observations_tensor = torch.FloatTensor(observations).to(self.device)
            
            # 批量分配技能
            team_skills_list = []
            agent_skills_list = []
            skill_log_probs_list = []
            
            for i in range(self.n_envs):
                team_skill, agent_skills, log_probs = self.agent.assign_skills(
                    states_tensor[i], observations_tensor[i], deterministic=False
                )
                
                # 转换为numpy
                if isinstance(team_skill, torch.Tensor):
                    team_skill = team_skill.cpu().item()
                if isinstance(agent_skills, torch.Tensor):
                    agent_skills = agent_skills.cpu().numpy()
                
                team_skills_list.append(team_skill)
                agent_skills_list.append(agent_skills)
                skill_log_probs_list.append(log_probs)
            
            return np.array(team_skills_list), np.array(agent_skills_list), skill_log_probs_list
            
        except Exception as e:
            self.logger.error(f"批量技能分配失败: {e}")
            # 返回默认技能
            return (np.zeros(self.n_envs, dtype=int), 
                   np.zeros((self.n_envs, self.config.n_agents), dtype=int),
                   [{'team_log_prob': 0.0, 'agent_log_probs': [0.0] * self.config.n_agents} for _ in range(self.n_envs)])
    
    def _select_actions_batch(self, observations, agent_skills):
        """批量动作选择"""
        try:
            observations_tensor = torch.FloatTensor(observations).to(self.device)
            agent_skills_tensor = torch.LongTensor(agent_skills).to(self.device)
            
            actions_list = []
            action_logprobs_list = []
            
            for i in range(self.n_envs):
                actions, action_logprobs = self.agent.select_action(
                    observations_tensor[i], agent_skills_tensor[i], 
                    deterministic=False, env_id=i
                )
                
                # 转换为numpy
                if isinstance(actions, torch.Tensor):
                    actions = actions.cpu().detach().numpy()
                if isinstance(action_logprobs, torch.Tensor):
                    action_logprobs = action_logprobs.cpu().detach().numpy()
                
                actions_list.append(actions)
                action_logprobs_list.append(action_logprobs)
            
            return np.array(actions_list), np.array(action_logprobs_list)
            
        except Exception as e:
            self.logger.error(f"批量动作选择失败: {e}")
            # 返回随机动作
            return (np.random.randn(self.n_envs, self.config.n_agents, self.config.action_dim),
                   np.zeros((self.n_envs, self.config.n_agents)))
    
    def _store_high_level_experiences_batch(self, states, team_skills, observations, agent_skills,
                                          accumulated_rewards, skill_log_probs, collected_experiences):
        """批量存储高层经验"""
        for i in range(self.n_envs):
            experience = {
                'experience_type': 'high_level',
                'env_id': i,
                'state': states[i].copy(),
                'team_skill': team_skills[i],
                'observations': observations[i].copy(),
                'agent_skills': agent_skills[i].copy(),
                'accumulated_reward': accumulated_rewards[i],
                'skill_log_probs': skill_log_probs[i] if skill_log_probs else None,
                'timestamp': time.time()
            }
            collected_experiences['high_level'].append(experience)
    
    def _store_low_level_experiences_batch(self, states, observations, actions, rewards, next_states,
                                         next_observations, dones, team_skills, agent_skills,
                                         action_logprobs, skill_log_probs, collected_experiences):
        """批量存储低层经验"""
        for i in range(self.n_envs):
            experience = {
                'experience_type': 'low_level',
                'env_id': i,
                'state': states[i].copy(),
                'observations': observations[i].copy(),
                'actions': actions[i].copy(),
                'rewards': rewards[i],
                'next_state': next_states[i].copy(),
                'next_observations': next_observations[i].copy(),
                'dones': dones[i],
                'team_skill': team_skills[i],
                'agent_skills': agent_skills[i].copy(),
                'action_logprobs': action_logprobs[i].copy(),
                'skill_log_probs': skill_log_probs[i] if skill_log_probs else None,
                'timestamp': time.time()
            }
            collected_experiences['low_level'].append(experience)
    
    def _store_state_skill_experiences_batch(self, states, team_skills, observations, agent_skills,
                                           collected_experiences):
        """批量存储状态技能经验"""
        for i in range(self.n_envs):
            experience = {
                'experience_type': 'state_skill',
                'env_id': i,
                'state': states[i].copy(),
                'team_skill': team_skills[i],
                'observations': observations[i].copy(),
                'agent_skills': agent_skills[i].copy(),
                'timestamp': time.time()
            }
            collected_experiences['state_skill'].append(experience)
    
    def _handle_environment_resets(self, dones):
        """处理环境重置"""
        reset_count = np.sum(dones)
        if reset_count > 0:
            self.total_episodes += reset_count
            self.logger.debug(f"重置了 {reset_count} 个环境")
    
    def store_experiences_to_agent(self, collected_experiences):
        """将收集的经验存储到代理"""
        store_start_time = time.time()
        
        stored_counts = {'high_level': 0, 'low_level': 0, 'state_skill': 0}
        
        # 存储高层经验
        for exp in collected_experiences['high_level']:
            success = self.agent.store_high_level_transition(
                state=exp['state'],
                team_skill=exp['team_skill'],
                observations=exp['observations'],
                agent_skills=exp['agent_skills'],
                accumulated_reward=exp['accumulated_reward'],
                skill_log_probs=exp.get('skill_log_probs'),
                worker_id=exp['env_id']
            )
            if success:
                stored_counts['high_level'] += 1
        
        # 存储低层经验
        for exp in collected_experiences['low_level']:
            success = self.agent.store_low_level_transition(
                state=exp['state'],
                next_state=exp['next_state'],
                observations=exp['observations'],
                next_observations=exp['next_observations'],
                actions=exp['actions'],
                rewards=exp['rewards'],
                dones=exp['dones'],
                team_skill=exp['team_skill'],
                agent_skills=exp['agent_skills'],
                action_logprobs=exp['action_logprobs'],
                skill_log_probs=exp.get('skill_log_probs'),
                worker_id=exp['env_id']
            )
            if success:
                stored_counts['low_level'] += 1
        
        # 存储状态技能数据
        for exp in collected_experiences['state_skill']:
            try:
                state_tensor = torch.FloatTensor(exp['state']).to(self.device)
                team_skill_tensor = torch.tensor(exp['team_skill'], device=self.device)
                observations_tensor = torch.FloatTensor(exp['observations']).to(self.device)
                agent_skills_tensor = torch.tensor(exp['agent_skills'], device=self.device)
                
                self.agent.state_skill_dataset.push(
                    state_tensor, team_skill_tensor, observations_tensor, agent_skills_tensor
                )
                stored_counts['state_skill'] += 1
            except Exception as e:
                self.logger.error(f"状态技能数据存储失败: {e}")
        
        store_time = time.time() - store_start_time
        
        self.logger.info(f"经验存储完成，耗时: {store_time:.3f}s")
        self.logger.info(f"存储统计: 高层={stored_counts['high_level']}, "
                        f"低层={stored_counts['low_level']}, "
                        f"状态技能={stored_counts['state_skill']}")
        
        return stored_counts
    
    def perform_update(self):
        """执行模型更新"""
        update_start_time = time.time()
        
        try:
            self.logger.info("开始模型更新...")
            
            update_info = self.agent.rollout_update()
            
            if update_info:
                self.total_updates += 1
                update_time = time.time() - update_start_time
                self.performance_stats['update_times'].append(update_time)
                
                self.logger.info(f"模型更新完成 #{self.total_updates}，耗时: {update_time:.3f}s")
                
                # 记录更新信息
                if isinstance(update_info, dict):
                    for key, value in update_info.items():
                        if isinstance(value, (int, float)):
                            self.logger.debug(f"更新指标 {key}: {value:.6f}")
                
                return True
            else:
                self.logger.warning("模型更新返回None")
                return False
                
        except Exception as e:
            self.logger.error(f"模型更新失败: {e}")
            return False
    
    def log_training_progress(self, current_samples, total_samples):
        """记录训练进度"""
        current_time = time.time()
        
        if current_time - self.performance_stats['last_log_time'] < 60:
            return  # 每分钟记录一次
        
        progress_percent = (current_samples / total_samples) * 100
        elapsed_time = current_time - self.start_time
        
        # 计算速度
        samples_per_second = current_samples / elapsed_time if elapsed_time > 0 else 0
        self.performance_stats['samples_per_second'].append(samples_per_second)
        
        # 计算平均性能
        avg_rollout_time = np.mean(self.performance_stats['rollout_times']) if self.performance_stats['rollout_times'] else 0
        avg_update_time = np.mean(self.performance_stats['update_times']) if self.performance_stats['update_times'] else 0
        avg_samples_per_sec = np.mean(self.performance_stats['samples_per_second']) if self.performance_stats['samples_per_second'] else 0
        
        # 估计剩余时间
        if samples_per_second > 0:
            remaining_samples = total_samples - current_samples
            remaining_time = remaining_samples / samples_per_second
        else:
            remaining_time = 0
        
        self.logger.info(f"训练进度: {progress_percent:.1f}% ({current_samples:,} / {total_samples:,} 样本)")
        self.logger.info(f"时间: 已用={elapsed_time/3600:.1f}h, 预计剩余={remaining_time/3600:.1f}h")
        self.logger.info(f"性能: 样本速度={avg_samples_per_sec:.1f}/s, "
                        f"平均rollout={avg_rollout_time:.2f}s, 平均更新={avg_update_time:.2f}s")
        self.logger.info(f"统计: 更新次数={self.total_updates}, Episodes={self.total_episodes}")
        
        # GPU内存使用（如果有GPU）
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated(self.device) / 1024**3
            memory_reserved = torch.cuda.memory_reserved(self.device) / 1024**3
            self.logger.info(f"GPU内存: 已分配={memory_allocated:.2f}GB, 已保留={memory_reserved:.2f}GB")
        
        self.performance_stats['last_log_time'] = current_time
    
    def save_model(self):
        """保存模型"""
        try:
            model_path = os.path.join(self.log_dir, 'vectorized_sb3_model.pt')
            self.agent.save_model(model_path)
            self.logger.info(f"模型已保存: {model_path}")
        except Exception as e:
            self.logger.error(f"保存模型失败: {e}")
    
    def cleanup(self):
        """清理资源"""
        try:
            # 关闭向量化环境
            if hasattr(self, 'vec_env'):
                self.vec_env.close()
                self.logger.info("向量化环境已关闭")
            
            # 关闭TensorBoard writer
            if hasattr(self.agent, 'writer') and self.agent.writer:
                self.agent.writer.close()
                self.logger.info("TensorBoard writer已关闭")
            
            # 清理代理缓冲区
            if hasattr(self.agent, 'high_level_buffer'):
                self.agent.high_level_buffer.clear()
            if hasattr(self.agent, 'low_level_buffer'):
                self.agent.low_level_buffer.clear()
            if hasattr(self.agent, 'state_skill_dataset'):
                self.agent.state_skill_dataset.clear()
            
            self.logger.info("所有资源清理完成")
            
        except Exception as e:
            print(f"清理资源时出错: {e}")
    
    def train(self, total_samples=100000):
        """
        执行完整的向量化训练
        
        参数:
            total_samples: 训练总样本数
        """
        self.logger.info(f"开始基于SB3向量化环境的HMASD训练: {total_samples:,} 样本")
        self.logger.info(f"配置: {self.n_envs} 个向量化环境")
        
        try:
            # 初始化代理
            self.initialize_agent()
            
            # 开始训练
            self.start_time = time.time()
            current_samples = 0
            
            while current_samples < total_samples:
                # 向量化rollout收集
                collected_experiences = self.collect_vectorized_rollout()
                
                # 存储经验到代理
                stored_counts = self.store_experiences_to_agent(collected_experiences)
                
                # 更新样本计数
                rollout_samples = self.n_envs * self.config.rollout_length
                current_samples += rollout_samples
                self.total_samples = current_samples
                
                # 执行模型更新
                if self.agent.should_rollout_update():
                    self.perform_update()
                
                # 记录训练进度
                self.log_training_progress(current_samples, total_samples)
            
            # 训练完成
            self.logger.info(f"向量化训练完成！总样本: {current_samples:,}")
            
        except KeyboardInterrupt:
            self.logger.info("训练被用户中断")
        except Exception as e:
            self.logger.error(f"训练过程中出错: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # 保存最终模型
            self.save_model()
            
            # 清理资源
            self.cleanup()
        
        # 训练完成统计
        if self.start_time:
            total_time = time.time() - self.start_time
            self.logger.info(f"\n向量化训练完成！")
            self.logger.info(f"总时间: {total_time/3600:.2f}小时")
            self.logger.info(f"总样本数: {self.total_samples:,}")
            self.logger.info(f"总更新数: {self.total_updates}")
            self.logger.info(f"总Episodes: {self.total_episodes}")
            
            if total_time > 0:
                self.logger.info(f"样本收集速度: {self.total_samples/total_time:.1f} 样本/秒")
                self.logger.info(f"Episode完成速度: {self.total_episodes/total_time:.1f} episodes/秒")

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='基于SB3向量化环境的HMASD训练')
    
    # 训练参数
    parser.add_argument('--samples', type=int, default=None, help='训练总样本数（如果不指定，将从config.py中读取）')
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cpu', 'cuda'])
    parser.add_argument('--debug', action='store_true', help='启用调试模式')
    
    # 向量化环境配置
    parser.add_argument('--n_envs', type=int, default=32, help='向量化环境数量')
    
    # 环境参数
    parser.add_argument('--scenario', type=int, default=2, choices=[1, 2], help='场景选择')
    parser.add_argument('--n_uavs', type=int, default=5, help='无人机数量')
    parser.add_argument('--n_users', type=int, default=50, help='用户数量')
    parser.add_argument('--user_distribution', type=str, default='uniform', 
                       choices=['uniform', 'cluster', 'hotspot'])
    parser.add_argument('--channel_model', type=str, default='3gpp-36777',
                       choices=['free_space', 'urban', 'suburban', '3gpp-36777'])
    parser.add_argument('--max_hops', type=int, default=3, help='最大跳数（仅场景2）')
    
    # 日志参数
    parser.add_argument('--log_level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])
    parser.add_argument('--console_log_level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])
    
    return parser.parse_args()

def main():
    """主函数"""
    args = parse_args()
    
    # 创建配置
    config = Config()
    
    # 确定训练样本数：优先使用命令行参数，其次使用配置文件中的值
    if args.samples is not None:
        total_samples = args.samples
        print(f"📈 使用命令行指定的训练样本数: {total_samples:,}")
    else:
        total_samples = int(config.total_timesteps)
        print(f"📈 从config.py读取训练样本数: {total_samples:,}")
    
    print("🚀 基于SB3向量化环境的HMASD训练")
    print("=" * 80)
    print(f"📊 向量化环境数量: {args.n_envs}")
    print(f"🎯 训练样本数: {total_samples:,}")
    print(f"🔧 设备: {args.device}")
    print(f"🌍 场景: {args.scenario}")
    print(f"🚁 无人机数量: {args.n_uavs}")
    print(f"👥 用户数量: {args.n_users}")
    
    # 验证并打印配置
    config.validate_training_mode()
    config.validate_rollout_config()
    print(config.get_rollout_summary())
    
    try:
        # 创建向量化训练器
        trainer = VectorizedHMASDTrainer(config, args)
        
        # 开始训练
        trainer.train(total_samples=total_samples)
        
        print("🎉 向量化训练成功完成！")
        
    except KeyboardInterrupt:
        print("\n⚠️ 训练被用户中断")
    except Exception as e:
        print(f"\n❌ 训练失败: {e}")
        import traceback
        traceback.print_exc()
    finally:
        try:
            shutdown_logging()
        except:
            pass

if __name__ == "__main__":
    main()
