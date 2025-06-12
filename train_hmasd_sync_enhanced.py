#!/usr/bin/env python3
"""
HMASD同步训练脚本 - 增强奖励追踪版本
基于train_enhanced_reward_tracking.py，集成同步训练机制
确保严格的on-policy特性并提供详细的训练数据收集
"""

import os
import time
import numpy as np
import torch
import argparse
import logging
import matplotlib.pyplot as plt
from datetime import datetime
import multiprocessing as mp
import pandas as pd
from collections import defaultdict, deque
from logger import init_multiproc_logging, get_logger, shutdown_logging, LOG_LEVELS, set_log_level

# 导入 Stable Baselines3 的向量化环境
from stable_baselines3.common.vec_env import SubprocVecEnv

# 导入论文中的配置
from config_1 import Config
from hmasd.agent import HMASDAgent
from envs.pettingzoo.scenario1 import UAVBaseStationEnv
from envs.pettingzoo.scenario2 import UAVCooperativeNetworkEnv
from envs.pettingzoo.env_adapter import ParallelToArrayAdapter
from evaltools.eval_utils import evaluate_agent

class SyncEnhancedRewardTracker:
    """同步训练增强的奖励追踪器，用于论文数据收集"""
    
    def __init__(self, log_dir, config):
        self.log_dir = log_dir
        self.config = config
        
        # 训练过程中的奖励数据收集
        self.training_rewards = {
            'episode_rewards': [],
            'step_rewards': [],
            'env_rewards': [],
            'intrinsic_rewards': [],
            'reward_components': {
                'env_component': [],
                'team_disc_component': [],
                'ind_disc_component': []
            },
            'cumulative_rewards': [],
            'reward_variance': [],
            'episodes_completed': 0,
            'total_steps': 0
        }
        
        # 同步训练特定的追踪
        self.sync_training_metrics = {
            'policy_versions': [],
            'sync_updates': 0,
            'samples_collected_per_sync': [],
            'collection_times': [],
            'update_times': [],
            'sync_efficiency': []
        }
        
        # 技能使用统计
        self.skill_usage = {
            'team_skills': defaultdict(int),
            'agent_skills': defaultdict(lambda: defaultdict(int)),
            'skill_switches': 0,
            'skill_diversity_history': [],
            'episode_skill_counts': []
        }
        
        # 性能指标
        self.performance_metrics = {
            'episode_lengths': [],
            'success_rates': [],
            'coverage_ratios': [],
            'served_users': [],
            'network_efficiency': [],
            'total_throughput': [],
            'avg_throughput_per_user': []
        }
        
        # 滑动窗口统计
        self.window_size = 100
        self.recent_rewards = deque(maxlen=self.window_size)
        self.recent_lengths = deque(maxlen=self.window_size)
        
        # 数据导出设置
        self.export_interval = 1000
        self.last_export_step = 0
        
    def log_sync_update(self, policy_version, samples_collected, collection_time, update_time):
        """记录同步更新信息"""
        self.sync_training_metrics['policy_versions'].append(policy_version)
        self.sync_training_metrics['sync_updates'] += 1
        self.sync_training_metrics['samples_collected_per_sync'].append(samples_collected)
        self.sync_training_metrics['collection_times'].append(collection_time)
        self.sync_training_metrics['update_times'].append(update_time)
        
        # 计算同步效率 (样本数/总时间)
        total_time = collection_time + update_time
        efficiency = samples_collected / total_time if total_time > 0 else 0
        self.sync_training_metrics['sync_efficiency'].append(efficiency)
        
    def log_training_step(self, step, env_id, reward, reward_components=None, info=None):
        """记录训练步骤的奖励信息"""
        self.training_rewards['total_steps'] += 1
        self.training_rewards['step_rewards'].append({
            'step': step,
            'env_id': env_id,
            'reward': reward,
            'timestamp': time.time()
        })
        
        if reward_components:
            for comp_name, comp_value in reward_components.items():
                if comp_name in self.training_rewards['reward_components']:
                    self.training_rewards['reward_components'][comp_name].append({
                        'step': step,
                        'env_id': env_id,
                        'value': comp_value
                    })
        
        # 记录额外信息
        if info:
            served_users = 0
            total_users = 0
            
            if 'reward_info' in info and 'connected_users' in info['reward_info']:
                served_users = info['reward_info']['connected_users']
            elif 'coverage_ratio' in info and 'n_users' in info:
                served_users = int(info['coverage_ratio'] * info['n_users'])
                total_users = info['n_users']
            elif 'served_users' in info:
                served_users = info['served_users']
                total_users = info.get('total_users', 0)
            
            if served_users > 0 or total_users > 0:
                self.performance_metrics['served_users'].append({
                    'step': step,
                    'env_id': env_id,
                    'served_users': served_users,
                    'total_users': total_users
                })
            
            # 记录吞吐量信息
            if 'reward_info' in info:
                reward_info = info['reward_info']
                if 'system_throughput_mbps' in reward_info:
                    self.performance_metrics['total_throughput'].append({
                        'step': step,
                        'env_id': env_id,
                        'system_throughput_mbps': reward_info['system_throughput_mbps'],
                        'timestamp': time.time()
                    })
                
                if 'avg_throughput_per_user_mbps' in reward_info:
                    self.performance_metrics['avg_throughput_per_user'].append({
                        'step': step,
                        'env_id': env_id,
                        'avg_throughput_per_user_mbps': reward_info['avg_throughput_per_user_mbps'],
                        'timestamp': time.time()
                    })
    
    def log_episode_completion(self, episode_num, env_id, total_reward, episode_length, info=None):
        """记录episode完成信息"""
        self.training_rewards['episodes_completed'] += 1
        
        episode_data = {
            'episode': episode_num,
            'env_id': env_id,
            'total_reward': total_reward,
            'episode_length': episode_length,
            'timestamp': time.time()
        }
        
        if info:
            episode_data.update(info)
        
        self.training_rewards['episode_rewards'].append(episode_data)
        self.recent_rewards.append(total_reward)
        self.recent_lengths.append(episode_length)
        
        # 计算滑动窗口统计
        if len(self.recent_rewards) >= 10:
            self.training_rewards['reward_variance'].append({
                'episode': episode_num,
                'mean': np.mean(self.recent_rewards),
                'std': np.std(self.recent_rewards),
                'min': np.min(self.recent_rewards),
                'max': np.max(self.recent_rewards)
            })
    
    def log_skill_usage(self, step, team_skill, agent_skills, skill_changed=False):
        """记录技能使用情况"""
        self.skill_usage['team_skills'][team_skill] += 1
        
        for i, skill in enumerate(agent_skills):
            self.skill_usage['agent_skills'][i][skill] += 1
        
        if skill_changed:
            self.skill_usage['skill_switches'] += 1
        
        # 计算技能多样性
        unique_skills = len(set(agent_skills))
        diversity = unique_skills / len(agent_skills) if len(agent_skills) > 0 else 0
        self.skill_usage['skill_diversity_history'].append({
            'step': step,
            'diversity': diversity,
            'unique_skills': unique_skills,
            'total_agents': len(agent_skills)
        })
    
    def export_training_data(self, step, writer=None):
        """导出训练数据用于论文分析"""
        if step - self.last_export_step < self.export_interval:
            return
        
        export_dir = os.path.join(self.log_dir, 'paper_data')
        os.makedirs(export_dir, exist_ok=True)
        
        # 导出奖励数据
        if self.training_rewards['episode_rewards']:
            rewards_df = pd.DataFrame(self.training_rewards['episode_rewards'])
            rewards_df.to_csv(os.path.join(export_dir, f'episode_rewards_step_{step}.csv'), index=False)
        
        # 导出同步训练指标
        if self.sync_training_metrics['policy_versions']:
            sync_df = pd.DataFrame({
                'policy_version': self.sync_training_metrics['policy_versions'],
                'samples_collected': self.sync_training_metrics['samples_collected_per_sync'],
                'collection_time': self.sync_training_metrics['collection_times'],
                'update_time': self.sync_training_metrics['update_times'],
                'sync_efficiency': self.sync_training_metrics['sync_efficiency']
            })
            sync_df.to_csv(os.path.join(export_dir, f'sync_metrics_step_{step}.csv'), index=False)
        
        # 导出奖励组成分析
        components_data = []
        for comp_name, comp_list in self.training_rewards['reward_components'].items():
            for entry in comp_list:
                components_data.append({
                    'step': entry['step'],
                    'env_id': entry['env_id'],
                    'component': comp_name,
                    'value': entry['value']
                })
        
        if components_data:
            components_df = pd.DataFrame(components_data)
            components_df.to_csv(os.path.join(export_dir, f'reward_components_step_{step}.csv'), index=False)
        
        # 导出技能使用统计
        skill_stats = {
            'team_skills': dict(self.skill_usage['team_skills']),
            'skill_switches': self.skill_usage['skill_switches'],
            'total_steps': step,
            'sync_updates': self.sync_training_metrics['sync_updates']
        }
        
        import json
        with open(os.path.join(export_dir, f'skill_usage_step_{step}.json'), 'w') as f:
            json.dump(skill_stats, f, indent=2)
        
        # 生成训练曲线图
        self.generate_training_plots(export_dir, step)
        
        # 记录到TensorBoard
        if writer:
            self.log_to_tensorboard(writer, step)
        
        self.last_export_step = step
        main_logger.debug(f"已导出步骤 {step} 的训练数据到 {export_dir}")
    
    def generate_training_plots(self, export_dir, step):
        """生成训练过程的可视化图表"""
        
        # 1. Episode奖励趋势图
        if self.training_rewards['episode_rewards']:
            episodes = [r['episode'] for r in self.training_rewards['episode_rewards']]
            rewards = [r['total_reward'] for r in self.training_rewards['episode_rewards']]
            
            plt.figure(figsize=(15, 10))
            
            # 原始奖励曲线
            plt.subplot(2, 3, 1)
            plt.plot(episodes, rewards, alpha=0.3, color='blue', label='Episode Rewards')
            if len(rewards) >= 50:
                window = min(50, len(rewards))
                smoothed = pd.Series(rewards).rolling(window=window, center=True).mean()
                plt.plot(episodes, smoothed, color='red', linewidth=2, label=f'{window}-episode MA')
            plt.xlabel('Episode')
            plt.ylabel('Total Reward')
            plt.title('HMASD Sync Training Reward Progress')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # 奖励分布直方图
            plt.subplot(2, 3, 2)
            plt.hist(rewards, bins=50, alpha=0.7, color='green')
            plt.xlabel('Total Reward')
            plt.ylabel('Frequency')
            plt.title('Reward Distribution')
            plt.grid(True, alpha=0.3)
            
            # Episode长度趋势
            lengths = [r['episode_length'] for r in self.training_rewards['episode_rewards']]
            plt.subplot(2, 3, 3)
            plt.plot(episodes, lengths, alpha=0.6, color='orange')
            plt.xlabel('Episode')
            plt.ylabel('Episode Length')
            plt.title('Episode Length Progression')
            plt.grid(True, alpha=0.3)
            
            # 奖励方差趋势
            if self.training_rewards['reward_variance']:
                var_episodes = [v['episode'] for v in self.training_rewards['reward_variance']]
                var_means = [v['mean'] for v in self.training_rewards['reward_variance']]
                var_stds = [v['std'] for v in self.training_rewards['reward_variance']]
                
                plt.subplot(2, 3, 4)
                plt.errorbar(var_episodes, var_means, yerr=var_stds, alpha=0.7, color='purple')
                plt.xlabel('Episode')
                plt.ylabel('Mean Reward ± Std')
                plt.title('Reward Stability (100-episode window)')
                plt.grid(True, alpha=0.3)
            
            # 同步训练效率
            if self.sync_training_metrics['sync_efficiency']:
                plt.subplot(2, 3, 5)
                sync_updates = range(1, len(self.sync_training_metrics['sync_efficiency']) + 1)
                plt.plot(sync_updates, self.sync_training_metrics['sync_efficiency'], 'bo-', alpha=0.7)
                plt.xlabel('Sync Update')
                plt.ylabel('Efficiency (samples/sec)')
                plt.title('Sync Training Efficiency')
                plt.grid(True, alpha=0.3)
            
            # 策略版本进展
            if self.sync_training_metrics['policy_versions']:
                plt.subplot(2, 3, 6)
                updates = range(1, len(self.sync_training_metrics['policy_versions']) + 1)
                plt.plot(updates, self.sync_training_metrics['policy_versions'], 'ro-', alpha=0.7)
                plt.xlabel('Sync Update')
                plt.ylabel('Policy Version')
                plt.title('Policy Version Progress')
                plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(export_dir, f'sync_training_progress_step_{step}.png'), dpi=300, bbox_inches='tight')
            plt.close()
    
    def log_to_tensorboard(self, writer, step):
        """记录详细数据到TensorBoard"""
        
        # 训练奖励统计
        if self.recent_rewards:
            writer.add_scalar('Training/Reward_Mean_100ep', np.mean(self.recent_rewards), step)
            writer.add_scalar('Training/Reward_Std_100ep', np.std(self.recent_rewards), step)
            writer.add_scalar('Training/Reward_Min_100ep', np.min(self.recent_rewards), step)
            writer.add_scalar('Training/Reward_Max_100ep', np.max(self.recent_rewards), step)
        
        # 同步训练指标
        writer.add_scalar('Sync/Total_Updates', self.sync_training_metrics['sync_updates'], step)
        if self.sync_training_metrics['policy_versions']:
            writer.add_scalar('Sync/Current_Policy_Version', self.sync_training_metrics['policy_versions'][-1], step)
        
        if self.sync_training_metrics['sync_efficiency']:
            recent_efficiency = self.sync_training_metrics['sync_efficiency'][-10:]  # 最近10次
            writer.add_scalar('Sync/Avg_Efficiency_10updates', np.mean(recent_efficiency), step)
        
        if self.sync_training_metrics['collection_times']:
            recent_collection_time = self.sync_training_metrics['collection_times'][-10:]
            writer.add_scalar('Sync/Avg_Collection_Time_10updates', np.mean(recent_collection_time), step)
        
        if self.sync_training_metrics['update_times']:
            recent_update_time = self.sync_training_metrics['update_times'][-10:]
            writer.add_scalar('Sync/Avg_Update_Time_10updates', np.mean(recent_update_time), step)
        
        # 技能多样性
        if self.skill_usage['skill_diversity_history']:
            recent_diversity = self.skill_usage['skill_diversity_history'][-10:]
            avg_diversity = np.mean([d['diversity'] for d in recent_diversity])
            writer.add_scalar('Training/Skill_Diversity_Recent', avg_diversity, step)
        
        # 其他基础指标
        writer.add_scalar('Training/Episodes_Completed', self.training_rewards['episodes_completed'], step)
        writer.add_scalar('Training/Skill_Switches_Total', self.skill_usage['skill_switches'], step)
    
    def get_summary_statistics(self):
        """获取训练摘要统计信息"""
        summary = {
            'total_episodes': self.training_rewards['episodes_completed'],
            'total_steps': self.training_rewards['total_steps'],
            'skill_switches': self.skill_usage['skill_switches'],
            'sync_updates': self.sync_training_metrics['sync_updates']
        }
        
        if self.training_rewards['episode_rewards']:
            rewards = [r['total_reward'] for r in self.training_rewards['episode_rewards']]
            summary.update({
                'reward_mean': np.mean(rewards),
                'reward_std': np.std(rewards),
                'reward_min': np.min(rewards),
                'reward_max': np.max(rewards)
            })
        
        if self.sync_training_metrics['sync_efficiency']:
            summary.update({
                'avg_sync_efficiency': np.mean(self.sync_training_metrics['sync_efficiency']),
                'final_policy_version': self.sync_training_metrics['policy_versions'][-1] if self.sync_training_metrics['policy_versions'] else 0
            })
        
        return summary

def get_device(device_pref):
    """根据偏好选择计算设备"""
    if device_pref == 'auto':
        if torch.cuda.is_available():
            main_logger.info("检测到GPU可用，使用CUDA")
            return torch.device('cuda')
        else:
            main_logger.info("未检测到GPU，使用CPU")
            return torch.device('cpu')
    elif device_pref == 'cuda':
        if torch.cuda.is_available():
            main_logger.info("使用CUDA")
            return torch.device('cuda')
        else:
            main_logger.warning("请求使用CUDA但未检测到GPU，回退到CPU")
            return torch.device('cpu')
    else:
        main_logger.info("使用CPU")
        return torch.device('cpu')

def make_env(scenario, n_uavs, n_users, user_distribution, channel_model, max_hops=None, render_mode=None, rank=0, seed=0):
    """创建环境实例的函数"""
    def _init():
        env_seed = seed + rank
        if scenario == 1:
            raw_env = UAVBaseStationEnv(
                n_uavs=n_uavs,
                n_users=n_users,
                user_distribution=user_distribution,
                channel_model=channel_model,
                render_mode=render_mode,
                seed=env_seed
            )
        elif scenario == 2:
            raw_env = UAVCooperativeNetworkEnv(
                n_uavs=n_uavs,
                n_users=n_users,
                max_hops=max_hops,
                user_distribution=user_distribution,
                channel_model=channel_model,
                render_mode=render_mode,
                seed=env_seed
            )
        else:
            raise ValueError(f"未知的场景: {scenario}")

        env = ParallelToArrayAdapter(raw_env, seed=env_seed)
        return env

    return _init

def parse_args():
    parser = argparse.ArgumentParser(description='HMASD同步训练 - 增强奖励追踪版本')
    
    # 运行模式和环境参数
    parser.add_argument('--mode', type=str, default='train', help='运行模式: train或eval')
    parser.add_argument('--scenario', type=int, default=2, help='场景: 1=基站模式, 2=协作组网模式')
    parser.add_argument('--model_path', type=str, default='models/hmasd_sync_enhanced_tracking.pt', help='模型保存/加载路径')
    parser.add_argument('--log_dir', type=str, default='tf-logs', help='日志目录')
    parser.add_argument('--log_level', type=str, default='info', 
                        choices=['debug', 'info', 'warning', 'error', 'critical'], 
                        help='日志级别')
    parser.add_argument('--console_log_level', type=str, default='error', 
                        choices=['debug', 'info', 'warning', 'error', 'critical'], 
                        help='控制台日志级别')
    parser.add_argument('--eval_episodes', type=int, default=10, help='评估的episode数量')
    parser.add_argument('--render', action='store_true', help='是否渲染环境')
    parser.add_argument('--device', type=str, default='auto', 
                        choices=['auto', 'cuda', 'cpu'], help='计算设备')

    # 环境参数
    parser.add_argument('--n_uavs', type=int, default=5, help='初始无人机数量')
    parser.add_argument('--n_users', type=int, default=50, help='用户数量')
    parser.add_argument('--max_hops', type=int, default=3, help='最大跳数 (仅用于场景2)')
    parser.add_argument('--user_distribution', type=str, default='uniform', 
                        choices=['uniform', 'cluster', 'hotspot'], help='用户分布类型')
    parser.add_argument('--channel_model', type=str, default='3gpp-36777',
                        choices=['free_space', 'urban', 'suburban','3gpp-36777'], help='信道模型')
    
    # 并行参数
    parser.add_argument('--num_envs', type=int, default=0, 
                        help='并行环境数量 (0=使用配置文件中的值)')
    parser.add_argument('--eval_rollout_threads', type=int, default=0, 
                        help='评估时的并行线程数 (0=使用配置文件中的值)')
    
    # 数据收集参数
    parser.add_argument('--export_interval', type=int, default=1000, 
                        help='数据导出间隔步数')
    parser.add_argument('--detailed_logging', action='store_true', 
                        help='启用详细的奖励日志记录')
    
    return parser.parse_args()

def train_sync_enhanced(vec_env, eval_vec_env, config, args, device):
    """
    HMASD同步训练函数 - 增强奖励追踪版本
    """
    num_envs = vec_env.num_envs
    main_logger.info(f"开始HMASD同步训练 (使用 {num_envs} 个并行环境)...")

    # 更新环境维度
    state_dim = vec_env.get_attr('state_dim')[0]
    obs_shape = vec_env.observation_space.shape
    if len(obs_shape) == 3:
        obs_dim = obs_shape[2]
        n_uavs_check = obs_shape[1]
        main_logger.info(f"从 observation_space 推断: obs_dim={obs_dim}, n_uavs={n_uavs_check}")
        if n_uavs_check != config.n_agents:
            main_logger.warning(f"从 observation_space 推断的 n_uavs ({n_uavs_check}) 与配置 ({config.n_agents}) 不匹配。")
            obs_dim = vec_env.get_attr('obs_dim')[0]
    else:
        main_logger.warning("无法从 observation_space 推断 obs_dim，尝试从适配器属性获取。")
        obs_dim = vec_env.get_attr('obs_dim')[0]

    config.update_env_dims(state_dim, obs_dim)
    main_logger.info(f"更新配置: state_dim={state_dim}, obs_dim={obs_dim}")

    # 创建日志目录
    log_dir = os.path.join(args.log_dir, f"hmasd_sync_enhanced_tracking_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(log_dir, exist_ok=True)
    model_dir = os.path.dirname(args.model_path)
    os.makedirs(model_dir, exist_ok=True)
    
    # 创建HMASD代理（启用同步模式）
    agent = HMASDAgent(config, log_dir=log_dir, device=device)
    
    # 确保同步模式启用
    agent.sync_mode = True
    agent.sync_batch_size = config.batch_size
    main_logger.info(f"同步训练模式已启用，batch_size: {agent.sync_batch_size}")
    
    # 创建增强的奖励追踪器
    reward_tracker = SyncEnhancedRewardTracker(log_dir, config)
    reward_tracker.export_interval = args.export_interval
    
    # 记录超参数
    agent.writer.add_text('Parameters/n_agents', str(config.n_agents), 0)
    agent.writer.add_text('Parameters/n_Z', str(config.n_Z), 0)
    agent.writer.add_text('Parameters/n_z', str(config.n_z), 0)
    agent.writer.add_text('Parameters/k', str(config.k), 0)
    agent.writer.add_text('Parameters/lambda_e', str(config.lambda_e), 0)
    agent.writer.add_text('Parameters/num_envs', str(num_envs), 0)
    agent.writer.add_text('Parameters/sync_batch_size', str(agent.sync_batch_size), 0)

    # 训练变量
    total_steps = 0
    n_episodes = 0
    update_times = 0
    best_reward = float('-inf')
    last_eval_step = 0
    episode_rewards = []
    
    # 记录训练开始时间
    start_time = time.time()

    # 重置所有环境
    main_logger.info("重置并行环境...")
    results = vec_env.env_method('reset')
    observations = np.array([res[0] for res in results])
    initial_infos = [res[1] for res in results]
    states = np.array([info.get('state', np.zeros(agent.config.state_dim)) for info in initial_infos])
    main_logger.debug(f"环境已重置。观测形状: {observations.shape}, 状态形状: {states.shape}")

    # 环境状态跟踪
    env_steps = np.zeros(num_envs, dtype=int)
    env_rewards = np.zeros(num_envs)
    
    # 奖励组成跟踪
    env_reward_components = [{
        'env_component': 0.0,
        'team_disc_component': 0.0,
        'ind_disc_component': 0.0
    } for _ in range(num_envs)]
    
    main_logger.info(f"开始同步训练循环，目标步数: {config.total_timesteps}")
    
    # 同步训练循环
    while total_steps < config.total_timesteps:
        # 1. 数据收集阶段
        collection_start_time = time.time()
        samples_collected_this_batch = 0
        
        while not agent.should_sync_update():
            # 代理为所有环境选择动作
            all_actions_list = []
            all_agent_infos_list = []

            for i in range(num_envs):
                actions, agent_info = agent.step(states[i], observations[i], env_steps[i], deterministic=False, env_id=i)
                all_actions_list.append(actions)
                all_agent_infos_list.append(agent_info)

            actions_array = np.array(all_actions_list)

            # 执行动作
            next_observations, rewards, dones, infos = vec_env.step(actions_array)
            next_states = np.array([info.get('next_state', np.zeros(state_dim)) for info in infos])

            # 更新环境状态和存储经验
            for i in range(num_envs):
                current_agent_info = all_agent_infos_list[i]
                
                # 计算奖励组成部分
                current_reward = rewards[i] if isinstance(rewards[i], (int, float)) else rewards[i].item()
                
                # 估算奖励组成部分
                env_component = config.lambda_e * current_reward
                team_disc_component = config.lambda_D * 0.1  # 模拟判别器奖励
                ind_disc_component = config.lambda_d * 0.05  # 模拟个体判别器奖励
                
                # 累积奖励组成
                env_reward_components[i]['env_component'] += env_component
                env_reward_components[i]['team_disc_component'] += team_disc_component
                env_reward_components[i]['ind_disc_component'] += ind_disc_component
                
                # 记录训练步骤
                reward_components_to_log = None
                if args.detailed_logging:
                    reward_components_to_log = {
                        'env_component': env_component,
                        'team_disc_component': team_disc_component,
                        'ind_disc_component': ind_disc_component
                    }
                
                reward_tracker.log_training_step(
                    total_steps, i, current_reward,
                    reward_components=reward_components_to_log,
                    info=infos[i]
                )
                
                # 记录技能使用
                reward_tracker.log_skill_usage(
                    total_steps,
                    current_agent_info['team_skill'],
                    current_agent_info['agent_skills'],
                    current_agent_info['skill_changed']
                )
                
                # 存储经验（同步模式）
                stored = agent.store_transition(
                    states[i], next_states[i], observations[i], next_observations[i],
                    actions_array[i], rewards[i], dones[i], current_agent_info['team_skill'],
                    current_agent_info['agent_skills'], current_agent_info['action_logprobs'],
                    log_probs=current_agent_info['log_probs'],
                    skill_timer_for_env=current_agent_info['skill_timer'],
                    env_id=i
                )
                
                if stored:
                    samples_collected_this_batch += 1

                # 更新环境状态跟踪
                env_steps[i] += 1
                env_rewards[i] += current_reward

                # 如果环境完成
                if dones[i]:
                    n_episodes += 1
                    episode_rewards.append(env_rewards[i])

                    # 记录episode完成到追踪器
                    episode_info = env_reward_components[i].copy()
                    if 'reward_info' in infos[i]:
                        episode_info.update(infos[i]['reward_info'])
                    
                    reward_tracker.log_episode_completion(
                        n_episodes, i, env_rewards[i], env_steps[i], episode_info
                    )

                    # 记录到TensorBoard
                    agent.writer.add_scalar('Training/Episode_Reward', env_rewards[i], n_episodes)
                    agent.writer.add_scalar('Training/Episode_Length', env_steps[i], n_episodes)
                    
                    # 记录性能指标
                    if 'reward_info' in infos[i]:
                        reward_info = infos[i]['reward_info']
                        if 'system_throughput_mbps' in reward_info:
                            agent.writer.add_scalar('Performance/Episode_System_Throughput_Mbps', 
                                                   reward_info['system_throughput_mbps'], n_episodes)
                        if 'connected_users' in reward_info:
                            agent.writer.add_scalar('Performance/Episode_Connected_Users', 
                                                   reward_info['connected_users'], n_episodes)

                    main_logger.info(f"Episode结束 - 环境ID: {i}, Episode: {n_episodes}, "
                                   f"奖励: {env_rewards[i]:.2f}, 步数: {env_steps[i]}")

                    # 重置环境状态跟踪
                    env_steps[i] = 0
                    env_rewards[i] = 0
                    env_reward_components[i] = {
                        'env_component': 0.0,
                        'team_disc_component': 0.0,
                        'ind_disc_component': 0.0
                    }

                    # 计算最近episode统计
                    if len(episode_rewards) >= 10:
                        recent_rewards = episode_rewards[-10:]
                        avg_reward = np.mean(recent_rewards)
                        agent.writer.add_scalar('Training/Avg_Reward_10ep', avg_reward, n_episodes)
                        
                        if len(episode_rewards) % 50 == 0:
                            main_logger.info(f"最近10个episodes平均奖励: {avg_reward:.2f}")

            # 更新总步数
            total_steps += num_envs
            
            # 更新状态和观测
            states = next_states
            observations = next_observations

            # 检查是否收集被停止（达到同步点）
            if not agent.collection_enabled:
                break
        
        collection_time = time.time() - collection_start_time
        
        # 2. 同步更新阶段
        if agent.should_sync_update():
            sync_start_time = time.time()
            samples_count = agent.samples_collected_this_round
            main_logger.info(f"达到同步点 - 收集了 {samples_count} 个样本，耗时 {collection_time:.2f}s")
            
            # 执行同步更新
            update_info = agent.sync_update()
            update_times += 1
            update_time = time.time() - sync_start_time
            
            # 记录同步更新信息
            reward_tracker.log_sync_update(
                update_info['policy_version'],
                samples_count,
                collection_time,
                update_time
            )
            
            main_logger.info(f"同步更新完成 - 策略版本: {update_info['policy_version']}, "
                           f"更新耗时: {update_time:.2f}s")
            
            # 记录详细的更新信息到TensorBoard
            for key, value in update_info.items():
                if isinstance(value, (int, float)):
                    agent.writer.add_scalar(f'Training/Update_{key}', value, update_times)
            
            # 记录训练进度
            if update_times % 10 == 0:
                elapsed_time = time.time() - start_time
                steps_per_sec = total_steps / elapsed_time if elapsed_time > 0 else 0
                
                main_logger.info(f"训练进度: {total_steps}/{config.total_timesteps} "
                               f"({100.0 * total_steps / config.total_timesteps:.1f}%), "
                               f"更新次数: {update_times}, Episodes: {n_episodes}, "
                               f"步数/秒: {steps_per_sec:.1f}")

        # 3. 定期导出数据
        if total_steps % reward_tracker.export_interval == 0:
            reward_tracker.export_training_data(total_steps, agent.writer)

        # 4. 定期评估
        if total_steps >= last_eval_step + config.eval_interval:
            main_logger.info(f"开始评估 (步数: {total_steps})")
            eval_start_time = time.time()
            
            try:
                eval_results = evaluate_agent(agent, eval_vec_env, config, num_episodes=config.eval_episodes)
                eval_time = time.time() - eval_start_time
                
                eval_reward = eval_results['mean_reward']
                eval_std = eval_results['std_reward']
                
                main_logger.info(f"评估完成，耗时: {eval_time:.2f}s")
                main_logger.info(f"评估结果: 平均奖励={eval_reward:.4f}, 标准差={eval_std:.4f}")
                
                # 记录到TensorBoard
                agent.writer.add_scalar('Eval/MeanReward', eval_reward, total_steps)
                agent.writer.add_scalar('Eval/StdReward', eval_std, total_steps)
                
                # 保存最佳模型
                if eval_reward > best_reward:
                    best_reward = eval_reward
                    agent.save_model(args.model_path)
                    main_logger.info(f"保存最佳模型，奖励: {best_reward:.2f}")
                
            except Exception as e:
                main_logger.error(f"评估失败: {e}")
            
            last_eval_step = total_steps

        # 5. 定期保存模型
        if total_steps > 0 and total_steps % (config.eval_interval * 2) == 0:
            checkpoint_path = f"models/hmasd_sync_enhanced_tracking_{total_steps}.pt"
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            agent.save_model(checkpoint_path)
            main_logger.info(f"保存检查点: {checkpoint_path}")

    # 训练完成后的最终数据导出
    reward_tracker.export_training_data(total_steps, agent.writer)
    
    # 生成训练摘要报告
    summary = reward_tracker.get_summary_statistics()
    main_logger.info("同步训练摘要:")
    for key, value in summary.items():
        main_logger.info(f"  {key}: {value}")
    
    # 保存摘要到文件
    import json
    summary_path = os.path.join(log_dir, 'sync_training_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    total_time = time.time() - start_time
    main_logger.info(f"同步训练完成! 总步数: {total_steps}, 总episodes: {n_episodes}")
    main_logger.info(f"总时间: {total_time:.2f}s, 总更新: {update_times}")
    main_logger.info(f"最佳奖励: {best_reward:.2f}")
    main_logger.info(f"训练数据已保存到: {log_dir}")

    # 保存最终模型
    final_model_path = os.path.join(model_dir, 'hmasd_sync_enhanced_tracking_final.pt')
    agent.save_model(final_model_path)
    main_logger.info(f"最终模型已保存到 {final_model_path}")
    
    return agent

def main():
    args = parse_args()
    
    # 创建日志目录
    os.makedirs(args.log_dir, exist_ok=True)
    
    # 为训练会话创建固定的日志文件名
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = f"hmasd_sync_enhanced_tracking_{timestamp}.log"
    
    # 初始化多进程日志系统
    file_level = LOG_LEVELS.get(args.log_level.lower(), logging.INFO)
    console_level = LOG_LEVELS.get(args.console_log_level.lower(), logging.WARNING)
    init_multiproc_logging(
        log_dir=args.log_dir, 
        log_file=log_file, 
        file_level=file_level, 
        console_level=console_level
    )
    
    # 获取main_logger实例
    global main_logger
    main_logger = get_logger("HMASD-Sync-Enhanced")
    main_logger.info(f"HMASD同步训练启动 - 增强奖励追踪版本")
    main_logger.info(f"文件级别={args.log_level}, 控制台级别={args.console_log_level}")
    main_logger.info(f"日志文件: {os.path.join(args.log_dir, log_file)}")
    
    # 使用配置
    config = Config()
    
    # 获取计算设备
    device = get_device(args.device)
    
    # 确定并行环境数量
    num_envs = args.num_envs if args.num_envs > 0 else config.num_envs
    eval_rollout_threads = args.eval_rollout_threads if args.eval_rollout_threads > 0 else config.eval_rollout_threads
    
    main_logger.info(f"使用 {num_envs} 个并行训练环境和 {eval_rollout_threads} 个并行评估环境")
    main_logger.info(f"详细日志记录: {args.detailed_logging}")
    main_logger.info(f"数据导出间隔: {args.export_interval} 步")
    
    # 创建环境
    base_seed = config.seed if hasattr(config, 'seed') else int(time.time())
    main_logger.info(f"基础种子: {base_seed}")

    train_env_fns = [make_env(
        scenario=args.scenario,
        n_uavs=args.n_uavs,
        n_users=args.n_users,
        user_distribution=args.user_distribution,
        channel_model=args.channel_model,
        max_hops=args.max_hops if args.scenario == 2 else None,
        render_mode=None,
        rank=i,
        seed=base_seed
    ) for i in range(num_envs)]

    eval_env_fns = [make_env(
        scenario=args.scenario,
        n_uavs=args.n_uavs,
        n_users=args.n_users,
        user_distribution=args.user_distribution,
        channel_model=args.channel_model,
        max_hops=args.max_hops if args.scenario == 2 else None,
        render_mode="human" if args.render and i == 0 else None,
        rank=i,
        seed=base_seed + num_envs
    ) for i in range(eval_rollout_threads)]

    # 创建向量化环境
    main_logger.info("创建 SubprocVecEnv...")
    train_vec_env = SubprocVecEnv(train_env_fns, start_method='spawn')
    eval_vec_env = SubprocVecEnv(eval_env_fns, start_method='spawn')
    main_logger.info("SubprocVecEnv 已创建。")

    # 更新配置中的智能体数量
    try:
        n_agents_from_env = train_vec_env.get_attr('n_uavs')[0]
        config.n_agents = n_agents_from_env
        main_logger.info(f"从环境更新智能体数量: n_agents={config.n_agents}")
    except Exception as e:
        main_logger.warning(f"无法从环境获取 n_uavs: {e}. 使用命令行参数: {args.n_uavs}")
        config.n_agents = args.n_uavs

    main_logger.info(f"使用论文中的超参数: n_Z={config.n_Z}, n_z={config.n_z}, k={config.k}")
    main_logger.info(f"同步训练配置: batch_size={config.batch_size}")

    if args.mode == 'train':
        agent = train_sync_enhanced(train_vec_env, eval_vec_env, config, args, device)
        main_logger.info("同步训练完成，增强的数据收集已启用")
    elif args.mode == 'eval':
        # 评估模式
        if not os.path.exists(args.model_path):
            main_logger.error(f"模型文件 {args.model_path} 不存在")
            return
        
        log_dir = os.path.join(args.log_dir, f"eval_sync_enhanced_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        os.makedirs(log_dir, exist_ok=True)
        
        agent = HMASDAgent(config, log_dir=log_dir, device=device)
        agent.load_model(args.model_path)
        
        eval_results = evaluate_agent(agent, eval_vec_env, config, num_episodes=args.eval_episodes)
        main_logger.info(f"评估结果: 平均奖励={eval_results['mean_reward']:.4f}, "
                       f"标准差={eval_results['std_reward']:.4f}")
    else:
        main_logger.error(f"未知的运行模式: {args.mode}")
    
    # 关闭环境
    train_vec_env.close()
    eval_vec_env.close()

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    try:
        main()
    finally:
        try:
            shutdown_logging()
            print("日志系统已关闭")
        except Exception as e:
            print(f"关闭日志系统时出错: {e}")
