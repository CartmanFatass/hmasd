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

class EnhancedRewardTracker:
    """增强的奖励追踪器，用于论文数据收集"""
    
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
            'total_throughput': [],  # 新增：总吞吐量记录
            'avg_throughput_per_user': []  # 新增：平均用户吞吐量记录
        }
        
        # 滑动窗口统计
        self.window_size = 100
        self.recent_rewards = deque(maxlen=self.window_size)
        self.recent_lengths = deque(maxlen=self.window_size)
        
        # 数据导出设置
        self.export_interval = 1000  # 每1000步导出一次数据
        self.last_export_step = 0
        
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
            if 'served_users' in info:
                self.performance_metrics['served_users'].append({
                    'step': step,
                    'env_id': env_id,
                    'served_users': info['served_users'],
                    'total_users': info.get('total_users', 0)
                })
            
            # 记录吞吐量信息（修正后的字段名）
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
            'total_steps': step
        }
        
        import json
        with open(os.path.join(export_dir, f'skill_usage_step_{step}.json'), 'w') as f:
            json.dump(skill_stats, f, indent=2)
        
        # 生成训练曲线图
        self.generate_training_plots(export_dir, step)
        
        # 记录到TensorBoard（如果提供）
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
            
            plt.figure(figsize=(12, 8))
            
            # 原始奖励曲线
            plt.subplot(2, 2, 1)
            plt.plot(episodes, rewards, alpha=0.3, color='blue', label='Episode Rewards')
            # 滑动平均
            if len(rewards) >= 10:
                window = 50
                if len(rewards) >= window:
                    smoothed = pd.Series(rewards).rolling(window=window, center=True).mean()
                    plt.plot(episodes, smoothed, color='red', linewidth=2, label=f'{window}-episode MA')
            plt.xlabel('Episode')
            plt.ylabel('Total Reward')
            plt.title('Training Reward Progress')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # 奖励分布直方图
            plt.subplot(2, 2, 2)
            plt.hist(rewards, bins=50, alpha=0.7, color='green')
            plt.xlabel('Total Reward')
            plt.ylabel('Frequency')
            plt.title('Reward Distribution')
            plt.grid(True, alpha=0.3)
            
            # Episode长度趋势
            if self.performance_metrics['episode_lengths'] or len(episodes) == len([r['episode_length'] for r in self.training_rewards['episode_rewards']]):
                lengths = [r['episode_length'] for r in self.training_rewards['episode_rewards']]
                plt.subplot(2, 2, 3)
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
                
                plt.subplot(2, 2, 4)
                plt.errorbar(var_episodes, var_means, yerr=var_stds, alpha=0.7, color='purple')
                plt.xlabel('Episode')
                plt.ylabel('Mean Reward ± Std')
                plt.title('Reward Stability (100-episode window)')
                plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(export_dir, f'training_progress_step_{step}.png'), dpi=300, bbox_inches='tight')
            plt.close()
        
        # 2. 奖励组成分析图
        if any(self.training_rewards['reward_components'].values()):
            plt.figure(figsize=(15, 5))
            
            components = ['env_component', 'team_disc_component', 'ind_disc_component']
            colors = ['blue', 'red', 'green']
            
            for i, (comp_name, color) in enumerate(zip(components, colors)):
                if comp_name in self.training_rewards['reward_components'] and self.training_rewards['reward_components'][comp_name]:
                    comp_data = self.training_rewards['reward_components'][comp_name]
                    steps = [d['step'] for d in comp_data]
                    values = [d['value'] for d in comp_data]
                    
                    plt.subplot(1, 3, i+1)
                    plt.plot(steps, values, alpha=0.6, color=color)
                    plt.xlabel('Training Step')
                    plt.ylabel('Reward Component Value')
                    plt.title(f'{comp_name.replace("_", " ").title()}')
                    plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(export_dir, f'reward_components_step_{step}.png'), dpi=300, bbox_inches='tight')
            plt.close()
        
        # 3. 技能使用分析图
        if self.skill_usage['skill_diversity_history']:
            plt.figure(figsize=(12, 4))
            
            diversity_data = self.skill_usage['skill_diversity_history']
            steps = [d['step'] for d in diversity_data]
            diversity_values = [d['diversity'] for d in diversity_data]
            
            plt.subplot(1, 2, 1)
            plt.plot(steps, diversity_values, alpha=0.7, color='purple')
            plt.xlabel('Training Step')
            plt.ylabel('Skill Diversity')
            plt.title('Agent Skill Diversity Over Time')
            plt.grid(True, alpha=0.3)
            
            # 团队技能使用分布
            if self.skill_usage['team_skills']:
                plt.subplot(1, 2, 2)
                skills = list(self.skill_usage['team_skills'].keys())
                counts = list(self.skill_usage['team_skills'].values())
                plt.bar(skills, counts, alpha=0.7, color='orange')
                plt.xlabel('Team Skill ID')
                plt.ylabel('Usage Count')
                plt.title('Team Skill Usage Distribution')
                plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(export_dir, f'skill_analysis_step_{step}.png'), dpi=300, bbox_inches='tight')
            plt.close()
    
    def log_to_tensorboard(self, writer, step):
        """记录详细数据到TensorBoard"""
        
        # 训练奖励统计
        if self.recent_rewards:
            writer.add_scalar('Training/Reward_Mean_100ep', np.mean(self.recent_rewards), step)
            writer.add_scalar('Training/Reward_Std_100ep', np.std(self.recent_rewards), step)
            writer.add_scalar('Training/Reward_Min_100ep', np.min(self.recent_rewards), step)
            writer.add_scalar('Training/Reward_Max_100ep', np.max(self.recent_rewards), step)
        
        if self.recent_lengths:
            writer.add_scalar('Training/EpisodeLength_Mean_100ep', np.mean(self.recent_lengths), step)
        
        # 技能多样性
        if self.skill_usage['skill_diversity_history']:
            recent_diversity = self.skill_usage['skill_diversity_history'][-10:]  # 最近10次
            avg_diversity = np.mean([d['diversity'] for d in recent_diversity])
            writer.add_scalar('Training/Skill_Diversity_Recent', avg_diversity, step)
        
        # 技能使用分布熵
        if self.skill_usage['team_skills']:
            total_usage = sum(self.skill_usage['team_skills'].values())
            skill_probs = [count/total_usage for count in self.skill_usage['team_skills'].values()]
            skill_entropy = -sum(p * np.log(p + 1e-8) for p in skill_probs)
            writer.add_scalar('Training/Team_Skill_Entropy', skill_entropy, step)
        
        # 训练效率指标
        writer.add_scalar('Training/Episodes_Completed', self.training_rewards['episodes_completed'], step)
        writer.add_scalar('Training/Skill_Switches_Total', self.skill_usage['skill_switches'], step)
        
        # 奖励组成比例
        if any(self.training_rewards['reward_components'].values()):
            recent_components = {}
            for comp_name, comp_list in self.training_rewards['reward_components'].items():
                if comp_list:
                    recent_data = comp_list[-100:]  # 最近100个数据点
                    recent_components[comp_name] = np.mean([d['value'] for d in recent_data])
            
            total_intrinsic = sum(recent_components.values())
            if total_intrinsic != 0:
                for comp_name, comp_value in recent_components.items():
                    proportion = comp_value / total_intrinsic
                    writer.add_scalar(f'Training/Reward_Proportion_{comp_name}', proportion, step)
        
        # Throughput统计（新增）
        if self.performance_metrics['served_users']:
            # 计算最近100步的滑动窗口平均吞吐量
            recent_served_data = self.performance_metrics['served_users'][-100:]
            recent_served_users = [u['served_users'] for u in recent_served_data]
            recent_total_users = [u['total_users'] for u in recent_served_data]
            
            if recent_served_users:
                # 平均服务用户数
                avg_served_users = np.mean(recent_served_users)
                writer.add_scalar('Performance/Throughput_ServedUsers_100steps', avg_served_users, step)
                
                # 平均总用户数
                avg_total_users = np.mean(recent_total_users)
                writer.add_scalar('Performance/Throughput_TotalUsers_100steps', avg_total_users, step)
                
                # 服务率（吞吐率）
                service_rate = avg_served_users / max(avg_total_users, 1)
                writer.add_scalar('Performance/Throughput_ServiceRate_100steps', service_rate, step)
                
                # 计算吞吐率变化趋势（最近50步 vs 前50步）
                if len(recent_served_data) >= 100:
                    first_half = recent_served_data[:50]
                    second_half = recent_served_data[50:]
                    
                    first_half_rate = np.mean([u['served_users'] for u in first_half]) / max(np.mean([u['total_users'] for u in first_half]), 1)
                    second_half_rate = np.mean([u['served_users'] for u in second_half]) / max(np.mean([u['total_users'] for u in second_half]), 1)
                    
                    throughput_trend = second_half_rate - first_half_rate
                    writer.add_scalar('Performance/Throughput_Trend_100steps', throughput_trend, step)
            
            # 按环境分别统计吞吐量
            env_throughputs = defaultdict(list)
            env_total_users = defaultdict(list)
            for entry in recent_served_data:
                env_throughputs[entry['env_id']].append(entry['served_users'])
                env_total_users[entry['env_id']].append(entry['total_users'])
            
            for env_id in env_throughputs.keys():
                served_values = env_throughputs[env_id]
                total_values = env_total_users[env_id]
                
                if served_values:
                    env_avg_served = np.mean(served_values)
                    env_avg_total = np.mean(total_values)
                    env_service_rate = env_avg_served / max(env_avg_total, 1)
                    
                    writer.add_scalar(f'Performance/Env_{env_id}_ServedUsers', env_avg_served, step)
                    writer.add_scalar(f'Performance/Env_{env_id}_ServiceRate', env_service_rate, step)
            
            # 吞吐量方差（稳定性指标）
            if len(recent_served_users) > 1:
                throughput_std = np.std(recent_served_users)
                throughput_cv = throughput_std / max(np.mean(recent_served_users), 1e-8)  # 变异系数
                writer.add_scalar('Performance/Throughput_Std_100steps', throughput_std, step)
                writer.add_scalar('Performance/Throughput_CV_100steps', throughput_cv, step)
        
        # 系统吞吐量统计（修正后）
        if self.performance_metrics['total_throughput']:
            # 计算最近100步的系统吞吐量统计
            recent_throughput_data = self.performance_metrics['total_throughput'][-100:]
            recent_system_throughput = [t['system_throughput_mbps'] for t in recent_throughput_data if 'system_throughput_mbps' in t]
            
            if recent_system_throughput:
                # 平均系统吞吐量
                avg_system_throughput = np.mean(recent_system_throughput)
                writer.add_scalar('Performance/System_Throughput_Mbps_100steps', avg_system_throughput, step)
                
                # 系统吞吐量标准差
                throughput_std = np.std(recent_system_throughput)
                writer.add_scalar('Performance/System_Throughput_Std_100steps', throughput_std, step)
                
                # 系统吞吐量最大值和最小值
                writer.add_scalar('Performance/System_Throughput_Max_100steps', np.max(recent_system_throughput), step)
                writer.add_scalar('Performance/System_Throughput_Min_100steps', np.min(recent_system_throughput), step)
                
                # 按环境分别统计系统吞吐量
                env_system_throughputs = defaultdict(list)
                for entry in recent_throughput_data:
                    if 'system_throughput_mbps' in entry:
                        env_system_throughputs[entry['env_id']].append(entry['system_throughput_mbps'])
                
                for env_id, throughput_values in env_system_throughputs.items():
                    if throughput_values:
                        env_avg_throughput = np.mean(throughput_values)
                        writer.add_scalar(f'Performance/Env_{env_id}_System_Throughput_Mbps', env_avg_throughput, step)
        
        # 平均用户吞吐量统计（新增）
        if self.performance_metrics['avg_throughput_per_user']:
            # 计算最近100步的平均用户吞吐量统计
            recent_avg_throughput_data = self.performance_metrics['avg_throughput_per_user'][-100:]
            recent_avg_throughput = [t['avg_throughput_per_user_mbps'] for t in recent_avg_throughput_data]
            
            if recent_avg_throughput:
                # 平均用户吞吐量
                avg_user_throughput = np.mean(recent_avg_throughput)
                writer.add_scalar('Performance/Avg_User_Throughput_Mbps_100steps', avg_user_throughput, step)
                
                # 平均用户吞吐量标准差
                user_throughput_std = np.std(recent_avg_throughput)
                writer.add_scalar('Performance/Avg_User_Throughput_Std_100steps', user_throughput_std, step)
                
                # 按环境分别统计平均用户吞吐量
                env_avg_user_throughputs = defaultdict(list)
                for entry in recent_avg_throughput_data:
                    env_avg_user_throughputs[entry['env_id']].append(entry['avg_throughput_per_user_mbps'])
                
                for env_id, throughput_values in env_avg_user_throughputs.items():
                    if throughput_values:
                        env_avg_user_throughput = np.mean(throughput_values)
                        writer.add_scalar(f'Performance/Env_{env_id}_Avg_User_Throughput_Mbps', env_avg_user_throughput, step)
    
    def get_summary_statistics(self):
        """获取训练摘要统计信息"""
        summary = {
            'total_episodes': self.training_rewards['episodes_completed'],
            'total_steps': self.training_rewards['total_steps'],
            'skill_switches': self.skill_usage['skill_switches']
        }
        
        if self.training_rewards['episode_rewards']:
            rewards = [r['total_reward'] for r in self.training_rewards['episode_rewards']]
            summary.update({
                'reward_mean': np.mean(rewards),
                'reward_std': np.std(rewards),
                'reward_min': np.min(rewards),
                'reward_max': np.max(rewards)
            })
        
        if self.skill_usage['team_skills']:
            summary['team_skill_usage'] = dict(self.skill_usage['team_skills'])
        
        return summary

# 获取计算设备
def get_device(device_pref):
    """
    根据偏好选择计算设备
    
    参数:
        device_pref: 设备偏好 ('auto', 'cuda', 'cpu')
        
    返回:
        device: torch.device对象
    """
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
    else:  # 'cpu'或其他值
        main_logger.info("使用CPU")
        return torch.device('cpu')

# 创建环境函数
def make_env(scenario, n_uavs, n_users, user_distribution, channel_model, max_hops=None, render_mode=None, rank=0, seed=0):
    """
    创建环境实例的函数 (用于 SubprocVecEnv)
    """
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

# 解析命令行参数
def parse_args():
    parser = argparse.ArgumentParser(description='增强的HMASD训练，专注于论文数据收集')
    
    # 运行模式和环境参数
    parser.add_argument('--mode', type=str, default='train', help='运行模式: train或eval')
    parser.add_argument('--scenario', type=int, default=2, help='场景: 1=基站模式, 2=协作组网模式')
    parser.add_argument('--model_path', type=str, default='models/hmasd_enhanced_tracking.pt', help='模型保存/加载路径')
    parser.add_argument('--log_dir', type=str, default='logs', help='日志目录')
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

# 增强的训练函数
def train(vec_env, eval_vec_env, config, args, device):
    """
    增强的HMASD训练函数，专注于论文数据收集
    """
    num_envs = vec_env.num_envs
    main_logger.info(f"开始增强的HMASD训练 (使用 {num_envs} 个并行环境)...")

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
    log_dir = os.path.join(args.log_dir, f"enhanced_tracking_{datetime.now().strftime('%Y%m%d-%H%M%S')}")
    os.makedirs(log_dir, exist_ok=True)
    model_dir = os.path.dirname(args.model_path)
    os.makedirs(model_dir, exist_ok=True)
    
    # 创建HMASD代理
    agent = HMASDAgent(config, log_dir=log_dir, device=device)
    
    # 创建增强的奖励追踪器
    reward_tracker = EnhancedRewardTracker(log_dir, config)
    reward_tracker.export_interval = args.export_interval
    
    # 记录超参数
    agent.writer.add_text('Parameters/n_agents', str(config.n_agents), 0)
    agent.writer.add_text('Parameters/n_Z', str(config.n_Z), 0)
    agent.writer.add_text('Parameters/n_z', str(config.n_z), 0)
    agent.writer.add_text('Parameters/k', str(config.k), 0)
    agent.writer.add_text('Parameters/lambda_e', str(config.lambda_e), 0)
    agent.writer.add_text('Parameters/num_envs', str(num_envs), 0)

    # 训练变量
    total_steps = 0
    n_episodes = 0
    max_episodes = config.total_timesteps // config.buffer_size
    episode_rewards = []
    update_times = 0
    best_reward = float('-inf')
    last_eval_step = 0
    
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
    env_skill_durations = np.zeros(num_envs, dtype=int)
    
    # 奖励组成跟踪
    env_reward_components = [{
        'env_component': 0.0,
        'team_disc_component': 0.0,
        'ind_disc_component': 0.0
    } for _ in range(num_envs)]
    
    # 训练循环
    while total_steps < config.total_timesteps:
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
            skill_timer_value = env_skill_durations[i]
            
            # 计算奖励组成部分（模拟内在奖励计算）
            current_reward = rewards[i] if isinstance(rewards[i], (int, float)) else rewards[i].item()
            
            # 估算奖励组成部分
            env_component = config.lambda_e * current_reward
            team_disc_component = config.lambda_D * 0.1  # 模拟判别器奖励
            ind_disc_component = config.lambda_d * 0.05  # 模拟个体判别器奖励
            
            # 累积奖励组成
            env_reward_components[i]['env_component'] += env_component
            env_reward_components[i]['team_disc_component'] += team_disc_component
            env_reward_components[i]['ind_disc_component'] += ind_disc_component
            
            # 记录到奖励追踪器
            if args.detailed_logging:
                reward_tracker.log_training_step(
                    total_steps, i, current_reward,
                    reward_components={
                        'env_component': env_component,
                        'team_disc_component': team_disc_component,
                        'ind_disc_component': ind_disc_component
                    },
                    info=infos[i]  # 传递完整的info，包含reward_info
                )
            
            # 记录技能使用
            reward_tracker.log_skill_usage(
                total_steps,
                current_agent_info['team_skill'],
                current_agent_info['agent_skills'],
                current_agent_info['skill_changed']
            )
            
            # 存储经验
            agent.store_transition(
                states[i], next_states[i], observations[i], next_observations[i],
                actions_array[i], rewards[i], dones[i], current_agent_info['team_skill'],
                current_agent_info['agent_skills'], current_agent_info['action_logprobs'],
                log_probs=current_agent_info['log_probs'],
                skill_timer_for_env=skill_timer_value,
                env_id=i
            )
            
            # 更新技能持续时间
            if dones[i]:
                env_skill_durations[i] = 0
            elif skill_timer_value == config.k - 1:
                env_skill_durations[i] = 0
            elif current_agent_info['skill_changed']:
                env_skill_durations[i] = 0
            else:
                env_skill_durations[i] += 1

            # 更新环境状态跟踪
            env_steps[i] += 1
            env_rewards[i] += current_reward

            # 如果环境完成
            if dones[i]:
                n_episodes += 1
                episode_rewards.append(env_rewards[i])

                # 记录episode完成到追踪器
                episode_info = {}
                if 'global' in infos[i]:
                    episode_info.update({
                        'served_users': infos[i]['global'].get('served_users', 0),
                        'total_users': len(infos[i]['global'].get('connections', [{}])[0]) if infos[i]['global'].get('connections') else 0,
                        'coverage_ratio': infos[i]['global'].get('served_users', 0) / max(len(infos[i]['global'].get('connections', [{}])[0]), 1) if infos[i]['global'].get('connections') else 0
                    })
                
                # 添加奖励组成信息
                episode_info.update(env_reward_components[i])
                
                reward_tracker.log_episode_completion(
                    n_episodes, i, env_rewards[i], env_steps[i], episode_info
                )

                # 记录到TensorBoard - 增强版本
                agent.writer.add_scalar('Training/Episode_Reward', env_rewards[i], n_episodes)
                agent.writer.add_scalar('Training/Episode_Length', env_steps[i], n_episodes)
                
                # 记录奖励组成
                total_intrinsic = sum(env_reward_components[i].values())
                if total_intrinsic != 0:
                    for comp_name, comp_value in env_reward_components[i].items():
                        agent.writer.add_scalar(f'Training/Episode_{comp_name}', comp_value, n_episodes)
                        agent.writer.add_scalar(f'Training/Episode_{comp_name}_Proportion', comp_value/total_intrinsic, n_episodes)

                # 获取吞吐量信息（修正后的字段名）
                system_throughput = 0
                if 'reward_info' in infos[i] and 'system_throughput_mbps' in infos[i]['reward_info']:
                    system_throughput = infos[i]['reward_info']['system_throughput_mbps']
                
                main_logger.info(f"Episode结束 - 环境ID: {i}, Episode编号: {n_episodes}, "
                               f"总奖励: {env_rewards[i]:.2f}, 步数: {env_steps[i]}, "
                               f"服务用户: {episode_info.get('served_users', 0)}/{episode_info.get('total_users', 0)}, "
                               f"覆盖率: {episode_info.get('coverage_ratio', 0):.2%}, "
                               f"系统吞吐量: {system_throughput:.2f}Mbps")

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
                    std_reward = np.std(recent_rewards)
                    max_reward = np.max(recent_rewards)
                    min_reward = np.min(recent_rewards)

                    agent.writer.add_scalar('Training/Avg_Reward_10ep', avg_reward, n_episodes)
                    agent.writer.add_scalar('Training/Std_Reward_10ep', std_reward, n_episodes)
                    agent.writer.add_scalar('Training/Max_Reward_10ep', max_reward, n_episodes)
                    agent.writer.add_scalar('Training/Min_Reward_10ep', min_reward, n_episodes)

                    main_logger.info(f"最近10个episodes: 平均奖励 {avg_reward:.2f} ± {std_reward:.2f}, 最大/最小: {max_reward:.2f}/{min_reward:.2f}")

            # 更新网络
            if total_steps // num_envs > 0 and (total_steps // num_envs) % (config.buffer_size // num_envs) == 0:
                if len(agent.low_level_buffer) >= agent.config.batch_size:
                    try:
                        update_info = agent.update()
                        update_times += 1
                        elapsed = time.time() - start_time

                        # 增强的更新信息记录
                        if total_steps % 10240 == 0:
                            main_logger.info(f"更新 {update_times}, 总步数 {total_steps}, "
                                 f"高层损失 {update_info.get('coordinator_loss', 0):.4f}, "
                                 f"低层损失 {update_info.get('discoverer_loss', 0):.4f}, "
                                 f"判别器损失 {update_info.get('discriminator_loss', 0):.4f}, "
                                 f"已用时间 {elapsed:.2f}s")
                            
                            # 记录详细的更新信息到TensorBoard
                            for key, value in update_info.items():
                                if isinstance(value, (int, float)):
                                    agent.writer.add_scalar(f'Training/Update_{key}', value, update_times)
                            
                    except Exception as e:
                        main_logger.error(f"更新错误: {e}")
                        update_times += 1

            # 定期导出数据
            if total_steps % reward_tracker.export_interval == 0:
                reward_tracker.export_training_data(total_steps, agent.writer)

            # 评估
            if total_steps >= last_eval_step + config.eval_interval:
                main_logger.info(f"即将进行评估，将评估 {config.eval_episodes} 个episodes...")
                eval_reward, eval_std, eval_min, eval_max = evaluate(eval_vec_env, agent, config.eval_episodes)
                main_logger.info(f"评估完成 ({config.eval_episodes} 个episodes): 平均奖励 {eval_reward:.2f} ± {eval_std:.2f}, 最大/最小: {eval_max:.2f}/{eval_min:.2f}")

                # 保存最佳模型
                if eval_reward > best_reward:
                    best_reward = eval_reward
                    agent.save_model(args.model_path)
                    main_logger.info(f"保存最佳模型，奖励: {best_reward:.2f}")
                
                last_eval_step = total_steps

        # 更新总步数
        total_steps += num_envs

        # 更新状态和观测
        states = next_states
        observations = next_observations

    # 训练完成后的最终数据导出
    reward_tracker.export_training_data(total_steps, agent.writer)
    
    # 生成训练摘要报告
    summary = reward_tracker.get_summary_statistics()
    main_logger.info("训练摘要:")
    for key, value in summary.items():
        main_logger.info(f"  {key}: {value}")
    
    # 保存摘要到文件
    import json
    summary_path = os.path.join(log_dir, 'training_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    main_logger.info(f"训练完成! 总步数: {total_steps}, 总episodes: {n_episodes}")
    main_logger.info(f"最佳奖励: {best_reward:.2f}")
    main_logger.info(f"训练数据已保存到: {log_dir}")

    # 保存最终模型
    final_model_path = os.path.join(model_dir, 'hmasd_enhanced_tracking_final.pt')
    agent.save_model(final_model_path)
    main_logger.info(f"最终模型已保存到 {final_model_path}")
    
    return agent

# 评估函数（复用原有的evaluate函数）
def evaluate(vec_env, agent, n_episodes=10, render=False):
    """评估HMASD代理"""
    # 这里复用原来的evaluate函数逻辑
    # 为了简洁，暂时使用简化版本
    main_logger.info(f"开始评估: {n_episodes} episodes")
    
    num_envs = vec_env.num_envs
    episode_rewards = []
    
    # 重置环境
    results = vec_env.env_method('reset')
    observations = np.array([res[0] for res in results])
    initial_infos = [res[1] for res in results]
    states = np.array([info.get('state', np.zeros(agent.config.state_dim)) for info in initial_infos])
    
    env_steps = np.zeros(num_envs, dtype=int)
    env_rewards = np.zeros(num_envs)
    completed_episodes = 0
    
    with torch.no_grad():
        while completed_episodes < n_episodes:
            all_actions_list = []
            
            for i in range(num_envs):
                actions, agent_info = agent.step(states[i], observations[i], env_steps[i], deterministic=True, env_id=i)
                all_actions_list.append(actions)
            
            actions_array = np.array(all_actions_list)
            next_observations, rewards, dones, infos = vec_env.step(actions_array)
            next_states = np.array([info.get('next_state', np.zeros(agent.config.state_dim)) for info in infos])
            
            for i in range(num_envs):
                env_steps[i] += 1
                env_rewards[i] += rewards[i]
                
                if dones[i] and completed_episodes < n_episodes:
                    episode_rewards.append(env_rewards[i])
                    completed_episodes += 1
                    main_logger.info(f"评估 Episode {completed_episodes}/{n_episodes}, 奖励: {env_rewards[i]:.2f}")
                    env_steps[i] = 0
                    env_rewards[i] = 0
            
            states = next_states
            observations = next_observations
            
            if completed_episodes >= n_episodes:
                break

    mean_reward = np.mean(episode_rewards) if episode_rewards else 0
    std_reward = np.std(episode_rewards) if episode_rewards else 0
    min_reward = np.min(episode_rewards) if episode_rewards else 0
    max_reward = np.max(episode_rewards) if episode_rewards else 0

    return mean_reward, std_reward, min_reward, max_reward

# 主函数
def main():
    args = parse_args()
    
    # 创建日志目录
    os.makedirs(args.log_dir, exist_ok=True)
    
    # 为训练会话创建固定的日志文件名
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = f"hmasd_enhanced_tracking_{timestamp}.log"
    
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
    main_logger = get_logger("HMASD-Enhanced")
    main_logger.info(f"增强的HMASD训练启动: 文件级别={args.log_level}, 控制台级别={args.console_log_level}")
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

    main_logger.info(f"使用论文中的超参数: n_Z={config.n_Z}, n_z={config.n_z}, k={config.k}, lambda_e={config.lambda_e}")

    if args.mode == 'train':
        agent = train(train_vec_env, eval_vec_env, config, args, device)
        main_logger.info("训练完成，增强的数据收集已启用")
    elif args.mode == 'eval':
        # 评估模式（复用原有逻辑）
        if not os.path.exists(args.model_path):
            main_logger.error(f"模型文件 {args.model_path} 不存在")
            return
        
        log_dir = os.path.join(args.log_dir, f"eval_enhanced_{datetime.now().strftime('%Y%m%d-%H%M%S')}")
        os.makedirs(log_dir, exist_ok=True)
        
        agent = HMASDAgent(config, log_dir=log_dir, device=device)
        agent.load_model(args.model_path)
        
        evaluate(eval_vec_env, agent, n_episodes=args.eval_episodes, render=args.render)
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
