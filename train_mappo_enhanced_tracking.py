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
import traceback
import sys
import gc
import psutil
from logger import init_multiproc_logging, get_logger, shutdown_logging, LOG_LEVELS, set_log_level

# 导入 PyTorch 相关库
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.distributions import Categorical, Normal

# 导入 Stable Baselines3 向量化环境
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor

# 导入配置和环境
from config_1 import Config
from envs.pettingzoo.scenario1 import UAVBaseStationEnv
from envs.pettingzoo.scenario2 import UAVCooperativeNetworkEnv
from envs.pettingzoo.env_adapter import ParallelToArrayAdapter

# 导入工具类
from hmasd.utils import ReplayBuffer, compute_gae, compute_ppo_loss

# 数值稳定性和监控工具
def check_tensor_health(tensor, name="", logger=None, raise_on_error=False):
    """检查张量的数值健康状况 - 增强版本，支持布尔张量"""
    if logger is None:
        logger = main_logger
        
    try:
        if tensor is None:
            msg = f"张量 {name} 为 None"
            logger.error(msg)
            if raise_on_error:
                raise ValueError(msg)
            return False
            
        if not isinstance(tensor, torch.Tensor):
            msg = f"张量 {name} 不是 torch.Tensor 类型: {type(tensor)}"
            logger.error(msg)
            if raise_on_error:
                raise TypeError(msg)
            return False
        
        # 检查张量是否为空
        if tensor.numel() == 0:
            msg = f"张量 {name} 为空张量"
            logger.error(msg)
            if raise_on_error:
                raise ValueError(msg)
            return False
        
        # 对于布尔张量，跳过数值统计，只检查基本属性
        if tensor.dtype == torch.bool:
            logger.debug(f"张量 {name} 是布尔类型，跳过数值统计检查")
            logger.debug(f"布尔张量 {name} 健康检查通过: 形状={tensor.shape}, "
                        f"True数量={tensor.sum().item()}, False数量={(~tensor).sum().item()}")
            return True
            
        # 检查是否包含 NaN 或 Inf (仅对数值类型)
        if tensor.dtype in [torch.float16, torch.float32, torch.float64, torch.complex64, torch.complex128]:
            has_nan = torch.isnan(tensor).any()
            has_inf = torch.isinf(tensor).any()
            
            if has_nan:
                nan_count = torch.isnan(tensor).sum().item()
                msg = f"张量 {name} 包含 {nan_count} 个 NaN 值"
                logger.error(msg)
                if raise_on_error:
                    raise ValueError(msg)
                return False
                
            if has_inf:
                inf_count = torch.isinf(tensor).sum().item()
                msg = f"张量 {name} 包含 {inf_count} 个 Inf 值"
                logger.error(msg)
                if raise_on_error:
                    raise ValueError(msg)
                return False
        
        # 数值统计检查 (仅对数值类型张量)
        if tensor.dtype in [torch.float16, torch.float32, torch.float64, torch.int8, torch.int16, torch.int32, torch.int64]:
            # 安全转换为浮点数进行统计
            float_tensor = tensor.float() if tensor.dtype != torch.float32 else tensor
            
            tensor_min = float_tensor.min().item()
            tensor_max = float_tensor.max().item()
            tensor_mean = float_tensor.mean().item()
            tensor_std = float_tensor.std().item()
            
            # 检查数值是否过大或过小
            if abs(tensor_max) > 1e6 or abs(tensor_min) > 1e6:
                logger.warning(f"张量 {name} 数值范围异常: 最小值={tensor_min:.6f}, 最大值={tensor_max:.6f}")
            
            if tensor_std > 1e3:
                logger.warning(f"张量 {name} 标准差过大: {tensor_std:.6f}")
            
            # 记录张量统计信息（仅在debug模式下）
            logger.debug(f"张量 {name} 健康检查通过: 形状={tensor.shape}, "
                        f"范围=[{tensor_min:.6f}, {tensor_max:.6f}], "
                        f"均值={tensor_mean:.6f}, 标准差={tensor_std:.6f}")
        else:
            # 对于其他类型（如整数），只记录基本信息
            logger.debug(f"张量 {name} 健康检查通过: 形状={tensor.shape}, 类型={tensor.dtype}")
        
        return True
        
    except Exception as e:
        msg = f"检查张量 {name} 时发生异常: {e}"
        logger.error(msg)
        if raise_on_error:
            raise
        return False

def safe_divide(numerator, denominator, epsilon=1e-8, logger=None):
    """安全除法，避免除零错误 - 增强版本"""
    if logger is None:
        logger = main_logger
        
    try:
        # 检查输入张量健康性
        if not check_tensor_health(numerator, "numerator", logger, raise_on_error=False):
            logger.warning("分子张量异常，返回零张量")
            return torch.zeros_like(numerator) if isinstance(numerator, torch.Tensor) else torch.tensor(0.0)
            
        if not check_tensor_health(denominator, "denominator", logger, raise_on_error=False):
            logger.warning("分母张量异常，返回零张量")
            return torch.zeros_like(numerator) if isinstance(numerator, torch.Tensor) else torch.tensor(0.0)
        
        # 检查分母是否接近零
        if isinstance(denominator, torch.Tensor):
            zero_mask = torch.abs(denominator) < epsilon
            if zero_mask.any():
                logger.debug(f"检测到 {zero_mask.sum().item()} 个接近零的分母值，将使用epsilon调整")
                safe_denominator = torch.where(zero_mask, epsilon, denominator)
            else:
                safe_denominator = denominator
        else:
            if abs(denominator) < epsilon:
                logger.debug("标量分母接近零，使用epsilon")
                safe_denominator = epsilon if denominator >= 0 else -epsilon
            else:
                safe_denominator = denominator
        
        result = numerator / safe_denominator
        
        # 检查结果健康性
        if not check_tensor_health(result, "division_result", logger):
            logger.warning("除法结果异常，使用备用值")
            return torch.zeros_like(numerator) if isinstance(numerator, torch.Tensor) else torch.tensor(0.0)
            
        return result
        
    except Exception as e:
        logger.error(f"安全除法操作失败: {e}")
        return torch.zeros_like(numerator) if isinstance(numerator, torch.Tensor) else torch.tensor(0.0)

def safe_log(tensor, epsilon=1e-8, logger=None):
    """安全对数运算，避免log(0)"""
    if logger is None:
        logger = main_logger
        
    try:
        # 确保输入大于零
        safe_tensor = torch.clamp(tensor, min=epsilon)
        result = torch.log(safe_tensor)
        
        if not check_tensor_health(result, "log_result", logger):
            logger.warning(f"对数运算结果异常")
            return torch.zeros_like(tensor)
            
        return result
        
    except Exception as e:
        logger.error(f"安全对数运算失败: {e}")
        return torch.zeros_like(tensor)

def safe_exp(tensor, max_value=50.0, logger=None):
    """安全指数运算，避免数值溢出"""
    if logger is None:
        logger = main_logger
        
    try:
        # 限制指数输入范围
        safe_tensor = torch.clamp(tensor, max=max_value)
        result = torch.exp(safe_tensor)
        
        if not check_tensor_health(result, "exp_result", logger):
            logger.warning(f"指数运算结果异常")
            return torch.ones_like(tensor)
            
        return result
        
    except Exception as e:
        logger.error(f"安全指数运算失败: {e}")
        return torch.ones_like(tensor)

def monitor_gradients(model, name="", logger=None, max_norm_threshold=10.0):
    """监控模型梯度"""
    if logger is None:
        logger = main_logger
        
    try:
        total_norm = 0.0
        param_count = 0
        grad_norms = []
        
        for param in model.parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
                grad_norms.append(param_norm.item())
                param_count += 1
        
        total_norm = total_norm ** (1. / 2)
        
        if param_count > 0:
            avg_grad_norm = np.mean(grad_norms)
            max_grad_norm = np.max(grad_norms)
            min_grad_norm = np.min(grad_norms)
            
            logger.debug(f"模型 {name} 梯度统计: 总范数={total_norm:.6f}, 平均={avg_grad_norm:.6f}, "
                        f"最大={max_grad_norm:.6f}, 最小={min_grad_norm:.6f}")
            
            # 检查梯度爆炸
            if total_norm > max_norm_threshold:
                logger.warning(f"模型 {name} 检测到梯度爆炸: 总范数={total_norm:.6f} > 阈值={max_norm_threshold}")
                
            # 检查梯度消失
            if total_norm < 1e-7:
                logger.warning(f"模型 {name} 检测到梯度消失: 总范数={total_norm:.6f}")
                
        return total_norm, grad_norms
        
    except Exception as e:
        logger.error(f"监控模型 {name} 梯度时发生异常: {e}")
        return 0.0, []

def log_memory_usage(logger=None, step=None):
    """记录内存使用情况 - 增强版本，包含内存泄漏检测"""
    if logger is None:
        logger = main_logger
        
    try:
        # GPU内存使用
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated() / (1024**3)  # GB
            gpu_memory_cached = torch.cuda.memory_reserved() / (1024**3)  # GB
            gpu_max_memory = torch.cuda.max_memory_allocated() / (1024**3)  # GB
            
            logger.debug(f"步骤 {step} GPU内存: 已分配={gpu_memory:.2f}GB, "
                        f"缓存={gpu_memory_cached:.2f}GB, 峰值={gpu_max_memory:.2f}GB")
            
            # GPU内存预警
            if gpu_memory > 8.0:  # 8GB阈值
                logger.warning(f"步骤 {step} GPU内存使用过高: {gpu_memory:.2f}GB")
            
            # 检测内存泄漏 - 如果缓存内存远大于已分配内存
            if gpu_memory_cached > gpu_memory * 2:
                logger.warning(f"步骤 {step} 检测到可能的GPU内存泄漏: "
                              f"缓存({gpu_memory_cached:.2f}GB) >> 已分配({gpu_memory:.2f}GB)")
        
        # CPU内存使用
        process = psutil.Process()
        memory_info = process.memory_info()
        cpu_memory = memory_info.rss / (1024**3)  # GB
        cpu_memory_percent = process.memory_percent()
        
        logger.debug(f"步骤 {step} CPU内存: {cpu_memory:.2f}GB ({cpu_memory_percent:.1f}%)")
        
        # CPU内存预警
        if cpu_memory > 16.0:  # 16GB阈值
            logger.warning(f"步骤 {step} CPU内存使用过高: {cpu_memory:.2f}GB")
        
        # 系统内存使用
        system_memory = psutil.virtual_memory()
        logger.debug(f"步骤 {step} 系统内存: 使用率={system_memory.percent}%, "
                    f"可用={system_memory.available / (1024**3):.2f}GB")
        
        # 系统内存预警
        if system_memory.percent > 85:
            logger.warning(f"步骤 {step} 系统内存使用率过高: {system_memory.percent}%")
            
        # 返回内存统计信息用于监控
        return {
            'gpu_memory': gpu_memory if torch.cuda.is_available() else 0,
            'gpu_memory_cached': gpu_memory_cached if torch.cuda.is_available() else 0,
            'cpu_memory': cpu_memory,
            'cpu_memory_percent': cpu_memory_percent,
            'system_memory_percent': system_memory.percent
        }
        
    except Exception as e:
        logger.error(f"记录内存使用时发生异常: {e}")
        return {}

def safe_tensor_ops_wrapper(func):
    """装饰器：为张量操作添加安全检查"""
    def wrapper(*args, **kwargs):
        try:
            # 检查输入张量
            for i, arg in enumerate(args):
                if isinstance(arg, torch.Tensor):
                    if not check_tensor_health(arg, f"input_{i}"):
                        raise ValueError(f"输入张量 {i} 健康检查失败")
            
            # 执行原函数
            result = func(*args, **kwargs)
            
            # 检查输出张量
            if isinstance(result, torch.Tensor):
                if not check_tensor_health(result, "output"):
                    raise ValueError("输出张量健康检查失败")
            elif isinstance(result, (list, tuple)):
                for i, item in enumerate(result):
                    if isinstance(item, torch.Tensor):
                        if not check_tensor_health(item, f"output_{i}"):
                            raise ValueError(f"输出张量 {i} 健康检查失败")
            
            return result
            
        except Exception as e:
            main_logger.error(f"安全张量操作失败: {e}")
            main_logger.error(f"函数: {func.__name__}, 参数: {args}, 关键字参数: {kwargs}")
            raise
    
    return wrapper

class EnhancedRewardTracker:
    """增强的奖励追踪器，用于MAPPO训练数据收集"""
    
    def __init__(self, log_dir, config):
        self.log_dir = log_dir
        self.config = config
        
        # 训练过程中的奖励数据收集
        self.training_rewards = {
            'episode_rewards': [],
            'step_rewards': [],
            'env_rewards': [],
            'agent_rewards': [],  # MAPPO特有：每个智能体的奖励
            'cumulative_rewards': [],
            'reward_variance': [],
            'episodes_completed': 0,
            'total_steps': 0
        }
        
        # 性能指标
        self.performance_metrics = {
            'episode_lengths': [],
            'success_rates': [],
            'coverage_ratios': [],
            'served_users': [],
            'network_efficiency': [],
            'total_throughput': [],
            'avg_throughput_per_user': [],
            'agent_coordination': []  # MAPPO特有：智能体协调指标
        }
        
        # 滑动窗口统计
        self.window_size = 100
        self.recent_rewards = deque(maxlen=self.window_size)
        self.recent_lengths = deque(maxlen=self.window_size)
        
        # 数据导出设置
        self.export_interval = 1000
        self.last_export_step = 0
    
    def log_training_step(self, step, env_id, reward, agent_rewards=None, info=None):
        """记录训练步骤的奖励信息"""
        self.training_rewards['total_steps'] += 1
        self.training_rewards['step_rewards'].append({
            'step': step,
            'env_id': env_id,
            'reward': reward,
            'timestamp': time.time()
        })
        
        # 记录每个智能体的奖励
        if agent_rewards is not None:
            self.training_rewards['agent_rewards'].append({
                'step': step,
                'env_id': env_id,
                'agent_rewards': agent_rewards.copy(),
                'timestamp': time.time()
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
            
            # 记录吞吐量信息
            if 'reward_info' in info:
                reward_info = info['reward_info']
                if 'total_throughput_mbps' in reward_info:
                    self.performance_metrics['total_throughput'].append({
                        'step': step,
                        'env_id': env_id,
                        'total_throughput_mbps': reward_info['total_throughput_mbps'],
                        'timestamp': time.time()
                    })
    
    def log_episode_completion(self, episode_num, env_id, total_reward, episode_length, agent_rewards=None, info=None):
        """记录episode完成信息"""
        self.training_rewards['episodes_completed'] += 1
        
        episode_data = {
            'episode': episode_num,
            'env_id': env_id,
            'total_reward': total_reward,
            'episode_length': episode_length,
            'timestamp': time.time()
        }
        
        if agent_rewards is not None:
            episode_data['agent_rewards'] = agent_rewards.copy()
            # 计算智能体协调指标
            reward_std = np.std(agent_rewards)
            reward_mean = np.mean(agent_rewards)
            coordination_metric = 1.0 / (1.0 + reward_std) if reward_std > 0 else 1.0
            episode_data['coordination_metric'] = coordination_metric
            
            self.performance_metrics['agent_coordination'].append({
                'episode': episode_num,
                'env_id': env_id,
                'coordination_metric': coordination_metric,
                'reward_std': reward_std,
                'reward_mean': reward_mean
            })
        
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
    
    def export_training_data(self, step):
        """导出训练数据用于论文分析"""
        if step - self.last_export_step < self.export_interval:
            return
        
        export_dir = os.path.join(self.log_dir, 'paper_data')
        os.makedirs(export_dir, exist_ok=True)
        
        # 导出奖励数据
        if self.training_rewards['episode_rewards']:
            rewards_df = pd.DataFrame(self.training_rewards['episode_rewards'])
            rewards_df.to_csv(os.path.join(export_dir, f'episode_rewards_step_{step}.csv'), index=False)
        
        # 导出智能体协调数据
        if self.performance_metrics['agent_coordination']:
            coord_df = pd.DataFrame(self.performance_metrics['agent_coordination'])
            coord_df.to_csv(os.path.join(export_dir, f'agent_coordination_step_{step}.csv'), index=False)
        
        # 生成训练曲线图
        self.generate_training_plots(export_dir, step)
        
        self.last_export_step = step
        main_logger.debug(f"已导出步骤 {step} 的训练数据到 {export_dir}")
    
    def generate_training_plots(self, export_dir, step):
        """生成训练过程的可视化图表"""
        
        # Episode奖励趋势图
        if self.training_rewards['episode_rewards']:
            episodes = [r['episode'] for r in self.training_rewards['episode_rewards']]
            rewards = [r['total_reward'] for r in self.training_rewards['episode_rewards']]
            
            plt.figure(figsize=(15, 10))
            
            # 原始奖励曲线
            plt.subplot(2, 3, 1)
            plt.plot(episodes, rewards, alpha=0.3, color='blue', label='Episode Rewards')
            # 滑动平均
            if len(rewards) >= 10:
                window = 50
                if len(rewards) >= window:
                    smoothed = pd.Series(rewards).rolling(window=window, center=True).mean()
                    plt.plot(episodes, smoothed, color='red', linewidth=2, label=f'{window}-episode MA')
            plt.xlabel('Episode')
            plt.ylabel('Total Reward')
            plt.title('MAPPO Training Reward Progress')
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
            if len(episodes) == len([r['episode_length'] for r in self.training_rewards['episode_rewards']]):
                lengths = [r['episode_length'] for r in self.training_rewards['episode_rewards']]
                plt.subplot(2, 3, 3)
                plt.plot(episodes, lengths, alpha=0.6, color='orange')
                plt.xlabel('Episode')
                plt.ylabel('Episode Length')
                plt.title('Episode Length Progression')
                plt.grid(True, alpha=0.3)
            
            # 智能体协调指标
            if self.performance_metrics['agent_coordination']:
                coord_episodes = [c['episode'] for c in self.performance_metrics['agent_coordination']]
                coord_metrics = [c['coordination_metric'] for c in self.performance_metrics['agent_coordination']]
                
                plt.subplot(2, 3, 4)
                plt.plot(coord_episodes, coord_metrics, alpha=0.7, color='purple')
                plt.xlabel('Episode')
                plt.ylabel('Coordination Metric')
                plt.title('Agent Coordination Over Time')
                plt.grid(True, alpha=0.3)
            
            # 奖励方差趋势
            if self.training_rewards['reward_variance']:
                var_episodes = [v['episode'] for v in self.training_rewards['reward_variance']]
                var_means = [v['mean'] for v in self.training_rewards['reward_variance']]
                var_stds = [v['std'] for v in self.training_rewards['reward_variance']]
                
                plt.subplot(2, 3, 5)
                plt.errorbar(var_episodes, var_means, yerr=var_stds, alpha=0.7, color='red')
                plt.xlabel('Episode')
                plt.ylabel('Mean Reward ± Std')
                plt.title('Reward Stability (100-episode window)')
                plt.grid(True, alpha=0.3)
            
            # 智能体奖励标准差趋势
            if self.performance_metrics['agent_coordination']:
                reward_stds = [c['reward_std'] for c in self.performance_metrics['agent_coordination']]
                
                plt.subplot(2, 3, 6)
                plt.plot(coord_episodes, reward_stds, alpha=0.7, color='brown')
                plt.xlabel('Episode')
                plt.ylabel('Agent Reward Std')
                plt.title('Agent Reward Variance')
                plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(export_dir, f'mappo_training_progress_step_{step}.png'), dpi=300, bbox_inches='tight')
            plt.close()
    
    def get_summary_statistics(self):
        """获取训练摘要统计信息"""
        summary = {
            'total_episodes': self.training_rewards['episodes_completed'],
            'total_steps': self.training_rewards['total_steps']
        }
        
        if self.training_rewards['episode_rewards']:
            rewards = [r['total_reward'] for r in self.training_rewards['episode_rewards']]
            summary.update({
                'reward_mean': np.mean(rewards),
                'reward_std': np.std(rewards),
                'reward_min': np.min(rewards),
                'reward_max': np.max(rewards)
            })
        
        if self.performance_metrics['agent_coordination']:
            coord_metrics = [c['coordination_metric'] for c in self.performance_metrics['agent_coordination']]
            summary.update({
                'avg_coordination': np.mean(coord_metrics),
                'coordination_std': np.std(coord_metrics)
            })
        
        return summary

class MAPPOActor(nn.Module):
    """MAPPO Actor网络 - 增强数值稳定性"""
    
    def __init__(self, obs_dim, action_dim, hidden_size=64, activation_fn=nn.ReLU):
        super(MAPPOActor, self).__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            activation_fn(),
            nn.Linear(hidden_size, hidden_size),
            activation_fn(),
            nn.Linear(hidden_size, action_dim)
        )
        
        # 连续动作空间的标准差 - 限制范围避免数值不稳定
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        
        # 数值稳定性参数
        self.epsilon = 1e-8
        self.log_std_min = -20
        self.log_std_max = 2
        
    def forward(self, obs):
        """前向传播 - 增强数值稳定性"""
        try:
            # 检查输入健康性
            if not check_tensor_health(obs, "actor_input", main_logger):
                main_logger.error("Actor输入张量异常，使用零张量")
                obs = torch.zeros_like(obs)
            
            mean = self.net(obs)
            
            # 检查均值健康性
            if not check_tensor_health(mean, "actor_mean", main_logger):
                main_logger.error("Actor均值异常，使用零张量")
                mean = torch.zeros_like(mean)
            
            # 限制log_std范围，防止数值不稳定
            log_std = torch.clamp(self.log_std, self.log_std_min, self.log_std_max)
            std = safe_exp(log_std, max_value=self.log_std_max, logger=main_logger)
            
            # 确保标准差不会太小
            std = torch.clamp(std, min=self.epsilon)
            
            # 检查标准差健康性
            if not check_tensor_health(std, "actor_std", main_logger):
                main_logger.error("Actor标准差异常，使用单位张量")
                std = torch.ones_like(std)
            
            return mean, std
            
        except Exception as e:
            main_logger.error(f"Actor前向传播失败: {e}")
            # 返回安全的默认值
            return torch.zeros(obs.shape[0], self.action_dim), torch.ones(obs.shape[0], self.action_dim)
    
    def get_action_and_log_prob(self, obs):
        """获取动作和对数概率 - 增强数值稳定性"""
        try:
            mean, std = self.forward(obs)
            
            # 创建分布时添加数值稳定性检查
            dist = Normal(mean, std)
            action = dist.sample()
            
            # 检查动作健康性
            if not check_tensor_health(action, "sampled_action", main_logger):
                main_logger.warning("采样动作异常，使用均值")
                action = mean
            
            # 安全计算对数概率
            log_prob = dist.log_prob(action)
            
            # 检查对数概率健康性
            if not check_tensor_health(log_prob, "log_prob", main_logger):
                main_logger.warning("对数概率异常，使用零值")
                log_prob = torch.zeros_like(log_prob)
            
            log_prob = log_prob.sum(dim=-1)
            
            return action, log_prob
            
        except Exception as e:
            main_logger.error(f"获取动作和对数概率失败: {e}")
            # 返回安全的默认值
            batch_size = obs.shape[0]
            return torch.zeros(batch_size, self.action_dim), torch.zeros(batch_size)
    
    def evaluate_actions(self, obs, actions):
        """评估动作的对数概率和熵 - 增强数值稳定性"""
        try:
            mean, std = self.forward(obs)
            
            # 检查动作健康性
            if not check_tensor_health(actions, "input_actions", main_logger):
                main_logger.error("输入动作异常")
                actions = torch.zeros_like(actions)
            
            # 创建分布
            dist = Normal(mean, std)
            
            # 安全计算对数概率
            log_prob = dist.log_prob(actions)
            if not check_tensor_health(log_prob, "eval_log_prob", main_logger):
                main_logger.warning("评估对数概率异常，使用零值")
                log_prob = torch.zeros_like(log_prob)
            
            log_prob = log_prob.sum(dim=-1)
            
            # 安全计算熵
            entropy = dist.entropy()
            if not check_tensor_health(entropy, "entropy", main_logger):
                main_logger.warning("熵计算异常，使用零值")
                entropy = torch.zeros_like(entropy)
            
            entropy = entropy.sum(dim=-1)
            
            return log_prob, entropy
            
        except Exception as e:
            main_logger.error(f"评估动作失败: {e}")
            # 返回安全的默认值
            batch_size = obs.shape[0]
            return torch.zeros(batch_size), torch.zeros(batch_size)

class MAPPOCritic(nn.Module):
    """MAPPO Critic网络 - 使用全局状态"""
    
    def __init__(self, state_dim, hidden_size=64, activation_fn=nn.ReLU):
        super(MAPPOCritic, self).__init__()
        self.state_dim = state_dim
        
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            activation_fn(),
            nn.Linear(hidden_size, hidden_size),
            activation_fn(),
            nn.Linear(hidden_size, 1)
        )
        
    def forward(self, state):
        """前向传播"""
        return self.net(state)

class MAPPOAgent:
    """MAPPO智能体"""
    
    def __init__(self, config, log_dir, device='cpu'):
        self.config = config
        self.device = device
        self.log_dir = log_dir
        
        # 创建网络
        self.actor = MAPPOActor(
            obs_dim=config.obs_dim,
            action_dim=config.action_dim,
            hidden_size=config.hidden_size
        ).to(device)
        
        self.critic = MAPPOCritic(
            state_dim=config.state_dim,
            hidden_size=config.hidden_size
        ).to(device)
        
        # 优化器
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=config.lr_coordinator)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=config.lr_coordinator)
        
        # 经验缓冲区 - 使用ReplayBuffer
        self.buffer = ReplayBuffer(capacity=config.buffer_size)
        
        # 为每个环境创建独立的缓冲区
        self.env_buffers = {}
        
        # TensorBoard
        self.writer = SummaryWriter(log_dir)
        self.update_step = 0
        
    def select_actions(self, obs, states, deterministic=False):
        """选择动作"""
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs).to(self.device)
            
            if deterministic:
                mean, _ = self.actor(obs_tensor)
                actions = mean
                log_probs = torch.zeros(obs_tensor.shape[0])
            else:
                actions, log_probs = self.actor.get_action_and_log_prob(obs_tensor)
            
            # 计算值函数
            states_tensor = torch.FloatTensor(states).to(self.device)
            values = self.critic(states_tensor).squeeze()
            
        return actions.cpu().numpy(), log_probs.cpu().numpy(), values.cpu().numpy()
    
    def store_transition(self, obs, next_obs, states, next_states, actions, rewards, dones, log_probs, values, env_id=0):
        """存储转换到ReplayBuffer - 增强张量类型处理"""
        try:
            # 统一转换为PyTorch张量，确保正确的数据类型
            obs = torch.tensor(obs, dtype=torch.float32) if not isinstance(obs, torch.Tensor) else obs.float()
            next_obs = torch.tensor(next_obs, dtype=torch.float32) if not isinstance(next_obs, torch.Tensor) else next_obs.float()
            states = torch.tensor(states, dtype=torch.float32) if not isinstance(states, torch.Tensor) else states.float()
            next_states = torch.tensor(next_states, dtype=torch.float32) if not isinstance(next_states, torch.Tensor) else next_states.float()
            actions = torch.tensor(actions, dtype=torch.float32) if not isinstance(actions, torch.Tensor) else actions.float()
            log_probs = torch.tensor(log_probs, dtype=torch.float32) if not isinstance(log_probs, torch.Tensor) else log_probs.float()
            values = torch.tensor(values, dtype=torch.float32) if not isinstance(values, torch.Tensor) else values.float()
            
            # 特殊处理奖励张量 - 确保为标量
            if isinstance(rewards, (list, np.ndarray)):
                if hasattr(rewards, '__len__') and len(rewards) == 1:
                    rewards = float(rewards[0])
                else:
                    rewards = float(rewards)
            elif isinstance(rewards, torch.Tensor):
                rewards = rewards.float()
                if rewards.numel() == 1:
                    rewards = rewards.item()
                    rewards = torch.tensor(rewards, dtype=torch.float32)
            else:
                rewards = torch.tensor(float(rewards), dtype=torch.float32)
            
            # 特殊处理布尔张量 - 确保正确的布尔类型转换
            if isinstance(dones, (list, np.ndarray)):
                if hasattr(dones, '__len__') and len(dones) == 1:
                    dones = bool(dones[0])
                else:
                    dones = bool(dones)
                dones = torch.tensor(dones, dtype=torch.bool)
            elif isinstance(dones, torch.Tensor):
                # 如果已经是张量，确保是布尔类型
                if dones.dtype != torch.bool:
                    dones = dones.bool()
                if dones.numel() == 1:
                    dones_value = dones.item()
                    dones = torch.tensor(dones_value, dtype=torch.bool)
            else:
                # 标量值直接转换
                dones = torch.tensor(bool(dones), dtype=torch.bool)
            
            # 验证所有张量的健康性
            tensors_to_check = [
                (obs, "obs"), (next_obs, "next_obs"), (states, "states"), 
                (next_states, "next_states"), (actions, "actions"), 
                (rewards, "rewards"), (dones, "dones"), 
                (log_probs, "log_probs"), (values, "values")
            ]
            
            for tensor, name in tensors_to_check:
                if not check_tensor_health(tensor, name, main_logger):
                    main_logger.warning(f"存储转换时张量 {name} 健康检查失败，跳过存储")
                    return
            
            # 将经验存储为元组
            experience = (
                obs.clone(),
                next_obs.clone(), 
                states.clone(),
                next_states.clone(),
                actions.clone(),
                rewards.clone(),
                dones.clone(),
                log_probs.clone(),
                values.clone()
            )
            self.buffer.push(experience)
            
            main_logger.debug(f"成功存储转换: obs={obs.shape}, rewards={rewards.item() if rewards.numel() == 1 else rewards}, "
                            f"dones={dones.item() if dones.numel() == 1 else dones}")
            
        except Exception as e:
            main_logger.error(f"存储转换失败: {e}")
            main_logger.error(f"输入类型: obs={type(obs)}, rewards={type(rewards)}, dones={type(dones)}")
            raise
    
    def update(self):
        """更新网络 - 增强异常处理和监控"""
        try:
            if len(self.buffer) < self.config.buffer_size // 4:  # 当缓冲区有足够数据时再更新
                main_logger.debug(f"缓冲区数据不足: {len(self.buffer)}/{self.config.buffer_size//4}")
                return {}
            
            main_logger.debug(f"开始网络更新，缓冲区大小: {len(self.buffer)}")
            
            # 记录内存使用情况
            log_memory_usage(main_logger, self.update_step)
            
            # 从ReplayBuffer获取批次数据
            batch_size = min(len(self.buffer), self.config.buffer_size // 2)
            experiences = self.buffer.sample(batch_size)
            
            if not experiences:
                main_logger.warning("获取的经验数据为空")
                return {}
            
            main_logger.debug(f"采样批次大小: {batch_size}")
            
            # 解包经验数据
            try:
                obs_list, next_obs_list, states_list, next_states_list, actions_list, rewards_list, dones_list, log_probs_list, values_list = zip(*experiences)
            except Exception as e:
                main_logger.error(f"解包经验数据失败: {e}")
                return {}
            
            # 转换为张量并移到设备
            try:
                obs = torch.stack(obs_list).to(self.device)
                next_obs = torch.stack(next_obs_list).to(self.device)
                states = torch.stack(states_list).to(self.device)
                next_states = torch.stack(next_states_list).to(self.device)
                actions = torch.stack(actions_list).to(self.device)
                rewards = torch.stack(rewards_list).to(self.device)
                dones = torch.stack(dones_list).to(self.device)
                old_log_probs = torch.stack(log_probs_list).to(self.device)
                values = torch.stack(values_list).to(self.device)
                
                # 检查所有张量的健康性
                tensors_to_check = [
                    (obs, "obs"), (next_obs, "next_obs"), (states, "states"), 
                    (next_states, "next_states"), (actions, "actions"), 
                    (rewards, "rewards"), (dones, "dones"), 
                    (old_log_probs, "old_log_probs"), (values, "values")
                ]
                
                for tensor, name in tensors_to_check:
                    if not check_tensor_health(tensor, name, main_logger):
                        main_logger.error(f"张量 {name} 健康检查失败，跳过此次更新")
                        return {}
                        
            except Exception as e:
                main_logger.error(f"转换张量失败: {e}")
                return {}
            
            # 确保所有张量具有相同的第一维度（批次维度）
            batch_dim = obs.shape[0]
            main_logger.debug(f"批次维度: {batch_dim}")
            
            # 处理多智能体情况 - 确保维度一致性
            if obs.dim() == 3:  # [batch_size, n_agents, obs_dim]
                batch_size, n_agents, obs_dim = obs.shape
                main_logger.debug(f"多智能体模式: batch_size={batch_size}, n_agents={n_agents}, obs_dim={obs_dim}")
                
                obs = obs.reshape(batch_size * n_agents, obs_dim)
                actions = actions.reshape(batch_size * n_agents, -1)
                old_log_probs = old_log_probs.reshape(batch_size * n_agents)
                
                # 对于rewards, dones, values - 确保正确展开
                if rewards.dim() == 2:  # [batch_size, n_agents]
                    rewards = rewards.reshape(batch_size * n_agents)
                if dones.dim() == 2:  # [batch_size, n_agents]
                    dones = dones.reshape(batch_size * n_agents)
                if values.dim() == 2:  # [batch_size, n_agents]
                    values = values.reshape(batch_size * n_agents)
                
                # 对状态进行重复以匹配智能体数量
                states = states.repeat_interleave(n_agents, dim=0)
                next_states = next_states.repeat_interleave(n_agents, dim=0)
            else:
                main_logger.debug("单智能体模式")
                # 单智能体情况，确保维度正确
                if rewards.dim() > 1:
                    rewards = rewards.squeeze()
                if dones.dim() > 1:
                    dones = dones.squeeze()
                if values.dim() > 1:
                    values = values.squeeze()
                if old_log_probs.dim() > 1:
                    old_log_probs = old_log_probs.squeeze()
            
            # 计算下一状态的值函数（在维度调整后）
            try:
                with torch.no_grad():
                    next_values = self.critic(next_states).squeeze()
                    
                if not check_tensor_health(next_values, "next_values", main_logger):
                    main_logger.error("下一状态值函数计算异常")
                    return {}
                    
            except Exception as e:
                main_logger.error(f"计算下一状态值函数失败: {e}")
                return {}
            
            # 确保所有张量维度匹配（使用最小尺寸）
            min_size = min(rewards.shape[0], values.shape[0], next_values.shape[0], dones.shape[0])
            main_logger.debug(f"张量最小尺寸: {min_size}")
            
            if min_size == 0:
                main_logger.error("张量尺寸为0，跳过更新")
                return {}
                
            rewards = rewards[:min_size]
            values = values[:min_size]
            next_values = next_values[:min_size]
            dones = dones[:min_size]
            obs = obs[:min_size]
            actions = actions[:min_size]
            old_log_probs = old_log_probs[:min_size]
            states = states[:min_size]
            
            # 使用utils中的compute_gae函数
            try:
                advantages, returns = compute_gae(
                    rewards, values, next_values, dones.float(),
                    self.config.gamma, self.config.gae_lambda
                )
                
                if not check_tensor_health(advantages, "advantages", main_logger) or \
                   not check_tensor_health(returns, "returns", main_logger):
                    main_logger.error("GAE计算结果异常")
                    return {}
                    
            except Exception as e:
                main_logger.error(f"GAE计算失败: {e}")
                return {}
            
            # 安全的优势标准化
            try:
                adv_std = advantages.std()
                if adv_std < 1e-8:
                    main_logger.warning(f"优势标准差过小: {adv_std}, 跳过标准化")
                    advantages_norm = advantages
                else:
                    advantages_norm = safe_divide(
                        advantages - advantages.mean(), 
                        adv_std, 
                        epsilon=1e-8, 
                        logger=main_logger
                    )
                    
                if not check_tensor_health(advantages_norm, "advantages_norm", main_logger):
                    main_logger.warning("优势标准化异常，使用原始值")
                    advantages_norm = advantages
                    
            except Exception as e:
                main_logger.error(f"优势标准化失败: {e}")
                advantages_norm = advantages
            
            total_actor_loss = 0
            total_critic_loss = 0
            
            # PPO多轮更新
            for epoch in range(self.config.ppo_epochs):
                try:
                    # Actor更新
                    log_probs, entropy = self.actor.evaluate_actions(obs, actions)
                    
                    if not check_tensor_health(log_probs, f"log_probs_epoch_{epoch}", main_logger) or \
                       not check_tensor_health(entropy, f"entropy_epoch_{epoch}", main_logger):
                        main_logger.warning(f"Epoch {epoch}: Actor评估结果异常，跳过此轮")
                        continue
                    
                    # 安全计算比率
                    ratio_exp = log_probs - old_log_probs
                    ratio_exp = torch.clamp(ratio_exp, min=-20, max=20)  # 限制指数范围
                    ratio = safe_exp(ratio_exp, max_value=20, logger=main_logger)
                    
                    if not check_tensor_health(ratio, f"ratio_epoch_{epoch}", main_logger):
                        main_logger.warning(f"Epoch {epoch}: 比率计算异常，跳过此轮")
                        continue
                    
                    surr1 = ratio * advantages_norm
                    surr2 = torch.clamp(ratio, 1 - self.config.clip_epsilon, 1 + self.config.clip_epsilon) * advantages_norm
                    
                    actor_loss = -torch.min(surr1, surr2).mean() - self.config.entropy_coef * entropy.mean()
                    
                    if not check_tensor_health(actor_loss, f"actor_loss_epoch_{epoch}", main_logger):
                        main_logger.warning(f"Epoch {epoch}: Actor损失异常，跳过此轮")
                        continue
                    
                    # 更新Actor
                    self.actor_optimizer.zero_grad()
                    actor_loss.backward()
                    
                    # 监控梯度
                    actor_grad_norm, _ = monitor_gradients(self.actor, "Actor", main_logger)
                    
                    torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.config.max_grad_norm)
                    self.actor_optimizer.step()
                    
                    # Critic更新
                    current_values = self.critic(states).squeeze()
                    
                    if not check_tensor_health(current_values, f"current_values_epoch_{epoch}", main_logger):
                        main_logger.warning(f"Epoch {epoch}: 当前值函数异常，跳过此轮")
                        continue
                    
                    critic_loss = F.mse_loss(current_values, returns)
                    
                    if not check_tensor_health(critic_loss, f"critic_loss_epoch_{epoch}", main_logger):
                        main_logger.warning(f"Epoch {epoch}: Critic损失异常，跳过此轮")
                        continue
                    
                    # 更新Critic
                    self.critic_optimizer.zero_grad()
                    critic_loss.backward()
                    
                    # 监控梯度
                    critic_grad_norm, _ = monitor_gradients(self.critic, "Critic", main_logger)
                    
                    torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.config.max_grad_norm)
                    self.critic_optimizer.step()
                    
                    total_actor_loss += actor_loss.item()
                    total_critic_loss += critic_loss.item()
                    
                    main_logger.debug(f"Epoch {epoch}: Actor损失={actor_loss.item():.6f}, "
                                    f"Critic损失={critic_loss.item():.6f}, "
                                    f"Actor梯度范数={actor_grad_norm:.6f}, "
                                    f"Critic梯度范数={critic_grad_norm:.6f}")
                    
                except Exception as e:
                    main_logger.error(f"PPO Epoch {epoch} 更新失败: {e}")
                    main_logger.error(f"异常详情: {traceback.format_exc()}")
                    continue
            
            # 清空缓冲区
            self.buffer.clear()
            
            self.update_step += 1
            
            # 计算平均损失
            if self.config.ppo_epochs > 0:
                avg_actor_loss = total_actor_loss / self.config.ppo_epochs
                avg_critic_loss = total_critic_loss / self.config.ppo_epochs
            else:
                avg_actor_loss = 0.0
                avg_critic_loss = 0.0
            
            # 记录到TensorBoard
            try:
                self.writer.add_scalar('Training/Actor_Loss', avg_actor_loss, self.update_step)
                self.writer.add_scalar('Training/Critic_Loss', avg_critic_loss, self.update_step)
                self.writer.add_scalar('Training/Advantages_Mean', advantages.mean().item(), self.update_step)
                self.writer.add_scalar('Training/Advantages_Std', advantages.std().item(), self.update_step)
                self.writer.add_scalar('Training/Returns_Mean', returns.mean().item(), self.update_step)
                
                # 记录内存使用情况
                if torch.cuda.is_available():
                    gpu_memory = torch.cuda.memory_allocated() / (1024**3)
                    self.writer.add_scalar('Memory/GPU_Memory_GB', gpu_memory, self.update_step)
                    
            except Exception as e:
                main_logger.warning(f"记录TensorBoard数据失败: {e}")
            
            main_logger.info(f"网络更新完成: Actor损失={avg_actor_loss:.6f}, Critic损失={avg_critic_loss:.6f}")
            
            return {
                'actor_loss': avg_actor_loss,
                'critic_loss': avg_critic_loss,
                'update_step': self.update_step,
                'advantages_mean': advantages.mean().item(),
                'advantages_std': advantages.std().item(),
                'returns_mean': returns.mean().item()
            }
            
        except Exception as e:
            main_logger.error(f"网络更新发生未捕获的异常: {e}")
            main_logger.error(f"异常详情: {traceback.format_exc()}")
            
            # 尝试清理GPU内存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # 强制垃圾回收
            gc.collect()
            
            return {
                'error': str(e),
                'update_step': self.update_step
            }
    
    
    def save_model(self, path):
        """保存模型"""
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'config': self.config
        }, path)
    
    def load_model(self, path):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])


def get_device(device_pref):
    """根据偏好选择计算设备"""
    if device_pref == 'auto':
        if torch.cuda.is_available():
            main_logger.info("检测到GPU可用，使用CUDA")
            return 'cuda'
        else:
            main_logger.info("未检测到GPU，使用CPU")
            return 'cpu'
    elif device_pref == 'cuda':
        if torch.cuda.is_available():
            main_logger.info("使用CUDA")
            return 'cuda'
        else:
            main_logger.warning("请求使用CUDA但未检测到GPU，回退到CPU")
            return 'cpu'
    else:
        main_logger.info("使用CPU")
        return 'cpu'

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
    parser = argparse.ArgumentParser(description='基于MAPPO的增强训练，专注于论文数据收集')
    
    # 运行模式和环境参数
    parser.add_argument('--mode', type=str, default='train', help='运行模式: train或eval')
    parser.add_argument('--scenario', type=int, default=2, help='场景: 1=基站模式, 2=协作组网模式')
    parser.add_argument('--model_path', type=str, default='models/mappo_enhanced_tracking.pt', help='模型保存/加载路径')
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
    parser.add_argument('--num_envs', type=int, default=8, help='并行环境数量')
    
    # MAPPO超参数
    parser.add_argument('--learning_rate', type=float, default=3e-4, help='学习率')
    parser.add_argument('--gamma', type=float, default=0.99, help='折扣因子')
    parser.add_argument('--gae_lambda', type=float, default=0.95, help='GAE参数')
    parser.add_argument('--clip_epsilon', type=float, default=0.2, help='PPO裁剪参数')
    parser.add_argument('--entropy_coef', type=float, default=0.01, help='熵系数')
    parser.add_argument('--max_grad_norm', type=float, default=0.5, help='最大梯度范数')
    parser.add_argument('--ppo_epochs', type=int, default=10, help='PPO更新轮数')
    parser.add_argument('--buffer_size', type=int, default=2048, help='缓冲区大小')
    
    # 数据收集参数
    parser.add_argument('--export_interval', type=int, default=1000, 
                        help='数据导出间隔步数')
    parser.add_argument('--detailed_logging', action='store_true', 
                        help='启用详细的奖励日志记录')
    
    return parser.parse_args()

def train(config, args, device):
    """MAPPO训练函数"""
    main_logger.info("开始MAPPO训练...")

    num_envs = args.num_envs
    main_logger.info(f"使用 {num_envs} 个并行训练环境")
    
    # 创建日志目录
    log_dir = os.path.join(args.log_dir, f"mappo_enhanced_tracking_{datetime.now().strftime('%Y%m%d-%H%M%S')}")
    os.makedirs(log_dir, exist_ok=True)
    model_dir = os.path.dirname(args.model_path)
    os.makedirs(model_dir, exist_ok=True)
    
    # 创建环境
    base_seed = getattr(config, 'seed', int(time.time()))
    main_logger.info(f"基础种子: {base_seed}")

    env_fns = [make_env(
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

    # 创建环境实例
    envs = [env_fn() for env_fn in env_fns]
    
    # 获取环境信息
    sample_env = envs[0]
    n_agents = sample_env.n_uavs
    obs_dim = sample_env.obs_dim
    state_dim = sample_env.state_dim
    action_dim = sample_env.action_dim
    
    # 更新配置
    config.n_agents = n_agents
    config.obs_dim = obs_dim
    config.state_dim = state_dim
    config.action_dim = action_dim
    config.lr_coordinator = args.learning_rate
    config.gamma = args.gamma
    config.gae_lambda = args.gae_lambda
    config.clip_epsilon = args.clip_epsilon
    config.entropy_coef = args.entropy_coef
    config.max_grad_norm = args.max_grad_norm
    config.ppo_epochs = args.ppo_epochs
    config.buffer_size = args.buffer_size
    
    main_logger.info(f"环境信息: n_agents={n_agents}, obs_dim={obs_dim}, state_dim={state_dim}, action_dim={action_dim}")

    # 创建MAPPO智能体
    agent = MAPPOAgent(config, log_dir, device)
    
    # 创建增强的奖励追踪器
    reward_tracker = EnhancedRewardTracker(log_dir, config)
    reward_tracker.export_interval = args.export_interval
    
    # 初始化环境
    observations = []
    states = []
    for env in envs:
        obs, info = env.reset()
        observations.append(obs)
        states.append(info.get('state', np.zeros(state_dim)))
    
    observations = np.array(observations)
    states = np.array(states)
    
    # 训练循环
    total_steps = 0
    episode_count = 0
    env_episode_rewards = np.zeros(num_envs)
    env_episode_lengths = np.zeros(num_envs, dtype=int)
    
    start_time = time.time()
    
    # 增强的训练循环 - 添加异常处理和监控
    consecutive_errors = 0
    max_consecutive_errors = 10
    last_save_step = 0
    save_interval = 5000
    
    while total_steps < config.total_timesteps:
        try:
            # 定期记录内存使用情况
            if total_steps % 1000 == 0:
                log_memory_usage(main_logger, total_steps)
            
            # 选择动作
            try:
                actions, log_probs, values = agent.select_actions(observations, states)
                
                # 检查动作的有效性
                if not isinstance(actions, np.ndarray) or np.isnan(actions).any() or np.isinf(actions).any():
                    main_logger.error(f"步骤 {total_steps}: 动作选择异常，跳过此步")
                    consecutive_errors += 1
                    if consecutive_errors >= max_consecutive_errors:
                        main_logger.error(f"连续错误达到 {max_consecutive_errors} 次，退出训练")
                        break
                    continue
                
            except Exception as e:
                main_logger.error(f"步骤 {total_steps}: 动作选择失败: {e}")
                consecutive_errors += 1
                if consecutive_errors >= max_consecutive_errors:
                    main_logger.error(f"连续错误达到 {max_consecutive_errors} 次，退出训练")
                    break
                continue
            
            # 执行动作
            next_observations = []
            next_states = []
            rewards = []
            dones = []
            infos = []
            
            env_step_success = True
            
            for i, env in enumerate(envs):
                try:
                    next_obs, reward, terminated, truncated, info = env.step(actions[i])
                    done = bool(terminated or truncated)  # 显式转换为Python bool
                    
                    # 验证环境返回值
                    if next_obs is None or np.isnan(next_obs).any() or np.isinf(next_obs).any():
                        main_logger.warning(f"步骤 {total_steps}: 环境{i}返回异常观察值")
                        next_obs = observations[i]  # 使用上一步的观察
                    
                    if not isinstance(reward, (int, float)) or np.isnan(reward) or np.isinf(reward):
                        main_logger.warning(f"步骤 {total_steps}: 环境{i}返回异常奖励值: {reward}")
                        reward = 0.0  # 使用默认奖励
                    
                    next_observations.append(next_obs)
                    next_states.append(info.get('next_state', np.zeros(state_dim)))
                    rewards.append(reward)
                    dones.append(done)
                    infos.append(info)
                    
                except Exception as e:
                    main_logger.error(f"步骤 {total_steps}: 环境{i}步骤执行失败: {e}")
                    # 使用安全的默认值
                    next_observations.append(observations[i])
                    next_states.append(states[i])
                    rewards.append(0.0)
                    dones.append(False)
                    infos.append({})
                    env_step_success = False
            
            if not env_step_success:
                consecutive_errors += 1
                if consecutive_errors >= max_consecutive_errors:
                    main_logger.error(f"连续错误达到 {max_consecutive_errors} 次，退出训练")
                    break
                continue
            
            # 转换为数组
            try:
                next_observations = np.array(next_observations)
                next_states = np.array(next_states)
                rewards = np.array(rewards)
                dones = np.array(dones)
                
                # 验证数组健康性
                if np.isnan(next_observations).any() or np.isinf(next_observations).any():
                    main_logger.error(f"步骤 {total_steps}: next_observations包含异常值")
                    consecutive_errors += 1
                    continue
                    
                if np.isnan(rewards).any() or np.isinf(rewards).any():
                    main_logger.error(f"步骤 {total_steps}: rewards包含异常值")
                    consecutive_errors += 1
                    continue
                    
            except Exception as e:
                main_logger.error(f"步骤 {total_steps}: 数组转换失败: {e}")
                consecutive_errors += 1
                continue
            
            # 存储经验 - 需要逐个环境存储
            storage_success = True
            for i in range(num_envs):
                try:
                    # 确保数据为单个标量值，而不是数组，并显式转换布尔类型
                    env_reward = float(rewards[i]) if isinstance(rewards[i], np.ndarray) else float(rewards[i])
                    env_done = bool(dones[i])  # 显式转换为Python bool，无论原始类型
                    
                    # 验证数据有效性
                    if np.isnan(env_reward) or np.isinf(env_reward):
                        main_logger.warning(f"步骤 {total_steps}: 环境{i}奖励异常，使用0.0")
                        env_reward = 0.0
                    
                    agent.store_transition(
                        observations[i], next_observations[i], states[i], next_states[i],
                        actions[i], env_reward, env_done, log_probs[i], values[i]
                    )
                    
                except Exception as e:
                    main_logger.error(f"步骤 {total_steps}: 存储环境{i}转换失败: {e}")
                    storage_success = False
            
            if not storage_success:
                consecutive_errors += 1
                if consecutive_errors >= max_consecutive_errors:
                    main_logger.error(f"连续错误达到 {max_consecutive_errors} 次，退出训练")
                    break
                continue
            
            # 更新环境奖励和长度
            env_episode_rewards += rewards
            env_episode_lengths += 1
            
            # 处理episode结束
            for i, done in enumerate(dones):
                if done:
                    episode_count += 1
                    
                    try:
                        # 记录episode完成
                        agent_rewards = [env_episode_rewards[i]] * n_agents  # 简化处理
                        reward_tracker.log_episode_completion(
                            episode_count, i, env_episode_rewards[i], 
                            env_episode_lengths[i], agent_rewards, infos[i]
                        )
                        
                        main_logger.info(f"Episode {episode_count}: 环境{i}, 奖励={env_episode_rewards[i]:.2f}, 长度={env_episode_lengths[i]}")
                        
                        # 重置环境
                        obs, info = envs[i].reset()
                        
                        # 验证重置后的观察
                        if obs is None or np.isnan(obs).any() or np.isinf(obs).any():
                            main_logger.error(f"环境{i}重置后观察异常")
                            obs = np.zeros_like(observations[i])
                            
                        observations[i] = obs
                        states[i] = info.get('state', np.zeros(state_dim))
                        env_episode_rewards[i] = 0
                        env_episode_lengths[i] = 0
                        
                    except Exception as e:
                        main_logger.error(f"处理环境{i} episode结束时失败: {e}")
                        # 强制重置
                        try:
                            obs, info = envs[i].reset()
                            observations[i] = obs if obs is not None else np.zeros_like(observations[i])
                            states[i] = info.get('state', np.zeros(state_dim)) if info else np.zeros(state_dim)
                            env_episode_rewards[i] = 0
                            env_episode_lengths[i] = 0
                        except:
                            main_logger.error(f"环境{i}强制重置也失败")
            
            # 更新状态
            observations = next_observations
            states = next_states
            total_steps += num_envs
            
            # 重置连续错误计数器
            consecutive_errors = 0
            
            # 更新网络
            try:
                if len(agent.buffer) >= config.buffer_size // 4:
                    update_info = agent.update()
                    if update_info and 'error' not in update_info:
                        main_logger.info(f"步骤 {total_steps}: Actor损失={update_info['actor_loss']:.4f}, Critic损失={update_info['critic_loss']:.4f}")
                    elif 'error' in update_info:
                        main_logger.error(f"步骤 {total_steps}: 网络更新失败: {update_info['error']}")
                        
            except Exception as e:
                main_logger.error(f"步骤 {total_steps}: 网络更新异常: {e}")
                main_logger.error(f"异常详情: {traceback.format_exc()}")
            
            # 定期保存模型和导出数据
            try:
                # 定期导出数据
                if total_steps % reward_tracker.export_interval == 0:
                    reward_tracker.export_training_data(total_steps)
                
                # 定期保存模型
                if total_steps - last_save_step >= save_interval:
                    agent.save_model(args.model_path)
                    main_logger.info(f"步骤 {total_steps}: 模型已保存")
                    last_save_step = total_steps
                    
            except Exception as e:
                main_logger.error(f"步骤 {total_steps}: 保存模型或导出数据失败: {e}")
            
            # 定期清理内存 - 增强版本
            if total_steps % 1000 == 0:  # 更频繁的清理
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
                main_logger.debug(f"步骤 {total_steps}: 定期内存清理完成")
            
            # 深度内存清理
            if total_steps % 10000 == 0:
                if torch.cuda.is_available():
                    # 重置GPU内存统计
                    torch.cuda.reset_peak_memory_stats()
                    torch.cuda.empty_cache()
                    main_logger.info(f"步骤 {total_steps}: 深度GPU内存清理完成")
                
                # 强制Python垃圾回收
                collected = gc.collect()
                main_logger.info(f"步骤 {total_steps}: 垃圾回收清理了 {collected} 个对象")
                
        except KeyboardInterrupt:
            main_logger.info("接收到中断信号，保存当前进度并退出...")
            try:
                agent.save_model(args.model_path)
                reward_tracker.export_training_data(total_steps)
                main_logger.info("进度保存完成")
            except Exception as e:
                main_logger.error(f"保存进度失败: {e}")
            break
            
        except Exception as e:
            main_logger.error(f"步骤 {total_steps}: 训练循环发生未捕获异常: {e}")
            main_logger.error(f"异常详情: {traceback.format_exc()}")
            consecutive_errors += 1
            
            if consecutive_errors >= max_consecutive_errors:
                main_logger.error(f"连续错误达到 {max_consecutive_errors} 次，退出训练")
                break
                
            # 尝试恢复
            try:
                # 清理GPU内存
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
                
                # 重置环境状态
                for i, env in enumerate(envs):
                    try:
                        obs, info = env.reset()
                        observations[i] = obs if obs is not None else np.zeros_like(observations[i])
                        states[i] = info.get('state', np.zeros(state_dim)) if info else np.zeros(state_dim)
                        env_episode_rewards[i] = 0
                        env_episode_lengths[i] = 0
                    except Exception as reset_e:
                        main_logger.error(f"重置环境{i}失败: {reset_e}")
                        
                main_logger.info("尝试恢复训练状态")
                
            except Exception as recovery_e:
                main_logger.error(f"恢复训练状态失败: {recovery_e}")
    
    # 训练完成
    elapsed_time = time.time() - start_time
    main_logger.info(f"MAPPO训练完成! 总用时: {elapsed_time:.2f}秒")
    
    # 最终保存
    agent.save_model(args.model_path)
    reward_tracker.export_training_data(total_steps)
    
    # 生成摘要
    summary = reward_tracker.get_summary_statistics()
    main_logger.info("训练摘要:")
    for key, value in summary.items():
        main_logger.info(f"  {key}: {value}")
    
    # 保存摘要到文件
    import json
    summary_path = os.path.join(log_dir, 'training_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    main_logger.info(f"训练数据已保存到: {log_dir}")
    
    # 关闭环境
    for env in envs:
        env.close()
    
    return agent

def evaluate(agent, envs, n_episodes=10, render=False):
    """评估MAPPO模型"""
    main_logger.info(f"开始评估: {n_episodes} episodes")
    
    num_envs = len(envs)
    episode_rewards = []
    completed_episodes = 0
    
    # 重置环境
    observations = []
    states = []
    for env in envs:
        obs, info = env.reset()
        observations.append(obs)
        states.append(info.get('state', np.zeros(agent.config.state_dim)))
    
    observations = np.array(observations)
    states = np.array(states)
    env_rewards = np.zeros(num_envs)
    
    while completed_episodes < n_episodes:
        # 预测动作
        actions, _, _ = agent.select_actions(observations, states, deterministic=True)
        
        # 执行动作
        next_observations = []
        next_states = []
        rewards = []
        dones = []
        
        for i, env in enumerate(envs):
            next_obs, reward, terminated, truncated, info = env.step(actions[i])
            done = bool(terminated or truncated)  # 显式转换为Python bool
            next_observations.append(next_obs)
            next_states.append(info.get('next_state', np.zeros(agent.config.state_dim)))
            rewards.append(reward)
            dones.append(done)
        
        next_observations = np.array(next_observations)
        next_states = np.array(next_states)
        rewards = np.array(rewards)
        dones = np.array(dones)
        
        env_rewards += rewards
        
        # 检查完成的环境
        for i, done in enumerate(dones):
            if done and completed_episodes < n_episodes:
                episode_rewards.append(env_rewards[i])
                completed_episodes += 1
                main_logger.info(f"评估 Episode {completed_episodes}/{n_episodes}, 奖励: {env_rewards[i]:.2f}")
                
                # 重置环境
                obs, info = envs[i].reset()
                observations[i] = obs
                states[i] = info.get('state', np.zeros(agent.config.state_dim))
                env_rewards[i] = 0
        
        observations = next_observations
        states = next_states

    mean_reward = np.mean(episode_rewards) if episode_rewards else 0
    std_reward = np.std(episode_rewards) if episode_rewards else 0
    min_reward = np.min(episode_rewards) if episode_rewards else 0
    max_reward = np.max(episode_rewards) if episode_rewards else 0

    return mean_reward, std_reward, min_reward, max_reward

def main():
    args = parse_args()
    
    # 创建日志目录
    os.makedirs(args.log_dir, exist_ok=True)
    
    # 为训练会话创建固定的日志文件名
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = f"mappo_enhanced_tracking_{timestamp}.log"
    
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
    main_logger = get_logger("MAPPO-Enhanced")
    main_logger.info(f"基于MAPPO的增强训练启动: 文件级别={args.log_level}, 控制台级别={args.console_log_level}")
    main_logger.info(f"日志文件: {os.path.join(args.log_dir, log_file)}")
    
    # 使用配置
    config = Config()
    
    # 获取计算设备
    device = get_device(args.device)
    
    main_logger.info(f"详细日志记录: {args.detailed_logging}")
    main_logger.info(f"数据导出间隔: {args.export_interval} 步")

    if args.mode == 'train':
        agent = train(config, args, device)
        main_logger.info("MAPPO训练完成，增强的数据收集已启用")
    elif args.mode == 'eval':
        # 评估模式
        if not os.path.exists(args.model_path):
            main_logger.error(f"模型文件 {args.model_path} 不存在")
            return
        
        # 创建评估环境
        base_seed = getattr(config, 'seed', int(time.time()))
        
        eval_env_fns = [make_env(
            scenario=args.scenario,
            n_uavs=args.n_uavs,
            n_users=args.n_users,
            user_distribution=args.user_distribution,
            channel_model=args.channel_model,
            max_hops=args.max_hops if args.scenario == 2 else None,
            render_mode="human" if args.render and i == 0 else None,
            rank=i,
            seed=base_seed
        ) for i in range(4)]  # 少量环境用于评估
        
        eval_envs = [env_fn() for env_fn in eval_env_fns]
        
        # 更新配置维度
        sample_env = eval_envs[0]
        config.n_agents = sample_env.n_uavs
        config.obs_dim = sample_env.obs_dim
        config.state_dim = sample_env.state_dim
        config.action_dim = sample_env.action_dim
        
        # 创建智能体并加载模型
        log_dir = os.path.join(args.log_dir, f"mappo_eval_{datetime.now().strftime('%Y%m%d-%H%M%S')}")
        os.makedirs(log_dir, exist_ok=True)
        
        agent = MAPPOAgent(config, log_dir, device)
        agent.load_model(args.model_path)
        main_logger.info(f"已加载模型: {args.model_path}")
        
        # 进行评估
        mean_reward, std_reward, min_reward, max_reward = evaluate(
            agent, eval_envs, n_episodes=args.eval_episodes, render=args.render
        )
        
        main_logger.info(f"评估结果: 平均奖励 {mean_reward:.2f} ± {std_reward:.2f}, 最大/最小: {max_reward:.2f}/{min_reward:.2f}")
        
        for env in eval_envs:
            env.close()
    else:
        main_logger.error(f"未知的运行模式: {args.mode}")

if __name__ == "__main__":
    try:
        main()
    finally:
        try:
            shutdown_logging()
            print("日志系统已关闭")
        except Exception as e:
            print(f"关闭日志系统时出错: {e}")
