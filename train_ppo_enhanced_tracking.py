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

# 导入 Stable Baselines3
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.utils import set_random_seed
import torch.nn as nn

# 导入配置和环境
from config_1 import Config
from envs.pettingzoo.scenario1 import UAVBaseStationEnv
from envs.pettingzoo.scenario2 import UAVCooperativeNetworkEnv
from envs.pettingzoo.env_adapter import ParallelToArrayAdapter

class EnhancedRewardTracker:
    """增强的奖励追踪器，用于PPO训练数据收集"""
    
    def __init__(self, log_dir, config):
        self.log_dir = log_dir
        self.config = config
        
        # 训练过程中的奖励数据收集
        self.training_rewards = {
            'episode_rewards': [],
            'step_rewards': [],
            'env_rewards': [],
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
            'avg_throughput_per_user': []
        }
        
        # 滑动窗口统计
        self.window_size = 100
        self.recent_rewards = deque(maxlen=self.window_size)
        self.recent_lengths = deque(maxlen=self.window_size)
        
        # 数据导出设置
        self.export_interval = 1000
        self.last_export_step = 0
        
    def log_training_step(self, step, env_id, reward, info=None):
        """记录训练步骤的奖励信息"""
        self.training_rewards['total_steps'] += 1
        self.training_rewards['step_rewards'].append({
            'step': step,
            'env_id': env_id,
            'reward': reward,
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
            plt.title('PPO Training Reward Progress')
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
            if len(episodes) == len([r['episode_length'] for r in self.training_rewards['episode_rewards']]):
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
            plt.savefig(os.path.join(export_dir, f'ppo_training_progress_step_{step}.png'), dpi=300, bbox_inches='tight')
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
        
        return summary

class CustomActorCriticPolicy(ActorCriticPolicy):
    """自定义的Actor-Critic策略网络"""
    
    def __init__(self, observation_space, action_space, lr_schedule, *args, **kwargs):
        # 设置网络架构
        kwargs['net_arch'] = dict(
            pi=[64, 64],  # Actor网络层
            vf=[64, 64]   # Critic网络层
        )
        kwargs['activation_fn'] = nn.ReLU
        
        super(CustomActorCriticPolicy, self).__init__(
            observation_space, action_space, lr_schedule, *args, **kwargs
        )

class TrainingCallback(BaseCallback):
    """自定义训练回调函数"""
    
    def __init__(self, reward_tracker, eval_freq=10000, verbose=0):
        super(TrainingCallback, self).__init__(verbose)
        self.reward_tracker = reward_tracker
        self.eval_freq = eval_freq
        self.n_calls = 0
        
    def _on_step(self) -> bool:
        self.n_calls += 1
        
        # 定期导出数据
        if self.n_calls % self.eval_freq == 0:
            self.reward_tracker.export_training_data(self.n_calls)
        
        return True

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
        env = Monitor(env)  # 添加Monitor包装器用于统计
        return env

    return _init

def parse_args():
    parser = argparse.ArgumentParser(description='基于PPO的增强训练，专注于论文数据收集')
    
    # 运行模式和环境参数
    parser.add_argument('--mode', type=str, default='train', help='运行模式: train或eval')
    parser.add_argument('--scenario', type=int, default=2, help='场景: 1=基站模式, 2=协作组网模式')
    parser.add_argument('--model_path', type=str, default='models/ppo_enhanced_tracking.zip', help='模型保存/加载路径')
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
    
    # PPO超参数
    parser.add_argument('--learning_rate', type=float, default=3e-4, help='学习率')
    parser.add_argument('--gamma', type=float, default=0.99, help='折扣因子')
    parser.add_argument('--gae_lambda', type=float, default=0.95, help='GAE参数')
    parser.add_argument('--clip_range', type=float, default=0.2, help='PPO裁剪参数')
    parser.add_argument('--ent_coef', type=float, default=0.01, help='熵系数')
    parser.add_argument('--vf_coef', type=float, default=0.5, help='值函数损失系数')
    parser.add_argument('--max_grad_norm', type=float, default=0.5, help='最大梯度范数')
    parser.add_argument('--n_steps', type=int, default=2048, help='每次更新收集的步数')
    parser.add_argument('--batch_size', type=int, default=64, help='小批量大小')
    parser.add_argument('--n_epochs', type=int, default=10, help='每次更新的优化轮数')
    
    # 数据收集参数
    parser.add_argument('--export_interval', type=int, default=1000, 
                        help='数据导出间隔步数')
    parser.add_argument('--detailed_logging', action='store_true', 
                        help='启用详细的奖励日志记录')
    
    return parser.parse_args()

def train(config, args, device):
    """PPO训练函数"""
    main_logger.info("开始PPO训练...")

    # 确定并行环境数量
    num_envs = args.num_envs if args.num_envs > 0 else config.num_envs
    eval_rollout_threads = args.eval_rollout_threads if args.eval_rollout_threads > 0 else config.eval_rollout_threads
    
    main_logger.info(f"使用 {num_envs} 个并行训练环境和 {eval_rollout_threads} 个并行评估环境")
    
    # 创建日志目录
    log_dir = os.path.join(args.log_dir, f"ppo_enhanced_tracking_{datetime.now().strftime('%Y%m%d-%H%M%S')}")
    os.makedirs(log_dir, exist_ok=True)
    model_dir = os.path.dirname(args.model_path)
    os.makedirs(model_dir, exist_ok=True)
    
    # 创建增强的奖励追踪器
    reward_tracker = EnhancedRewardTracker(log_dir, config)
    reward_tracker.export_interval = args.export_interval
    
    # 创建环境
    base_seed = getattr(config, 'seed', int(time.time()))
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
    
    # 归一化环境（可选）
    # train_vec_env = VecNormalize(train_vec_env, norm_obs=True, norm_reward=True)
    # eval_vec_env = VecNormalize(eval_vec_env, norm_obs=True, norm_reward=False, training=False)
    
    main_logger.info("SubprocVecEnv 已创建。")

    # 设置随机种子
    set_random_seed(base_seed, using_cuda=(device == 'cuda'))

    # 创建PPO模型
    model = PPO(
        CustomActorCriticPolicy,
        train_vec_env,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        clip_range_vf=None,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        max_grad_norm=args.max_grad_norm,
        use_sde=False,
        sde_sample_freq=-1,
        target_kl=None,
        tensorboard_log=log_dir,
        policy_kwargs=None,
        verbose=1,
        seed=base_seed,
        device=device,
        _init_setup_model=True
    )

    main_logger.info(f"PPO模型已创建，设备: {device}")
    main_logger.info(f"模型参数: 学习率={args.learning_rate}, 折扣因子={args.gamma}, 裁剪参数={args.clip_range}")

    # 创建回调函数
    training_callback = TrainingCallback(
        reward_tracker=reward_tracker,
        eval_freq=args.export_interval,
        verbose=1
    )
    
    eval_callback = EvalCallback(
        eval_vec_env,
        best_model_save_path=model_dir,
        log_path=log_dir,
        eval_freq=args.export_interval,
        deterministic=True,
        render=False,
        verbose=1
    )

    # 开始训练
    start_time = time.time()
    total_timesteps = config.total_timesteps
    
    main_logger.info(f"开始训练，总时间步数: {total_timesteps}")
    
    model.learn(
        total_timesteps=total_timesteps,
        callback=[training_callback, eval_callback],
        tb_log_name="ppo_enhanced_tracking",
        reset_num_timesteps=True,
        progress_bar=True
    )

    elapsed_time = time.time() - start_time
    main_logger.info(f"训练完成! 总用时: {elapsed_time:.2f}秒")

    # 保存最终模型
    final_model_path = args.model_path
    model.save(final_model_path)
    main_logger.info(f"最终模型已保存到 {final_model_path}")
    
    # 训练完成后的最终数据导出
    reward_tracker.export_training_data(total_timesteps)
    
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
    
    main_logger.info(f"训练数据已保存到: {log_dir}")
    
    # 关闭环境
    train_vec_env.close()
    eval_vec_env.close()
    
    return model

def evaluate(model, eval_vec_env, n_episodes=10, render=False):
    """评估PPO模型"""
    main_logger.info(f"开始评估: {n_episodes} episodes")
    
    num_envs = eval_vec_env.num_envs
    episode_rewards = []
    completed_episodes = 0
    
    # 重置环境
    obs = eval_vec_env.reset()
    env_rewards = np.zeros(num_envs)
    
    while completed_episodes < n_episodes:
        # 预测动作
        actions, _ = model.predict(obs, deterministic=True)
        
        # 执行动作
        obs, rewards, terminated, truncated, infos = eval_vec_env.step(actions)
        dones = terminated | truncated  # 合并terminated和truncated为dones
        env_rewards += rewards
        
        # 检查完成的环境
        for i, done in enumerate(dones):
            if done and completed_episodes < n_episodes:
                episode_rewards.append(env_rewards[i])
                completed_episodes += 1
                main_logger.info(f"评估 Episode {completed_episodes}/{n_episodes}, 奖励: {env_rewards[i]:.2f}")
                env_rewards[i] = 0

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
    log_file = f"ppo_enhanced_tracking_{timestamp}.log"
    
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
    main_logger = get_logger("PPO-Enhanced")
    main_logger.info(f"基于PPO的增强训练启动: 文件级别={args.log_level}, 控制台级别={args.console_log_level}")
    main_logger.info(f"日志文件: {os.path.join(args.log_dir, log_file)}")
    
    # 使用配置
    config = Config()
    
    # 获取计算设备
    device = get_device(args.device)
    
    main_logger.info(f"详细日志记录: {args.detailed_logging}")
    main_logger.info(f"数据导出间隔: {args.export_interval} 步")

    if args.mode == 'train':
        model = train(config, args, device)
        main_logger.info("PPO训练完成，增强的数据收集已启用")
    elif args.mode == 'eval':
        # 评估模式
        if not os.path.exists(args.model_path):
            main_logger.error(f"模型文件 {args.model_path} 不存在")
            return
        
        # 创建评估环境
        base_seed = getattr(config, 'seed', int(time.time()))
        eval_rollout_threads = args.eval_rollout_threads if args.eval_rollout_threads > 0 else config.eval_rollout_threads
        
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
        ) for i in range(eval_rollout_threads)]
        
        eval_vec_env = SubprocVecEnv(eval_env_fns, start_method='spawn')
        
        # 加载模型
        model = PPO.load(args.model_path, device=device)
        main_logger.info(f"已加载模型: {args.model_path}")
        
        # 进行评估
        mean_reward, std_reward, min_reward, max_reward = evaluate(
            model, eval_vec_env, n_episodes=args.eval_episodes, render=args.render
        )
        
        main_logger.info(f"评估结果: 平均奖励 {mean_reward:.2f} ± {std_reward:.2f}, 最大/最小: {max_reward:.2f}/{min_reward:.2f}")
        
        eval_vec_env.close()
    else:
        main_logger.error(f"未知的运行模式: {args.mode}")

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
