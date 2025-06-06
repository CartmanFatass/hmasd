import os
import time
import numpy as np
import torch
import argparse
import matplotlib.pyplot as plt
from datetime import datetime

from config import Config
from hmasd.agent import HMASDAgent
from envs.pettingzoo.scenario1 import UAVBaseStationEnv
from envs.pettingzoo.scenario2 import UAVCooperativeNetworkEnv

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
            print("检测到GPU可用，使用CUDA")
            return torch.device('cuda')
        else:
            print("未检测到GPU，使用CPU")
            return torch.device('cpu')
    elif device_pref == 'cuda':
        if torch.cuda.is_available():
            print("使用CUDA")
            return torch.device('cuda')
        else:
            print("警告: 请求使用CUDA但未检测到GPU，回退到CPU")
            return torch.device('cpu')
    else:  # 'cpu'或其他值
        print("使用CPU")
        return torch.device('cpu')

# 解析命令行参数
def parse_args():
    parser = argparse.ArgumentParser(description='HMASD: 层次化多智能体技能发现')
    parser.add_argument('--mode', type=str, default='train', help='运行模式: train或eval')
    parser.add_argument('--scenario', type=int, default=1, help='场景: 1=基站模式, 2=协作组网模式')
    parser.add_argument('--model_path', type=str, default='models/hmasd_model.pt', help='模型保存/加载路径')
    parser.add_argument('--log_dir', type=str, default='logs', help='日志目录')
    parser.add_argument('--state_dim', type=int, help='全局状态维度')
    parser.add_argument('--obs_dim', type=int, help='观测维度')
    parser.add_argument('--eval_episodes', type=int, default=10, help='评估的episode数量')
    parser.add_argument('--render', action='store_true', help='是否渲染环境')
    parser.add_argument('--n_uavs', type=int, default=5, help='无人机数量')
    parser.add_argument('--n_users', type=int, default=50, help='用户数量')
    parser.add_argument('--max_hops', type=int, default=3, help='最大跳数 (仅用于场景2)')
    parser.add_argument('--user_distribution', type=str, default='uniform', 
                        choices=['uniform', 'cluster', 'hotspot'], help='用户分布类型')
    parser.add_argument('--channel_model', type=str, default='free_space',
                        choices=['free_space', 'urban', 'suburban'], help='信道模型')
    parser.add_argument('--device', type=str, default='auto', 
                        choices=['auto', 'cuda', 'cpu'], help='计算设备: auto=自动选择, cuda=GPU, cpu=CPU')
    
    return parser.parse_args()

# 训练函数
def train(env, config, args, device):
    """
    训练HMASD代理
    
    参数:
        env: 环境实例
        config: 配置对象
        args: 命令行参数
    """
    print("开始训练HMASD...")
    
    # 更新环境维度
    if args.state_dim is not None and args.obs_dim is not None:
        config.update_env_dims(args.state_dim, args.obs_dim)
    else:
        # 尝试从环境中获取
        try:
            state_dim = env.get_state_dim()
            obs_dim = env.get_obs_dim()
            config.update_env_dims(state_dim, obs_dim)
        except:
            raise ValueError("必须提供state_dim和obs_dim参数或环境必须具有get_state_dim()和get_obs_dim()方法")
            
    # 创建日志目录
    log_dir = os.path.join(args.log_dir, datetime.now().strftime("%Y%m%d-%H%M%S"))
    os.makedirs(log_dir, exist_ok=True)
    model_dir = os.path.dirname(args.model_path)
    os.makedirs(model_dir, exist_ok=True)
    
    # 创建HMASD代理
    agent = HMASDAgent(config, log_dir=log_dir, device=device)
    
    # 记录超参数
    agent.writer.add_text('Parameters/n_agents', str(config.n_agents), 0)
    agent.writer.add_text('Parameters/n_Z', str(config.n_Z), 0)
    agent.writer.add_text('Parameters/n_z', str(config.n_z), 0)
    agent.writer.add_text('Parameters/k', str(config.k), 0)
    agent.writer.add_text('Parameters/gamma', str(config.gamma), 0)
    
    # 训练变量
    total_steps = 0
    n_episodes = 0
    max_episodes = config.total_timesteps // config.buffer_size  # 估计的最大episode数量
    episode_rewards = []
    update_times = 0
    best_reward = float('-inf')
    
    # 记录训练开始时间
    start_time = time.time()
    
    # 训练循环
    while total_steps < config.total_timesteps:
        # 重置环境
        state, observations = env.reset()
        episode_reward = 0
        ep_t = 0
        done = False
        
        # 单个episode循环
        while not done:
            # 代理选择动作
            actions, info = agent.step(state, observations, ep_t)
            
            # 如果技能发生变化，记录技能分布
            if info['skill_changed']:
                agent.log_skill_distribution(
                    info['team_skill'], 
                    info['agent_skills'], 
                    episode=n_episodes
                )
            
            # 执行动作
            next_state, next_observations, reward, done, _ = env.step(actions)
            
            # 存储经验
            agent.store_transition(
                state, next_state, observations, next_observations,
                actions, reward, done, info['team_skill'], info['agent_skills'], info['action_logprobs']
            )
            
            # 更新状态
            state = next_state
            observations = next_observations
            episode_reward += reward
            ep_t += 1
            total_steps += 1
            
            # 更新网络
            if total_steps % config.buffer_size == 0:
                update_info = agent.update()
                update_times += 1
                
                # 打印训练信息
                elapsed = time.time() - start_time
                print(f"更新 {update_times}, 总步数 {total_steps}, "
                     f"高层损失 {update_info['coordinator_loss']:.4f}, "
                     f"低层损失 {update_info['discoverer_loss']:.4f}, "
                     f"判别器损失 {update_info['discriminator_loss']:.4f}, "
                     f"已用时间 {elapsed:.2f}s")
            
            # 评估
            if total_steps % config.eval_interval == 0:
                eval_reward = evaluate(env, agent, 5)
                print(f"评估: 平均奖励 {eval_reward:.2f}")
                
                # 保存最佳模型
                if eval_reward > best_reward:
                    best_reward = eval_reward
                    agent.save_model(args.model_path)
                    print(f"保存最佳模型，奖励: {best_reward:.2f}")
        
        # Episode结束后的处理
        n_episodes += 1
        episode_rewards.append(episode_reward)
        
        # 记录episode奖励到TensorBoard
        agent.training_info['episode_rewards'].append(episode_reward)
        agent.writer.add_scalar('Reward/episode_reward', episode_reward, n_episodes)
        agent.writer.add_scalar('Reward/episode_length', ep_t, n_episodes)
        
        # 如果已经有多个episodes，计算移动平均
        if len(episode_rewards) >= 10:
            recent_rewards = episode_rewards[-10:]
            avg_reward = sum(recent_rewards) / len(recent_rewards)
            agent.writer.add_scalar('Reward/avg_reward_10', avg_reward, n_episodes)
        
        print(f"Episode {n_episodes}/{max_episodes}, 奖励: {episode_reward:.2f}, 步数: {ep_t}")
        
        # 每10个episodes绘制一次奖励曲线
        if n_episodes % 10 == 0:
            plt.figure(figsize=(10, 5))
            plt.plot(episode_rewards)
            plt.title('Episode Rewards')
            plt.xlabel('Episode')
            plt.ylabel('Reward')
            plt.savefig(os.path.join(log_dir, 'rewards.png'))
            plt.close()
    
    print(f"训练完成! 总步数: {total_steps}, 总episodes: {n_episodes}")
    print(f"最佳奖励: {best_reward:.2f}")
    
    # 保存最终模型
    final_model_path = os.path.join(model_dir, 'hmasd_final.pt')
    agent.save_model(final_model_path)
    print(f"最终模型已保存到 {final_model_path}")
    
    return agent

# 评估函数
def evaluate(env, agent, n_episodes=10, render=False):
    """
    评估HMASD代理
    
    参数:
        env: 环境实例
        agent: HMASD代理实例
        n_episodes: 评估的episode数量
        render: 是否渲染环境
        
    返回:
        mean_reward: 平均奖励
    """
    episode_rewards = []
    episode_lengths = []
    eval_step = getattr(agent, 'global_step', 0)
    
    for i in range(n_episodes):
        state, observations = env.reset()
        episode_reward = 0
        ep_t = 0
        done = False
        
        while not done:
            actions, info = agent.step(state, observations, ep_t)
            
            # 如果技能发生变化，记录技能分布
            if info['skill_changed'] and hasattr(agent, 'writer'):
                agent.log_skill_distribution(
                    info['team_skill'], 
                    info['agent_skills'], 
                    episode=eval_step + i
                )
                
            next_state, next_observations, reward, done, _ = env.step(actions)
            
            if render:
                env.render()
                
            state = next_state
            observations = next_observations
            episode_reward += reward
            ep_t += 1
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(ep_t)
        print(f"评估 Episode {i+1}/{n_episodes}, 奖励: {episode_reward:.2f}, 步数: {ep_t}")
        
        # 记录到TensorBoard
        if hasattr(agent, 'writer'):
            agent.writer.add_scalar('Eval/episode_reward', episode_reward, eval_step + i)
            agent.writer.add_scalar('Eval/episode_length', ep_t, eval_step + i)
    
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    mean_length = np.mean(episode_lengths)
    
    # 记录评估统计信息
    if hasattr(agent, 'writer'):
        agent.writer.add_scalar('Eval/mean_reward', mean_reward, eval_step)
        agent.writer.add_scalar('Eval/reward_std', std_reward, eval_step)
        agent.writer.add_scalar('Eval/mean_episode_length', mean_length, eval_step)
        agent.writer.flush()  # 确保数据写入磁盘
    
    print(f"评估完成: 平均奖励 {mean_reward:.2f} ± {std_reward:.2f}, 平均步数: {mean_length:.2f}")
    
    return mean_reward

# 主函数
def main():
    args = parse_args()
    config = Config()
    
    # 获取计算设备
    device = get_device(args.device)
    
    # 根据选择的场景创建环境
    if args.scenario == 1:
        print(f"创建场景1: 无人机基站环境")
        env = UAVBaseStationEnv(
            n_uavs=args.n_uavs,
            n_users=args.n_users,
            user_distribution=args.user_distribution,
            channel_model=args.channel_model,
            render_mode="human" if args.render else None,
        )
    elif args.scenario == 2:
        print(f"创建场景2: 无人机协作组网环境")
        env = UAVCooperativeNetworkEnv(
            n_uavs=args.n_uavs,
            n_users=args.n_users,
            max_hops=args.max_hops,
            user_distribution=args.user_distribution,
            channel_model=args.channel_model,
            render_mode="human" if args.render else None,
        )
    else:
        raise ValueError(f"未知的场景: {args.scenario}")
    
    # 更新配置中的智能体数量
    config.n_agents = env.n_uavs
    
    print(f"环境已创建: n_agents={env.n_uavs}, state_dim={env.state_dim}, obs_dim={env.obs_dim}")
    
    if args.mode == 'train':
        agent = train(env, config, args, device)
    elif args.mode == 'eval':
        # 加载模型
        if not os.path.exists(args.model_path):
            print(f"模型文件 {args.model_path} 不存在")
            return
        
        # 更新环境维度
        if args.state_dim is not None and args.obs_dim is not None:
            config.update_env_dims(args.state_dim, args.obs_dim)
        else:
            # 尝试从环境中获取
            try:
                state_dim = env.get_state_dim()
                obs_dim = env.get_obs_dim()
                config.update_env_dims(state_dim, obs_dim)
            except:
                raise ValueError("必须提供state_dim和obs_dim参数或环境必须具有get_state_dim()和get_obs_dim()方法")
        
        # 创建日志目录
        log_dir = os.path.join(args.log_dir, f"eval_{datetime.now().strftime('%Y%m%d-%H%M%S')}")
        os.makedirs(log_dir, exist_ok=True)
        
        # 创建代理并加载模型
        agent = HMASDAgent(config, log_dir=log_dir, device=device)
        agent.load_model(args.model_path)
        
        # 记录模型配置
        agent.writer.add_text('Eval/model_path', args.model_path, 0)
        agent.writer.add_text('Eval/scenario', str(args.scenario), 0)
        agent.writer.add_text('Eval/n_agents', str(config.n_agents), 0)
        
        # 评估模型
        evaluate(env, agent, n_episodes=args.eval_episodes, render=args.render)
    else:
        print(f"未知的运行模式: {args.mode}")

if __name__ == "__main__":
    main()
