import numpy as np
import matplotlib.pyplot as plt
import argparse

from envs.pettingzoo.scenario1 import UAVBaseStationEnv
from envs.pettingzoo.scenario2 import UAVCooperativeNetworkEnv

def parse_args():
    parser = argparse.ArgumentParser(description='测试多无人机基站环境')
    parser.add_argument('--scenario', type=int, default=1, help='场景: 1=基站模式, 2=协作组网模式')
    parser.add_argument('--n_uavs', type=int, default=5, help='无人机数量')
    parser.add_argument('--n_users', type=int, default=50, help='用户数量')
    parser.add_argument('--max_hops', type=int, default=3, help='最大跳数 (仅用于场景2)')
    parser.add_argument('--user_distribution', type=str, default='uniform', 
                        choices=['uniform', 'cluster', 'hotspot'], help='用户分布类型')
    parser.add_argument('--channel_model', type=str, default='free_space',
                        choices=['free_space', 'urban', 'suburban'], help='信道模型')
    parser.add_argument('--steps', type=int, default=100, help='测试步数')
    parser.add_argument('--render', action='store_true', help='是否渲染环境')
    
    return parser.parse_args()

def random_policy(env, steps=100, render=False):
    """
    使用随机策略测试环境
    
    参数:
        env: 环境实例
        steps: 测试步数
        render: 是否渲染环境
    """
    print("使用随机策略测试环境...")
    
    # 重置环境
    state, observations = env.reset()
    print(f"状态维度: {state.shape}")
    print(f"观测维度: {observations.shape}")
    
    # 记录奖励
    rewards = []
    coverage_ratios = []
    
    # 运行随机策略
    for i in range(steps):
        # 随机动作
        actions = np.random.uniform(-1, 1, (env.n_uavs, 3))
        
        # 执行动作
        next_state, next_observations, reward, done, info = env.step(actions)
        
        # 记录奖励和覆盖率
        rewards.append(reward)
        coverage_ratio = np.sum(env.connections) / env.n_users
        coverage_ratios.append(coverage_ratio)
        
        # 打印信息
        if i % 10 == 0:
            print(f"步数: {i}, 奖励: {reward:.4f}, 覆盖率: {coverage_ratio:.2%}")
            if 'connectivity_ratio' in info:
                print(f"连通性: {info['connectivity_ratio']:.2%}")
        
        # 渲染
        if render:
            env.render()
        
        # 更新状态
        state = next_state
        observations = next_observations
        
        # 检查是否结束
        if done:
            print("环境结束")
            break
    
    # 绘制奖励和覆盖率曲线
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(rewards)
    plt.title('奖励')
    plt.xlabel('步数')
    plt.ylabel('奖励')
    
    plt.subplot(1, 2, 2)
    plt.plot(coverage_ratios)
    plt.title('用户覆盖率')
    plt.xlabel('步数')
    plt.ylabel('覆盖率')
    
    plt.tight_layout()
    plt.savefig(f'test_scenario{args.scenario}.png')
    print(f"结果已保存到 test_scenario{args.scenario}.png")
    
    if render:
        plt.show()
    
    return rewards, coverage_ratios

def main():
    args = parse_args()
    
    # 创建环境
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
    
    print(f"环境已创建: n_uavs={env.n_uavs}, n_users={env.n_users}")
    print(f"状态维度: {env.state_dim}, 观测维度: {env.obs_dim}")
    
    # 测试环境
    rewards, coverage_ratios = random_policy(env, steps=args.steps, render=args.render)
    
    # 关闭环境
    env.close()
    
    # 打印统计信息
    print("\n测试结果:")
    print(f"平均奖励: {np.mean(rewards):.4f}")
    print(f"平均覆盖率: {np.mean(coverage_ratios):.2%}")
    print(f"最大覆盖率: {np.max(coverage_ratios):.2%}")

if __name__ == "__main__":
    args = parse_args()
    main()
