import numpy as np
import matplotlib.pyplot as plt
import argparse
import time

from envs.pettingzoo.scenario1 import UAVBaseStationEnv

def parse_args():
    parser = argparse.ArgumentParser(description='测试场景1: 无人机基站环境')
    parser.add_argument('--n_uavs', type=int, default=5, help='无人机数量')
    parser.add_argument('--n_users', type=int, default=50, help='用户数量')
    parser.add_argument('--user_distribution', type=str, default='uniform', 
                        choices=['uniform', 'cluster', 'hotspot'], help='用户分布类型')
    parser.add_argument('--channel_model', type=str, default='free_space',
                        choices=['free_space', 'urban', 'suburban'], help='信道模型')
    parser.add_argument('--steps', type=int, default=100, help='测试步数')
    parser.add_argument('--render', action='store_true', help='是否渲染环境')
    parser.add_argument('--save_fig', action='store_true', help='是否保存结果图表')
    
    return parser.parse_args()

def test_scenario1_basic():
    """基本功能测试"""
    print("==== 测试场景1: 基本功能 ====")
    env = UAVBaseStationEnv(n_uavs=3, n_users=20)
    state, obs = env.reset()
    
    print(f"状态维度: {state.shape}")
    print(f"观测维度: {obs.shape}")
    print(f"动作空间: {env.action_spaces}")
    
    # 测试单步
    actions = np.random.uniform(-1, 1, (3, 3))
    next_state, next_obs, reward, done, info = env.step(actions)
    
    print(f"奖励: {reward}")
    print(f"完成: {done}")
    print(f"信息: {info}")
    
    env.close()
    print("基本功能测试完成\n")

def test_different_distributions(render=False, save_fig=False):
    """测试不同用户分布"""
    print("==== 测试场景1: 不同用户分布 ====")
    
    distributions = ['uniform', 'cluster', 'hotspot']
    
    plt.figure(figsize=(15, 5))
    
    for i, dist in enumerate(distributions):
        print(f"测试 {dist} 分布...")
        env = UAVBaseStationEnv(
            n_uavs=5,
            n_users=50,
            user_distribution=dist,
            render_mode="human" if render else None
        )
        
        state, obs = env.reset()
        
        # 绘制用户分布
        plt.subplot(1, 3, i+1)
        plt.scatter(env.user_positions[:, 0], env.user_positions[:, 1], c='blue', alpha=0.5)
        plt.title(f'{dist} 分布')
        plt.xlabel('X (m)')
        plt.ylabel('Y (m)')
        
        # 执行几步
        for _ in range(5):
            actions = np.random.uniform(-1, 1, (5, 3))
            state, obs, reward, done, info = env.step(actions)
            if render:
                time.sleep(0.5)
        
        print(f"覆盖率: {np.sum(env.connections) / env.n_users:.2%}")
        env.close()
    
    plt.tight_layout()
    if save_fig:
        plt.savefig("scenario1_user_distributions.png")
        print("已保存用户分布图到 scenario1_user_distributions.png")
    if render:
        plt.show()
    
    print("不同用户分布测试完成\n")

def test_coverage_maximization(steps=100, render=False, save_fig=False):
    """测试覆盖率最大化"""
    print("==== 测试场景1: 覆盖率最大化 ====")
    
    env = UAVBaseStationEnv(
        n_uavs=5,
        n_users=50,
        user_distribution='cluster',
        render_mode="human" if render else None
    )
    
    state, obs = env.reset()
    
    # 记录数据
    rewards = []
    coverage_ratios = []
    quality_rewards = []
    
    # 执行策略：无人机向用户聚集区域移动
    for i in range(steps):
        # 计算每个无人机的目标位置（朝向最近的未覆盖用户群）
        actions = np.zeros((env.n_uavs, 3))
        
        for uav_idx in range(env.n_uavs):
            # 获取未连接的用户位置
            connected_users = np.any(env.connections, axis=0)
            unconnected_user_positions = env.user_positions[~connected_users]
            
            if len(unconnected_user_positions) > 0:
                # 计算到最近未连接用户群的中心向量
                uav_pos = env.uav_positions[uav_idx]
                
                # 使用简单聚类找到用户群
                if len(unconnected_user_positions) > 10:
                    # 随机选择一个未连接用户作为中心
                    center_idx = np.random.randint(len(unconnected_user_positions))
                    center = unconnected_user_positions[center_idx]
                    
                    # 找到附近的用户
                    dists = np.sqrt(np.sum((unconnected_user_positions - center) ** 2, axis=1))
                    nearby_users = unconnected_user_positions[dists < 200]
                    
                    if len(nearby_users) > 0:
                        target = np.mean(nearby_users, axis=0)
                    else:
                        target = center
                else:
                    # 用户较少时直接使用均值
                    target = np.mean(unconnected_user_positions, axis=0)
                
                # 计算方向向量
                direction = np.zeros(3)
                direction[:2] = target - uav_pos[:2]
                
                # 归一化
                norm = np.linalg.norm(direction)
                if norm > 0:
                    direction = direction / norm
                
                # 设置高度（尝试保持在中等高度）
                direction[2] = (100 - uav_pos[2]) * 0.01
                
                # 设置动作
                actions[uav_idx] = direction * 0.5  # 缩小动作幅度
            else:
                # 所有用户都已连接，保持位置
                actions[uav_idx] = np.zeros(3)
        
        # 执行动作
        state, obs, reward, done, info = env.step(actions)
        
        # 记录数据
        rewards.append(reward)
        coverage_ratio = np.sum(env.connections) / env.n_users
        coverage_ratios.append(coverage_ratio)
        
        if hasattr(env, 'reward_info'):
            quality_rewards.append(env.reward_info['quality_reward'])
        
        # 打印进度
        if i % 10 == 0 or i == steps - 1:
            print(f"步数: {i}, 奖励: {reward:.4f}, 覆盖率: {coverage_ratio:.2%}")
        
        # 渲染
        if render:
            time.sleep(0.1)
    
    # 绘制结果
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(rewards, label='总奖励')
    if len(quality_rewards) > 0:
        plt.plot(quality_rewards, label='服务质量奖励')
    plt.title('奖励随时间变化')
    plt.xlabel('步数')
    plt.ylabel('奖励')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(coverage_ratios)
    plt.title('用户覆盖率随时间变化')
    plt.xlabel('步数')
    plt.ylabel('覆盖率')
    
    plt.tight_layout()
    if save_fig:
        plt.savefig("scenario1_coverage_maximization.png")
        print("已保存覆盖率最大化结果到 scenario1_coverage_maximization.png")
    if render:
        plt.show()
    
    env.close()
    print("覆盖率最大化测试完成\n")
    
    return np.mean(rewards), np.max(coverage_ratios)

def main():
    args = parse_args()
    
    # 基本功能测试
    test_scenario1_basic()
    
    # 不同用户分布测试
    test_different_distributions(render=args.render, save_fig=args.save_fig)
    
    # 覆盖率最大化测试
    avg_reward, max_coverage = test_coverage_maximization(
        steps=args.steps, 
        render=args.render,
        save_fig=args.save_fig
    )
    
    print("\n==== 场景1测试结果汇总 ====")
    print(f"平均奖励: {avg_reward:.4f}")
    print(f"最大覆盖率: {max_coverage:.2%}")

if __name__ == "__main__":
    args = parse_args()
    main()
