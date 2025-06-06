import numpy as np
import matplotlib.pyplot as plt
import argparse
import time
from sklearn.cluster import KMeans

# 导入中文字体配置
from fix_matplotlib_font import configure_chinese_font

# 配置matplotlib支持中文
configure_chinese_font()

from envs.pettingzoo.scenario2 import UAVCooperativeNetworkEnv

def parse_args():
    parser = argparse.ArgumentParser(description='测试场景2: 无人机协作组网环境')
    parser.add_argument('--n_uavs', type=int, default=8, help='无人机数量')
    parser.add_argument('--n_users', type=int, default=100, help='用户数量')
    parser.add_argument('--max_hops', type=int, default=3, help='最大跳数')
    parser.add_argument('--user_distribution', type=str, default='hotspot', 
                        choices=['uniform', 'cluster', 'hotspot'], help='用户分布类型')
    parser.add_argument('--channel_model', type=str, default='free_space',
                        choices=['free_space', 'urban', 'suburban'], help='信道模型')
    parser.add_argument('--n_ground_bs', type=int, default=1, help='地面基站数量')
    parser.add_argument('--steps', type=int, default=100, help='测试步数')
    parser.add_argument('--render', action='store_true', help='是否渲染环境')
    parser.add_argument('--save_fig', action='store_true', help='是否保存结果图表')
    
    return parser.parse_args()

def test_scenario2_basic():
    """基本功能测试"""
    print("==== 测试场景2: 基本功能 ====")
    env = UAVCooperativeNetworkEnv(n_uavs=5, n_users=50)
    state, obs = env.reset()
    
    print(f"状态维度: {state.shape}")
    print(f"观测维度: {obs.shape}")
    print(f"动作空间: {env._action_spaces}")
    
    # 测试单步
    actions = np.random.uniform(-1, 1, (5, 3))
    next_state, next_obs, reward, done, info = env.step(actions)
    
    print(f"奖励: {reward}")
    print(f"完成: {done}")
    print(f"信息字段: {list(info.keys())}")
    print(f"UAV角色: {info['uav_roles']}")
    print(f"覆盖率: {info['coverage_ratio']:.2%}")
    print(f"连通性: {info['connectivity_ratio']:.2%}")
    
    env.close()
    print("基本功能测试完成\n")

def test_role_assignment(steps=50, render=False, save_fig=False):
    """测试UAV角色分配"""
    print("==== 测试场景2: UAV角色分配 ====")
    
    env = UAVCooperativeNetworkEnv(
        n_uavs=8,
        n_users=100,
        user_distribution='hotspot',
        render_mode="human" if render else None
    )
    
    state, obs = env.reset()
    
    # 记录角色数据
    timesteps = []
    base_station_counts = []
    relay_counts = []
    unassigned_counts = []
    coverage_ratios = []
    connectivity_ratios = []
    
    # 执行随机策略
    for i in range(steps):
        # 随机动作
        actions = np.random.uniform(-1, 1, (env.n_uavs, 3))
        
        # 执行动作
        next_state, next_obs, reward, done, info = env.step(actions)
        
        # 记录数据
        timesteps.append(i)
        roles = info['uav_roles']
        base_station_counts.append(np.sum(roles == 1))
        relay_counts.append(np.sum(roles == 2))
        unassigned_counts.append(np.sum(roles == 0))
        coverage_ratios.append(info['coverage_ratio'])
        connectivity_ratios.append(info['connectivity_ratio'])
        
        # 打印信息
        if i % 10 == 0 or i == steps - 1:
            print(f"步数: {i}, 基站数: {base_station_counts[-1]}, 中继数: {relay_counts[-1]}, " +
                  f"未分配: {unassigned_counts[-1]}, 连通性: {connectivity_ratios[-1]:.2%}")
        
        # 渲染
        if render:
            time.sleep(0.1)
    
    # 绘制角色分配变化
    plt.figure(figsize=(12, 10))
    
    plt.subplot(2, 2, 1)
    plt.stackplot(timesteps, 
                 base_station_counts, 
                 relay_counts, 
                 unassigned_counts,
                 labels=['基站', '中继', '未分配'],
                 colors=['red', 'orange', 'gray'],
                 alpha=0.7)
    plt.title('UAV角色分配随时间变化')
    plt.xlabel('步数')
    plt.ylabel('UAV数量')
    plt.legend(loc='upper right')
    
    plt.subplot(2, 2, 2)
    plt.plot(timesteps, coverage_ratios, label='覆盖率')
    plt.plot(timesteps, connectivity_ratios, label='连通性')
    plt.title('覆盖率与连通性随时间变化')
    plt.xlabel('步数')
    plt.ylabel('比率')
    plt.legend()
    
    # 绘制最终状态的UAV位置和角色
    plt.subplot(2, 2, 3)
    roles = info['uav_roles']
    colors = np.array(['gray', 'red', 'orange'])[roles]
    
    # 绘制无人机位置
    plt.scatter(env.uav_positions[:, 0], env.uav_positions[:, 1], c=colors, s=100, marker='^')
    
    # 绘制用户位置
    plt.scatter(env.user_positions[:, 0], env.user_positions[:, 1], c='blue', s=10, alpha=0.5)
    
    # 绘制地面基站
    plt.scatter(env.ground_bs_positions[:, 0], env.ground_bs_positions[:, 1], c='black', s=150, marker='s')
    
    # 添加连接线
    for i in range(env.n_uavs):
        for j in range(i+1, env.n_uavs):
            if env.uav_connections[i, j]:
                plt.plot([env.uav_positions[i, 0], env.uav_positions[j, 0]],
                         [env.uav_positions[i, 1], env.uav_positions[j, 1]],
                         'y-', alpha=0.5)
        
        # UAV到地面基站的连接
        for j in range(env.n_ground_bs):
            if env.uav_bs_connections[i, j]:
                plt.plot([env.uav_positions[i, 0], env.ground_bs_positions[j, 0]],
                         [env.uav_positions[i, 1], env.ground_bs_positions[j, 1]],
                         'c-', alpha=0.7, linewidth=2)
    
    plt.title('UAV位置和角色')
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    
    # 添加图例
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='^', color='w', markerfacecolor='red', markersize=10, label='基站UAV'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='orange', markersize=10, label='中继UAV'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='gray', markersize=10, label='未分配UAV'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=6, label='用户'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='black', markersize=10, label='地面基站'),
        Line2D([0], [0], color='y', lw=2, alpha=0.5, label='UAV-UAV连接'),
        Line2D([0], [0], color='c', lw=2, alpha=0.7, label='UAV-BS连接')
    ]
    plt.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.3, 1.0))
    
    # 绘制奖励组成
    plt.subplot(2, 2, 4)
    if 'reward_info' in info and len(info['reward_info']) > 0:
        reward_info = info['reward_info']
        reward_keys = list(reward_info.keys())
        reward_values = list(reward_info.values())
        
        plt.bar(range(len(reward_keys)), reward_values, tick_label=reward_keys)
        plt.title('奖励组成')
        plt.xticks(rotation=45)
        plt.ylabel('奖励值')
    
    plt.tight_layout()
    if save_fig:
        plt.savefig("scenario2_role_assignment.png")
        print("已保存角色分配结果到 scenario2_role_assignment.png")
    if render:
        plt.show()
    
    env.close()
    print("UAV角色分配测试完成\n")
    
    return base_station_counts[-1], relay_counts[-1]

def test_routing_policy(steps=100, render=False, save_fig=False):
    """测试路由策略"""
    print("==== 测试场景2: 路由策略 ====")
    
    env = UAVCooperativeNetworkEnv(
        n_uavs=8,
        n_users=80,
        max_hops=3,
        user_distribution='hotspot',
        render_mode="human" if render else None
    )
    
    state, obs = env.reset()
    
    # 记录数据
    rewards = []
    coverage_ratios = []
    connectivity_ratios = []
    hop_counts = []
    
    # 执行策略：一部分无人机作为中继围绕地面基站，一部分无人机作为基站覆盖用户
    for i in range(steps):
        # 计算每个无人机的目标位置
        actions = np.zeros((env.n_uavs, 3))
        
        for uav_idx in range(env.n_uavs):
            uav_pos = env.uav_positions[uav_idx]
            
            # 获取当前UAV的角色
            current_roles = env.uav_roles if hasattr(env, "uav_roles") and len(env.uav_roles) > 0 else np.zeros(env.n_uavs)
            
            if uav_idx < env.n_uavs // 3:
                # 前1/3的UAV作为中继围绕地面基站
                # 计算到最近地面基站的方向
                bs_dists = []
                for j in range(env.n_ground_bs):
                    bs_pos = env.ground_bs_positions[j]
                    dist = np.sqrt(np.sum((uav_pos[:2] - bs_pos[:2]) ** 2))
                    bs_dists.append((j, dist))
                
                # 选择最近的基站
                nearest_bs_idx, nearest_bs_dist = min(bs_dists, key=lambda x: x[1])
                nearest_bs_pos = env.ground_bs_positions[nearest_bs_idx]
                
                # 计算目标位置：距离基站200米的圆周上
                if nearest_bs_dist > 0:
                    # 计算当前方向向量
                    direction = (uav_pos[:2] - nearest_bs_pos[:2]) / nearest_bs_dist
                    
                    # 稍微旋转以形成围绕基站的轨迹
                    angle = 0.2  # 旋转角度
                    rotation = np.array([
                        [np.cos(angle), -np.sin(angle)],
                        [np.sin(angle), np.cos(angle)]
                    ])
                    direction = rotation.dot(direction)
                    
                    # 目标距离
                    target_dist = 200
                    
                    # 计算目标位置
                    target_pos = nearest_bs_pos[:2] + direction * target_dist
                    
                    # 计算方向向量
                    move_direction = target_pos - uav_pos[:2]
                    dist = np.linalg.norm(move_direction)
                    if dist > 0:
                        move_direction = move_direction / dist
                    
                    # 设置动作
                    actions[uav_idx, :2] = move_direction * 0.5
                    
                    # 保持合适的高度
                    actions[uav_idx, 2] = (80 - uav_pos[2]) * 0.02
            else:
                # 其余UAV作为基站覆盖用户
                connected_users = np.any(env.connections, axis=0)
                unconnected_user_positions = env.user_positions[~connected_users]
                
                if len(unconnected_user_positions) > 0:
                    # 找到最近的未连接用户群
                    if len(unconnected_user_positions) > 10:
                        # 聚类用户
                        
                        try:
                            n_clusters = min(5, len(unconnected_user_positions) // 10 + 1)
                            kmeans = KMeans(n_clusters=n_clusters)
                            kmeans.fit(unconnected_user_positions)
                            clusters = kmeans.cluster_centers_
                            
                            # 找到最近的群
                            cluster_dists = np.array([np.linalg.norm(uav_pos[:2] - cluster) for cluster in clusters])
                            nearest_cluster_idx = np.argmin(cluster_dists)
                            target = clusters[nearest_cluster_idx]
                        except:
                            # 如果KMeans失败，使用简单方法
                            target = np.mean(unconnected_user_positions, axis=0)
                    else:
                        # 用户较少时，直接计算均值
                        target = np.mean(unconnected_user_positions, axis=0)
                    
                    # 计算方向向量
                    direction = np.zeros(3)
                    direction[:2] = target - uav_pos[:2]
                    
                    # 归一化
                    norm = np.linalg.norm(direction[:2])
                    if norm > 0:
                        direction[:2] = direction[:2] / norm
                    
                    # 设置适当的高度
                    if current_roles[uav_idx] == 1:  # 如果当前是基站
                        target_height = 100
                    else:
                        target_height = 120
                    
                    direction[2] = (target_height - uav_pos[2]) * 0.01
                    
                    # 设置动作
                    actions[uav_idx] = direction * 0.5
                else:
                    # 所有用户都已连接，保持位置或稍作移动以优化连接质量
                    actions[uav_idx] = np.random.uniform(-0.1, 0.1, 3)
        
        # 执行动作
        state, obs, reward, done, info = env.step(actions)
        
        # 记录数据
        rewards.append(reward)
        coverage_ratios.append(info['coverage_ratio'])
        connectivity_ratios.append(info['connectivity_ratio'])
        
        # 计算平均跳数
        total_hops = 0
        paths = info.get('routing_paths', {})
        if paths:
            for path in paths.values():
                total_hops += len(path)
            avg_hops = total_hops / max(len(paths), 1)
            hop_counts.append(avg_hops)
        else:
            hop_counts.append(0)
        
        # 打印进度
        if i % 10 == 0 or i == steps - 1:
            print(f"步数: {i}, 奖励: {reward:.4f}, 覆盖率: {coverage_ratios[-1]:.2%}, " +
                  f"连通性: {connectivity_ratios[-1]:.2%}")
        
        # 渲染
        if render:
            time.sleep(0.1)
    
    # 绘制结果
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 2, 1)
    plt.plot(rewards)
    plt.title('奖励随时间变化')
    plt.xlabel('步数')
    plt.ylabel('奖励')
    
    plt.subplot(2, 2, 2)
    plt.plot(coverage_ratios, label='覆盖率')
    plt.plot(connectivity_ratios, label='连通性')
    plt.title('覆盖率与连通性随时间变化')
    plt.xlabel('步数')
    plt.ylabel('比率')
    plt.legend()
    
    plt.subplot(2, 2, 3)
    if hop_counts:
        plt.plot(hop_counts)
        plt.title('平均跳数随时间变化')
        plt.xlabel('步数')
        plt.ylabel('平均跳数')
    
    plt.subplot(2, 2, 4)
    # 绘制最终状态下的路由路径
    # 2D平面图
    plt.scatter(env.user_positions[:, 0], env.user_positions[:, 1], c='blue', s=10, alpha=0.3, label='用户')
    plt.scatter(env.ground_bs_positions[:, 0], env.ground_bs_positions[:, 1], c='black', s=150, marker='s', label='地面基站')
    
    roles = info['uav_roles']
    role_colors = {0: 'gray', 1: 'red', 2: 'orange'}
    
    # 绘制无人机及其角色
    for i, role in enumerate(roles):
        plt.scatter(env.uav_positions[i, 0], env.uav_positions[i, 1], 
                   c=role_colors[role], s=100, marker='^', label=f'UAV {i}')
    
    # 绘制UAV之间的连接
    for i in range(env.n_uavs):
        for j in range(i+1, env.n_uavs):
            if env.uav_connections[i, j]:
                plt.plot([env.uav_positions[i, 0], env.uav_positions[j, 0]],
                         [env.uav_positions[i, 1], env.uav_positions[j, 1]],
                         'y-', alpha=0.5)
        
        # UAV到地面基站的连接
        for j in range(env.n_ground_bs):
            if env.uav_bs_connections[i, j]:
                plt.plot([env.uav_positions[i, 0], env.ground_bs_positions[j, 0]],
                         [env.uav_positions[i, 1], env.ground_bs_positions[j, 1]],
                         'c-', alpha=0.7, linewidth=2)
        
        # UAV到用户的连接
        for j in range(env.n_users):
            if env.connections[i, j]:
                plt.plot([env.uav_positions[i, 0], env.user_positions[j, 0]],
                         [env.uav_positions[i, 1], env.user_positions[j, 1]],
                         'g-', alpha=0.1)
    
    plt.title('最终网络拓扑')
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    
    # 添加自定义图例
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='^', color='w', markerfacecolor='red', markersize=10, label='基站UAV'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='orange', markersize=10, label='中继UAV'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='gray', markersize=10, label='未分配UAV'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=6, label='用户'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='black', markersize=10, label='地面基站'),
        Line2D([0], [0], color='y', lw=2, alpha=0.5, label='UAV-UAV连接'),
        Line2D([0], [0], color='c', lw=2, alpha=0.7, label='UAV-BS连接')
    ]
    
    plt.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    if save_fig:
        plt.savefig("scenario2_routing_policy.png")
        print("已保存路由策略结果到 scenario2_routing_policy.png")
    if render:
        plt.show()
    
    env.close()
    print("路由策略测试完成\n")
    
    return np.mean(rewards), np.mean(connectivity_ratios)

def test_max_hops_impact(render=False, save_fig=False):
    """测试最大跳数的影响"""
    print("==== 测试场景2: 最大跳数的影响 ====")
    
    max_hops_values = [2, 3, 4, 5]
    results = []
    
    for max_hops in max_hops_values:
        print(f"测试最大跳数 = {max_hops}")
        env = UAVCooperativeNetworkEnv(
            n_uavs=8,
            n_users=80,
            max_hops=max_hops,
            user_distribution='hotspot',
            render_mode="human" if render else None
        )
        
        # 重置环境
        state, obs = env.reset()
        
        # 记录数据
        rewards = []
        coverage_ratios = []
        connectivity_ratios = []
        avg_hops_list = []
        
        # 运行随机策略
        for i in range(30):
            # 随机动作
            actions = np.random.uniform(-1, 1, (env.n_uavs, 3))
            
            # 执行动作
            state, obs, reward, done, info = env.step(actions)
            
            # 记录数据
            rewards.append(reward)
            coverage_ratios.append(info['coverage_ratio'])
            connectivity_ratios.append(info['connectivity_ratio'])
            
            # 计算平均跳数
            total_hops = 0
            paths = info.get('routing_paths', {})
            if paths:
                for path in paths.values():
                    total_hops += len(path)
                avg_hops = total_hops / max(len(paths), 1)
                avg_hops_list.append(avg_hops)
            
            # 渲染
            if render:
                time.sleep(0.1)
        
        # 记录结果
        results.append({
            'max_hops': max_hops,
            'avg_reward': np.mean(rewards),
            'avg_coverage': np.mean(coverage_ratios),
            'avg_connectivity': np.mean(connectivity_ratios),
            'avg_hops': np.mean(avg_hops_list) if avg_hops_list else 0
        })
        
        env.close()
    
    # 绘制结果对比
    plt.figure(figsize=(15, 5))
    
    # 奖励对比
    plt.subplot(1, 3, 1)
    plt.plot([r['max_hops'] for r in results], [r['avg_reward'] for r in results], 'o-')
    plt.title('最大跳数对奖励的影响')
    plt.xlabel('最大跳数')
    plt.ylabel('平均奖励')
    
    # 连通性对比
    plt.subplot(1, 3, 2)
    plt.plot([r['max_hops'] for r in results], [r['avg_connectivity'] for r in results], 'o-')
    plt.title('最大跳数对连通性的影响')
    plt.xlabel('最大跳数')
    plt.ylabel('平均连通性')
    
    # 实际平均跳数对比
    plt.subplot(1, 3, 3)
    plt.plot([r['max_hops'] for r in results], [r['avg_hops'] for r in results], 'o-')
    plt.title('最大跳数对实际平均跳数的影响')
    plt.xlabel('最大跳数')
    plt.ylabel('实际平均跳数')
    
    plt.tight_layout()
    if save_fig:
        plt.savefig("scenario2_max_hops_impact.png")
        print("已保存最大跳数影响结果到 scenario2_max_hops_impact.png")
    if render:
        plt.show()
    
    print("最大跳数影响测试完成\n")
    
    return results

def main():
    args = parse_args()
    
    # 基本功能测试
    test_scenario2_basic()
    
    # UAV角色分配测试
    base_stations, relays = test_role_assignment(
        steps=min(args.steps, 50),
        render=args.render,
        save_fig=args.save_fig
    )
    
    # 路由策略测试
    avg_reward, avg_connectivity = test_routing_policy(
        steps=args.steps,
        render=args.render,
        save_fig=args.save_fig
    )
    
    # 最大跳数影响测试
    if args.save_fig:
        hop_results = test_max_hops_impact(render=args.render, save_fig=args.save_fig)
    
    print("\n==== 场景2测试结果汇总 ====")
    print(f"最终基站数量: {base_stations}")
    print(f"最终中继数量: {relays}")
    print(f"路由策略平均奖励: {avg_reward:.4f}")
    print(f"路由策略平均连通性: {avg_connectivity:.2%}")

if __name__ == "__main__":
    args = parse_args()
    main()
