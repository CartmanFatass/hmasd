import numpy as np
import sys
import traceback

# 导入场景
from envs.pettingzoo.scenario1 import UAVBaseStationEnv
from envs.pettingzoo.scenario2 import UAVCooperativeNetworkEnv

def test_scenario1():
    print("\n=== 测试场景1：无人机基站环境 ===")
    env = UAVBaseStationEnv(n_uavs=3, n_users=20)
    state, obs = env.reset()
    
    print(f"状态维度: {state.shape}")
    print(f"观测维度: {obs.shape}")
    
    # 运行几个步骤
    for i in range(3):
        actions = np.array([[-0.1, 0.2, 0], [0.1, 0.1, -0.1], [0.2, -0.1, 0.1]])
        next_state, next_obs, reward, done, info = env.step(actions)
        print(f"步骤 {i+1}:")
        print(f"  奖励: {reward:.4f}")
        print(f"  连接用户数: {info['served_users']}/{env.n_users}")
    
    env.close()
    print("场景1测试完成!")
    return True

def test_scenario2():
    print("\n=== 测试场景2：无人机协作组网环境 ===")
    env = UAVCooperativeNetworkEnv(n_uavs=5, n_users=30, max_hops=3, n_ground_bs=1)
    state, obs = env.reset()
    
    print(f"状态维度: {state.shape}")
    print(f"观测维度: {obs.shape}")
    
    # 运行几个步骤
    for i in range(3):
        actions = np.array([[0.1, 0.1, 0], [-0.1, 0.1, -0.1], [0.2, -0.1, 0.1], [-0.2, -0.2, 0], [0, 0, 0.2]])
        next_state, next_obs, reward, done, info = env.step(actions)
        print(f"步骤 {i+1}:")
        print(f"  奖励: {reward:.4f}")
        print(f"  连接用户数: {info['served_users']}/{env.n_users}")
        print(f"  连通性: {info['connectivity_ratio']:.2%}")
        
        # 检查UAV角色
        if 'uav_roles' in info:
            roles = info['uav_roles']
            print(f"  UAV角色: 基站={sum(roles == 1)}, 中继={sum(roles == 2)}, 未分配={sum(roles == 0)}")
    
    env.close()
    print("场景2测试完成!")
    return True

if __name__ == "__main__":
    print("======= 多无人机基站环境测试 =======")
    sys.stdout.flush()
    
    # 测试场景1
    s1_success = False
    try:
        s1_success = test_scenario1()
        sys.stdout.flush()
    except Exception as e:
        print(f"场景1测试失败: {str(e)}")
        traceback.print_exc()
        sys.stdout.flush()
    
    # 测试场景2
    s2_success = False
    try:
        s2_success = test_scenario2()
        sys.stdout.flush()
    except Exception as e:
        print(f"场景2测试失败: {str(e)}")
        traceback.print_exc()
        sys.stdout.flush()
    
    # 测试结果
    print("\n======= 测试结果 =======")
    print(f"场景1: {'成功' if s1_success else '失败'}")
    print(f"场景2: {'成功' if s2_success else '失败'}")
    sys.stdout.flush()
    
    if s1_success and s2_success:
        print("\n所有测试通过!")
    else:
        print("\n部分测试失败，请查看错误信息。")
    sys.stdout.flush()
