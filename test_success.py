import sys
import numpy as np
from envs.pettingzoo.scenario1 import UAVBaseStationEnv
from envs.pettingzoo.scenario2 import UAVCooperativeNetworkEnv

def test_scenario1():
    print("=== 测试场景1开始 ===")
    sys.stdout.flush()
    
    try:
        env = UAVBaseStationEnv(n_uavs=3, n_users=20)
        state, obs = env.reset()
        
        print(f"状态维度: {state.shape}")
        print(f"观测维度: {obs.shape}")
        sys.stdout.flush()
        
        # 测试几个步骤
        for i in range(3):
            actions = np.random.uniform(-1, 1, (3, 3))
            next_state, next_obs, reward, done, info = env.step(actions)
            print(f"步骤 {i}: 奖励={reward:.4f}, 已连接用户数={info['served_users']}")
            sys.stdout.flush()
        
        env.close()
        print("场景1测试成功!")
        sys.stdout.flush()
        return True
    except Exception as e:
        print(f"场景1测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
        return False

def test_scenario2():
    print("=== 测试场景2开始 ===")
    sys.stdout.flush()
    
    try:
        env = UAVCooperativeNetworkEnv(n_uavs=5, n_users=30, max_hops=3)
        state, obs = env.reset()
        
        print(f"状态维度: {state.shape}")
        print(f"观测维度: {obs.shape}")
        sys.stdout.flush()
        
        # 测试几个步骤
        for i in range(3):
            actions = np.random.uniform(-1, 1, (5, 3))
            next_state, next_obs, reward, done, info = env.step(actions)
            print(f"步骤 {i}: 奖励={reward:.4f}, 覆盖率={info['coverage_ratio']:.2%}, 连通性={info['connectivity_ratio']:.2%}")
            roles = info['uav_roles']
            base_stations = np.sum(roles == 1)
            relays = np.sum(roles == 2)
            print(f"UAV角色: 基站={base_stations}, 中继={relays}, 未分配={np.sum(roles == 0)}")
            sys.stdout.flush()
        
        env.close()
        print("场景2测试成功!")
        sys.stdout.flush()
        return True
    except Exception as e:
        print(f"场景2测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
        return False

if __name__ == "__main__":
    print("开始环境测试...")
    sys.stdout.flush()
    
    scenario1_success = test_scenario1()
    scenario2_success = test_scenario2()
    
    if scenario1_success and scenario2_success:
        print("\n所有测试均通过！环境已成功设置。")
    else:
        print("\n有测试未通过，请查看上面的错误信息。")
    sys.stdout.flush()
