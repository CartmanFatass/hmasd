import sys
import os
import numpy as np
from datetime import datetime
import io

# 导入场景
from envs.pettingzoo.scenario1 import UAVBaseStationEnv
from envs.pettingzoo.scenario2 import UAVCooperativeNetworkEnv

# 使用UTF-8编码创建日志文件
log_file = f"test_log_fixed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
log_stream = io.StringIO()

# 定义一个用于同时输出到控制台和日志的函数
def log_print(message):
    print(message)
    print(message, file=log_stream)
    sys.stdout.flush()

def test_scenario1():
    log_print("\n=== 测试场景1：无人机基站环境 ===")
    
    try:
        env = UAVBaseStationEnv(n_uavs=3, n_users=20, channel_model="3gpp-36777")
        state, obs = env.reset()
        
        log_print(f"状态维度: {state.shape}")
        log_print(f"观测维度: {obs.shape}")
        
        # 运行几个步骤
        for i in range(3):
            actions = np.array([[-0.1, 0.2, 0], [0.1, 0.1, -0.1], [0.2, -0.1, 0.1]])
            next_state, next_obs, reward, done, info = env.step(actions)
            log_print(f"步骤 {i+1}:")
            log_print(f"  奖励: {reward:.4f}")
            log_print(f"  连接用户数: {info['served_users']}/{env.n_users}")
        
        env.close()
        log_print("场景1测试成功!")
        return True
    except Exception as e:
        import traceback
        log_print(f"场景1测试失败: {str(e)}")
        log_print(traceback.format_exc())
        return False

def test_scenario2():
    log_print("\n=== 测试场景2：无人机协作组网环境 ===")
    
    try:
        env = UAVCooperativeNetworkEnv(n_uavs=5, n_users=30, max_hops=3, n_ground_bs=1, channel_model="3gpp-36777")
        state, obs = env.reset()
        
        log_print(f"状态维度: {state.shape}")
        log_print(f"观测维度: {obs.shape}")
        
        # 运行几个步骤
        for i in range(3):
            actions = np.array([[0.1, 0.1, 0], [-0.1, 0.1, -0.1], [0.2, -0.1, 0.1], [-0.2, -0.2, 0], [0, 0, 0.2]])
            next_state, next_obs, reward, done, info = env.step(actions)
            log_print(f"步骤 {i+1}:")
            log_print(f"  奖励: {reward:.4f}")
            log_print(f"  连接用户数: {info['served_users']}/{env.n_users}")
            log_print(f"  连通性: {info['connectivity_ratio']:.2%}")
            
            # 检查UAV角色
            if 'uav_roles' in info:
                roles = info['uav_roles']
                log_print(f"  UAV角色: 基站={sum(roles == 1)}, 中继={sum(roles == 2)}, 未分配={sum(roles == 0)}")
        
        env.close()
        log_print("场景2测试成功!")
        return True
    except Exception as e:
        import traceback
        log_print(f"场景2测试失败: {str(e)}")
        log_print(traceback.format_exc())
        return False

if __name__ == "__main__":
    log_print("======= 多无人机基站环境测试 =======")
    
    # 测试场景1
    s1_success = test_scenario1()
    
    # 测试场景2
    s2_success = test_scenario2()
    
    # 测试结果
    log_print("\n======= 测试结果 =======")
    log_print(f"场景1: {'成功' if s1_success else '失败'}")
    log_print(f"场景2: {'成功' if s2_success else '失败'}")
    
    if s1_success and s2_success:
        log_print("\n所有测试通过!")
    else:
        log_print("\n部分测试失败，请查看错误信息。")
    
    # 保存日志到文件
    with open(log_file, 'w', encoding='utf-8') as f:
        f.write(log_stream.getvalue())
    
    print(f"\n日志已保存到 {log_file}")
