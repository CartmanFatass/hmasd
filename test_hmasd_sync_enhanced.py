#!/usr/bin/env python3
"""
HMASD同步训练增强版本测试脚本
用于快速验证train_hmasd_sync_enhanced.py的基本功能
"""

import os
import sys
import time
import torch
import numpy as np
from datetime import datetime

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 全局导入所有必要的模块
try:
    from config_1 import Config
    from hmasd.agent import HMASDAgent
    from envs.pettingzoo.scenario2 import UAVCooperativeNetworkEnv
    from envs.pettingzoo.env_adapter import ParallelToArrayAdapter
    from train_hmasd_sync_enhanced import SyncEnhancedRewardTracker, get_device
    IMPORTS_SUCCESSFUL = True
    IMPORT_ERROR = None
except ImportError as e:
    IMPORTS_SUCCESSFUL = False
    IMPORT_ERROR = str(e)
    # 创建占位符，避免NameError
    Config = None
    HMASDAgent = None
    UAVCooperativeNetworkEnv = None
    ParallelToArrayAdapter = None
    evaluate_agent = None
    SyncEnhancedRewardTracker = None
    get_device = None

def test_imports():
    """测试必要的导入"""
    print("测试模块导入...")
    if IMPORTS_SUCCESSFUL:
        print("✅ 所有必要模块导入成功")
        return True
    else:
        print(f"❌ 模块导入失败: {IMPORT_ERROR}")
        return False

def test_config():
    """测试配置"""
    print("\n测试配置...")
    if not IMPORTS_SUCCESSFUL:
        print("❌ 配置测试跳过：导入失败")
        return False, None
    
    try:
        config = Config()
        print(f"✅ 配置加载成功")
        print(f"   - n_agents: {config.n_agents}")
        print(f"   - n_Z: {config.n_Z}")
        print(f"   - n_z: {config.n_z}")
        print(f"   - batch_size: {config.batch_size}")
        print(f"   - num_envs: {config.num_envs}")
        return True, config
    except Exception as e:
        print(f"❌ 配置测试失败: {e}")
        return False, None

def test_environment():
    """测试环境创建"""
    print("\n测试环境创建...")
    if not IMPORTS_SUCCESSFUL:
        print("❌ 环境测试跳过：导入失败")
        return False, None, None
    
    try:
        # 创建单个环境测试
        raw_env = UAVCooperativeNetworkEnv(
            n_uavs=3,
            n_users=10,
            max_hops=2,
            user_distribution='uniform',
            channel_model='3gpp-36777',
            render_mode=None,
            seed=42
        )
        
        env = ParallelToArrayAdapter(raw_env, seed=42)
        obs, info = env.reset()
        
        print(f"✅ 环境创建成功")
        print(f"   - 观测形状: {obs.shape}")
        print(f"   - 状态维度: {len(info.get('state', []))}")
        print(f"   - 智能体数量: {obs.shape[0] if len(obs.shape) > 1 else 1}")
        
        # 测试一步环境交互
        action = np.random.uniform(-1, 1, size=obs.shape)
        next_obs, reward, done, truncated, next_info = env.step(action)
        print(f"   - 环境步进测试成功，奖励: {reward:.4f}")
        
        env.close()
        return True, len(info.get('state', [])), obs.shape[-1] if len(obs.shape) > 1 else len(obs)
    except Exception as e:
        print(f"❌ 环境测试失败: {e}")
        return False, None, None

def test_reward_tracker():
    """测试奖励追踪器"""
    print("\n测试奖励追踪器...")
    if not IMPORTS_SUCCESSFUL:
        print("❌ 奖励追踪器测试跳过：导入失败")
        return False
    
    try:
        config = Config()
        
        # 创建临时日志目录
        log_dir = f"test_logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(log_dir, exist_ok=True)
        
        tracker = SyncEnhancedRewardTracker(log_dir, config)
        
        # 测试基本功能
        tracker.log_training_step(1, 0, 10.5, None, {'reward_info': {'connected_users': 5}})
        tracker.log_skill_usage(1, 0, [0, 1, 2], True)
        tracker.log_episode_completion(1, 0, 100.0, 50, {'env_component': 95.0})
        tracker.log_sync_update(1, 1024, 15.2, 2.1)
        
        print("✅ 奖励追踪器测试成功")
        print(f"   - 总步数: {tracker.training_rewards['total_steps']}")
        print(f"   - 完成episodes: {tracker.training_rewards['episodes_completed']}")
        print(f"   - 同步更新: {tracker.sync_training_metrics['sync_updates']}")
        
        # 清理测试目录
        import shutil
        shutil.rmtree(log_dir, ignore_errors=True)
        
        return True
    except Exception as e:
        print(f"❌ 奖励追踪器测试失败: {e}")
        return False

def test_agent_creation():
    """测试Agent创建"""
    print("\n测试Agent创建...")
    if not IMPORTS_SUCCESSFUL:
        print("❌ Agent创建测试跳过：导入失败")
        return False
    
    try:
        config = Config()
        
        # 更新环境维度
        config.update_env_dims(20, 15)  # 使用测试维度
        
        # 创建临时日志目录
        log_dir = f"test_agent_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(log_dir, exist_ok=True)
        
        device = get_device('cpu')  # 使用CPU进行测试
        agent = HMASDAgent(config, log_dir=log_dir, device=device)
        
        # 测试同步模式设置
        agent.sync_mode = True
        agent.sync_batch_size = config.batch_size
        
        print("✅ Agent创建成功")
        print(f"   - 设备: {device}")
        print(f"   - 同步模式: {agent.sync_mode}")
        print(f"   - 同步batch大小: {agent.sync_batch_size}")
        print(f"   - 策略版本: {agent.policy_version}")
        
        # 清理
        import shutil
        shutil.rmtree(log_dir, ignore_errors=True)
        
        return True
    except Exception as e:
        print(f"❌ Agent创建测试失败: {e}")
        return False

def test_sync_mechanism():
    """测试同步机制"""
    print("\n测试同步机制...")
    if not IMPORTS_SUCCESSFUL:
        print("❌ 同步机制测试跳过：导入失败")
        return False
    
    try:
        config = Config()
        config.update_env_dims(20, 15)
        
        log_dir = f"test_sync_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(log_dir, exist_ok=True)
        
        device = get_device('cpu')
        agent = HMASDAgent(config, log_dir=log_dir, device=device)
        
        # 启用同步模式
        agent.sync_mode = True
        agent.sync_batch_size = 8  # 小批次用于测试
        
        print(f"   - 初始样本数: {agent.samples_collected_this_round}")
        print(f"   - 应该同步更新: {agent.should_sync_update()}")
        
        # 模拟存储一些样本
        dummy_state = np.zeros(config.state_dim)
        dummy_obs = np.zeros((config.n_agents, config.obs_dim))
        dummy_action = np.zeros((config.n_agents, config.action_dim))
        
        for i in range(10):
            stored = agent.store_transition(
                dummy_state, dummy_state, dummy_obs, dummy_obs,
                dummy_action, 1.0, False, 0, [0, 1, 2], 
                np.zeros((config.n_agents, config.action_dim)),
                log_probs={'team_skill_log_prob': 0.0, 'agent_skills_log_probs': np.zeros(3)},
                skill_timer_for_env=0, env_id=0
            )
            if stored:
                print(f"   - 存储样本 {i+1}, 当前计数: {agent.samples_collected_this_round}")
                
            if agent.should_sync_update():
                print(f"   - 达到同步点，执行同步更新")
                break
        
        print("✅ 同步机制测试成功")
        
        # 清理
        import shutil
        shutil.rmtree(log_dir, ignore_errors=True)
        
        return True
    except Exception as e:
        print(f"❌ 同步机制测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("=" * 60)
    print("    HMASD同步训练增强版本 - 功能测试")
    print("=" * 60)
    
    # 运行所有测试
    tests = [
        ("模块导入", test_imports),
        ("配置加载", lambda: test_config()[0]),
        ("环境创建", lambda: test_environment()[0]),
        ("奖励追踪器", test_reward_tracker),
        ("Agent创建", test_agent_creation),
        ("同步机制", test_sync_mechanism),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            if test_func():
                passed += 1
            else:
                print(f"❌ {test_name} 测试失败")
        except Exception as e:
            print(f"❌ {test_name} 测试异常: {e}")
    
    # 测试结果
    print("\n" + "=" * 60)
    print("                    测试结果")
    print("=" * 60)
    print(f"通过: {passed}/{total}")
    
    if passed == total:
        print("🎉 所有测试通过！train_hmasd_sync_enhanced.py 准备就绪")
        print("\n推荐使用方法:")
        print("  chmod +x start_hmasd_sync_enhanced.sh")
        print("  ./start_hmasd_sync_enhanced.sh")
    else:
        print("⚠️  部分测试失败，请检查环境配置")
        
    print("=" * 60)

if __name__ == "__main__":
    main()
