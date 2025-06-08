import os
import sys
import argparse
import logging
from datetime import datetime

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from logger import init_multiproc_logging, get_logger, shutdown_logging, LOG_LEVELS

def test_ppo_imports():
    """测试PPO训练脚本的导入"""
    print("测试导入...")
    
    try:
        # 测试基础导入
        import torch
        import numpy as np
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import SubprocVecEnv
        print("✓ 基础库导入成功")
        
        # 测试项目导入
        from config_1 import Config
        from envs.pettingzoo.scenario1 import UAVBaseStationEnv
        from envs.pettingzoo.scenario2 import UAVCooperativeNetworkEnv
        from envs.pettingzoo.env_adapter import ParallelToArrayAdapter
        print("✓ 项目模块导入成功")
        
        # 测试PPO训练脚本导入
        from train_ppo_enhanced_tracking import (
            EnhancedRewardTracker, 
            CustomActorCriticPolicy,
            TrainingCallback,
            make_env,
            get_device
        )
        print("✓ PPO训练脚本导入成功")
        
        return True
        
    except ImportError as e:
        print(f"✗ 导入失败: {e}")
        return False

def test_environment_creation():
    """测试环境创建"""
    print("\n测试环境创建...")
    
    try:
        from config_1 import Config
        from train_ppo_enhanced_tracking import make_env
        
        config = Config()
        
        # 测试环境创建函数
        env_fn = make_env(
            scenario=2,
            n_uavs=3,
            n_users=10,
            user_distribution='uniform',
            channel_model='3gpp-36777',
            max_hops=3,
            render_mode=None,
            rank=0,
            seed=42
        )
        
        # 创建环境实例
        env = env_fn()
        print(f"✓ 环境创建成功: 类型 {type(env)}")
        
        # 测试环境重置
        obs, info = env.reset()
        print(f"✓ 环境重置成功: 观测形状 {obs.shape}")
        
        # 测试随机动作
        action_space = env.action_space
        random_action = action_space.sample()
        next_obs, reward, terminated, truncated, info = env.step(random_action)
        print(f"✓ 环境步骤成功: 奖励 {reward:.3f}")
        
        env.close()
        print("✓ 环境关闭成功")
        
        return True
        
    except Exception as e:
        print(f"✗ 环境测试失败: {e}")
        return False

def test_ppo_model_creation():
    """测试PPO模型创建"""
    print("\n测试PPO模型创建...")
    
    try:
        import torch
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv
        from train_ppo_enhanced_tracking import CustomActorCriticPolicy, make_env
        
        # 创建单个环境用于测试
        env_fn = make_env(
            scenario=1,
            n_uavs=3,
            n_users=10,
            user_distribution='uniform',
            channel_model='3gpp-36777',
            max_hops=None,
            render_mode=None,
            rank=0,
            seed=42
        )
        
        # 包装为向量化环境
        vec_env = DummyVecEnv([env_fn])
        print(f"✓ 向量化环境创建成功")
        
        # 创建PPO模型
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = PPO(
            CustomActorCriticPolicy,
            vec_env,
            learning_rate=3e-4,
            n_steps=64,  # 小值用于测试
            batch_size=32,
            n_epochs=2,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            verbose=0,
            device=device
        )
        
        print(f"✓ PPO模型创建成功，设备: {device}")
        print(f"✓ 策略网络: {type(model.policy)}")
        
        # 测试模型预测
        obs = vec_env.reset()
        actions, _ = model.predict(obs, deterministic=True)
        print(f"✓ 模型预测成功: 动作形状 {actions.shape}")
        
        vec_env.close()
        return True
        
    except Exception as e:
        print(f"✗ PPO模型测试失败: {e}")
        return False

def test_reward_tracker():
    """测试奖励追踪器"""
    print("\n测试奖励追踪器...")
    
    try:
        import tempfile
        from train_ppo_enhanced_tracking import EnhancedRewardTracker
        from config_1 import Config
        
        config = Config()
        
        # 创建临时目录
        with tempfile.TemporaryDirectory() as temp_dir:
            tracker = EnhancedRewardTracker(temp_dir, config)
            print("✓ 奖励追踪器创建成功")
            
            # 测试记录训练步骤
            tracker.log_training_step(
                step=1,
                env_id=0,
                reward=10.5,
                info={'served_users': 8, 'total_users': 10}
            )
            print("✓ 训练步骤记录成功")
            
            # 测试记录episode完成
            tracker.log_episode_completion(
                episode_num=1,
                env_id=0,
                total_reward=100.0,
                episode_length=50,
                info={'coverage_ratio': 0.8}
            )
            print("✓ Episode完成记录成功")
            
            # 测试获取摘要统计
            summary = tracker.get_summary_statistics()
            print(f"✓ 摘要统计获取成功: {len(summary)} 项指标")
            
        return True
        
    except Exception as e:
        print(f"✗ 奖励追踪器测试失败: {e}")
        return False

def test_training_callback():
    """测试训练回调函数"""
    print("\n测试训练回调函数...")
    
    try:
        import tempfile
        from train_ppo_enhanced_tracking import TrainingCallback, EnhancedRewardTracker
        from config_1 import Config
        
        config = Config()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            tracker = EnhancedRewardTracker(temp_dir, config)
            callback = TrainingCallback(
                reward_tracker=tracker,
                eval_freq=100,
                verbose=0
            )
            print("✓ 训练回调函数创建成功")
            
            # 模拟回调调用
            result = callback._on_step()
            print(f"✓ 回调函数执行成功: 返回 {result}")
            
        return True
        
    except Exception as e:
        print(f"✗ 训练回调函数测试失败: {e}")
        return False

def run_quick_training_test():
    """运行快速训练测试"""
    print("\n运行快速训练测试...")
    
    try:
        import tempfile
        import torch
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv
        from train_ppo_enhanced_tracking import (
            CustomActorCriticPolicy,
            EnhancedRewardTracker,
            TrainingCallback,
            make_env
        )
        from config_1 import Config
        
        config = Config()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # 创建简单环境
            env_fn = make_env(
                scenario=1,
                n_uavs=2,
                n_users=5,
                user_distribution='uniform',
                channel_model='3gpp-36777',
                max_hops=None,
                render_mode=None,
                rank=0,
                seed=42
            )
            
            vec_env = DummyVecEnv([env_fn])
            
            # 创建奖励追踪器和回调
            tracker = EnhancedRewardTracker(temp_dir, config)
            callback = TrainingCallback(tracker, eval_freq=50, verbose=0)
            
            # 创建PPO模型
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            model = PPO(
                CustomActorCriticPolicy,
                vec_env,
                learning_rate=3e-4,
                n_steps=32,  # 小值用于快速测试
                batch_size=16,
                n_epochs=2,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
                ent_coef=0.01,
                vf_coef=0.5,
                max_grad_norm=0.5,
                verbose=0,
                device=device
            )
            
            print("✓ 测试环境和模型设置完成")
            
            # 运行短期训练
            model.learn(
                total_timesteps=128,  # 很小的步数用于测试
                callback=callback,
                progress_bar=False
            )
            
            print("✓ 快速训练测试成功完成")
            
            # 测试保存和加载
            model_path = os.path.join(temp_dir, "test_model.zip")
            model.save(model_path)
            print("✓ 模型保存成功")
            
            loaded_model = PPO.load(model_path, device=device)
            print("✓ 模型加载成功")
            
            vec_env.close()
        
        return True
        
    except Exception as e:
        print(f"✗ 快速训练测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("=" * 60)
    print("PPO训练脚本测试")
    print("=" * 60)
    
    # 初始化日志系统
    init_multiproc_logging(
        log_dir="logs",
        log_file=f"test_ppo_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
        file_level=logging.INFO,
        console_level=logging.WARNING
    )
    
    logger = get_logger("PPO-Test")
    logger.info("开始PPO训练脚本测试")
    
    tests = [
        ("导入测试", test_ppo_imports),
        ("环境创建测试", test_environment_creation),
        ("PPO模型创建测试", test_ppo_model_creation),
        ("奖励追踪器测试", test_reward_tracker),
        ("训练回调函数测试", test_training_callback),
        ("快速训练测试", run_quick_training_test)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            success = test_func()
            results.append((test_name, success))
            if success:
                logger.info(f"{test_name}: 通过")
            else:
                logger.error(f"{test_name}: 失败")
        except Exception as e:
            print(f"✗ {test_name} 出现异常: {e}")
            results.append((test_name, False))
            logger.error(f"{test_name}: 异常 - {e}")
    
    # 总结结果
    print("\n" + "=" * 60)
    print("测试结果总结:")
    print("=" * 60)
    
    passed = 0
    failed = 0
    
    for test_name, success in results:
        status = "✓ 通过" if success else "✗ 失败"
        print(f"{test_name:.<40} {status}")
        if success:
            passed += 1
        else:
            failed += 1
    
    print(f"\n总计: {passed} 通过, {failed} 失败")
    
    if failed == 0:
        print("\n🎉 所有测试都通过了！PPO训练脚本准备就绪。")
        logger.info("所有测试通过")
    else:
        print(f"\n⚠️  有 {failed} 个测试失败，请检查相关问题。")
        logger.warning(f"有 {failed} 个测试失败")
    
    shutdown_logging()
    return failed == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
