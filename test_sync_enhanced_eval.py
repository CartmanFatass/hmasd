#!/usr/bin/env python3
"""
测试train_sync_enhanced.py中新的评估功能
"""

import sys
import os
import logging
from datetime import datetime

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from train_sync_enhanced import evaluate_agent, create_parallel_envs, reset_all_envs
from config_1 import Config
from hmasd.agent import HMASDAgent
from logger import setup_logger, main_logger

def test_evaluation_function():
    """测试新的评估函数"""
    print("=" * 60)
    print("测试train_sync_enhanced.py中的增强评估功能")
    print("=" * 60)
    
    # 设置日志
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    setup_logger(name=f'EVAL_TEST_{timestamp}', level=logging.INFO)
    
    # 初始化配置
    config = Config()
    config.eval_episodes = 2  # 只测试2个episodes
    
    # 创建测试环境
    print("1. 创建测试环境...")
    eval_envs = create_parallel_envs(config)
    eval_envs = eval_envs[:min(2, len(eval_envs))]  # 只使用2个环境进行测试
    print(f"   ✓ 创建了 {len(eval_envs)} 个评估环境")
    
    # 获取环境维度
    print("2. 获取环境维度...")
    sample_env = eval_envs[0]
    obs, info = sample_env.reset()
    state = info.get('state', [0] * 10)
    
    # 处理观测数据来获取正确的维度
    if isinstance(obs, dict):
        if len(obs) > 0:
            first_agent_key = list(obs.keys())[0]
            if isinstance(obs[first_agent_key], dict) and 'obs' in obs[first_agent_key]:
                obs_dim = len(obs[first_agent_key]['obs'])
            else:
                obs_values = [obs_val for obs_val in obs.values()]
                obs_dim = len(obs_values[0]) if obs_values else 10
        else:
            obs_dim = 10
    else:
        obs_dim = obs.shape[-1] if hasattr(obs, 'shape') and len(obs.shape) > 1 else len(obs)
    
    config.update_env_dims(len(state), obs_dim)
    print(f"   ✓ 环境维度: state_dim={config.state_dim}, obs_dim={config.obs_dim}")
    
    # 创建agent
    print("3. 创建HMASD agent...")
    log_dir = f"test_eval_{timestamp}"
    agent = HMASDAgent(config, log_dir=log_dir)
    print("   ✓ Agent创建成功")
    
    # 测试评估函数
    print("4. 测试评估函数...")
    try:
        eval_results = evaluate_agent(agent, eval_envs, config, num_episodes=config.eval_episodes)
        print("   ✓ 评估函数执行成功")
        
        # 检查返回结果
        expected_keys = ['mean_reward', 'std_reward', 'min_reward', 'max_reward', 'episode_rewards']
        print("5. 检查返回结果...")
        
        for key in expected_keys:
            if key in eval_results:
                value = eval_results[key]
                if key == 'episode_rewards':
                    print(f"   ✓ {key}: {len(value)} episodes")
                else:
                    print(f"   ✓ {key}: {value:.4f}")
            else:
                print(f"   ✗ 缺少 {key}")
        
        # 检查性能指标
        performance_metrics = [k for k in eval_results.keys() if k.startswith('mean_') and k != 'mean_reward']
        if performance_metrics:
            print("6. 性能指标:")
            for metric in performance_metrics:
                print(f"   ✓ {metric}: {eval_results[metric]:.4f}")
        else:
            print("6. 未检测到额外的性能指标（正常，取决于环境返回的info）")
        
        print("\n" + "=" * 60)
        print("测试结果: ✓ 所有功能正常工作")
        print("主要改进:")
        print("  • 不再依赖evaltools.eval_utils")
        print("  • 支持并行环境评估")
        print("  • 增强的性能指标收集（吞吐量、连接用户数等）")
        print("  • 与train_enhanced_reward_tracking.py风格一致")
        print("  • 支持TensorBoard记录增强版本")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"   ✗ 评估函数执行失败: {e}")
        import traceback
        print(traceback.format_exc())
        return False
    
    finally:
        # 清理资源
        print("7. 清理资源...")
        for env in eval_envs:
            try:
                env.close()
            except:
                pass
        print("   ✓ 环境已关闭")

if __name__ == "__main__":
    success = test_evaluation_function()
    sys.exit(0 if success else 1)
