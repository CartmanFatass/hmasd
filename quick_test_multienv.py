#!/usr/bin/env python3
"""
快速验证多环境支持的简单测试脚本
"""

import os
import sys
import numpy as np
import torch
import tempfile
import shutil

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config_1 import Config
from hmasd.agent import HMASDAgent

def quick_test():
    """快速测试多环境支持"""
    print("🚀 开始快速验证HMASD多环境支持...")
    
    # 创建配置
    config = Config()
    config.state_dim = 8
    config.obs_dim = 6
    config.n_agents = 3
    config.action_dim = 4
    config.k = 4
    
    # 创建临时目录
    temp_dir = tempfile.mkdtemp()
    
    try:
        # 创建代理
        agent = HMASDAgent(config, log_dir=temp_dir, device=torch.device('cpu'))
        
        # 测试数据
        state = np.random.randn(config.state_dim)
        observations = np.random.randn(config.n_agents, config.obs_dim)
        
        print("✅ 代理创建成功")
        
        # 测试1: 环境状态初始化
        print("\n📋 测试1: 环境状态初始化")
        assert hasattr(agent, 'env_team_skills'), "缺少env_team_skills属性"
        assert hasattr(agent, 'env_agent_skills'), "缺少env_agent_skills属性"
        assert hasattr(agent, 'env_timers'), "缺少env_timers属性"
        print("✅ 环境状态字典已正确初始化")
        
        # 测试2: 多环境step调用
        print("\n📋 测试2: 多环境step调用")
        num_test_envs = 4
        env_results = {}
        
        for env_id in range(num_test_envs):
            actions, info = agent.step(state, observations, 0, deterministic=False, env_id=env_id)
            env_results[env_id] = {
                'actions': actions,
                'team_skill': info['team_skill'],
                'agent_skills': info['agent_skills'],
                'env_id': info['env_id']
            }
            
            # 验证返回值
            assert actions.shape == (config.n_agents, config.action_dim), f"动作形状错误: {actions.shape}"
            assert info['env_id'] == env_id, f"环境ID不匹配: {info['env_id']} != {env_id}"
            
        print(f"✅ 成功测试{num_test_envs}个环境的step调用")
        
        # 显示每个环境的技能分配
        for env_id, result in env_results.items():
            print(f"  环境{env_id}: 团队技能={result['team_skill']}, 个体技能={result['agent_skills']}")
        
        # 测试3: 技能计时器
        print("\n📋 测试3: 技能计时器管理")
        env_id = 0
        
        for step in range(config.k + 1):
            actions, info = agent.step(state, observations, step, deterministic=False, env_id=env_id)
            skill_changed = info['skill_changed']
            skill_timer = info['skill_timer']
            
            if step == 0:
                assert skill_changed, f"第一步应该分配技能"
            
            print(f"  步骤{step}: 技能变化={skill_changed}, 计时器={skill_timer}")
        
        print("✅ 技能计时器管理正确")
        
        # 测试4: 经验存储
        print("\n📋 测试4: 经验存储")
        
        # 准备存储数据
        next_state = np.random.randn(config.state_dim)
        next_observations = np.random.randn(config.n_agents, config.obs_dim)
        actions = np.random.randn(config.n_agents, config.action_dim)
        rewards = 1.0
        dones = False
        
        initial_buffer_size = len(agent.low_level_buffer)
        
        # 为多个环境存储经验
        for env_id in range(num_test_envs):
            # 获取技能信息
            _, info = agent.step(state, observations, 0, deterministic=False, env_id=env_id)
            
            # 存储转换
            agent.store_transition(
                state, next_state, observations, next_observations,
                actions, rewards, dones,
                info['team_skill'], info['agent_skills'], info['action_logprobs'],
                log_probs=info['log_probs'], skill_timer_for_env=0, env_id=env_id
            )
        
        final_buffer_size = len(agent.low_level_buffer)
        expected_increase = num_test_envs * config.n_agents
        actual_increase = final_buffer_size - initial_buffer_size
        
        assert actual_increase == expected_increase, f"经验存储数量错误: 期望{expected_increase}, 实际{actual_increase}"
        print(f"✅ 经验存储正确，增加了{actual_increase}个低层经验")
        
        # 测试5: 向后兼容性
        print("\n📋 测试5: 向后兼容性")
        
        # 不传递env_id参数
        actions1, info1 = agent.step(state, observations, 0, deterministic=False)
        # 传递env_id=0
        actions2, info2 = agent.step(state, observations, 1, deterministic=False, env_id=0)
        
        assert actions1.shape == actions2.shape, "向后兼容性测试失败"
        assert 'team_skill' in info1 and 'team_skill' in info2, "信息结构不一致"
        
        print("✅ 向后兼容性正确")
        
        print("\n🎉 所有快速测试通过！多环境支持实现正确！")
        return True
        
    except Exception as e:
        print(f"\n❌ 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # 清理
        if hasattr(agent, 'writer'):
            agent.writer.close()
        shutil.rmtree(temp_dir, ignore_errors=True)

def main():
    """主函数"""
    print("HMASD多环境支持快速验证")
    print("=" * 40)
    
    success = quick_test()
    
    if success:
        print("\n✅ 验证结论: 多环境支持实现正确！")
        print("\n📖 使用方法:")
        print("  # 在训练循环中为每个环境调用:")
        print("  actions, info = agent.step(state, obs, step, env_id=env_id)")
        print("  agent.store_transition(..., env_id=env_id)")
        print("\n🚀 现在可以开始多环境并行训练了！")
        return 0
    else:
        print("\n❌ 验证失败，需要检查实现！")
        return 1

if __name__ == "__main__":
    sys.exit(main())
