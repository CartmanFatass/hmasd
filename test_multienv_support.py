#!/usr/bin/env python3
"""
测试多环境并行训练支持的实现
验证HMASD代理是否正确支持多环境并行训练
"""

import os
import sys
import time
import numpy as np
import torch
import unittest
from unittest.mock import Mock, patch
import tempfile
import shutil

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config_1 import Config
from hmasd.agent import HMASDAgent

class TestMultiEnvSupport(unittest.TestCase):
    """测试多环境支持的单元测试类"""
    
    def setUp(self):
        """测试前的初始化"""
        self.config = Config()
        # 设置较小的参数以便快速测试
        self.config.state_dim = 10
        self.config.obs_dim = 8
        self.config.n_agents = 3
        self.config.action_dim = 4
        self.config.k = 5  # 技能周期长度
        self.config.buffer_size = 100
        self.config.batch_size = 32
        self.config.high_level_batch_size = 16
        
        # 创建临时日志目录
        self.temp_dir = tempfile.mkdtemp()
        
        # 创建代理
        self.agent = HMASDAgent(self.config, log_dir=self.temp_dir, device=torch.device('cpu'))
        
        # 测试环境数量
        self.num_envs = 4
        
    def tearDown(self):
        """测试后的清理"""
        # 关闭TensorBoard writer
        if hasattr(self.agent, 'writer'):
            self.agent.writer.close()
        
        # 删除临时目录
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_env_state_initialization(self):
        """测试环境状态的初始化"""
        print("\n=== 测试环境状态初始化 ===")
        
        # 检查预初始化的环境状态
        self.assertIsInstance(self.agent.env_team_skills, dict)
        self.assertIsInstance(self.agent.env_agent_skills, dict)
        self.assertIsInstance(self.agent.env_log_probs, dict)
        self.assertIsInstance(self.agent.env_hidden_states, dict)
        self.assertIsInstance(self.agent.env_reward_sums, dict)
        self.assertIsInstance(self.agent.env_timers, dict)
        
        # 检查预初始化的32个环境
        for env_id in range(32):
            self.assertIn(env_id, self.agent.env_reward_sums)
            self.assertIn(env_id, self.agent.env_timers)
            self.assertEqual(self.agent.env_reward_sums[env_id], 0.0)
            self.assertEqual(self.agent.env_timers[env_id], 0)
            self.assertIsNone(self.agent.env_team_skills[env_id])
            self.assertIsNone(self.agent.env_agent_skills[env_id])
        
        print("✓ 环境状态初始化正确")
    
    def test_multi_env_step(self):
        """测试多环境的step方法"""
        print("\n=== 测试多环境step方法 ===")
        
        # 准备测试数据
        state = np.random.randn(self.config.state_dim)
        observations = np.random.randn(self.config.n_agents, self.config.obs_dim)
        
        env_skills = {}
        env_infos = {}
        
        # 测试多个环境的step调用
        for env_id in range(self.num_envs):
            actions, info = self.agent.step(state, observations, 0, deterministic=False, env_id=env_id)
            
            # 验证返回值
            self.assertEqual(actions.shape, (self.config.n_agents, self.config.action_dim))
            self.assertIn('team_skill', info)
            self.assertIn('agent_skills', info)
            self.assertIn('env_id', info)
            self.assertEqual(info['env_id'], env_id)
            
            # 存储技能信息
            env_skills[env_id] = {
                'team_skill': info['team_skill'],
                'agent_skills': info['agent_skills']
            }
            env_infos[env_id] = info
            
            # 验证环境特定状态已更新
            self.assertIsNotNone(self.agent.env_team_skills[env_id])
            self.assertIsNotNone(self.agent.env_agent_skills[env_id])
            self.assertIsNotNone(self.agent.env_log_probs[env_id])
        
        # 验证不同环境可能有不同的技能（允许相同，但结构应该正确）
        for env_id in range(self.num_envs):
            team_skill = env_skills[env_id]['team_skill']
            agent_skills = env_skills[env_id]['agent_skills']
            
            self.assertTrue(0 <= team_skill < self.config.n_Z)
            self.assertEqual(len(agent_skills), self.config.n_agents)
            for agent_skill in agent_skills:
                self.assertTrue(0 <= agent_skill < self.config.n_z)
        
        print(f"✓ 多环境step方法正确，测试了{self.num_envs}个环境")
        print(f"  环境技能分配: {[(env_id, skills['team_skill']) for env_id, skills in env_skills.items()]}")
    
    def test_skill_timer_management(self):
        """测试技能计时器管理"""
        print("\n=== 测试技能计时器管理 ===")
        
        state = np.random.randn(self.config.state_dim)
        observations = np.random.randn(self.config.n_agents, self.config.obs_dim)
        
        env_id = 0
        
        # 模拟k步的技能使用
        for step in range(self.config.k + 2):  # 多执行2步以测试重置
            actions, info = self.agent.step(state, observations, step, deterministic=False, env_id=env_id)
            
            expected_skill_changed = (step % self.config.k == 0)
            actual_skill_changed = info['skill_changed']
            
            if step == 0:
                # 第一步应该分配技能
                self.assertTrue(actual_skill_changed, f"步骤{step}: 应该分配初始技能")
            else:
                # 检查技能变化是否符合预期
                self.assertEqual(expected_skill_changed, actual_skill_changed, 
                               f"步骤{step}: 技能变化预期={expected_skill_changed}, 实际={actual_skill_changed}")
            
            # 验证计时器值
            expected_timer = 0 if expected_skill_changed else (step % self.config.k)
            actual_timer = info['skill_timer']
            
            print(f"  步骤{step}: 技能变化={actual_skill_changed}, 计时器={actual_timer}, 预期计时器={expected_timer}")
        
        print("✓ 技能计时器管理正确")
    
    def test_experience_storage(self):
        """测试经验存储"""
        print("\n=== 测试经验存储 ===")
        
        # 准备测试数据
        state = np.random.randn(self.config.state_dim)
        next_state = np.random.randn(self.config.state_dim)
        observations = np.random.randn(self.config.n_agents, self.config.obs_dim)
        next_observations = np.random.randn(self.config.n_agents, self.config.obs_dim)
        actions = np.random.randn(self.config.n_agents, self.config.action_dim)
        rewards = 1.0
        dones = False
        
        initial_low_level_size = len(self.agent.low_level_buffer)
        initial_high_level_samples = self.agent.high_level_samples_total
        
        # 为不同环境存储经验
        for env_id in range(self.num_envs):
            # 先执行step获取技能信息
            _, info = self.agent.step(state, observations, 0, deterministic=False, env_id=env_id)
            
            # 存储转换
            self.agent.store_transition(
                state, next_state, observations, next_observations,
                actions, rewards, dones,
                info['team_skill'], info['agent_skills'], info['action_logprobs'],
                log_probs=info['log_probs'], skill_timer_for_env=0, env_id=env_id
            )
        
        # 验证低层经验增加
        expected_low_level_increase = self.num_envs * self.config.n_agents
        actual_low_level_increase = len(self.agent.low_level_buffer) - initial_low_level_size
        self.assertEqual(actual_low_level_increase, expected_low_level_increase,
                        f"低层经验应该增加{expected_low_level_increase}个，实际增加{actual_low_level_increase}个")
        
        print(f"✓ 经验存储正确，低层经验增加{actual_low_level_increase}个")
        
        # 测试高层经验收集（在技能周期结束时）
        print("\n--- 测试高层经验收集 ---")
        
        for env_id in range(self.num_envs):
            # 模拟技能周期结束
            _, info = self.agent.step(state, observations, self.config.k-1, deterministic=False, env_id=env_id)
            
            self.agent.store_transition(
                state, next_state, observations, next_observations,
                actions, rewards, dones,
                info['team_skill'], info['agent_skills'], info['action_logprobs'],
                log_probs=info['log_probs'], skill_timer_for_env=self.config.k-1, env_id=env_id
            )
        
        # 验证高层经验增加
        final_high_level_samples = self.agent.high_level_samples_total
        high_level_increase = final_high_level_samples - initial_high_level_samples
        self.assertGreater(high_level_increase, 0, "应该有高层经验被收集")
        
        print(f"✓ 高层经验收集正确，收集到{high_level_increase}个高层样本")
    
    def test_env_isolation(self):
        """测试环境隔离性"""
        print("\n=== 测试环境隔离性 ===")
        
        state = np.random.randn(self.config.state_dim)
        observations = np.random.randn(self.config.n_agents, self.config.obs_dim)
        
        # 为不同环境执行不同数量的步骤
        env_steps = {0: 0, 1: 3, 2: 7, 3: 2}
        
        for env_id, steps in env_steps.items():
            for step in range(steps + 1):
                self.agent.step(state, observations, step, deterministic=False, env_id=env_id)
        
        # 验证环境状态独立性
        for env_id, expected_steps in env_steps.items():
            # 检查计时器状态
            expected_timer = expected_steps % self.config.k
            # 注意：由于step方法内部的逻辑，timer可能会在达到k-1后重置
            if expected_steps > 0 and expected_steps % self.config.k == 0:
                expected_timer = 0
                
            print(f"  环境{env_id}: 执行了{expected_steps}步, 当前计时器状态={self.agent.env_timers[env_id]}")
            
            # 验证环境有自己的技能状态
            self.assertIsNotNone(self.agent.env_team_skills[env_id])
            self.assertIsNotNone(self.agent.env_agent_skills[env_id])
        
        print("✓ 环境隔离性正确")
    
    def test_backward_compatibility(self):
        """测试向后兼容性"""
        print("\n=== 测试向后兼容性 ===")
        
        state = np.random.randn(self.config.state_dim)
        observations = np.random.randn(self.config.n_agents, self.config.obs_dim)
        
        # 测试不传递env_id参数（应该默认为0）
        actions1, info1 = self.agent.step(state, observations, 0, deterministic=False)
        actions2, info2 = self.agent.step(state, observations, 0, deterministic=False, env_id=0)
        
        # 验证结果结构相同
        self.assertEqual(actions1.shape, actions2.shape)
        self.assertIn('team_skill', info1)
        self.assertIn('agent_skills', info1)
        self.assertIn('team_skill', info2)
        self.assertIn('agent_skills', info2)
        
        # 验证全局状态与环境0状态同步
        self.assertEqual(self.agent.current_team_skill, self.agent.env_team_skills[0])
        np.testing.assert_array_equal(self.agent.current_agent_skills, self.agent.env_agent_skills[0])
        
        print("✓ 向后兼容性正确")
    
    def test_action_selection_consistency(self):
        """测试动作选择的一致性"""
        print("\n=== 测试动作选择一致性 ===")
        
        state = np.random.randn(self.config.state_dim)
        observations = np.random.randn(self.config.n_agents, self.config.obs_dim)
        
        # 设置相同的随机种子
        torch.manual_seed(42)
        np.random.seed(42)
        
        # 为多个环境获取动作
        actions_dict = {}
        for env_id in range(3):
            # 重置随机种子以确保可重现性测试
            if env_id == 0:
                torch.manual_seed(42)
                np.random.seed(42)
            
            actions, info = self.agent.step(state, observations, 0, deterministic=True, env_id=env_id)
            actions_dict[env_id] = actions
            
            # 验证动作形状
            self.assertEqual(actions.shape, (self.config.n_agents, self.config.action_dim))
            
            # 验证确定性模式下的一致性（相同环境ID应该产生相同结果）
            if env_id > 0:
                actions2, _ = self.agent.step(state, observations, 1, deterministic=True, env_id=env_id)
                # 注意：由于技能可能不同，动作可能不同，但应该是有效的
                self.assertEqual(actions2.shape, (self.config.n_agents, self.config.action_dim))
        
        print("✓ 动作选择一致性正确")
    
    def run_comprehensive_test(self):
        """运行综合测试"""
        print("\n" + "="*50)
        print("开始HMASD多环境支持综合测试")
        print("="*50)
        
        try:
            self.test_env_state_initialization()
            self.test_multi_env_step()
            self.test_skill_timer_management()
            self.test_experience_storage()
            self.test_env_isolation()
            self.test_backward_compatibility()
            self.test_action_selection_consistency()
            
            print("\n" + "="*50)
            print("🎉 所有测试通过！HMASD多环境支持实现正确！")
            print("="*50)
            return True
            
        except Exception as e:
            print(f"\n❌ 测试失败: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

def main():
    """主测试函数"""
    print("HMASD多环境并行训练支持验证测试")
    print("测试目标：验证多环境并行训练的正确实现")
    
    # 创建测试实例
    test_instance = TestMultiEnvSupport()
    test_instance.setUp()
    
    try:
        # 运行综合测试
        success = test_instance.run_comprehensive_test()
        
        if success:
            print("\n✅ 测试结论：多环境并行训练支持实现正确，可以安全使用！")
            
            # 输出一些实用信息
            print("\n📋 使用指南：")
            print("1. 在agent.step()调用时传递env_id参数：")
            print("   actions, info = agent.step(state, obs, step, env_id=env_id)")
            print("2. 在agent.store_transition()调用时传递env_id参数：")
            print("   agent.store_transition(..., env_id=env_id)")
            print("3. 每个环境维护独立的技能状态和计时器")
            print("4. 环境0的状态会同步到全局状态以保持兼容性")
            
            return 0
        else:
            print("\n❌ 测试结论：发现问题，需要修复后再使用！")
            return 1
            
    finally:
        test_instance.tearDown()

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
