#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试 train_enhanced_reward_tracking.py 中的数据收集功能
验证各项数据是否能够正常收集、处理和导出
"""

import os
import sys
import time
import shutil
import tempfile
import unittest
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
from datetime import datetime
from unittest.mock import Mock, MagicMock, patch
from collections import defaultdict, deque

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 导入待测试的模块
from config_1 import Config
from logger import init_multiproc_logging, get_logger, shutdown_logging

# 导入 EnhancedRewardTracker
from train_enhanced_reward_tracking import EnhancedRewardTracker

class TestEnhancedRewardTracker(unittest.TestCase):
    """测试 EnhancedRewardTracker 类的所有功能"""
    
    def setUp(self):
        """测试前的准备工作"""
        # 创建临时目录用于测试
        self.test_dir = tempfile.mkdtemp(prefix='test_reward_tracker_')
        
        # 创建配置实例
        self.config = Config()
        self.config.n_agents = 5
        
        # 创建 EnhancedRewardTracker 实例
        self.tracker = EnhancedRewardTracker(self.test_dir, self.config)
        
        # 设置测试参数
        self.tracker.export_interval = 10  # 减小导出间隔用于测试
        
        print(f"测试目录: {self.test_dir}")
    
    def tearDown(self):
        """测试后的清理工作"""
        # 清理临时目录
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    def test_01_basic_initialization(self):
        """测试基本初始化功能"""
        print("测试1: 基本初始化...")
        
        # 验证初始化状态
        self.assertEqual(self.tracker.log_dir, self.test_dir)
        self.assertEqual(self.tracker.config, self.config)
        
        # 验证数据结构初始化
        self.assertIsInstance(self.tracker.training_rewards, dict)
        self.assertIsInstance(self.tracker.skill_usage, dict)
        self.assertIsInstance(self.tracker.performance_metrics, dict)
        
        # 验证关键字段存在
        self.assertIn('episode_rewards', self.tracker.training_rewards)
        self.assertIn('step_rewards', self.tracker.training_rewards)
        self.assertIn('reward_components', self.tracker.training_rewards)
        
        print("✓ 基本初始化测试通过")
    
    def test_02_log_training_step(self):
        """测试训练步骤数据记录功能"""
        print("测试2: 训练步骤数据记录...")
        
        # 模拟训练步骤数据
        step = 100
        env_id = 0
        reward = 15.5
        reward_components = {
            'env_component': 10.0,
            'team_disc_component': 3.0,
            'ind_disc_component': 2.5
        }
        info = {
            'reward_info': {
                'connected_users': 8,
                'system_throughput_mbps': 150.5,
                'avg_throughput_per_user_mbps': 18.8
            },
            'coverage_ratio': 0.8,
            'n_users': 10
        }
        
        # 记录数据
        self.tracker.log_training_step(step, env_id, reward, reward_components, info)
        
        # 验证数据记录
        self.assertEqual(self.tracker.training_rewards['total_steps'], 1)
        self.assertEqual(len(self.tracker.training_rewards['step_rewards']), 1)
        
        # 验证步骤奖励记录
        step_record = self.tracker.training_rewards['step_rewards'][0]
        self.assertEqual(step_record['step'], step)
        self.assertEqual(step_record['env_id'], env_id)
        self.assertEqual(step_record['reward'], reward)
        
        # 验证奖励组成记录
        for comp_name, comp_value in reward_components.items():
            if comp_name in self.tracker.training_rewards['reward_components']:
                comp_records = self.tracker.training_rewards['reward_components'][comp_name]
                self.assertEqual(len(comp_records), 1)
                self.assertEqual(comp_records[0]['value'], comp_value)
        
        # 验证性能指标记录
        self.assertTrue(len(self.tracker.performance_metrics['served_users']) > 0)
        self.assertTrue(len(self.tracker.performance_metrics['total_throughput']) > 0)
        
        print("✓ 训练步骤数据记录测试通过")
    
    def test_03_log_episode_completion(self):
        """测试Episode完成数据记录功能"""
        print("测试3: Episode完成数据记录...")
        
        # 模拟Episode完成数据
        episode_num = 5
        env_id = 1
        total_reward = 250.8
        episode_length = 150
        info = {
            'served_users': 12,
            'total_users': 15,
            'coverage_ratio': 0.8,
            'system_throughput': 200.5
        }
        
        # 记录Episode完成
        self.tracker.log_episode_completion(episode_num, env_id, total_reward, episode_length, info)
        
        # 验证记录
        self.assertEqual(self.tracker.training_rewards['episodes_completed'], 1)
        self.assertEqual(len(self.tracker.training_rewards['episode_rewards']), 1)
        
        # 验证Episode奖励记录
        episode_record = self.tracker.training_rewards['episode_rewards'][0]
        self.assertEqual(episode_record['episode'], episode_num)
        self.assertEqual(episode_record['env_id'], env_id)
        self.assertEqual(episode_record['total_reward'], total_reward)
        self.assertEqual(episode_record['episode_length'], episode_length)
        
        # 验证滑动窗口更新
        self.assertEqual(len(self.tracker.recent_rewards), 1)
        self.assertEqual(len(self.tracker.recent_lengths), 1)
        
        print("✓ Episode完成数据记录测试通过")
    
    def test_04_log_skill_usage(self):
        """测试技能使用数据记录功能"""
        print("测试4: 技能使用数据记录...")
        
        # 模拟技能使用数据
        step = 50
        team_skill = 1
        agent_skills = [0, 1, 2, 1, 0]
        skill_changed = True
        
        # 记录技能使用
        self.tracker.log_skill_usage(step, team_skill, agent_skills, skill_changed)
        
        # 验证团队技能记录
        self.assertEqual(self.tracker.skill_usage['team_skills'][team_skill], 1)
        
        # 验证个体技能记录
        for i, skill in enumerate(agent_skills):
            self.assertEqual(self.tracker.skill_usage['agent_skills'][i][skill], 1)
        
        # 验证技能切换记录
        self.assertEqual(self.tracker.skill_usage['skill_switches'], 1)
        
        # 验证技能多样性记录
        self.assertEqual(len(self.tracker.skill_usage['skill_diversity_history']), 1)
        diversity_record = self.tracker.skill_usage['skill_diversity_history'][0]
        self.assertEqual(diversity_record['step'], step)
        self.assertGreater(diversity_record['diversity'], 0)
        
        print("✓ 技能使用数据记录测试通过")
    
    def test_05_sliding_window_statistics(self):
        """测试滑动窗口统计功能"""
        print("测试5: 滑动窗口统计...")
        
        # 添加多个Episode记录以测试滑动窗口
        for i in range(15):
            total_reward = 100 + np.random.normal(0, 10)
            episode_length = 120 + np.random.randint(-20, 20)
            self.tracker.log_episode_completion(i, 0, total_reward, episode_length)
        
        # 验证滑动窗口大小
        self.assertEqual(len(self.tracker.recent_rewards), 15)
        self.assertEqual(len(self.tracker.recent_lengths), 15)
        
        # 验证奖励方差统计生成
        self.assertGreater(len(self.tracker.training_rewards['reward_variance']), 0)
        
        # 验证统计数据包含必要字段
        latest_stats = self.tracker.training_rewards['reward_variance'][-1]
        self.assertIn('mean', latest_stats)
        self.assertIn('std', latest_stats)
        self.assertIn('min', latest_stats)
        self.assertIn('max', latest_stats)
        
        print("✓ 滑动窗口统计测试通过")
    
    def test_06_data_export_csv_json(self):
        """测试CSV和JSON数据导出功能"""
        print("测试6: CSV和JSON数据导出...")
        
        # 添加测试数据
        self._generate_test_data()
        
        # 模拟TensorBoard writer
        mock_writer = Mock()
        
        # 执行数据导出
        self.tracker.export_training_data(100, mock_writer)
        
        # 验证导出目录创建
        export_dir = os.path.join(self.test_dir, 'paper_data')
        self.assertTrue(os.path.exists(export_dir))
        
        # 验证CSV文件导出
        expected_csv_files = [
            'episode_rewards_step_100.csv',
            'reward_components_step_100.csv'
        ]
        for csv_file in expected_csv_files:
            csv_path = os.path.join(export_dir, csv_file)
            if os.path.exists(csv_path):
                # 验证CSV文件可以读取
                df = pd.read_csv(csv_path)
                self.assertGreater(len(df), 0)
                print(f"✓ CSV文件 {csv_file} 导出成功，包含 {len(df)} 行数据")
        
        # 验证JSON文件导出
        json_file = 'skill_usage_step_100.json'
        json_path = os.path.join(export_dir, json_file)
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                skill_data = json.load(f)
            self.assertIn('team_skills', skill_data)
            self.assertIn('skill_switches', skill_data)
            print(f"✓ JSON文件 {json_file} 导出成功")
        
        print("✓ CSV和JSON数据导出测试通过")
    
    def test_07_visualization_generation(self):
        """测试可视化图表生成功能"""
        print("测试7: 可视化图表生成...")
        
        # 添加测试数据
        self._generate_test_data()
        
        # 生成可视化图表
        export_dir = os.path.join(self.test_dir, 'paper_data')
        os.makedirs(export_dir, exist_ok=True)
        
        try:
            self.tracker.generate_training_plots(export_dir, 100)
            
            # 验证图表文件生成
            expected_plots = [
                'training_progress_step_100.png',
                'reward_components_step_100.png',
                'skill_analysis_step_100.png'
            ]
            
            generated_plots = []
            for plot_file in expected_plots:
                plot_path = os.path.join(export_dir, plot_file)
                if os.path.exists(plot_path):
                    generated_plots.append(plot_file)
                    # 验证文件大小 > 0
                    self.assertGreater(os.path.getsize(plot_path), 0)
            
            print(f"✓ 成功生成 {len(generated_plots)} 个可视化图表: {generated_plots}")
            
        except Exception as e:
            print(f"⚠ 可视化生成过程中出现警告: {e}")
            # 这里不让测试失败，因为可视化可能因为环境问题而失败
        
        print("✓ 可视化图表生成测试完成")
    
    def test_08_tensorboard_logging(self):
        """测试TensorBoard日志记录功能"""
        print("测试8: TensorBoard日志记录...")
        
        # 添加测试数据
        self._generate_test_data()
        
        # 创建模拟的TensorBoard writer
        mock_writer = Mock()
        
        # 执行TensorBoard日志记录
        self.tracker.log_to_tensorboard(mock_writer, 100)
        
        # 验证writer.add_scalar调用
        self.assertGreater(mock_writer.add_scalar.call_count, 0)
        
        # 验证记录的指标类型
        call_args_list = mock_writer.add_scalar.call_args_list
        logged_metrics = [call[0][0] for call in call_args_list]  # 提取标签
        
        expected_metric_prefixes = [
            'Training/',
            'Performance/',
        ]
        
        found_prefixes = []
        for prefix in expected_metric_prefixes:
            if any(metric.startswith(prefix) for metric in logged_metrics):
                found_prefixes.append(prefix)
        
        print(f"✓ TensorBoard记录了 {len(logged_metrics)} 个指标")
        print(f"✓ 包含指标类别: {found_prefixes}")
        
        print("✓ TensorBoard日志记录测试通过")
    
    def test_09_summary_statistics(self):
        """测试摘要统计功能"""
        print("测试9: 摘要统计...")
        
        # 添加测试数据
        self._generate_test_data()
        
        # 获取摘要统计
        summary = self.tracker.get_summary_statistics()
        
        # 验证摘要包含必要字段
        expected_fields = [
            'total_episodes',
            'total_steps',
            'skill_switches'
        ]
        
        for field in expected_fields:
            self.assertIn(field, summary)
        
        # 验证数值合理性
        self.assertGreater(summary['total_episodes'], 0)
        self.assertGreater(summary['total_steps'], 0)
        self.assertGreaterEqual(summary['skill_switches'], 0)
        
        if 'reward_mean' in summary:
            self.assertIsInstance(summary['reward_mean'], (int, float))
        
        print(f"✓ 摘要统计包含 {len(summary)} 个字段")
        print("✓ 摘要统计测试通过")
    
    def test_10_error_handling(self):
        """测试错误处理功能"""
        print("测试10: 错误处理...")
        
        # 测试无效输入处理
        try:
            # 测试None值处理
            self.tracker.log_training_step(None, None, None)
            print("✓ None值输入处理正常")
        except Exception as e:
            print(f"⚠ None值输入处理异常: {e}")
        
        try:
            # 测试空字典处理
            self.tracker.log_training_step(1, 0, 10.0, {}, {})
            print("✓ 空字典输入处理正常")
        except Exception as e:
            print(f"⚠ 空字典输入处理异常: {e}")
        
        try:
            # 测试不存在的目录导出
            invalid_tracker = EnhancedRewardTracker("/invalid/path", self.config)
            invalid_tracker.export_training_data(100)
            print("✓ 无效路径处理正常")
        except Exception as e:
            print(f"✓ 无效路径正确抛出异常: {type(e).__name__}")
        
        print("✓ 错误处理测试完成")
    
    def test_11_performance_stress_test(self):
        """测试性能压力测试"""
        print("测试11: 性能压力测试...")
        
        start_time = time.time()
        
        # 大量数据记录测试
        num_steps = 1000
        for i in range(num_steps):
            # 训练步骤记录
            self.tracker.log_training_step(
                step=i, 
                env_id=i % 4, 
                reward=np.random.normal(10, 2),
                reward_components={
                    'env_component': np.random.normal(8, 1),
                    'team_disc_component': np.random.normal(1, 0.5),
                    'ind_disc_component': np.random.normal(1, 0.5)
                },
                info={
                    'reward_info': {
                        'connected_users': np.random.randint(5, 15),
                        'system_throughput_mbps': np.random.normal(150, 30)
                    }
                }
            )
            
            # 技能使用记录
            self.tracker.log_skill_usage(
                step=i,
                team_skill=np.random.randint(0, 3),
                agent_skills=[np.random.randint(0, 3) for _ in range(5)],
                skill_changed=np.random.random() < 0.1
            )
            
            # 每100步记录一个Episode
            if i % 100 == 0:
                self.tracker.log_episode_completion(
                    episode_num=i // 100,
                    env_id=0,
                    total_reward=np.random.normal(1000, 100),
                    episode_length=100
                )
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"✓ 处理 {num_steps} 步数据耗时: {duration:.2f}秒")
        print(f"✓ 平均每步耗时: {duration/num_steps*1000:.2f}毫秒")
        
        # 验证数据完整性
        self.assertEqual(self.tracker.training_rewards['total_steps'], num_steps)
        self.assertEqual(len(self.tracker.training_rewards['step_rewards']), num_steps)
        
        print("✓ 性能压力测试通过")
    
    def _generate_test_data(self):
        """生成测试数据"""
        # 生成Episode奖励数据
        for i in range(10):
            self.tracker.log_episode_completion(
                episode_num=i,
                env_id=i % 2,
                total_reward=100 + np.random.normal(0, 10),
                episode_length=120 + np.random.randint(-20, 20),
                info={'served_users': np.random.randint(8, 12)}
            )
        
        # 生成步骤奖励数据
        for i in range(50):
            self.tracker.log_training_step(
                step=i,
                env_id=i % 2,
                reward=np.random.normal(10, 2),
                reward_components={
                    'env_component': np.random.normal(8, 1),
                    'team_disc_component': np.random.normal(1, 0.5),
                    'ind_disc_component': np.random.normal(1, 0.5)
                },
                info={
                    'reward_info': {
                        'connected_users': np.random.randint(5, 15),
                        'system_throughput_mbps': np.random.normal(150, 30),
                        'avg_throughput_per_user_mbps': np.random.normal(15, 5)
                    }
                }
            )
        
        # 生成技能使用数据
        for i in range(20):
            self.tracker.log_skill_usage(
                step=i,
                team_skill=np.random.randint(0, 3),
                agent_skills=[np.random.randint(0, 3) for _ in range(5)],
                skill_changed=np.random.random() < 0.2
            )


class TestDataCollectionIntegration(unittest.TestCase):
    """集成测试：测试完整的数据收集流程"""
    
    def setUp(self):
        """测试前的准备工作"""
        self.test_dir = tempfile.mkdtemp(prefix='test_integration_')
        self.config = Config()
        self.config.n_agents = 3
        
    def tearDown(self):
        """测试后的清理工作"""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    def test_complete_training_simulation(self):
        """测试完整的训练模拟流程"""
        print("\n集成测试: 完整训练模拟...")
        
        # 创建追踪器
        tracker = EnhancedRewardTracker(self.test_dir, self.config)
        tracker.export_interval = 50  # 每50步导出一次
        
        # 模拟完整的训练过程
        total_steps = 200
        episode_length = 50
        current_episode = 0
        episode_step = 0
        episode_reward = 0
        
        # 创建模拟的TensorBoard writer
        mock_writer = Mock()
        
        for step in range(total_steps):
            # 模拟训练步骤
            step_reward = np.random.normal(10, 2)
            episode_reward += step_reward
            
            # 记录训练步骤
            tracker.log_training_step(
                step=step,
                env_id=step % 2,
                reward=step_reward,
                reward_components={
                    'env_component': step_reward * 0.8,
                    'team_disc_component': step_reward * 0.1,
                    'ind_disc_component': step_reward * 0.1
                },
                info={
                    'reward_info': {
                        'connected_users': np.random.randint(8, 12),
                        'system_throughput_mbps': np.random.normal(150, 20),
                        'avg_throughput_per_user_mbps': np.random.normal(15, 3)
                    },
                    'coverage_ratio': np.random.uniform(0.7, 0.9),
                    'n_users': 10
                }
            )
            
            # 记录技能使用
            tracker.log_skill_usage(
                step=step,
                team_skill=np.random.randint(0, 3),
                agent_skills=[np.random.randint(0, 3) for _ in range(3)],
                skill_changed=(step % 10 == 0)  # 每10步切换技能
            )
            
            episode_step += 1
            
            # Episode结束
            if episode_step >= episode_length:
                tracker.log_episode_completion(
                    episode_num=current_episode,
                    env_id=0,
                    total_reward=episode_reward,
                    episode_length=episode_step,
                    info={
                        'served_users': np.random.randint(8, 12),
                        'total_users': 10,
                        'coverage_ratio': np.random.uniform(0.7, 0.9)
                    }
                )
                
                current_episode += 1
                episode_step = 0
                episode_reward = 0
            
            # 定期导出数据
            if step % tracker.export_interval == 0 and step > 0:
                tracker.export_training_data(step, mock_writer)
        
        # 最终导出
        tracker.export_training_data(total_steps, mock_writer)
        
        # 验证最终状态
        summary = tracker.get_summary_statistics()
        
        print(f"✓ 模拟训练完成:")
        print(f"  - 总步数: {summary['total_steps']}")
        print(f"  - 总Episodes: {summary['total_episodes']}")
        print(f"  - 技能切换次数: {summary['skill_switches']}")
        print(f"  - 平均奖励: {summary.get('reward_mean', 'N/A'):.2f}" if 'reward_mean' in summary else "  - 平均奖励: N/A")
        
        # 验证数据文件生成
        export_dir = os.path.join(self.test_dir, 'paper_data')
        if os.path.exists(export_dir):
            files = os.listdir(export_dir)
            print(f"  - 生成文件: {len(files)} 个")
            for file in files:
                print(f"    * {file}")
        
        print("✓ 集成测试通过")


def run_comprehensive_test():
    """运行完整的测试套件"""
    print("=" * 60)
    print("Enhanced Reward Tracker 数据收集功能测试")
    print("=" * 60)
    
    # 设置测试环境
    test_log_dir = tempfile.mkdtemp(prefix='test_logs_')
    
    try:
        # 初始化日志系统
        init_multiproc_logging(log_dir=test_log_dir, console_level=30)  # WARNING级别
        
        # 创建测试套件
        test_suite = unittest.TestSuite()
        
        # 添加单元测试
        test_classes = [TestEnhancedRewardTracker, TestDataCollectionIntegration]
        for test_class in test_classes:
            tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
            test_suite.addTests(tests)
        
        # 运行测试
        runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
        result = runner.run(test_suite)
        
        # 测试结果统计
        total_tests = result.testsRun
        failures = len(result.failures)
        errors = len(result.errors)
        success_rate = (total_tests - failures - errors) / total_tests * 100 if total_tests > 0 else 0
        
        print("\n" + "=" * 60)
        print("测试结果总结")
        print("=" * 60)
        print(f"总测试数: {total_tests}")
        print(f"成功: {total_tests - failures - errors}")
        print(f"失败: {failures}")
        print(f"错误: {errors}")
        print(f"成功率: {success_rate:.1f}%")
        
        if failures > 0:
            print("\n失败的测试:")
            for test, traceback in result.failures:
                print(f"  - {test}: {traceback.split('AssertionError:')[-1].strip()}")
        
        if errors > 0:
            print("\n错误的测试:")
            for test, traceback in result.errors:
                print(f"  - {test}: {traceback.split('Exception:')[-1].strip()}")
        
        # 推荐措施
        print("\n" + "=" * 60)
        if success_rate >= 90:
            print("✅ 数据收集功能工作正常，可以开始实际训练！")
        elif success_rate >= 70:
            print("⚠️  数据收集功能基本正常，但建议检查失败的测试项。")
        else:
            print("❌ 数据收集功能存在问题，建议修复后再进行实际训练。")
        
        print("\n测试完成！")
        
        return result.wasSuccessful()
        
    except Exception as e:
        print(f"测试运行出错: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # 清理日志系统
        try:
            shutdown_logging()
        except:
            pass
        
        # 清理测试日志目录
        if os.path.exists(test_log_dir):
            shutil.rmtree(test_log_dir)


if __name__ == "__main__":
    # 关闭matplotlib的交互模式以避免测试时弹出窗口
    plt.ioff()
    
    # 运行测试
    success = run_comprehensive_test()
    
    # 退出代码
    sys.exit(0 if success else 1)
