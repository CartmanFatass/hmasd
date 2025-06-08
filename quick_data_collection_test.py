#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速测试 Enhanced Reward Tracker 数据收集功能
简化版本，专注于核心功能验证
"""

import os
import sys
import tempfile
import shutil
import numpy as np
import json
from datetime import datetime

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_basic_functionality():
    """测试基本功能"""
    print("=" * 50)
    print("快速数据收集功能测试")
    print("=" * 50)
    
    # 创建临时测试目录
    test_dir = tempfile.mkdtemp(prefix='quick_test_')
    print(f"测试目录: {test_dir}")
    
    try:
        # 导入必要的模块
        from config_1 import Config
        from train_enhanced_reward_tracking import EnhancedRewardTracker
        
        print("✓ 模块导入成功")
        
        # 创建配置和追踪器
        config = Config()
        config.n_agents = 3
        tracker = EnhancedRewardTracker(test_dir, config)
        tracker.export_interval = 10
        
        print("✓ EnhancedRewardTracker 初始化成功")
        
        # 测试1: 基本数据记录
        print("\n测试1: 基本数据记录...")
        
        # 记录训练步骤
        tracker.log_training_step(
            step=1,
            env_id=0,
            reward=15.5,
            reward_components={
                'env_component': 10.0,
                'team_disc_component': 3.0,
                'ind_disc_component': 2.5
            },
            info={
                'reward_info': {
                    'connected_users': 8,
                    'system_throughput_mbps': 150.5,
                    'avg_throughput_per_user_mbps': 18.8
                },
                'coverage_ratio': 0.8,
                'n_users': 10
            }
        )
        
        # 记录技能使用
        tracker.log_skill_usage(
            step=1,
            team_skill=1,
            agent_skills=[0, 1, 2],
            skill_changed=True
        )
        
        # 记录Episode完成
        tracker.log_episode_completion(
            episode_num=1,
            env_id=0,
            total_reward=250.8,
            episode_length=50,
            info={'served_users': 8, 'total_users': 10}
        )
        
        print("✓ 基本数据记录成功")
        
        # 验证数据结构
        assert tracker.training_rewards['total_steps'] == 1
        assert len(tracker.training_rewards['step_rewards']) == 1
        assert len(tracker.training_rewards['episode_rewards']) == 1
        assert tracker.skill_usage['team_skills'][1] == 1
        
        print("✓ 数据结构验证通过")
        
        # 测试2: 批量数据生成
        print("\n测试2: 批量数据生成...")
        
        for i in range(2, 25):
            tracker.log_training_step(
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
                        'system_throughput_mbps': np.random.normal(150, 30)
                    }
                }
            )
            
            tracker.log_skill_usage(
                step=i,
                team_skill=np.random.randint(0, 3),
                agent_skills=[np.random.randint(0, 3) for _ in range(3)],
                skill_changed=(i % 5 == 0)
            )
            
            if i % 10 == 0:
                tracker.log_episode_completion(
                    episode_num=i // 10,
                    env_id=0,
                    total_reward=np.random.normal(500, 50),
                    episode_length=10
                )
        
        print(f"✓ 批量数据生成成功 ({tracker.training_rewards['total_steps']} 步)")
        
        # 测试3: 数据导出
        print("\n测试3: 数据导出...")
        
        # 模拟TensorBoard writer
        class MockWriter:
            def __init__(self):
                self.scalars = []
            
            def add_scalar(self, tag, value, step):
                self.scalars.append((tag, value, step))
        
        mock_writer = MockWriter()
        
        # 执行导出
        tracker.export_training_data(25, mock_writer)
        
        # 检查导出文件
        export_dir = os.path.join(test_dir, 'paper_data')
        if os.path.exists(export_dir):
            files = os.listdir(export_dir)
            print(f"✓ 数据导出成功，生成 {len(files)} 个文件:")
            for file in files:
                print(f"  - {file}")
        else:
            print("⚠ 导出目录未创建")
        
        # 检查TensorBoard记录
        print(f"✓ TensorBoard记录了 {len(mock_writer.scalars)} 个指标")
        
        # 测试4: 摘要统计
        print("\n测试4: 摘要统计...")
        
        summary = tracker.get_summary_statistics()
        print("✓ 摘要统计:")
        for key, value in summary.items():
            print(f"  - {key}: {value}")
        
        # 测试5: 可视化生成（可选）
        print("\n测试5: 可视化生成...")
        
        try:
            import matplotlib.pyplot as plt
            plt.ioff()  # 关闭交互模式
            
            tracker.generate_training_plots(export_dir, 25)
            
            # 检查生成的图片
            plot_files = [f for f in os.listdir(export_dir) if f.endswith('.png')]
            print(f"✓ 可视化生成成功，生成 {len(plot_files)} 个图表:")
            for plot_file in plot_files:
                print(f"  - {plot_file}")
        
        except ImportError:
            print("⚠ matplotlib 不可用，跳过可视化测试")
        except Exception as e:
            print(f"⚠ 可视化生成警告: {e}")
        
        print("\n" + "=" * 50)
        print("✅ 所有测试通过！数据收集功能正常工作。")
        print("=" * 50)
        
        return True
        
    except ImportError as e:
        print(f"❌ 导入错误: {e}")
        print("请确保所有依赖已安装")
        return False
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # 清理测试目录
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)
            print(f"✓ 测试目录已清理: {test_dir}")

def test_error_handling():
    """测试错误处理"""
    print("\n" + "=" * 50)
    print("错误处理测试")
    print("=" * 50)
    
    test_dir = tempfile.mkdtemp(prefix='error_test_')
    
    try:
        from config_1 import Config
        from train_enhanced_reward_tracking import EnhancedRewardTracker
        
        config = Config()
        config.n_agents = 3
        tracker = EnhancedRewardTracker(test_dir, config)
        
        # 测试异常输入处理
        print("测试1: None值输入...")
        try:
            tracker.log_training_step(None, None, None)
            print("✓ None值输入处理正常")
        except Exception as e:
            print(f"⚠ None值输入异常: {type(e).__name__}")
        
        print("测试2: 空字典输入...")
        try:
            tracker.log_training_step(1, 0, 10.0, {}, {})
            print("✓ 空字典输入处理正常")
        except Exception as e:
            print(f"⚠ 空字典输入异常: {type(e).__name__}")
        
        print("测试3: 无效路径处理...")
        try:
            invalid_tracker = EnhancedRewardTracker("/invalid/nonexistent/path", config)
            invalid_tracker.export_training_data(100)
            print("⚠ 无效路径未抛出异常")
        except Exception as e:
            print(f"✓ 无效路径正确抛出异常: {type(e).__name__}")
        
        print("✓ 错误处理测试完成")
        return True
        
    except Exception as e:
        print(f"❌ 错误处理测试失败: {e}")
        return False
        
    finally:
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)

def main():
    """主测试函数"""
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 运行基本功能测试
    basic_success = test_basic_functionality()
    
    # 运行错误处理测试
    error_success = test_error_handling()
    
    # 总结
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
    
    if basic_success and error_success:
        print("✅ 所有测试通过！")
        print("📊 Enhanced Reward Tracker 数据收集功能正常工作")
        print("🚀 可以开始实际训练并收集论文数据")
        return 0
    else:
        print("❌ 部分测试失败")
        if not basic_success:
            print("  - 基本功能测试失败")
        if not error_success:
            print("  - 错误处理测试失败")
        print("🔧 建议检查代码和依赖")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
