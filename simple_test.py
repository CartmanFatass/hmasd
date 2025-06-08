#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简单直接的数据收集功能验证脚本
"""

import os
import sys
import tempfile
import shutil

print("开始验证 Enhanced Reward Tracker 数据收集功能...")
print("=" * 50)

try:
    # 1. 导入测试
    print("1. 测试模块导入...")
    from config_1 import Config
    from train_enhanced_reward_tracking import EnhancedRewardTracker
    print("✓ 模块导入成功")
    
    # 2. 初始化测试
    print("\n2. 测试初始化...")
    test_dir = tempfile.mkdtemp(prefix='simple_test_')
    config = Config()
    config.n_agents = 3
    tracker = EnhancedRewardTracker(test_dir, config)
    print("✓ EnhancedRewardTracker 初始化成功")
    
    # 3. 基本数据记录测试
    print("\n3. 测试基本数据记录...")
    
    # 记录训练步骤
    tracker.log_training_step(
        step=1,
        env_id=0,
        reward=10.5,
        reward_components={
            'env_component': 8.0,
            'team_disc_component': 1.5,
            'ind_disc_component': 1.0
        },
        info={
            'reward_info': {
                'connected_users': 8,
                'system_throughput_mbps': 120.5
            }
        }
    )
    
    # 记录技能使用
    tracker.log_skill_usage(1, 0, [0, 1, 2], True)
    
    # 记录Episode完成
    tracker.log_episode_completion(1, 0, 200.0, 40)
    
    print("✓ 基本数据记录成功")
    
    # 4. 数据验证
    print("\n4. 验证记录的数据...")
    assert tracker.training_rewards['total_steps'] == 1, "步数记录错误"
    assert len(tracker.training_rewards['episode_rewards']) == 1, "Episode记录错误"
    assert tracker.skill_usage['team_skills'][0] == 1, "技能记录错误"
    print("✓ 数据验证通过")
    
    # 5. 批量数据测试
    print("\n5. 测试批量数据处理...")
    import numpy as np
    
    for i in range(2, 21):
        tracker.log_training_step(i, 0, np.random.normal(10, 1))
        if i % 5 == 0:
            tracker.log_skill_usage(i, i % 3, [i % 3] * 3, True)
        if i % 10 == 0:
            tracker.log_episode_completion(i // 10, 0, np.random.normal(500, 50), 10)
    
    print(f"✓ 批量数据处理成功 (总步数: {tracker.training_rewards['total_steps']})")
    
    # 6. 数据导出测试
    print("\n6. 测试数据导出...")
    
    class MockWriter:
        def __init__(self):
            self.call_count = 0
        def add_scalar(self, tag, value, step):
            self.call_count += 1
    
    mock_writer = MockWriter()
    tracker.export_training_data(20, mock_writer)
    
    # 检查导出结果
    export_dir = os.path.join(test_dir, 'paper_data')
    if os.path.exists(export_dir):
        files = os.listdir(export_dir)
        print(f"✓ 数据导出成功，生成 {len(files)} 个文件")
        for file in sorted(files):
            print(f"  - {file}")
    else:
        print("⚠ 导出目录未创建")
    
    print(f"✓ TensorBoard记录了 {mock_writer.call_count} 个指标")
    
    # 7. 摘要统计测试
    print("\n7. 测试摘要统计...")
    summary = tracker.get_summary_statistics()
    print("✓ 摘要统计生成成功:")
    for key, value in summary.items():
        if isinstance(value, dict):
            print(f"  - {key}: {len(value)} 项")
        else:
            print(f"  - {key}: {value}")
    
    # 8. 清理
    print("\n8. 清理测试环境...")
    shutil.rmtree(test_dir)
    print("✓ 测试环境清理完成")
    
    # 成功总结
    print("\n" + "=" * 50)
    print("🎉 所有测试通过！")
    print("📊 Enhanced Reward Tracker 数据收集功能正常")
    print("✅ 可以开始实际训练并收集数据")
    print("=" * 50)
    
except ImportError as e:
    print(f"❌ 模块导入失败: {e}")
    print("请检查以下文件是否存在:")
    print("  - config_1.py")
    print("  - train_enhanced_reward_tracking.py")
    print("  - logger.py")
    sys.exit(1)
    
except Exception as e:
    print(f"❌ 测试失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n测试完成，数据收集功能验证成功！")
