#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
运行数据收集测试的便捷脚本
"""

import os
import sys
import subprocess
import argparse
from datetime import datetime

def main():
    parser = argparse.ArgumentParser(description='运行Enhanced Reward Tracker数据收集功能测试')
    parser.add_argument('--verbose', '-v', action='store_true', help='详细输出')
    parser.add_argument('--specific-test', '-t', type=str, help='运行特定测试 (例如: test_01_basic_initialization)')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Enhanced Reward Tracker 数据收集功能测试")
    print(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    # 构建命令
    cmd = [sys.executable, 'test_enhanced_reward_tracking.py']
    
    if args.specific_test:
        # 运行特定测试
        cmd = [sys.executable, '-m', 'unittest', f'test_enhanced_reward_tracking.TestEnhancedRewardTracker.{args.specific_test}']
        if args.verbose:
            cmd.append('-v')
    elif args.verbose:
        cmd.append('-v')
    
    try:
        # 运行测试
        print("开始运行测试...")
        print(f"执行命令: {' '.join(cmd)}")
        print("-" * 60)
        
        result = subprocess.run(cmd, cwd=os.getcwd(), capture_output=False)
        
        print("-" * 60)
        if result.returncode == 0:
            print("✅ 所有测试通过！数据收集功能正常。")
        else:
            print("❌ 部分测试失败，请检查上面的输出。")
        
        return result.returncode
        
    except KeyboardInterrupt:
        print("\n⚠️ 测试被用户中断")
        return 1
    except Exception as e:
        print(f"❌ 运行测试时出错: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
