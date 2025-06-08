#!/usr/bin/env python3
"""
增强的MAPPO训练启动脚本 - 提供实时进度反馈
"""

import os
import sys
import subprocess
import threading
import time
import argparse
from datetime import datetime

def run_training(train_args):
    """在子进程中运行训练"""
    cmd = [sys.executable, 'train_mappo_enhanced_tracking.py'] + train_args
    print(f"执行训练命令: {' '.join(cmd)}")
    print("=" * 60)
    
    # 启动训练进程
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        bufsize=1
    )
    
    # 实时显示输出
    for line in iter(process.stdout.readline, ''):
        print(line, end='')
    
    process.wait()
    return process.returncode

def run_monitor(log_file_path, delay=30):
    """运行监控器"""
    time.sleep(delay)  # 等待训练启动和日志文件创建
    
    # 等待日志文件创建
    max_wait = 60  # 最多等待60秒
    wait_count = 0
    while not os.path.exists(log_file_path) and wait_count < max_wait:
        time.sleep(1)
        wait_count += 1
    
    if not os.path.exists(log_file_path):
        print(f"警告: 等待 {max_wait} 秒后仍未找到日志文件 {log_file_path}")
        return
    
    # 启动监控
    cmd = [sys.executable, 'monitor_training.py', '--monitor', '--log_file', log_file_path]
    subprocess.run(cmd)

def main():
    parser = argparse.ArgumentParser(description='启动MAPPO训练并提供实时监控')
    
    # 训练参数
    parser.add_argument('--mode', type=str, default='train', help='运行模式: train或eval')
    parser.add_argument('--scenario', type=int, default=2, help='场景: 1=基站模式, 2=协作组网模式')
    parser.add_argument('--num_envs', type=int, default=32, help='并行环境数量')
    parser.add_argument('--n_uavs', type=int, default=5, help='初始无人机数量')
    parser.add_argument('--n_users', type=int, default=50, help='用户数量')
    parser.add_argument('--console_log_level', type=str, default='info', 
                        choices=['debug', 'info', 'warning', 'error', 'critical'], 
                        help='控制台日志级别')
    parser.add_argument('--log_level', type=str, default='info', 
                        choices=['debug', 'info', 'warning', 'error', 'critical'], 
                        help='文件日志级别')
    parser.add_argument('--device', type=str, default='auto', 
                        choices=['auto', 'cuda', 'cpu'], help='计算设备')
    
    # 监控参数
    parser.add_argument('--no_monitor', action='store_true', help='禁用实时监控')
    parser.add_argument('--monitor_delay', type=int, default=10, help='监控启动延迟（秒）')
    
    args = parser.parse_args()
    
    print("MAPPO增强训练启动器")
    print("=" * 60)
    print(f"模式: {args.mode}")
    print(f"场景: {args.scenario}")
    print(f"并行环境: {args.num_envs}")
    print(f"控制台日志级别: {args.console_log_level}")
    print(f"文件日志级别: {args.log_level}")
    print(f"设备: {args.device}")
    
    # 构建训练参数
    train_args = [
        '--mode', args.mode,
        '--scenario', str(args.scenario),
        '--num_envs', str(args.num_envs),
        '--n_uavs', str(args.n_uavs),
        '--n_users', str(args.n_users),
        '--console_log_level', args.console_log_level,
        '--log_level', args.log_level,
        '--device', args.device
    ]
    
    if args.mode == 'train' and not args.no_monitor:
        # 预测日志文件名
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        expected_log_file = f"logs/mappo_enhanced_tracking_{timestamp}.log"
        
        print(f"启用实时监控，预期日志文件: {expected_log_file}")
        print("注意: 监控器将在训练启动后开始运行")
        print("=" * 60)
        
        # 在单独的线程中启动监控
        monitor_thread = threading.Thread(
            target=run_monitor, 
            args=(expected_log_file, args.monitor_delay),
            daemon=True
        )
        monitor_thread.start()
        
        # 启动训练
        return_code = run_training(train_args)
        
        if return_code == 0:
            print("\n训练成功完成!")
        else:
            print(f"\n训练异常结束，返回码: {return_code}")
        
        # 给监控线程一些时间完成
        monitor_thread.join(timeout=5)
        
    else:
        # 直接运行训练，不启用监控
        return_code = run_training(train_args)
        
        if return_code == 0:
            print("\n训练成功完成!")
        else:
            print(f"\n训练异常结束，返回码: {return_code}")
    
    return return_code

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\n训练被用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n启动器出错: {e}")
        sys.exit(1)
