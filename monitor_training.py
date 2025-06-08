#!/usr/bin/env python3
"""
训练监控脚本 - 实时显示MAPPO训练进度
"""

import os
import time
import re
from datetime import datetime

def get_latest_log_file(log_dir='logs'):
    """获取最新的日志文件"""
    log_files = [f for f in os.listdir(log_dir) if f.startswith('mappo_enhanced_tracking') and f.endswith('.log')]
    if not log_files:
        return None
    
    # 按修改时间排序，获取最新的
    log_files.sort(key=lambda x: os.path.getmtime(os.path.join(log_dir, x)), reverse=True)
    return os.path.join(log_dir, log_files[0])

def monitor_training_progress(log_file_path, update_interval=10):
    """监控训练进度"""
    if not os.path.exists(log_file_path):
        print(f"错误: 日志文件 {log_file_path} 不存在")
        return
    
    print(f"开始监控训练进度: {log_file_path}")
    print("=" * 60)
    
    last_position = 0
    step_pattern = re.compile(r'步骤 (\d+): Actor损失=([\d.]+), Critic损失=([\d.]+)')
    episode_pattern = re.compile(r'Episode (\d+): 环境(\d+), 奖励=([-\d.]+), 长度=(\d+)')
    
    last_step = 0
    last_episode = 0
    
    try:
        while True:
            with open(log_file_path, 'r', encoding='utf-8') as f:
                f.seek(last_position)
                new_lines = f.readlines()
                last_position = f.tell()
            
            # 解析新的日志行
            for line in new_lines:
                line = line.strip()
                
                # 检查步骤更新
                step_match = step_pattern.search(line)
                if step_match:
                    step, actor_loss, critic_loss = step_match.groups()
                    step = int(step)
                    if step > last_step:
                        last_step = step
                        current_time = datetime.now().strftime('%H:%M:%S')
                        print(f"[{current_time}] 步骤 {step:,}: Actor损失={float(actor_loss):.4f}, Critic损失={float(critic_loss):.4f}")
                
                # 检查episode完成
                episode_match = episode_pattern.search(line)
                if episode_match:
                    episode, env_id, reward, length = episode_match.groups()
                    episode = int(episode)
                    if episode > last_episode:
                        last_episode = episode
                        current_time = datetime.now().strftime('%H:%M:%S')
                        print(f"[{current_time}] Episode {episode}: 环境{env_id}, 奖励={float(reward):.2f}, 长度={length}")
            
            time.sleep(update_interval)
            
    except KeyboardInterrupt:
        print("\n监控已停止")
    except Exception as e:
        print(f"监控错误: {e}")

def show_training_summary(log_file_path):
    """显示训练摘要"""
    if not os.path.exists(log_file_path):
        print(f"错误: 日志文件 {log_file_path} 不存在")
        return
    
    print(f"训练摘要 - {log_file_path}")
    print("=" * 60)
    
    step_pattern = re.compile(r'步骤 (\d+): Actor损失=([\d.]+), Critic损失=([\d.]+)')
    episode_pattern = re.compile(r'Episode (\d+): 环境(\d+), 奖励=([-\d.]+), 长度=(\d+)')
    
    steps = []
    episodes = []
    
    with open(log_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            step_match = step_pattern.search(line)
            if step_match:
                step, actor_loss, critic_loss = step_match.groups()
                steps.append({
                    'step': int(step),
                    'actor_loss': float(actor_loss),
                    'critic_loss': float(critic_loss)
                })
            
            episode_match = episode_pattern.search(line)
            if episode_match:
                episode, env_id, reward, length = episode_match.groups()
                episodes.append({
                    'episode': int(episode),
                    'env_id': int(env_id),
                    'reward': float(reward),
                    'length': int(length)
                })
    
    if steps:
        latest_step = steps[-1]
        print(f"最新步骤: {latest_step['step']:,}")
        print(f"最新Actor损失: {latest_step['actor_loss']:.4f}")
        print(f"最新Critic损失: {latest_step['critic_loss']:.4f}")
        
        # 计算损失趋势
        if len(steps) >= 10:
            recent_actor = [s['actor_loss'] for s in steps[-10:]]
            recent_critic = [s['critic_loss'] for s in steps[-10:]]
            print(f"近10次更新Actor损失均值: {sum(recent_actor)/len(recent_actor):.4f}")
            print(f"近10次更新Critic损失均值: {sum(recent_critic)/len(recent_critic):.4f}")
    
    if episodes:
        print(f"\n总Episode数: {len(episodes)}")
        recent_rewards = [e['reward'] for e in episodes[-20:]] if len(episodes) >= 20 else [e['reward'] for e in episodes]
        if recent_rewards:
            print(f"近期平均奖励: {sum(recent_rewards)/len(recent_rewards):.2f}")
            print(f"近期最大奖励: {max(recent_rewards):.2f}")
            print(f"近期最小奖励: {min(recent_rewards):.2f}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='MAPPO训练监控工具')
    parser.add_argument('--log_file', type=str, help='指定日志文件路径')
    parser.add_argument('--summary', action='store_true', help='显示训练摘要')
    parser.add_argument('--monitor', action='store_true', help='实时监控训练进度')
    parser.add_argument('--interval', type=int, default=10, help='监控更新间隔（秒）')
    
    args = parser.parse_args()
    
    # 确定日志文件
    if args.log_file:
        log_file_path = args.log_file
    else:
        log_file_path = get_latest_log_file()
        if not log_file_path:
            print("错误: 未找到训练日志文件")
            return
        print(f"自动选择最新日志文件: {log_file_path}")
    
    if args.summary:
        show_training_summary(log_file_path)
    elif args.monitor:
        monitor_training_progress(log_file_path, args.interval)
    else:
        # 默认显示摘要然后开始监控
        show_training_summary(log_file_path)
        print("\n开始实时监控...")
        monitor_training_progress(log_file_path, args.interval)

if __name__ == "__main__":
    main()
