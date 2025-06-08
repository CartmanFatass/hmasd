#!/usr/bin/env python3
"""
PPO训练启动脚本
提供预设配置的快速启动选项
"""

import os
import sys
import argparse
import subprocess
from datetime import datetime

def get_preset_configs():
    """获取预设配置"""
    configs = {
        "quick": {
            "description": "快速测试配置 (小规模，快速验证)",
            "args": [
                "--scenario", "1",
                "--n_uavs", "3",
                "--n_users", "10",
                "--num_envs", "2",
                "--n_steps", "512",
                "--batch_size", "32",
                "--learning_rate", "3e-4",
                "--export_interval", "200"
            ]
        },
        "standard": {
            "description": "标准配置 (中等规模，平衡性能和速度)",
            "args": [
                "--scenario", "2",
                "--n_uavs", "5",
                "--n_users", "50",
                "--num_envs", "8",
                "--n_steps", "2048",
                "--batch_size", "64",
                "--learning_rate", "3e-4",
                "--export_interval", "1000"
            ]
        },
        "performance": {
            "description": "高性能配置 (大规模，需要高性能硬件)",
            "args": [
                "--scenario", "2",
                "--n_uavs", "8",
                "--n_users", "100",
                "--num_envs", "16",
                "--n_steps", "4096",
                "--batch_size", "128",
                "--learning_rate", "5e-4",
                "--export_interval", "2000",
                "--device", "cuda"
            ]
        },
        "debug": {
            "description": "调试配置 (启用详细日志)",
            "args": [
                "--scenario", "1",
                "--n_uavs", "2",
                "--n_users", "5",
                "--num_envs", "1",
                "--n_steps", "256",
                "--batch_size", "16",
                "--learning_rate", "1e-3",
                "--export_interval", "50",
                "--detailed_logging",
                "--log_level", "debug",
                "--console_log_level", "info"
            ]
        }
    }
    return configs

def run_training(preset=None, custom_args=None, test_first=False):
    """运行PPO训练"""
    
    # 检查测试脚本
    if test_first:
        print("🧪 首先运行测试脚本验证环境...")
        test_cmd = [sys.executable, "test_ppo_training.py"]
        result = subprocess.run(test_cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print("❌ 测试失败！请检查环境配置:")
            print(result.stdout)
            print(result.stderr)
            return False
        else:
            print("✅ 测试通过！")
    
    # 构建命令
    cmd = [sys.executable, "train_ppo_enhanced_tracking.py", "--mode", "train"]
    
    if preset:
        configs = get_preset_configs()
        if preset in configs:
            cmd.extend(configs[preset]["args"])
            print(f"🚀 使用预设配置: {preset}")
            print(f"📝 描述: {configs[preset]['description']}")
        else:
            print(f"❌ 未知预设: {preset}")
            return False
    
    if custom_args:
        cmd.extend(custom_args)
    
    # 显示完整命令
    print(f"\n📋 执行命令:")
    print(" ".join(cmd))
    print(f"\n⏰ 开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    # 执行训练
    try:
        result = subprocess.run(cmd)
        return result.returncode == 0
    except KeyboardInterrupt:
        print("\n\n⚠️ 训练被用户中断")
        return False
    except Exception as e:
        print(f"\n❌ 执行失败: {e}")
        return False

def run_evaluation(model_path, preset=None, custom_args=None):
    """运行模型评估"""
    
    if not os.path.exists(model_path):
        print(f"❌ 模型文件不存在: {model_path}")
        return False
    
    cmd = [
        sys.executable, "train_ppo_enhanced_tracking.py",
        "--mode", "eval",
        "--model_path", model_path
    ]
    
    if preset:
        configs = get_preset_configs()
        if preset in configs:
            # 对于评估，只使用环境相关的参数
            env_args = []
            config_args = configs[preset]["args"]
            i = 0
            while i < len(config_args):
                if config_args[i] in ["--scenario", "--n_uavs", "--n_users", "--max_hops", 
                                     "--user_distribution", "--channel_model"]:
                    env_args.extend([config_args[i], config_args[i+1]])
                    i += 2
                else:
                    i += 2
            cmd.extend(env_args)
    
    if custom_args:
        cmd.extend(custom_args)
    
    print(f"🔍 评估模型: {model_path}")
    print(f"📋 执行命令:")
    print(" ".join(cmd))
    print("=" * 60)
    
    try:
        result = subprocess.run(cmd)
        return result.returncode == 0
    except KeyboardInterrupt:
        print("\n\n⚠️ 评估被用户中断")
        return False
    except Exception as e:
        print(f"\n❌ 执行失败: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(
        description="PPO训练启动脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
预设配置:
  quick      - 快速测试配置 (小规模，快速验证)
  standard   - 标准配置 (中等规模，平衡性能和速度)
  performance- 高性能配置 (大规模，需要高性能硬件)
  debug      - 调试配置 (启用详细日志)

使用示例:
  python run_ppo_training.py --preset standard
  python run_ppo_training.py --preset quick --test-first
  python run_ppo_training.py --eval models/ppo_enhanced_tracking.zip
  python run_ppo_training.py --custom -- --scenario 2 --n_uavs 6
        """
    )
    
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--preset", 
        choices=list(get_preset_configs().keys()),
        help="使用预设配置"
    )
    group.add_argument(
        "--eval",
        metavar="MODEL_PATH",
        help="评估指定的模型文件"
    )
    group.add_argument(
        "--custom",
        action="store_true",
        help="使用自定义参数 (在 -- 后指定)"
    )
    
    parser.add_argument(
        "--test-first",
        action="store_true",
        help="训练前先运行测试脚本"
    )
    
    parser.add_argument(
        "--list-presets",
        action="store_true",
        help="列出所有预设配置"
    )
    
    # 解析已知参数，剩余的传递给训练脚本
    args, unknown_args = parser.parse_known_args()
    
    if args.list_presets:
        print("可用的预设配置:")
        print("=" * 50)
        configs = get_preset_configs()
        for name, config in configs.items():
            print(f"\n🔧 {name}:")
            print(f"   {config['description']}")
            print(f"   参数: {' '.join(config['args'])}")
        return
    
    if args.eval:
        # 评估模式
        success = run_evaluation(args.eval, custom_args=unknown_args)
    elif args.preset:
        # 预设配置训练
        success = run_training(preset=args.preset, custom_args=unknown_args, test_first=args.test_first)
    elif args.custom:
        # 自定义参数训练
        success = run_training(custom_args=unknown_args, test_first=args.test_first)
    else:
        # 默认使用标准配置
        print("ℹ️  未指定配置，使用标准预设")
        success = run_training(preset="standard", test_first=args.test_first)
    
    if success:
        print(f"\n✅ 完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("🎉 任务成功完成！")
    else:
        print(f"\n❌ 结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("⚠️  任务执行失败")
        sys.exit(1)

if __name__ == "__main__":
    main()
