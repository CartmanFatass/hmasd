#!/usr/bin/env python3
"""
HMASD增强训练系统测试脚本
用于验证增强的奖励追踪和数据收集功能是否正常工作
"""

import os
import sys
import time
import subprocess
import tempfile
from pathlib import Path
import argparse

def test_imports():
    """测试所需模块是否能正常导入"""
    print("1. 测试模块导入...")
    
    try:
        import torch
        print(f"   ✓ PyTorch {torch.__version__}")
    except ImportError as e:
        print(f"   ✗ PyTorch 导入失败: {e}")
        return False
    
    try:
        import pandas as pd
        print(f"   ✓ Pandas {pd.__version__}")
    except ImportError as e:
        print(f"   ✗ Pandas 导入失败: {e}")
        return False
    
    try:
        import matplotlib
        print(f"   ✓ Matplotlib {matplotlib.__version__}")
    except ImportError as e:
        print(f"   ✗ Matplotlib 导入失败: {e}")
        return False
    
    try:
        import seaborn as sns
        print(f"   ✓ Seaborn {sns.__version__}")
    except ImportError as e:
        print(f"   ✗ Seaborn 导入失败: {e}")
        return False
    
    try:
        from stable_baselines3.common.vec_env import SubprocVecEnv
        print("   ✓ Stable Baselines3")
    except ImportError as e:
        print(f"   ✗ Stable Baselines3 导入失败: {e}")
        return False
    
    return True

def test_config():
    """测试配置文件"""
    print("2. 测试配置文件...")
    
    try:
        from config_1 import Config
        config = Config()
        
        # 检查关键配置
        assert hasattr(config, 'n_Z'), "缺少 n_Z 配置"
        assert hasattr(config, 'n_z'), "缺少 n_z 配置" 
        assert hasattr(config, 'k'), "缺少 k 配置"
        assert hasattr(config, 'lambda_e'), "缺少 lambda_e 配置"
        
        print(f"   ✓ 配置加载成功: n_Z={config.n_Z}, n_z={config.n_z}, k={config.k}")
        return True
        
    except Exception as e:
        print(f"   ✗ 配置文件测试失败: {e}")
        return False

def test_agent_creation():
    """测试代理创建"""
    print("3. 测试HMASD代理创建...")
    
    try:
        from config_1 import Config
        from hmasd.agent import HMASDAgent
        
        config = Config()
        # 设置测试用的环境维度
        config.update_env_dims(state_dim=20, obs_dim=10)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            agent = HMASDAgent(config, log_dir=temp_dir, device='cpu')
            print("   ✓ HMASD代理创建成功")
            return True
            
    except Exception as e:
        print(f"   ✗ HMASD代理创建失败: {e}")
        return False

def test_reward_tracker():
    """测试奖励追踪器"""
    print("4. 测试奖励追踪器...")
    
    try:
        # 需要添加当前目录到路径以导入模块
        sys.path.insert(0, os.getcwd())
        from train_enhanced_reward_tracking import EnhancedRewardTracker
        from config_1 import Config
        
        config = Config()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            tracker = EnhancedRewardTracker(temp_dir, config)
            
            # 测试记录功能
            tracker.log_training_step(100, 0, 1.5, {'env_component': 1.0, 'team_disc_component': 0.3, 'ind_disc_component': 0.2})
            tracker.log_episode_completion(1, 0, 150.0, 1000)
            tracker.log_skill_usage(100, 1, [0, 1, 2], True)
            
            print("   ✓ 奖励追踪器测试成功")
            return True
            
    except Exception as e:
        print(f"   ✗ 奖励追踪器测试失败: {e}")
        return False

def test_data_analyzer():
    """测试数据分析器"""
    print("5. 测试数据分析器...")
    
    try:
        # 需要添加当前目录到路径以导入模块
        sys.path.insert(0, os.getcwd())
        from paper_data_analysis import PaperDataAnalyzer
        
        with tempfile.TemporaryDirectory() as temp_dir:
            analyzer = PaperDataAnalyzer(temp_dir)
            print("   ✓ 数据分析器创建成功")
            return True
            
    except Exception as e:
        print(f"   ✗ 数据分析器测试失败: {e}")
        return False

def test_short_training(duration_minutes=2):
    """测试短时间训练"""
    print(f"6. 测试短时间训练 ({duration_minutes} 分钟)...")
    
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            # 构建训练命令
            cmd = [
                sys.executable, "train_enhanced_reward_tracking.py",
                "--mode", "train",
                "--scenario", "2",
                "--n_uavs", "3",
                "--n_users", "10", 
                "--num_envs", "2",
                "--log_dir", temp_dir,
                "--detailed_logging",
                "--export_interval", "100",
                "--log_level", "warning",
                "--console_log_level", "error",
                "--device", "cpu"
            ]
            
            print(f"   运行命令: {' '.join(cmd)}")
            
            # 设置超时
            timeout_seconds = duration_minutes * 60
            
            # 运行训练
            process = subprocess.Popen(
                cmd, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE,
                text=True
            )
            
            try:
                stdout, stderr = process.communicate(timeout=timeout_seconds)
                
                if process.returncode == 0:
                    print("   ✓ 短时间训练完成")
                    
                    # 检查是否生成了预期的文件
                    log_dirs = list(Path(temp_dir).glob("enhanced_tracking_*"))
                    if log_dirs:
                        log_dir = log_dirs[0]
                        paper_data_dir = log_dir / "paper_data"
                        
                        if paper_data_dir.exists():
                            print("   ✓ 论文数据目录已创建")
                            
                            # 检查数据文件
                            csv_files = list(paper_data_dir.glob("*.csv"))
                            json_files = list(paper_data_dir.glob("*.json"))
                            
                            print(f"   ✓ 生成数据文件: {len(csv_files)} CSV, {len(json_files)} JSON")
                        else:
                            print("   ! 论文数据目录未创建（可能训练时间太短）")
                    
                    return True
                else:
                    print(f"   ✗ 训练失败，返回码: {process.returncode}")
                    print(f"   错误输出: {stderr}")
                    return False
                    
            except subprocess.TimeoutExpired:
                process.kill()
                print(f"   ✓ 训练在 {duration_minutes} 分钟后正常超时")
                return True
                
    except Exception as e:
        print(f"   ✗ 短时间训练测试失败: {e}")
        return False

def test_data_analysis():
    """测试数据分析功能"""
    print("7. 测试数据分析功能...")
    
    try:
        # 创建模拟数据进行测试
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # 创建模拟的episode数据
            import pandas as pd
            import json
            
            episode_data = {
                'episode': list(range(1, 101)),
                'env_id': [i % 4 for i in range(100)],
                'total_reward': [10 + i * 0.1 + (i % 10) * 0.5 for i in range(100)],
                'episode_length': [1000 + i * 2 for i in range(100)],
                'timestamp': [time.time() + i for i in range(100)]
            }
            
            episode_df = pd.DataFrame(episode_data)
            episode_file = temp_path / "episode_rewards_step_1000.csv"
            episode_df.to_csv(episode_file, index=False)
            
            # 创建模拟的技能使用数据
            skill_data = {
                'team_skills': {'0': 30, '1': 35, '2': 35},
                'skill_switches': 25,
                'total_steps': 1000
            }
            
            skill_file = temp_path / "skill_usage_step_1000.json"
            with open(skill_file, 'w') as f:
                json.dump(skill_data, f)
            
            # 测试数据分析器
            sys.path.insert(0, os.getcwd())
            from paper_data_analysis import PaperDataAnalyzer
            
            analyzer = PaperDataAnalyzer(temp_path)
            generated_files = analyzer.run_full_analysis()
            
            if generated_files:
                print(f"   ✓ 数据分析成功，生成 {len(generated_files)} 个文件")
                return True
            else:
                print("   ✗ 数据分析未生成预期文件")
                return False
                
    except Exception as e:
        print(f"   ✗ 数据分析测试失败: {e}")
        return False

def main():
    """主测试函数"""
    parser = argparse.ArgumentParser(description='HMASD增强训练系统测试')
    parser.add_argument('--skip-training', action='store_true', 
                        help='跳过训练测试（仅测试导入和创建）')
    parser.add_argument('--training-minutes', type=int, default=2,
                        help='训练测试持续时间（分钟）')
    
    args = parser.parse_args()
    
    print("=== HMASD增强训练系统测试 ===\n")
    
    tests = [
        test_imports,
        test_config,
        test_agent_creation,
        test_reward_tracker,
        test_data_analyzer,
    ]
    
    if not args.skip_training:
        tests.extend([
            lambda: test_short_training(args.training_minutes),
            test_data_analysis
        ])
    
    passed = 0
    total = len(tests)
    
    for i, test_func in enumerate(tests):
        try:
            if test_func():
                passed += 1
            print()
        except Exception as e:
            print(f"   ✗ 测试异常: {e}\n")
    
    print("=" * 50)
    print(f"测试结果: {passed}/{total} 通过")
    
    if passed == total:
        print("🎉 所有测试通过！增强训练系统可以正常使用。")
        print("\n建议的下一步:")
        print("1. 运行完整训练：")
        print("   python train_enhanced_reward_tracking.py --mode train --detailed_logging")
        print("2. 分析训练结果：")
        print("   python paper_data_analysis.py logs/enhanced_tracking_*/")
        return True
    else:
        print("❌ 部分测试失败，请检查错误信息并修复问题。")
        print("\n常见解决方案:")
        print("1. 安装缺失的依赖: pip install torch pandas matplotlib seaborn stable-baselines3")
        print("2. 检查Python环境版本兼容性")
        print("3. 确保所有项目文件都在正确位置")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
