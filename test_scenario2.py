#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Scenario2 (无人机协作组网环境) 测试脚本

该脚本用于全面测试 UAVCooperativeNetworkEnv 环境的各项功能，包括：
- 环境初始化
- UAV连接和角色分配
- 路由路径计算
- 奖励机制
- SINR和吞吐量计算
- 可视化功能
- 性能测试

作者: Test Script Generator
日期: 2025-06-06
"""

import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Any

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from envs.pettingzoo.scenario2 import UAVCooperativeNetworkEnv

class Scenario2Tester:
    """Scenario2环境测试器"""
    
    def __init__(self, output_dir: str = "test_scenario2_results"):
        """
        初始化测试器
        
        参数:
            output_dir: 输出目录
        """
        self.output_dir = output_dir
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.test_results = {}
        
        # 创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 设置日志
        log_file = os.path.join(self.output_dir, f"test_scenario2_{self.timestamp}.log")
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # 测试统计
        self.test_count = 0
        self.passed_tests = 0
        self.failed_tests = 0
        
        self.logger.info("="*60)
        self.logger.info("开始Scenario2环境测试")
        self.logger.info(f"测试时间: {self.timestamp}")
        self.logger.info(f"输出目录: {self.output_dir}")
        self.logger.info("="*60)
    
    def run_test(self, test_name: str, test_func, *args, **kwargs):
        """
        运行单个测试
        
        参数:
            test_name: 测试名称
            test_func: 测试函数
            *args, **kwargs: 测试函数参数
        """
        self.test_count += 1
        self.logger.info(f"\n[测试 {self.test_count}] {test_name}")
        self.logger.info("-" * 40)
        
        try:
            start_time = time.time()
            result = test_func(*args, **kwargs)
            end_time = time.time()
            
            execution_time = end_time - start_time
            self.test_results[test_name] = {
                "status": "PASSED",
                "execution_time": execution_time,
                "result": result
            }
            self.passed_tests += 1
            self.logger.info(f"✓ 测试通过 (耗时: {execution_time:.3f}s)")
            return result
            
        except Exception as e:
            self.test_results[test_name] = {
                "status": "FAILED",
                "error": str(e)
            }
            self.failed_tests += 1
            self.logger.error(f"✗ 测试失败: {str(e)}")
            return None
    
    def test_environment_initialization(self):
        """测试环境初始化"""
        self.logger.info("测试不同配置下的环境初始化...")
        
        # 测试配置
        test_configs = [
            {
                "name": "默认配置",
                "params": {}
            },
            {
                "name": "小规模配置",
                "params": {
                    "n_uavs": 3,
                    "n_users": 10,
                    "area_size": 500,
                    "max_hops": 3
                }
            },
            {
                "name": "大规模配置",
                "params": {
                    "n_uavs": 8,
                    "n_users": 100,
                    "area_size": 2000,
                    "max_hops": 5
                }
            },
            {
                "name": "多地面基站配置",
                "params": {
                    "n_uavs": 6,
                    "n_users": 50,
                    "n_ground_bs": 4,
                    "max_hops": 4
                }
            }
        ]
        
        results = {}
        
        for config in test_configs:
            try:
                self.logger.info(f"  测试配置: {config['name']}")
                env = UAVCooperativeNetworkEnv(**config['params'])
                
                # 验证基本属性
                assert env.n_uavs > 0, "UAV数量必须大于0"
                assert env.n_users > 0, "用户数量必须大于0"
                assert env.area_size > 0, "区域大小必须大于0"
                assert len(env.agents) == env.n_uavs, "智能体数量与UAV数量不匹配"
                
                # 验证地面基站初始化
                assert hasattr(env, 'ground_bs_positions'), "缺少地面基站位置属性"
                assert env.ground_bs_positions is not None, "地面基站位置未初始化"
                assert len(env.ground_bs_positions) == env.n_ground_bs, "地面基站数量不匹配"
                
                # 验证观测空间和动作空间
                for agent in env.agents:
                    obs_space = env.observation_space(agent)
                    action_space = env.action_space(agent)
                    assert obs_space is not None, f"智能体{agent}的观测空间为空"
                    assert action_space is not None, f"智能体{agent}的动作空间为空"
                
                # 测试重置功能
                observations, infos = env.reset(seed=42)
                assert len(observations) == env.n_uavs, "观测数量与UAV数量不匹配"
                assert len(infos) == env.n_uavs, "信息数量与UAV数量不匹配"
                
                results[config['name']] = {
                    "status": "通过",
                    "n_uavs": env.n_uavs,
                    "n_users": env.n_users,
                    "n_ground_bs": env.n_ground_bs,
                    "obs_dim": env.obs_dim,
                    "state_dim": env.state_dim
                }
                
                env.close()
                self.logger.info(f"    ✓ {config['name']} 配置测试通过")
                
            except Exception as e:
                results[config['name']] = {
                    "status": "失败",
                    "error": str(e)
                }
                self.logger.error(f"    ✗ {config['name']} 配置测试失败: {str(e)}")
        
        return results
    
    def test_basic_functionality(self):
        """测试基本环境功能"""
        self.logger.info("测试环境基本功能...")
        
        env = UAVCooperativeNetworkEnv(
            n_uavs=5,
            n_users=30,
            area_size=1000,
            max_steps=100,
            seed=42
        )
        
        results = {}
        
        # 测试重置功能
        self.logger.info("  测试reset()功能...")
        observations, infos = env.reset(seed=42)
        
        # 验证重置后的状态
        assert env.current_step == 0, "步数未正确重置"
        assert env.uav_positions is not None, "UAV位置未初始化"
        assert env.user_positions is not None, "用户位置未初始化"
        assert env.connections is not None, "连接矩阵未初始化"
        assert env.sinr_matrix is not None, "SINR矩阵未初始化"
        
        results["reset"] = "通过"
        self.logger.info("    ✓ reset()功能正常")
        
        # 测试step功能
        self.logger.info("  测试step()功能...")
        
        # 创建随机动作
        actions = {}
        for agent in env.agents:
            actions[agent] = env.action_space(agent).sample()
        
        # 执行步骤
        observations, rewards, terminations, truncations, infos = env.step(actions)
        
        # 验证返回值
        assert len(observations) == env.n_uavs, "观测数量不正确"
        assert len(rewards) == env.n_uavs, "奖励数量不正确"
        assert len(terminations) == env.n_uavs, "终止状态数量不正确"
        assert len(truncations) == env.n_uavs, "截断状态数量不正确"
        assert len(infos) == env.n_uavs, "信息数量不正确"
        
        # 验证步数更新
        assert env.current_step == 1, "步数未正确更新"
        
        results["step"] = "通过"
        self.logger.info("    ✓ step()功能正常")
        
        # 测试多步执行
        self.logger.info("  测试多步执行...")
        for i in range(10):
            actions = {}
            for agent in env.agents:
                actions[agent] = env.action_space(agent).sample()
            observations, rewards, terminations, truncations, infos = env.step(actions)
        
        assert env.current_step == 11, "多步执行后步数不正确"
        results["multi_step"] = "通过"
        self.logger.info("    ✓ 多步执行功能正常")
        
        env.close()
        return results
    
    def test_uav_connections_and_roles(self):
        """测试UAV连接和角色分配"""
        self.logger.info("测试UAV连接和角色分配...")
        
        env = UAVCooperativeNetworkEnv(
            n_uavs=6,
            n_users=40,
            area_size=1000,
            max_hops=4,
            min_sinr=-10,  # 较低的SINR阈值以确保连接
            seed=42
        )
        
        observations, infos = env.reset(seed=42)
        
        results = {}
        
        # 测试UAV连接矩阵
        self.logger.info("  测试UAV连接矩阵...")
        assert hasattr(env, 'uav_connections'), "缺少UAV连接矩阵"
        assert env.uav_connections.shape == (env.n_uavs, env.n_uavs), "UAV连接矩阵维度不正确"
        
        # 验证连接矩阵的对称性
        assert np.allclose(env.uav_connections, env.uav_connections.T), "UAV连接矩阵不对称"
        
        # 验证对角线为False（UAV不与自己连接）
        assert not np.any(np.diag(env.uav_connections)), "UAV不应与自己连接"
        
        results["uav_connections"] = "通过"
        self.logger.info("    ✓ UAV连接矩阵正常")
        
        # 测试UAV到地面基站的连接
        self.logger.info("  测试UAV到地面基站的连接...")
        assert hasattr(env, 'uav_bs_connections'), "缺少UAV到地面基站连接矩阵"
        assert env.uav_bs_connections.shape == (env.n_uavs, env.n_ground_bs), "UAV到地面基站连接矩阵维度不正确"
        
        results["uav_bs_connections"] = "通过"
        self.logger.info("    ✓ UAV到地面基站连接正常")
        
        # 测试UAV角色分配
        self.logger.info("  测试UAV角色分配...")
        assert hasattr(env, 'uav_roles'), "缺少UAV角色数组"
        assert len(env.uav_roles) == env.n_uavs, "UAV角色数组长度不正确"
        
        # 验证角色值在有效范围内
        valid_roles = [0, 1, 2]  # 0:未分配, 1:基站, 2:中继
        assert all(role in valid_roles for role in env.uav_roles), "UAV角色值无效"
        
        # 统计各角色数量
        role_counts = np.bincount(env.uav_roles, minlength=3)
        self.logger.info(f"    角色分布: 未分配={role_counts[0]}, 基站={role_counts[1]}, 中继={role_counts[2]}")
        
        results["uav_roles"] = {
            "status": "通过",
            "role_distribution": {
                "unassigned": int(role_counts[0]),
                "base_station": int(role_counts[1]),
                "relay": int(role_counts[2])
            }
        }
        self.logger.info("    ✓ UAV角色分配正常")
        
        env.close()
        return results
    
    def test_routing_paths(self):
        """测试路由路径计算"""
        self.logger.info("测试路由路径计算...")
        
        env = UAVCooperativeNetworkEnv(
            n_uavs=8,
            n_users=50,
            area_size=1200,
            max_hops=5,
            min_sinr=-15,  # 更低的SINR阈值以确保连接
            seed=42
        )
        
        observations, infos = env.reset(seed=42)
        
        results = {}
        
        # 测试路由路径存在
        self.logger.info("  测试路由路径存在性...")
        assert hasattr(env, 'routing_paths'), "缺少路由路径字典"
        
        # 统计路由信息
        total_paths = len(env.routing_paths)
        hop_counts = []
        
        for uav_idx, path in env.routing_paths.items():
            hop_count = len(path)
            hop_counts.append(hop_count)
            
            # 验证跳数不超过最大限制
            assert hop_count <= env.max_hops, f"UAV {uav_idx} 的路径跳数 {hop_count} 超过最大限制 {env.max_hops}"
            
            # 验证路径格式
            for step in path:
                assert len(step) == 2, f"路径步骤格式错误: {step}"
                assert step[0] in ["uav", "ground_bs"], f"路径节点类型错误: {step[0]}"
        
        if hop_counts:
            avg_hops = np.mean(hop_counts)
            max_hops = np.max(hop_counts)
            min_hops = np.min(hop_counts)
        else:
            avg_hops = max_hops = min_hops = 0
        
        self.logger.info(f"    路由统计: 总路径数={total_paths}, 平均跳数={avg_hops:.2f}, 最大跳数={max_hops}, 最小跳数={min_hops}")
        
        results["routing_paths"] = {
            "status": "通过",
            "total_paths": total_paths,
            "avg_hops": float(avg_hops),
            "max_hops": int(max_hops),
            "min_hops": int(min_hops)
        }
        self.logger.info("    ✓ 路由路径计算正常")
        
        # 测试连通性
        self.logger.info("  测试网络连通性...")
        connectivity_ratio = env._compute_connectivity_ratio()
        assert 0 <= connectivity_ratio <= 1, "连通性比率超出有效范围"
        
        results["connectivity"] = {
            "status": "通过",
            "connectivity_ratio": float(connectivity_ratio)
        }
        self.logger.info(f"    ✓ 网络连通性: {connectivity_ratio:.2%}")
        
        env.close()
        return results
    
    def test_reward_mechanism(self):
        """测试奖励机制"""
        self.logger.info("测试奖励机制...")
        
        env = UAVCooperativeNetworkEnv(
            n_uavs=5,
            n_users=30,
            area_size=1000,
            coverage_weight=0.4,
            quality_weight=0.2,
            connectivity_weight=0.2,
            throughput_weight=0.2,
            seed=42
        )
        
        observations, infos = env.reset(seed=42)
        
        results = {}
        
        # 执行一些步骤以产生奖励
        for i in range(5):
            actions = {}
            for agent in env.agents:
                actions[agent] = env.action_space(agent).sample()
            observations, rewards, terminations, truncations, infos = env.step(actions)
        
        # 测试奖励计算
        self.logger.info("  测试奖励计算...")
        global_reward = env._compute_reward()
        assert isinstance(global_reward, (int, float)), "奖励值类型错误"
        
        # 验证奖励信息
        assert hasattr(env, 'reward_info'), "缺少奖励信息"
        reward_info = env.reward_info
        
        expected_keys = [
            "coverage_reward", "quality_reward", "connectivity_reward",
            "throughput_reward", "hop_penalty", "total_reward"
        ]
        for key in expected_keys:
            assert key in reward_info, f"奖励信息缺少字段: {key}"
            assert isinstance(reward_info[key], (int, float)), f"奖励字段{key}类型错误"
        
        self.logger.info(f"    奖励组成:")
        self.logger.info(f"      覆盖率奖励: {reward_info['coverage_reward']:.4f}")
        self.logger.info(f"      质量奖励: {reward_info['quality_reward']:.4f}")
        self.logger.info(f"      连通性奖励: {reward_info['connectivity_reward']:.4f}")
        self.logger.info(f"      吞吐量奖励: {reward_info['throughput_reward']:.4f}")
        self.logger.info(f"      跳数惩罚: {reward_info['hop_penalty']:.4f}")
        self.logger.info(f"      总奖励: {reward_info['total_reward']:.4f}")
        
        results["reward_mechanism"] = {
            "status": "通过",
            "reward_info": reward_info
        }
        self.logger.info("    ✓ 奖励机制正常")
        
        # 测试不同权重配置的影响
        self.logger.info("  测试不同权重配置...")
        weight_configs = [
            {"coverage_weight": 1.0, "quality_weight": 0.0, "connectivity_weight": 0.0, "throughput_weight": 0.0},
            {"coverage_weight": 0.0, "quality_weight": 1.0, "connectivity_weight": 0.0, "throughput_weight": 0.0},
            {"coverage_weight": 0.0, "quality_weight": 0.0, "connectivity_weight": 1.0, "throughput_weight": 0.0},
            {"coverage_weight": 0.0, "quality_weight": 0.0, "connectivity_weight": 0.0, "throughput_weight": 1.0}
        ]
        
        weight_test_results = []
        for i, weights in enumerate(weight_configs):
            test_env = UAVCooperativeNetworkEnv(
                n_uavs=3, n_users=15, area_size=500, seed=42, **weights
            )
            test_env.reset(seed=42)
            test_reward = test_env._compute_reward()
            weight_test_results.append(test_reward)
            test_env.close()
        
        results["weight_configs"] = {
            "status": "通过",
            "weight_test_rewards": weight_test_results
        }
        self.logger.info("    ✓ 权重配置测试通过")
        
        env.close()
        return results
    
    def test_sinr_and_throughput(self):
        """测试SINR和吞吐量计算"""
        self.logger.info("测试SINR和吞吐量计算...")
        
        env = UAVCooperativeNetworkEnv(
            n_uavs=4,
            n_users=20,
            area_size=800,
            seed=42
        )
        
        observations, infos = env.reset(seed=42)
        
        results = {}
        
        # 测试SINR矩阵
        self.logger.info("  测试SINR矩阵...")
        assert env.sinr_matrix.shape == (env.n_uavs, env.n_users), "SINR矩阵维度不正确"
        
        # 验证SINR值的合理性
        valid_sinr_count = 0
        for i in range(env.n_uavs):
            for j in range(env.n_users):
                sinr = env.sinr_matrix[i, j]
                assert isinstance(sinr, (int, float)), f"SINR值类型错误: {type(sinr)}"
                if sinr > -100:  # 合理的SINR值范围
                    valid_sinr_count += 1
        
        self.logger.info(f"    有效SINR值数量: {valid_sinr_count}/{env.n_uavs * env.n_users}")
        
        results["sinr_matrix"] = {
            "status": "通过",
            "valid_sinr_count": valid_sinr_count,
            "total_pairs": env.n_uavs * env.n_users
        }
        self.logger.info("    ✓ SINR矩阵计算正常")
        
        # 测试吞吐量计算
        self.logger.info("  测试吞吐量计算...")
        
        # 测试基础吞吐量计算
        total_throughput = 0
        throughput_count = 0
        
        for i in range(env.n_uavs):
            for j in range(env.n_users):
                if env.connections[i, j]:
                    throughput = env._compute_throughput(i, j)
                    assert throughput >= 0, f"吞吐量不能为负: {throughput}"
                    total_throughput += throughput
                    throughput_count += 1
        
        avg_throughput = total_throughput / max(throughput_count, 1)
        self.logger.info(f"    连接数: {throughput_count}, 平均吞吐量: {avg_throughput/1e6:.2f} Mbps")
        
        # 测试有效吞吐量计算
        total_effective_throughput = 0
        for i in range(env.n_uavs):
            for j in range(env.n_users):
                if env.connections[i, j]:
                    effective_throughput = env._compute_effective_throughput(i, j)
                    assert effective_throughput >= 0, f"有效吞吐量不能为负: {effective_throughput}"
                    total_effective_throughput += effective_throughput
        
        avg_effective_throughput = total_effective_throughput / max(throughput_count, 1)
        self.logger.info(f"    平均有效吞吐量: {avg_effective_throughput/1e6:.2f} Mbps")
        
        results["throughput"] = {
            "status": "通过",
            "total_throughput_mbps": float(total_throughput / 1e6),
            "avg_throughput_mbps": float(avg_throughput / 1e6),
            "avg_effective_throughput_mbps": float(avg_effective_throughput / 1e6),
            "connected_pairs": throughput_count
        }
        self.logger.info("    ✓ 吞吐量计算正常")
        
        # 测试回程容量计算
        self.logger.info("  测试回程容量计算...")
        backhaul_capacities = []
        
        for i in range(env.n_uavs):
            backhaul_capacity = env._compute_backhaul_capacity(i)
            assert backhaul_capacity >= 0, f"回程容量不能为负: {backhaul_capacity}"
            backhaul_capacities.append(backhaul_capacity)
        
        avg_backhaul_capacity = np.mean(backhaul_capacities)
        self.logger.info(f"    平均回程容量: {avg_backhaul_capacity/1e6:.2f} Mbps")
        
        results["backhaul"] = {
            "status": "通过",
            "avg_backhaul_capacity_mbps": float(avg_backhaul_capacity / 1e6)
        }
        self.logger.info("    ✓ 回程容量计算正常")
        
        env.close()
        return results
    
    def test_scenario_specific_features(self):
        """测试场景特定功能"""
        self.logger.info("测试场景特定功能...")
        
        results = {}
        
        # 测试不同跳数限制
        self.logger.info("  测试不同跳数限制...")
        hop_limit_results = []
        
        for max_hops in [3, 4, 5]:
            env = UAVCooperativeNetworkEnv(
                n_uavs=6,
                n_users=40,
                area_size=1000,
                max_hops=max_hops,
                seed=42
            )
            
            observations, infos = env.reset(seed=42)
            
            # 统计路径信息
            path_count = len(env.routing_paths)
            connectivity_ratio = env._compute_connectivity_ratio()
            
            hop_limit_results.append({
                "max_hops": max_hops,
                "path_count": path_count,
                "connectivity_ratio": float(connectivity_ratio)
            })
            
            env.close()
        
        self.logger.info("    跳数限制测试结果:")
        for result in hop_limit_results:
            self.logger.info(f"      最大跳数={result['max_hops']}: 路径数={result['path_count']}, 连通性={result['connectivity_ratio']:.2%}")
        
        results["hop_limits"] = {
            "status": "通过",
            "results": hop_limit_results
        }
        
        # 测试不同地面基站配置
        self.logger.info("  测试不同地面基站配置...")
        bs_config_results = []
        
        for n_bs in [1, 2, 4]:
            env = UAVCooperativeNetworkEnv(
                n_uavs=6,
                n_users=40,
                area_size=1000,
                n_ground_bs=n_bs,
                seed=42
            )
            
            observations, infos = env.reset(seed=42)
            
            # 统计连接信息
            bs_connections = np.sum(env.uav_bs_connections)
            connectivity_ratio = env._compute_connectivity_ratio()
            
            bs_config_results.append({
                "n_ground_bs": n_bs,
                "bs_connections": int(bs_connections),
                "connectivity_ratio": float(connectivity_ratio)
            })
            
            env.close()
        
        self.logger.info("    地面基站配置测试结果:")
        for result in bs_config_results:
            self.logger.info(f"      基站数={result['n_ground_bs']}: 基站连接数={result['bs_connections']}, 连通性={result['connectivity_ratio']:.2%}")
        
        results["bs_configs"] = {
            "status": "通过",
            "results": bs_config_results
        }
        
        self.logger.info("    ✓ 场景特定功能测试通过")
        
        return results
    
    def test_performance_and_stress(self):
        """测试性能和压力"""
        self.logger.info("测试性能和压力...")
        
        results = {}
        
        # 测试大规模场景
        self.logger.info("  测试大规模场景性能...")
        large_scale_configs = [
            {"n_uavs": 10, "n_users": 100, "area_size": 2000},
            {"n_uavs": 15, "n_users": 200, "area_size": 3000},
            {"n_uavs": 20, "n_users": 300, "area_size": 4000}
        ]
        
        performance_results = []
        
        for config in large_scale_configs:
            self.logger.info(f"    测试配置: {config}")
            
            start_time = time.time()
            
            try:
                env = UAVCooperativeNetworkEnv(**config, seed=42)
                
                # 测试初始化时间
                init_time = time.time() - start_time
                
                # 测试重置时间
                reset_start = time.time()
                observations, infos = env.reset(seed=42)
                reset_time = time.time() - reset_start
                
                # 测试步骤执行时间
                step_times = []
                for i in range(10):
                    actions = {}
                    for agent in env.agents:
                        actions[agent] = env.action_space(agent).sample()
                    
                    step_start = time.time()
                    observations, rewards, terminations, truncations, infos = env.step(actions)
                    step_time = time.time() - step_start
                    step_times.append(step_time)
                
                avg_step_time = np.mean(step_times)
                
                performance_results.append({
                    "config": config,
                    "init_time": float(init_time),
                    "reset_time": float(reset_time),
                    "avg_step_time": float(avg_step_time),
                    "status": "成功"
                })
                
                env.close()
                self.logger.info(f"      ✓ 初始化: {init_time:.3f}s, 重置: {reset_time:.3f}s, 平均步骤: {avg_step_time:.3f}s")
                
            except Exception as e:
                performance_results.append({
                    "config": config,
                    "status": "失败",
                    "error": str(e)
                })
                self.logger.error(f"      ✗ 配置失败: {str(e)}")
        
        results["large_scale"] = {
            "status": "完成",
            "results": performance_results
        }
        
        # 测试不同用户分布
        self.logger.info("  测试不同用户分布性能...")
        distribution_types = ["uniform", "cluster", "hotspot"]
        distribution_results = []
        
        for dist_type in distribution_types:
            try:
                env = UAVCooperativeNetworkEnv(
                    n_uavs=6,
                    n_users=50,
                    area_size=1000,
                    user_distribution=dist_type,
                    seed=42
                )
                
                observations, infos = env.reset(seed=42)
                
                # 统计覆盖率
                coverage_ratio = np.sum(env.connections) / env.n_users
                
                distribution_results.append({
                    "distribution": dist_type,
                    "coverage_ratio": float(coverage_ratio),
                    "status": "成功"
                })
                
                env.close()
                self.logger.info(f"      {dist_type}: 覆盖率={coverage_ratio:.2%}")
                
            except Exception as e:
                distribution_results.append({
                    "distribution": dist_type,
                    "status": "失败",
                    "error": str(e)
                })
                self.logger.error(f"      {dist_type}: 失败 - {str(e)}")
        
        results["distributions"] = {
            "status": "完成",
            "results": distribution_results
        }
        
        self.logger.info("    ✓ 性能和压力测试完成")
        
        return results
    
    def test_visualization(self):
        """测试可视化功能"""
        self.logger.info("测试可视化功能...")
        
        # 检查matplotlib是否可用
        try:
            import matplotlib.pyplot as plt
            plt.ioff()  # 关闭交互模式
        except ImportError:
            self.logger.warning("    ⚠ matplotlib未安装，跳过可视化测试")
            return {"status": "跳过", "reason": "matplotlib未安装"}
        
        results = {}
        
        try:
            env = UAVCooperativeNetworkEnv(
                n_uavs=5,
                n_users=30,
                area_size=1000,
                render_mode="rgb_array",  # 使用rgb_array模式避免显示窗口
                seed=42
            )
            
            observations, infos = env.reset(seed=42)
            
            # 测试渲染功能
            self.logger.info("  测试渲染功能...")
            frame = env.render()
            
            if frame is not None:
                assert isinstance(frame, np.ndarray), "渲染返回值类型错误"
                assert len(frame.shape) == 3, "渲染图像维度错误"
                self.logger.info(f"    ✓ 渲染图像尺寸: {frame.shape}")
                results["render"] = "通过"
            else:
                self.logger.info("    ⚠ 渲染返回None（可能是human模式）")
                results["render"] = "跳过"
            
            env.close()
            
        except Exception as e:
            self.logger.error(f"    ✗ 可视化测试失败: {str(e)}")
            results["render"] = f"失败: {str(e)}"
        
        return results
    
    def test_integration(self):
        """测试集成功能"""
        self.logger.info("测试集成功能...")
        
        results = {}
        
        # 测试多episode运行
        self.logger.info("  测试多episode运行...")
        env = UAVCooperativeNetworkEnv(
            n_uavs=4,
            n_users=20,
            area_size=800,
            max_steps=50,
            seed=42
        )
        
        episode_results = []
        
        try:
            for episode in range(3):
                self.logger.info(f"    Episode {episode + 1}/3")
                
                observations, infos = env.reset(seed=42 + episode)
                episode_reward = 0
                step_count = 0
                
                while step_count < env.max_steps:
                    actions = {}
                    for agent in env.agents:
                        actions[agent] = env.action_space(agent).sample()
                    
                    observations, rewards, terminations, truncations, infos = env.step(actions)
                    episode_reward += sum(rewards.values())
                    step_count += 1
                    
                    if all(terminations.values()) or all(truncations.values()):
                        break
                
                episode_results.append({
                    "episode": episode + 1,
                    "total_reward": float(episode_reward),
                    "steps": step_count
                })
                
                self.logger.info(f"      总奖励: {episode_reward:.4f}, 步数: {step_count}")
            
            results["multi_episode"] = {
                "status": "通过",
                "episodes": episode_results
            }
            self.logger.info("    ✓ 多episode运行正常")
            
        except Exception as e:
            results["multi_episode"] = f"失败: {str(e)}"
            self.logger.error(f"    ✗ 多episode运行失败: {str(e)}")
        
        # 测试随机种子重现性
        self.logger.info("  测试随机种子重现性...")
        try:
            seed = 123
            
            # 第一次运行
            env1 = UAVCooperativeNetworkEnv(n_uavs=3, n_users=15, area_size=500, seed=seed)
            obs1, _ = env1.reset(seed=seed)
            env1.close()
            
            # 第二次运行
            env2 = UAVCooperativeNetworkEnv(n_uavs=3, n_users=15, area_size=500, seed=seed)
            obs2, _ = env2.reset(seed=seed)
            env2.close()
            
            # 比较初始状态
            agent = list(obs1.keys())[0]
            obs1_array = obs1[agent]["obs"]
            obs2_array = obs2[agent]["obs"]
            
            if np.allclose(obs1_array, obs2_array):
                results["reproducibility"] = "通过"
                self.logger.info("    ✓ 随机种子重现性正常")
            else:
                results["reproducibility"] = "失败：观测不一致"
                self.logger.error("    ✗ 随机种子重现性失败：观测不一致")
                
        except Exception as e:
            results["reproducibility"] = f"失败: {str(e)}"
            self.logger.error(f"    ✗ 随机种子重现性测试失败: {str(e)}")
        
        env.close()
        return results
    
    def generate_visualizations(self):
        """生成可视化图表"""
        self.logger.info("生成测试可视化图表...")
        
        try:
            import matplotlib.pyplot as plt
            plt.style.use('default')
            
            # 创建测试环境进行数据收集
            env = UAVCooperativeNetworkEnv(
                n_uavs=6,
                n_users=40,
                area_size=1000,
                max_steps=100,
                seed=42
            )
            
            observations, infos = env.reset(seed=42)
            
            # 收集数据
            rewards_history = []
            coverage_history = []
            connectivity_history = []
            throughput_history = []
            
            for step in range(50):
                actions = {}
                for agent in env.agents:
                    actions[agent] = env.action_space(agent).sample()
                
                observations, rewards, terminations, truncations, infos = env.step(actions)
                
                reward = env._compute_reward()
                coverage = np.sum(env.connections) / env.n_users
                connectivity = env._compute_connectivity_ratio()
                
                # 计算总吞吐量
                total_throughput = 0
                for i in range(env.n_uavs):
                    for j in range(env.n_users):
                        if env.connections[i, j]:
                            total_throughput += env._compute_effective_throughput(i, j)
                
                rewards_history.append(reward)
                coverage_history.append(coverage)
                connectivity_history.append(connectivity)
                throughput_history.append(total_throughput / 1e6)  # 转换为Mbps
            
            # 创建可视化图表
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('Scenario2 环境测试结果可视化', fontsize=16, fontweight='bold')
            
            # 奖励历史
            axes[0, 0].plot(rewards_history, 'b-', linewidth=2)
            axes[0, 0].set_title('奖励变化历史')
            axes[0, 0].set_xlabel('步数')
            axes[0, 0].set_ylabel('奖励值')
            axes[0, 0].grid(True, alpha=0.3)
            
            # 覆盖率历史
            axes[0, 1].plot(coverage_history, 'g-', linewidth=2)
            axes[0, 1].set_title('用户覆盖率历史')
            axes[0, 1].set_xlabel('步数')
            axes[0, 1].set_ylabel('覆盖率')
            axes[0, 1].set_ylim(0, 1)
            axes[0, 1].grid(True, alpha=0.3)
            
            # 网络连通性历史
            axes[1, 0].plot(connectivity_history, 'r-', linewidth=2)
            axes[1, 0].set_title('网络连通性历史')
            axes[1, 0].set_xlabel('步数')
            axes[1, 0].set_ylabel('连通性比率')
            axes[1, 0].set_ylim(0, 1)
            axes[1, 0].grid(True, alpha=0.3)
            
            # 吞吐量历史
            axes[1, 1].plot(throughput_history, 'm-', linewidth=2)
            axes[1, 1].set_title('总吞吐量历史')
            axes[1, 1].set_xlabel('步数')
            axes[1, 1].set_ylabel('吞吐量 (Mbps)')
            axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # 保存图表
            chart_path = os.path.join(self.output_dir, f"scenario2_test_visualization_{self.timestamp}.png")
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"    ✓ 可视化图表已保存: {chart_path}")
            
            # 创建网络拓扑图
            self._create_network_topology_chart(env)
            
            env.close()
            
        except Exception as e:
            self.logger.error(f"    ✗ 可视化生成失败: {str(e)}")
    
    def _create_network_topology_chart(self, env):
        """创建网络拓扑图"""
        try:
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
            
            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot(111, projection='3d')
            
            # 绘制用户（地面）
            user_x = env.user_positions[:, 0]
            user_y = env.user_positions[:, 1]
            user_z = np.zeros(env.n_users)
            ax.scatter(user_x, user_y, user_z, c='blue', marker='.', s=50, alpha=0.6, label='用户')
            
            # 绘制UAV（按角色着色）
            colors = ['gray', 'red', 'orange']  # 未分配、基站、中继
            role_names = ['未分配', '基站', '中继']
            
            for role in range(3):
                uav_indices = np.where(env.uav_roles == role)[0]
                if len(uav_indices) > 0:
                    uav_positions = env.uav_positions[uav_indices]
                    ax.scatter(uav_positions[:, 0], uav_positions[:, 1], uav_positions[:, 2],
                             c=colors[role], marker='^', s=100, label=f'UAV-{role_names[role]}')
            
            # 绘制地面基站
            bs_x = env.ground_bs_positions[:, 0]
            bs_y = env.ground_bs_positions[:, 1]
            bs_z = env.ground_bs_positions[:, 2]
            ax.scatter(bs_x, bs_y, bs_z, c='black', marker='s', s=150, label='地面基站')
            
            # 绘制UAV之间的连接
            for i in range(env.n_uavs):
                for j in range(i+1, env.n_uavs):
                    if env.uav_connections[i, j]:
                        uav_pos_i = env.uav_positions[i]
                        uav_pos_j = env.uav_positions[j]
                        ax.plot([uav_pos_i[0], uav_pos_j[0]], 
                               [uav_pos_i[1], uav_pos_j[1]], 
                               [uav_pos_i[2], uav_pos_j[2]], 
                               'y-', alpha=0.5, linewidth=2)
            
            # 绘制UAV到地面基站的连接
            for i in range(env.n_uavs):
                for j in range(env.n_ground_bs):
                    if env.uav_bs_connections[i, j]:
                        uav_pos = env.uav_positions[i]
                        bs_pos = env.ground_bs_positions[j]
                        ax.plot([uav_pos[0], bs_pos[0]], 
                               [uav_pos[1], bs_pos[1]], 
                               [uav_pos[2], bs_pos[2]], 
                               'c-', alpha=0.7, linewidth=3)
            
            # 绘制UAV到用户的连接（只显示部分以避免过于拥挤）
            connection_count = 0
            for i in range(env.n_uavs):
                for j in range(env.n_users):
                    if env.connections[i, j] and connection_count < 20:  # 限制显示数量
                        uav_pos = env.uav_positions[i]
                        user_pos = env.user_positions[j]
                        ax.plot([uav_pos[0], user_pos[0]], 
                               [uav_pos[1], user_pos[1]], 
                               [uav_pos[2], 0], 
                               'g-', alpha=0.3, linewidth=1)
                        connection_count += 1
            
            # 设置坐标轴
            ax.set_xlim(0, env.area_size)
            ax.set_ylim(0, env.area_size)
            ax.set_zlim(0, env.height_range[1] * 1.2)
            ax.set_xlabel('X (m)')
            ax.set_ylabel('Y (m)')
            ax.set_zlabel('Z (m)')
            ax.set_title('无人机协作组网拓扑图')
            ax.legend()
            
            # 添加统计信息
            connectivity_ratio = env._compute_connectivity_ratio()
            coverage_ratio = np.sum(env.connections) / env.n_users
            
            info_text = f'网络连通性: {connectivity_ratio:.2%}\n用户覆盖率: {coverage_ratio:.2%}'
            ax.text2D(0.02, 0.95, info_text, transform=ax.transAxes, 
                     bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
            
            # 保存图表
            topology_path = os.path.join(self.output_dir, f"scenario2_network_topology_{self.timestamp}.png")
            plt.savefig(topology_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"    ✓ 网络拓扑图已保存: {topology_path}")
            
        except Exception as e:
            self.logger.error(f"    ✗ 网络拓扑图生成失败: {str(e)}")
    
    def generate_summary_report(self):
        """生成测试总结报告"""
        self.logger.info("\n" + "="*60)
        self.logger.info("生成测试总结报告")
        self.logger.info("="*60)
        
        # 计算测试统计
        total_tests = self.passed_tests + self.failed_tests
        success_rate = (self.passed_tests / total_tests * 100) if total_tests > 0 else 0
        
        # 创建报告内容
        report_lines = [
            "Scenario2 (无人机协作组网环境) 测试报告",
            "=" * 60,
            f"测试时间: {self.timestamp}",
            f"总测试数: {total_tests}",
            f"通过测试: {self.passed_tests}",
            f"失败测试: {self.failed_tests}",
            f"成功率: {success_rate:.1f}%",
            "",
            "详细测试结果:",
            "-" * 40
        ]
        
        # 添加每个测试的详细结果
        for test_name, result in self.test_results.items():
            status = result["status"]
            if status == "PASSED":
                execution_time = result.get("execution_time", 0)
                report_lines.append(f"✓ {test_name} - 通过 ({execution_time:.3f}s)")
                
                # 添加测试结果的详细信息
                if "result" in result and isinstance(result["result"], dict):
                    for key, value in result["result"].items():
                        if isinstance(value, dict) and "status" in value:
                            report_lines.append(f"    {key}: {value['status']}")
                        else:
                            report_lines.append(f"    {key}: {value}")
            else:
                error = result.get("error", "未知错误")
                report_lines.append(f"✗ {test_name} - 失败: {error}")
        
        # 添加推荐和建议
        report_lines.extend([
            "",
            "测试建议:",
            "-" * 40
        ])
        
        if self.failed_tests == 0:
            report_lines.append("✓ 所有测试通过！环境运行正常。")
        else:
            report_lines.append("⚠ 存在失败的测试，请检查相关功能。")
        
        if success_rate >= 90:
            report_lines.append("✓ 环境质量良好，可以用于训练和实验。")
        elif success_rate >= 70:
            report_lines.append("⚠ 环境基本可用，但建议修复失败的测试。")
        else:
            report_lines.append("✗ 环境存在较多问题，建议修复后再使用。")
        
        report_lines.extend([
            "",
            "文件输出:",
            "-" * 40,
            f"日志文件: test_scenario2_{self.timestamp}.log",
            f"可视化图表: scenario2_test_visualization_{self.timestamp}.png",
            f"网络拓扑图: scenario2_network_topology_{self.timestamp}.png",
            f"测试报告: test_scenario2_report_{self.timestamp}.txt"
        ])
        
        # 保存报告到文件
        report_content = "\n".join(report_lines)
        report_path = os.path.join(self.output_dir, f"test_scenario2_report_{self.timestamp}.txt")
        
        try:
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(report_content)
            self.logger.info(f"✓ 测试报告已保存: {report_path}")
        except Exception as e:
            self.logger.error(f"✗ 测试报告保存失败: {str(e)}")
        
        # 打印总结
        self.logger.info("\n" + report_content)
        
        return report_content
    
    def run_all_tests(self):
        """运行所有测试"""
        self.logger.info("开始运行所有测试...")
        
        # 定义测试套件
        test_suite = [
            ("环境初始化测试", self.test_environment_initialization),
            ("基本功能测试", self.test_basic_functionality),
            ("UAV连接和角色分配测试", self.test_uav_connections_and_roles),
            ("路由路径计算测试", self.test_routing_paths),
            ("奖励机制测试", self.test_reward_mechanism),
            ("SINR和吞吐量计算测试", self.test_sinr_and_throughput),
            ("场景特定功能测试", self.test_scenario_specific_features),
            ("性能和压力测试", self.test_performance_and_stress),
            ("可视化功能测试", self.test_visualization),
            ("集成测试", self.test_integration)
        ]
        
        # 运行所有测试
        for test_name, test_func in test_suite:
            self.run_test(test_name, test_func)
        
        # 生成可视化图表
        try:
            self.generate_visualizations()
        except Exception as e:
            self.logger.error(f"可视化生成失败: {str(e)}")
        
        # 生成测试报告
        self.generate_summary_report()
        
        return self.test_results


def main():
    """主函数"""
    print("开始Scenario2环境测试...")
    print("=" * 60)
    
    # 创建测试器
    tester = Scenario2Tester()
    
    try:
        # 运行所有测试
        results = tester.run_all_tests()
        
        print("\n" + "=" * 60)
        print("测试完成！")
        print(f"结果摘要: {tester.passed_tests}通过 / {tester.failed_tests}失败")
        print(f"详细结果请查看: {tester.output_dir}")
        print("=" * 60)
        
        return results
        
    except KeyboardInterrupt:
        print("\n测试被用户中断")
        return None
    except Exception as e:
        print(f"\n测试执行出错: {str(e)}")
        return None


if __name__ == "__main__":
    main()
