"""
场景2吞吐量测试脚本
专门测试UAV协作组网环境中吞吐量是否符合回程容量限制

重点验证：
1. 回程容量计算正确性
2. 有效吞吐量的回程限制应用
3. 多跳路径瓶颈识别
4. 系统级吞吐量统计准确性
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 添加环境路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from envs.pettingzoo.scenario2 import UAVCooperativeNetworkEnv

class ThroughputTester:
    """吞吐量测试器"""
    
    def __init__(self):
        self.test_results = {}
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def create_test_env(self, **kwargs):
        """创建测试环境"""
        default_params = {
            'n_uavs': 5,
            'n_users': 20,
            'area_size': 1000,
            'height_range': (50, 150),
            'max_speed': 30,
            'time_step': 1.0,
            'max_steps': 100,
            'user_distribution': "uniform",
            'channel_model': "free_space",
            'render_mode': None,
            'seed': 42,
            'min_sinr': 0,
            'max_connections': 10,
            'max_hops': 3,
            'n_ground_bs': 1,
        }
        default_params.update(kwargs)
        return UAVCooperativeNetworkEnv(**default_params)
    
    def test_backhaul_capacity_calculation(self):
        """测试1: 回程容量计算正确性"""
        print("=== 测试1: 回程容量计算正确性 ===")
        
        env = self.create_test_env(n_uavs=3, n_users=10, max_hops=3)
        obs, info = env.reset()
        
        # 手动设置UAV位置创建不同的网络拓扑
        test_cases = [
            {
                'name': '直连基站场景',
                'uav_positions': np.array([
                    [500, 500, 100],  # UAV0 - 接近基站中心
                    [200, 200, 80],   # UAV1 - 远离基站
                    [800, 800, 120],  # UAV2 - 远离基站
                ]),
                'expected_direct_connection': [True, False, False]
            },
            {
                'name': '多跳路径场景',
                'uav_positions': np.array([
                    [500, 500, 100],  # UAV0 - 接近基站
                    [600, 600, 90],   # UAV1 - 中间节点
                    [700, 700, 110],  # UAV2 - 末端节点
                ]),
                'expected_direct_connection': [True, False, False]
            },
            {
                'name': '长距离场景',
                'uav_positions': np.array([
                    [500, 500, 100],  # UAV0 - 接近基站
                    [100, 100, 80],   # UAV1 - 很远
                    [900, 900, 120],  # UAV2 - 很远
                ]),
                'expected_direct_connection': [True, False, False]
            }
        ]
        
        results = []
        
        for case in test_cases:
            print(f"\n--- {case['name']} ---")
            
            # 设置UAV位置
            env.uav_positions = case['uav_positions'].copy()
            
            # 更新连接和路由
            env._update_uav_connections()
            env._assign_uav_roles()
            env._compute_routing_paths()
            
            case_result = {
                'name': case['name'],
                'uav_positions': case['uav_positions'].copy(),
                'backhaul_capacities': [],
                'direct_connections': [],
                'routing_paths': {},
                'validation_results': []
            }
            
            for i in range(env.n_uavs):
                # 检查直连状态
                direct_connected = np.any(env.uav_bs_connections[i])
                case_result['direct_connections'].append(direct_connected)
                
                # 计算回程容量
                backhaul_capacity = env._compute_backhaul_capacity(i)
                case_result['backhaul_capacities'].append(backhaul_capacity)
                
                # 记录路由路径
                if i in env.routing_paths:
                    case_result['routing_paths'][i] = env.routing_paths[i].copy()
                
                print(f"UAV {i}:")
                print(f"  位置: {case['uav_positions'][i]}")
                print(f"  直连基站: {direct_connected}")
                print(f"  回程容量: {backhaul_capacity/1e6:.2f} Mbps")
                if i in env.routing_paths:
                    print(f"  路由路径: {env.routing_paths[i]}")
                else:
                    print(f"  路由路径: 无路径")
                
                # 验证回程容量的合理性
                validation_msg = []
                if direct_connected:
                    if backhaul_capacity <= 0:
                        validation_msg.append("❌ 直连基站但回程容量为0")
                    else:
                        validation_msg.append("✅ 直连基站且有回程容量")
                else:
                    if i in env.routing_paths:
                        if backhaul_capacity <= 0:
                            validation_msg.append("❌ 有路径但回程容量为0")
                        else:
                            validation_msg.append("✅ 有路径且有回程容量")
                    else:
                        if backhaul_capacity == 0:
                            validation_msg.append("✅ 无路径且回程容量为0")
                        else:
                            validation_msg.append("❌ 无路径但回程容量不为0")
                
                case_result['validation_results'].extend(validation_msg)
                for msg in validation_msg:
                    print(f"  {msg}")
            
            results.append(case_result)
        
        self.test_results['backhaul_capacity_test'] = results
        return results
    
    def test_effective_throughput_constraints(self):
        """测试2: 有效吞吐量的回程限制验证"""
        print("\n=== 测试2: 有效吞吐量的回程限制验证 ===")
        
        env = self.create_test_env(n_uavs=2, n_users=5, max_hops=2)
        obs, info = env.reset()
        
        # 设置特定的网络拓扑
        env.uav_positions = np.array([
            [500, 500, 100],  # UAV0 - 接近基站
            [600, 600, 90],   # UAV1 - 通过UAV0连基站
        ])
        
        # 设置用户位置（分别靠近不同UAV）
        env.user_positions = np.array([
            [480, 480],  # 用户0 - 靠近UAV0
            [490, 490],  # 用户1 - 靠近UAV0
            [580, 580],  # 用户2 - 靠近UAV1
            [590, 590],  # 用户3 - 靠近UAV1
            [600, 610],  # 用户4 - 靠近UAV1
        ])
        
        # 更新环境状态
        env._update_channel_state()
        env._update_uav_connections()
        env._assign_uav_roles()
        env._compute_routing_paths()
        
        results = {
            'uav_analysis': [],
            'constraint_violations': [],
            'system_analysis': {}
        }
        
        print("\n--- 逐UAV分析 ---")
        system_frontend_demand = 0
        system_backhaul_capacity = 0
        system_effective_throughput = 0
        
        for i in range(env.n_uavs):
            print(f"\nUAV {i} 分析:")
            
            # 计算前端需求总量（UAV到所有连接用户的吞吐量需求）
            frontend_demand = 0
            connected_users = []
            for j in range(env.n_users):
                if env.connections[i, j]:
                    user_throughput = env._compute_throughput(i, j)
                    frontend_demand += user_throughput
                    connected_users.append(j)
                    print(f"  连接用户{j}: {user_throughput/1e6:.2f} Mbps")
            
            # 计算回程容量
            backhaul_capacity = env._compute_backhaul_capacity(i)
            
            # 计算每个用户的有效吞吐量
            total_effective_throughput = 0
            for j in connected_users:
                effective_throughput = env._compute_effective_throughput(i, j)
                total_effective_throughput += effective_throughput
                print(f"  用户{j}有效吞吐量: {effective_throughput/1e6:.2f} Mbps")
            
            print(f"  前端需求总量: {frontend_demand/1e6:.2f} Mbps")
            print(f"  回程容量: {backhaul_capacity/1e6:.2f} Mbps")
            print(f"  有效吞吐量总量: {total_effective_throughput/1e6:.2f} Mbps")
            
            # 验证约束条件
            constraint_check = []
            
            # 检查1: 有效吞吐量不应超过回程容量
            if total_effective_throughput > backhaul_capacity * 1.01:  # 允许1%的数值误差
                constraint_check.append(f"❌ 有效吞吐量({total_effective_throughput/1e6:.2f})超过回程容量({backhaul_capacity/1e6:.2f})")
            else:
                constraint_check.append(f"✅ 有效吞吐量符合回程容量限制")
            
            # 检查2: 有效吞吐量不应超过前端需求
            if total_effective_throughput > frontend_demand * 1.01:
                constraint_check.append(f"❌ 有效吞吐量({total_effective_throughput/1e6:.2f})超过前端需求({frontend_demand/1e6:.2f})")
            else:
                constraint_check.append(f"✅ 有效吞吐量不超过前端需求")
            
            # 检查3: 有效吞吐量应该等于min(前端需求, 回程容量)
            expected_effective = min(frontend_demand, backhaul_capacity)
            if abs(total_effective_throughput - expected_effective) > expected_effective * 0.1:  # 允许10%误差
                constraint_check.append(f"❌ 有效吞吐量({total_effective_throughput/1e6:.2f})不等于预期值({expected_effective/1e6:.2f})")
            else:
                constraint_check.append(f"✅ 有效吞吐量符合min(前端需求, 回程容量)原则")
            
            for check in constraint_check:
                print(f"  {check}")
            
            # 记录结果
            uav_result = {
                'uav_id': i,
                'frontend_demand_mbps': frontend_demand / 1e6,
                'backhaul_capacity_mbps': backhaul_capacity / 1e6,
                'effective_throughput_mbps': total_effective_throughput / 1e6,
                'connected_users': connected_users,
                'constraint_checks': constraint_check
            }
            results['uav_analysis'].append(uav_result)
            
            # 累加系统级统计
            system_frontend_demand += frontend_demand
            system_backhaul_capacity += backhaul_capacity
            system_effective_throughput += total_effective_throughput
        
        # 系统级分析
        print(f"\n--- 系统级分析 ---")
        print(f"系统前端需求总量: {system_frontend_demand/1e6:.2f} Mbps")
        print(f"系统回程容量总量: {system_backhaul_capacity/1e6:.2f} Mbps")
        print(f"系统有效吞吐量总量: {system_effective_throughput/1e6:.2f} Mbps")
        
        # 验证系统级约束
        system_constraint_checks = []
        if system_effective_throughput > system_backhaul_capacity * 1.01:
            system_constraint_checks.append(f"❌ 系统有效吞吐量超过系统回程容量")
        else:
            system_constraint_checks.append(f"✅ 系统有效吞吐量符合系统回程容量限制")
        
        for check in system_constraint_checks:
            print(f"{check}")
        
        results['system_analysis'] = {
            'system_frontend_demand_mbps': system_frontend_demand / 1e6,
            'system_backhaul_capacity_mbps': system_backhaul_capacity / 1e6,
            'system_effective_throughput_mbps': system_effective_throughput / 1e6,
            'system_constraint_checks': system_constraint_checks
        }
        
        self.test_results['effective_throughput_test'] = results
        return results
    
    def test_multihop_bottleneck_identification(self):
        """测试3: 多跳路径瓶颈识别"""
        print("\n=== 测试3: 多跳路径瓶颈识别 ===")
        
        env = self.create_test_env(n_uavs=4, n_users=8, max_hops=4)
        obs, info = env.reset()
        
        # 创建一个线性的多跳拓扑: UAV0(基站附近) -> UAV1 -> UAV2 -> UAV3
        env.uav_positions = np.array([
            [500, 500, 100],  # UAV0 - 靠近基站
            [400, 400, 90],   # UAV1 - 第一跳
            [300, 300, 80],   # UAV2 - 第二跳  
            [200, 200, 70],   # UAV3 - 第三跳（最远）
        ])
        
        # 设置用户靠近UAV3，这样UAV3需要通过多跳回传
        env.user_positions = np.array([
            [190, 190], [210, 210], [180, 220], [220, 180],  # 靠近UAV3
            [490, 490], [510, 510], [480, 520], [520, 480],  # 靠近UAV0
        ])
        
        # 更新环境状态
        env._update_channel_state()
        env._update_uav_connections()
        env._assign_uav_roles()
        env._compute_routing_paths()
        
        results = {
            'topology_analysis': {},
            'bottleneck_analysis': [],
            'path_analysis': {}
        }
        
        print("\n--- 网络拓扑分析 ---")
        print("UAV连接矩阵:")
        for i in range(env.n_uavs):
            connections = [j for j in range(env.n_uavs) if env.uav_connections[i, j]]
            print(f"  UAV{i} 连接到: {connections}")
        
        print("\n路由路径:")
        for i in range(env.n_uavs):
            if i in env.routing_paths:
                print(f"  UAV{i} -> 基站: {env.routing_paths[i]}")
            else:
                print(f"  UAV{i} -> 基站: 无路径")
        
        results['topology_analysis'] = {
            'uav_connections': env.uav_connections.tolist(),
            'routing_paths': {k: v for k, v in env.routing_paths.items()}
        }
        
        print("\n--- 瓶颈分析 ---")
        
        # 对每个有多跳路径的UAV分析瓶颈
        for i in range(env.n_uavs):
            if i not in env.routing_paths:
                continue
                
            path = env.routing_paths[i]
            if len(path) <= 1:  # 直连或无路径
                continue
                
            print(f"\nUAV{i} 多跳路径分析:")
            print(f"  路径: {path}")
            
            # 计算路径上每一跳的容量
            link_capacities = []
            
            for hop_idx in range(len(path) - 1):
                current_node = path[hop_idx]
                next_node = path[hop_idx + 1]
                
                print(f"  跳 {hop_idx + 1}: {current_node} -> {next_node}")
                
                if current_node[0] == "uav" and next_node[0] == "uav":
                    # UAV到UAV的链路
                    uav1_idx = current_node[1]
                    uav2_idx = next_node[1]
                    
                    distance = env._compute_distance(env.uav_positions[uav1_idx], env.uav_positions[uav2_idx])
                    safe_distance = max(distance, 1e-6)
                    path_loss = 20 * np.log10(safe_distance) + 20 * np.log10(4 * np.pi * env.carrier_frequency / 3e8)
                    rx_power = env.tx_power - path_loss
                    sinr_db = rx_power - env.noise_power
                    sinr_linear = 10 ** (sinr_db / 10)
                    link_capacity = env.bandwidth * np.log2(1 + sinr_linear)
                    
                    print(f"    距离: {distance:.1f}m, 容量: {link_capacity/1e6:.2f} Mbps")
                    
                elif current_node[0] == "uav" and next_node[0] == "ground_bs":
                    # UAV到地面基站的链路
                    uav_idx_link = current_node[1]
                    bs_idx = next_node[1]
                    
                    distance = env._compute_distance(env.uav_positions[uav_idx_link], env.ground_bs_positions[bs_idx])
                    safe_distance = max(distance, 1e-6)
                    path_loss = 20 * np.log10(safe_distance) + 20 * np.log10(4 * np.pi * env.carrier_frequency / 3e8)
                    rx_power = env.ground_bs_tx_power - path_loss
                    sinr_db = rx_power - env.noise_power
                    sinr_linear = 10 ** (sinr_db / 10)
                    link_capacity = env.bandwidth * np.log2(1 + sinr_linear)
                    
                    print(f"    距离: {distance:.1f}m, 容量: {link_capacity/1e6:.2f} Mbps")
                
                link_capacities.append(link_capacity)
            
            # 找到瓶颈链路
            min_capacity = min(link_capacities)
            bottleneck_idx = link_capacities.index(min_capacity)
            
            print(f"  瓶颈链路: 跳 {bottleneck_idx + 1}, 容量: {min_capacity/1e6:.2f} Mbps")
            
            # 验证回程容量计算
            calculated_backhaul = env._compute_backhaul_capacity(i)
            hop_efficiency = 1.0 / len(path)
            expected_backhaul = min_capacity * hop_efficiency
            
            print(f"  计算的回程容量: {calculated_backhaul/1e6:.2f} Mbps")
            print(f"  预期回程容量: {expected_backhaul/1e6:.2f} Mbps")
            
            # 验证瓶颈识别正确性
            if abs(calculated_backhaul - expected_backhaul) < expected_backhaul * 0.1:
                print(f"  ✅ 瓶颈识别正确")
            else:
                print(f"  ❌ 瓶颈识别错误")
            
            bottleneck_result = {
                'uav_id': i,
                'path': path,
                'link_capacities_mbps': [c/1e6 for c in link_capacities],
                'bottleneck_hop': bottleneck_idx + 1,
                'bottleneck_capacity_mbps': min_capacity / 1e6,
                'calculated_backhaul_mbps': calculated_backhaul / 1e6,
                'expected_backhaul_mbps': expected_backhaul / 1e6,
                'validation': abs(calculated_backhaul - expected_backhaul) < expected_backhaul * 0.1
            }
            results['bottleneck_analysis'].append(bottleneck_result)
        
        self.test_results['bottleneck_test'] = results
        return results
    
    def test_system_throughput_aggregation(self):
        """测试4: 系统级吞吐量统计准确性"""
        print("\n=== 测试4: 系统级吞吐量统计准确性 ===")
        
        env = self.create_test_env(n_uavs=3, n_users=12, max_hops=3)
        obs, info = env.reset()
        
        # 执行几步以获得稳定的奖励信息
        for _ in range(5):
            actions = {agent: env.action_space(agent).sample() for agent in env.agents}
            obs, rewards, dones, truncs, infos = env.step(actions)
        
        # 手动计算系统吞吐量进行验证
        manual_system_throughput = 0
        uav_throughputs = []
        
        print("\n--- 手动计算各UAV吞吐量 ---")
        for i in range(env.n_uavs):
            uav_throughput = 0
            connected_users = []
            
            for j in range(env.n_users):
                if env.connections[i, j]:
                    effective_throughput = env._compute_effective_throughput(i, j)
                    uav_throughput += effective_throughput
                    connected_users.append(j)
            
            print(f"UAV{i}: {uav_throughput/1e6:.2f} Mbps (用户: {connected_users})")
            uav_throughputs.append(uav_throughput)
            manual_system_throughput += uav_throughput
        
        print(f"手动计算系统吞吐量: {manual_system_throughput/1e6:.2f} Mbps")
        
        # 从奖励信息中获取系统吞吐量
        if hasattr(env, 'reward_info') and 'system_throughput_mbps' in env.reward_info:
            env_system_throughput = env.reward_info['system_throughput_mbps']
            print(f"环境计算系统吞吐量: {env_system_throughput:.2f} Mbps")
            
            # 验证一致性
            if abs(manual_system_throughput/1e6 - env_system_throughput) < env_system_throughput * 0.05:
                print("✅ 系统吞吐量统计一致")
                consistency_check = True
            else:
                print("❌ 系统吞吐量统计不一致")
                consistency_check = False
        else:
            print("⚠️  环境未提供系统吞吐量信息")
            env_system_throughput = None
            consistency_check = None
        
        # 验证奖励计算中的吞吐量部分
        print("\n--- 奖励计算验证 ---")
        if hasattr(env, 'reward_info'):
            print("奖励信息:")
            for key, value in env.reward_info.items():
                if isinstance(value, (int, float)):
                    print(f"  {key}: {value:.4f}")
        
        results = {
            'manual_system_throughput_mbps': manual_system_throughput / 1e6,
            'env_system_throughput_mbps': env_system_throughput,
            'uav_throughputs_mbps': [t/1e6 for t in uav_throughputs],
            'consistency_check': consistency_check,
            'reward_info': env.reward_info if hasattr(env, 'reward_info') else {}
        }
        
        self.test_results['system_aggregation_test'] = results
        return results
    
    def test_edge_cases(self):
        """测试5: 边界条件测试"""
        print("\n=== 测试5: 边界条件测试 ===")
        
        test_cases = [
            {
                'name': '无连接场景',
                'setup': lambda env: self._setup_no_connections(env),
                'expected_throughput': 0
            },
            {
                'name': '单UAV场景',
                'setup': lambda env: self._setup_single_uav(env),
                'expected_properties': ['有连接', '有回程']
            },
            {
                'name': '最大跳数场景',
                'setup': lambda env: self._setup_max_hops(env),
                'expected_properties': ['路径长度=4']
            },
            {
                'name': '极近距离场景',
                'setup': lambda env: self._setup_very_close(env),
                'expected_properties': ['高容量']
            },
            {
                'name': '极远距离场景',
                'setup': lambda env: self._setup_very_far(env),
                'expected_properties': ['低容量或无连接']
            }
        ]
        
        results = []
        
        for case in test_cases:
            print(f"\n--- {case['name']} ---")
            
            env = self.create_test_env(n_uavs=4, n_users=8, max_hops=4)
            obs, info = env.reset()
            
            # 执行特定设置
            case['setup'](env)
            
            # 更新环境状态
            env._update_channel_state()
            env._update_uav_connections()
            env._assign_uav_roles()
            env._compute_routing_paths()
            
            # 计算吞吐量
            total_throughput = 0
            for i in range(env.n_uavs):
                for j in range(env.n_users):
                    if env.connections[i, j]:
                        effective_throughput = env._compute_effective_throughput(i, j)
                        total_throughput += effective_throughput
            
            print(f"总吞吐量: {total_throughput/1e6:.2f} Mbps")
            print(f"连接数: {np.sum(env.connections)}")
            print(f"有路径的UAV数: {len(env.routing_paths)}")
            
            # 验证预期属性
            validation_results = []
            if 'expected_throughput' in case:
                if abs(total_throughput - case['expected_throughput']) < 1e-6:
                    validation_results.append("✅ 吞吐量符合预期")
                else:
                    validation_results.append(f"❌ 吞吐量不符合预期(实际:{total_throughput/1e6:.2f}, 预期:{case['expected_throughput']})")
            
            if 'expected_properties' in case:
                for prop in case['expected_properties']:
                    if prop == '有连接' and np.sum(env.connections) > 0:
                        validation_results.append("✅ 存在连接")
                    elif prop == '有回程' and len(env.routing_paths) > 0:
                        validation_results.append("✅ 存在回程路径")
                    elif prop.startswith('路径长度='):
                        expected_length_str = prop.split('=')[1]
                        if expected_length_str.isdigit():
                            expected_length = int(expected_length_str)
                            max_path_length = max([len(path) for path in env.routing_paths.values()]) if env.routing_paths else 0
                            if max_path_length == expected_length:
                                validation_results.append(f"✅ 最大路径长度为{expected_length}")
                            else:
                                validation_results.append(f"❌ 最大路径长度为{max_path_length}，预期{expected_length}")
                        else:
                            # 如果是变量名，使用环境的max_hops值
                            expected_length = env.max_hops
                            max_path_length = max([len(path) for path in env.routing_paths.values()]) if env.routing_paths else 0
                            if max_path_length <= expected_length:
                                validation_results.append(f"✅ 最大路径长度({max_path_length})不超过限制({expected_length})")
                            else:
                                validation_results.append(f"❌ 最大路径长度({max_path_length})超过限制({expected_length})")
                    elif prop == '高容量' and total_throughput > 100e6:  # >100Mbps
                        validation_results.append("✅ 高容量连接")
                    elif prop == '低容量或无连接' and total_throughput < 10e6:  # <10Mbps
                        validation_results.append("✅ 低容量或无连接")
            
            for result in validation_results:
                print(f"  {result}")
            
            case_result = {
                'name': case['name'],
                'total_throughput_mbps': total_throughput / 1e6,
                'connections_count': int(np.sum(env.connections)),
                'routing_paths_count': len(env.routing_paths),
                'validation_results': validation_results
            }
            results.append(case_result)
        
        self.test_results['edge_cases_test'] = results
        return results
    
    def _setup_no_connections(self, env):
        """设置无连接场景"""
        # 将所有UAV放得很远，无法连接用户和基站
        env.uav_positions = np.array([
            [50, 50, 150],    # 角落
            [50, 950, 150],   # 角落
            [950, 50, 150],   # 角落
            [950, 950, 150],  # 角落
        ])
        env.user_positions = np.array([
            [500, 500], [510, 510], [490, 490], [520, 480],  # 中心区域
            [480, 520], [530, 470], [470, 530], [540, 460],
        ])
    
    def _setup_single_uav(self, env):
        """设置单UAV有效场景"""
        # 只有一个UAV在有效位置，其他都很远
        env.uav_positions = np.array([
            [500, 500, 100],  # 中心有效位置
            [50, 50, 150],    # 远离
            [50, 950, 150],   # 远离
            [950, 950, 150],  # 远离
        ])
        env.user_positions = np.array([
            [480, 480], [520, 520], [490, 510], [510, 490],  # 靠近UAV0
            [50, 50], [950, 950], [50, 950], [950, 50],      # 远离所有UAV
        ])
    
    def _setup_max_hops(self, env):
        """设置最大跳数场景"""
        # 创建链式拓扑达到最大跳数
        env.uav_positions = np.array([
            [500, 500, 100],  # UAV0 - 基站附近
            [400, 400, 100],  # UAV1
            [300, 300, 100],  # UAV2
            [200, 200, 100],  # UAV3 - 最远端
        ])
        env.user_positions = np.array([
            [190, 190], [210, 210], [180, 220], [220, 180],  # 靠近UAV3
            [490, 490], [510, 510], [480, 520], [520, 480],  # 靠近UAV0
        ])
    
    def _setup_very_close(self, env):
        """设置极近距离场景"""
        # 所有UAV都靠近基站
        env.uav_positions = np.array([
            [500, 500, 100],  # 基站中心
            [505, 505, 100],  # 非常靠近
            [495, 495, 100],  # 非常靠近
            [505, 495, 100],  # 非常靠近
        ])
        env.user_positions = np.array([
            [500, 500], [505, 505], [495, 495], [505, 495],
            [495, 505], [510, 510], [490, 490], [510, 490],
        ])
    
    def _setup_very_far(self, env):
        """设置极远距离场景"""
        # 所有UAV都很远离基站
        env.uav_positions = np.array([
            [100, 100, 50],   # 远角落
            [100, 900, 50],   # 远角落
            [900, 100, 50],   # 远角落
            [900, 900, 50],   # 远角落
        ])
        env.user_positions = np.array([
            [90, 90], [110, 110], [90, 110], [110, 90],      # 靠近UAV0
            [890, 890], [910, 910], [890, 910], [910, 890],  # 靠近UAV3
        ])
    
    def generate_report(self):
        """生成测试报告"""
        report_filename = f"throughput_test_report_{self.timestamp}.txt"
        
        with open(report_filename, 'w', encoding='utf-8') as f:
            f.write("场景2吞吐量测试报告\n")
            f.write("=" * 50 + "\n")
            f.write(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # 测试1结果
            if 'backhaul_capacity_test' in self.test_results:
                f.write("测试1: 回程容量计算正确性\n")
                f.write("-" * 30 + "\n")
                for case in self.test_results['backhaul_capacity_test']:
                    f.write(f"场景: {case['name']}\n")
                    for i, capacity in enumerate(case['backhaul_capacities']):
                        f.write(f"  UAV{i}: {capacity/1e6:.2f} Mbps\n")
                    f.write("\n")
            
            # 测试2结果
            if 'effective_throughput_test' in self.test_results:
                f.write("测试2: 有效吞吐量的回程限制验证\n")
                f.write("-" * 30 + "\n")
                test2_data = self.test_results['effective_throughput_test']
                for uav_data in test2_data['uav_analysis']:
                    f.write(f"UAV{uav_data['uav_id']}:\n")
                    f.write(f"  前端需求: {uav_data['frontend_demand_mbps']:.2f} Mbps\n")
                    f.write(f"  回程容量: {uav_data['backhaul_capacity_mbps']:.2f} Mbps\n")
                    f.write(f"  有效吞吐量: {uav_data['effective_throughput_mbps']:.2f} Mbps\n")
                    for check in uav_data['constraint_checks']:
                        f.write(f"  {check}\n")
                    f.write("\n")
                
                sys_data = test2_data['system_analysis']
                f.write("系统级统计:\n")
                f.write(f"  系统前端需求: {sys_data['system_frontend_demand_mbps']:.2f} Mbps\n")
                f.write(f"  系统回程容量: {sys_data['system_backhaul_capacity_mbps']:.2f} Mbps\n")
                f.write(f"  系统有效吞吐量: {sys_data['system_effective_throughput_mbps']:.2f} Mbps\n")
                f.write("\n")
            
            # 测试3结果
            if 'bottleneck_test' in self.test_results:
                f.write("测试3: 多跳路径瓶颈识别\n")
                f.write("-" * 30 + "\n")
                for bottleneck in self.test_results['bottleneck_test']['bottleneck_analysis']:
                    f.write(f"UAV{bottleneck['uav_id']}:\n")
                    f.write(f"  路径: {bottleneck['path']}\n")
                    f.write(f"  瓶颈跳: {bottleneck['bottleneck_hop']}\n")
                    f.write(f"  瓶颈容量: {bottleneck['bottleneck_capacity_mbps']:.2f} Mbps\n")
                    f.write(f"  验证结果: {'通过' if bottleneck['validation'] else '失败'}\n")
                    f.write("\n")
            
            # 测试4结果
            if 'system_aggregation_test' in self.test_results:
                f.write("测试4: 系统级吞吐量统计准确性\n")
                f.write("-" * 30 + "\n")
                test4_data = self.test_results['system_aggregation_test']
                f.write(f"手动计算: {test4_data['manual_system_throughput_mbps']:.2f} Mbps\n")
                if test4_data['env_system_throughput_mbps'] is not None:
                    f.write(f"环境计算: {test4_data['env_system_throughput_mbps']:.2f} Mbps\n")
                    f.write(f"一致性检查: {'通过' if test4_data['consistency_check'] else '失败'}\n")
                f.write("\n")
            
            # 测试5结果
            if 'edge_cases_test' in self.test_results:
                f.write("测试5: 边界条件测试\n")
                f.write("-" * 30 + "\n")
                for case in self.test_results['edge_cases_test']:
                    f.write(f"场景: {case['name']}\n")
                    f.write(f"  总吞吐量: {case['total_throughput_mbps']:.2f} Mbps\n")
                    f.write(f"  连接数: {case['connections_count']}\n")
                    f.write(f"  路径数: {case['routing_paths_count']}\n")
                    for result in case['validation_results']:
                        f.write(f"  {result}\n")
                    f.write("\n")
        
        print(f"\n测试报告已保存: {report_filename}")
        return report_filename
    
    def visualize_results(self):
        """可视化测试结果"""
        if not self.test_results:
            print("没有测试结果可供可视化")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('场景2吞吐量测试结果可视化', fontsize=16)
        
        # 图1: 回程容量对比
        if 'backhaul_capacity_test' in self.test_results:
            ax1 = axes[0, 0]
            test1_data = self.test_results['backhaul_capacity_test']
            
            scenarios = []
            uav_capacities = []
            
            for case in test1_data:
                scenarios.append(case['name'])
                uav_capacities.append([c/1e6 for c in case['backhaul_capacities']])
            
            x = np.arange(len(scenarios))
            width = 0.2
            
            for i in range(len(uav_capacities[0])):
                capacities = [cap[i] if i < len(cap) else 0 for cap in uav_capacities]
                ax1.bar(x + i*width, capacities, width, label=f'UAV{i}')
            
            ax1.set_xlabel('测试场景')
            ax1.set_ylabel('回程容量 (Mbps)')
            ax1.set_title('各场景回程容量对比')
            ax1.set_xticks(x + width)
            ax1.set_xticklabels(scenarios, rotation=45)
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        
        # 图2: 有效吞吐量 vs 回程容量
        if 'effective_throughput_test' in self.test_results:
            ax2 = axes[0, 1]
            test2_data = self.test_results['effective_throughput_test']
            
            uav_ids = []
            frontend_demands = []
            backhaul_capacities = []
            effective_throughputs = []
            
            for uav_data in test2_data['uav_analysis']:
                uav_ids.append(f"UAV{uav_data['uav_id']}")
                frontend_demands.append(uav_data['frontend_demand_mbps'])
                backhaul_capacities.append(uav_data['backhaul_capacity_mbps'])
                effective_throughputs.append(uav_data['effective_throughput_mbps'])
            
            x = np.arange(len(uav_ids))
            width = 0.25
            
            ax2.bar(x - width, frontend_demands, width, label='前端需求', alpha=0.8)
            ax2.bar(x, backhaul_capacities, width, label='回程容量', alpha=0.8)
            ax2.bar(x + width, effective_throughputs, width, label='有效吞吐量', alpha=0.8)
            
            ax2.set_xlabel('UAV')
            ax2.set_ylabel('吞吐量 (Mbps)')
            ax2.set_title('前端需求 vs 回程容量 vs 有效吞吐量')
            ax2.set_xticks(x)
            ax2.set_xticklabels(uav_ids)
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # 图3: 多跳路径瓶颈分析
        if 'bottleneck_test' in self.test_results:
            ax3 = axes[1, 0]
            test3_data = self.test_results['bottleneck_test']
            
            if 'bottleneck_analysis' in test3_data and test3_data['bottleneck_analysis']:
                uav_ids = []
                bottleneck_capacities = []
                calculated_backhual = []
                
                for bottleneck in test3_data['bottleneck_analysis']:
                    uav_ids.append(f"UAV{bottleneck['uav_id']}")
                    bottleneck_capacities.append(bottleneck['bottleneck_capacity_mbps'])
                    calculated_backhual.append(bottleneck['calculated_backhaul_mbps'])
                
                x = np.arange(len(uav_ids))
                width = 0.35
                
                ax3.bar(x - width/2, bottleneck_capacities, width, label='瓶颈容量', alpha=0.8)
                ax3.bar(x + width/2, calculated_backhual, width, label='计算回程容量', alpha=0.8)
                
                ax3.set_xlabel('UAV')
                ax3.set_ylabel('容量 (Mbps)')
                ax3.set_title('多跳路径瓶颈分析')
                ax3.set_xticks(x)
                ax3.set_xticklabels(uav_ids)
                ax3.legend()
                ax3.grid(True, alpha=0.3)
            else:
                ax3.text(0.5, 0.5, '无多跳路径数据', ha='center', va='center', transform=ax3.transAxes)
                ax3.set_title('多跳路径瓶颈分析')
        
        # 图4: 边界条件测试结果
        if 'edge_cases_test' in self.test_results:
            ax4 = axes[1, 1]
            test5_data = self.test_results['edge_cases_test']
            
            case_names = []
            throughputs = []
            connections = []
            
            for case in test5_data:
                case_names.append(case['name'].replace('场景', ''))
                throughputs.append(case['total_throughput_mbps'])
                connections.append(case['connections_count'])
            
            x = np.arange(len(case_names))
            
            # 双y轴
            ax4_twin = ax4.twinx()
            
            bars1 = ax4.bar(x - 0.2, throughputs, 0.4, label='吞吐量 (Mbps)', color='skyblue', alpha=0.8)
            bars2 = ax4_twin.bar(x + 0.2, connections, 0.4, label='连接数', color='lightcoral', alpha=0.8)
            
            ax4.set_xlabel('测试场景')
            ax4.set_ylabel('吞吐量 (Mbps)', color='skyblue')
            ax4_twin.set_ylabel('连接数', color='lightcoral')
            ax4.set_title('边界条件测试结果')
            ax4.set_xticks(x)
            ax4.set_xticklabels(case_names, rotation=45)
            
            # 合并图例
            lines1, labels1 = ax4.get_legend_handles_labels()
            lines2, labels2 = ax4_twin.get_legend_handles_labels()
            ax4.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
            
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图片
        plot_filename = f"throughput_test_visualization_{self.timestamp}.png"
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"可视化结果已保存: {plot_filename}")
        
        plt.show()
        return plot_filename

def main():
    """主函数"""
    print("场景2吞吐量测试开始")
    print("=" * 50)
    
    tester = ThroughputTester()
    
    try:
        # 执行所有测试
        print("开始执行测试套件...")
        
        tester.test_backhaul_capacity_calculation()
        tester.test_effective_throughput_constraints()
        tester.test_multihop_bottleneck_identification()
        tester.test_system_throughput_aggregation()
        tester.test_edge_cases()
        
        # 生成报告
        print("\n" + "="*50)
        print("生成测试报告和可视化结果...")
        
        report_file = tester.generate_report()
        plot_file = tester.visualize_results()
        
        print("\n" + "="*50)
        print("测试完成!")
        print(f"报告文件: {report_file}")
        print(f"可视化文件: {plot_file}")
        
        # 总结关键发现
        print("\n=== 关键发现总结 ===")
        
        # 检查是否有违反回程限制的情况
        violations = []
        if 'effective_throughput_test' in tester.test_results:
            for uav_data in tester.test_results['effective_throughput_test']['uav_analysis']:
                for check in uav_data['constraint_checks']:
                    if '❌' in check:
                        violations.append(f"UAV{uav_data['uav_id']}: {check}")
        
        if violations:
            print("⚠️  发现以下回程容量限制违反:")
            for violation in violations:
                print(f"  {violation}")
        else:
            print("✅ 所有UAV的有效吞吐量都符合回程容量限制")
        
        # 系统级一致性检查
        if 'system_aggregation_test' in tester.test_results:
            consistency = tester.test_results['system_aggregation_test']['consistency_check']
            if consistency:
                print("✅ 系统级吞吐量统计准确")
            elif consistency is False:
                print("❌ 系统级吞吐量统计存在问题")
            else:
                print("⚠️  无法验证系统级吞吐量统计")
        
    except Exception as e:
        print(f"测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 吞吐量测试成功完成!")
    else:
        print("\n❌ 吞吐量测试失败!")
        sys.exit(1)
