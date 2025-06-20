import numpy as np
from envs.pettingzoo.uav_env import MultiUAVEnv

class UAVCooperativeNetworkEnv(MultiUAVEnv):
    """
    场景2：无人机协作组网环境
    
    特点：
    - 无人机可根据情况合作组网，分别担任基站以及中继
    - 需要回程到地面基站
    - 跳数最多为3-5可调
    - 优化目标是最大化用户覆盖率、服务质量和网络连通性
    """
    
    def __init__(
        self,
        n_uavs=5,
        n_users=50,
        area_size=1000,
        height_range=(50, 150),
        max_speed=30,
        time_step=1.0,
        max_steps=5000,
        user_distribution="uniform",
        channel_model="free_space",
        render_mode=None,
        seed=None,
        min_sinr=0,  # 最小SINR阈值 (dB)
        max_connections=10,  # 每个无人机最大连接数
        max_hops=3,  # 最大跳数 (3-5可调)
        coverage_weight=0.4,  # 覆盖率权重
        quality_weight=0.2,  # 服务质量权重
        connectivity_weight=0.2,  # 网络连通性权重
        throughput_weight=0.2,  # 吞吐量权重
        n_ground_bs=1,  # 地面基站数量
    ):
        """
        初始化UAV协作组网环境
        
        参数:
            n_uavs: 无人机数量
            n_users: 用户数量
            area_size: 区域大小 (m)
            height_range: 无人机高度范围 (m)
            max_speed: 最大速度 (m/s)
            time_step: 时间步长 (s)
            max_steps: 最大步数
            user_distribution: 用户分布类型 ("uniform", "cluster", "hotspot")
            channel_model: 信道模型 ("free_space", "urban", "suburban")
            render_mode: 渲染模式
            seed: 随机种子
            min_sinr: 最小SINR阈值 (dB)
            max_connections: 每个无人机最大连接数
            max_hops: 最大跳数 (3-5可调)
            coverage_weight: 覆盖率权重
            quality_weight: 服务质量权重
            connectivity_weight: 网络连通性权重
            throughput_weight: 吞吐量权重
            n_ground_bs: 地面基站数量
        """
        # 先保存关键参数，防止在父类初始化时就需要使用
        self.n_ground_bs = n_ground_bs
        self.max_hops = max_hops
        self.min_sinr = min_sinr
        self.max_connections = max_connections
        self.coverage_weight = coverage_weight
        self.quality_weight = quality_weight
        self.connectivity_weight = connectivity_weight
        self.throughput_weight = throughput_weight
        self.area_size = area_size  # 需要在初始化地面基站之前设置
        
        # 初始化地面基站位置（在调用父类初始化之前）
        self._init_ground_bs()
        
        # 调用父类初始化
        super().__init__(
            n_uavs=n_uavs,
            n_users=n_users,
            area_size=area_size,
            height_range=height_range,
            max_speed=max_speed,
            time_step=time_step,
            max_steps=max_steps,
            user_distribution=user_distribution,
            channel_model=channel_model,
            render_mode=render_mode,
            seed=seed,
        )
        
        # 场景名称
        self.metadata["name"] = "uav_cooperative_network_v0"
        
        # 初始化UAV连接矩阵和角色
        self.uav_connections = np.zeros((self.n_uavs, self.n_uavs), dtype=bool)  # UAV之间的连接
        self.uav_bs_connections = np.zeros((self.n_uavs, self.n_ground_bs), dtype=bool)  # UAV到地面基站的连接
        self.uav_roles = np.zeros(self.n_uavs, dtype=int)  # 0: 未分配, 1: 基站, 2: 中继
        self.routing_paths = {}  # 路由路径 {uav_idx: [path_to_ground_bs]}
        
        # 扩展观测空间
        self.obs_dim += 3 + self.n_ground_bs + 1  # 添加UAV角色(3)、到地面基站的连接(n_ground_bs)和跳数信息(1)
    
    def _init_ground_bs(self):
        """初始化地面基站位置"""
        # 确保总是创建正确数量的地面基站
        self.ground_bs_positions = np.zeros((self.n_ground_bs, 3))
        
        if self.n_ground_bs == 1:
            # 单个地面基站放在中心
            self.ground_bs_positions[0] = [self.area_size/2, self.area_size/2, 30]
        elif self.n_ground_bs == 2:
            # 两个基站分布在对角
            self.ground_bs_positions[0] = [self.area_size * 0.25, self.area_size * 0.25, 30]
            self.ground_bs_positions[1] = [self.area_size * 0.75, self.area_size * 0.75, 30]
        elif self.n_ground_bs == 3:
            # 三个基站三角分布
            self.ground_bs_positions[0] = [self.area_size * 0.5, self.area_size * 0.2, 30]
            self.ground_bs_positions[1] = [self.area_size * 0.2, self.area_size * 0.8, 30]
            self.ground_bs_positions[2] = [self.area_size * 0.8, self.area_size * 0.8, 30]
        elif self.n_ground_bs == 4:
            # 四个角落
            self.ground_bs_positions[0] = [self.area_size * 0.2, self.area_size * 0.2, 30]
            self.ground_bs_positions[1] = [self.area_size * 0.2, self.area_size * 0.8, 30]
            self.ground_bs_positions[2] = [self.area_size * 0.8, self.area_size * 0.2, 30]
            self.ground_bs_positions[3] = [self.area_size * 0.8, self.area_size * 0.8, 30]
        else:
            # 更多地面基站时使用网格或随机分布
            if self.n_ground_bs <= 9:
                # 使用网格分布（最多3x3）
                grid_size = int(np.ceil(np.sqrt(self.n_ground_bs)))
                positions_generated = 0
                
                for i in range(grid_size):
                    for j in range(grid_size):
                        if positions_generated >= self.n_ground_bs:
                            break
                        
                        x = self.area_size * (0.2 + 0.6 * i / max(1, grid_size - 1))
                        y = self.area_size * (0.2 + 0.6 * j / max(1, grid_size - 1))
                        self.ground_bs_positions[positions_generated] = [x, y, 30]
                        positions_generated += 1
                    
                    if positions_generated >= self.n_ground_bs:
                        break
            else:
                # 随机分布
                for i in range(self.n_ground_bs):
                    self.ground_bs_positions[i] = [
                        np.random.uniform(self.area_size * 0.1, self.area_size * 0.9),
                        np.random.uniform(self.area_size * 0.1, self.area_size * 0.9),
                        30  # 固定高度
                    ]
    
    def reset(self, seed=None, options=None):
        """
        重置环境
        
        返回:
            observations: 所有智能体的观测字典
            infos: 所有智能体的信息字典
        """
        # 由于在父类的reset中会用到self.n_ground_bs，我们需要先确保它被正确设置
        # 确保地面基站位置被初始化
        if not hasattr(self, 'ground_bs_positions') or self.ground_bs_positions is None:
            self._init_ground_bs()
            
        # 调用父类的reset
        observations, infos = super().reset(seed, options)
        
        # 重置UAV连接矩阵和角色
        self.uav_connections = np.zeros((self.n_uavs, self.n_uavs), dtype=bool)
        self.uav_bs_connections = np.zeros((self.n_uavs, self.n_ground_bs), dtype=bool)
        self.uav_roles = np.zeros(self.n_uavs, dtype=int)
        self.routing_paths = {}
        
        # 更新UAV连接和角色
        self._update_uav_connections()
        self._assign_uav_roles()
        self._compute_routing_paths()
        
        # 更新观测 (使用字典版本)
        observations = self._update_observations_dict(observations)
        
        return observations, infos
    
    def _update_observations(self, observations):
        """
        更新观测，添加UAV角色和连接信息（用于数组格式的观测）
        
        参数:
            observations: 原始观测
            
        返回:
            updated_observations: 更新后的观测
        """
        updated_observations = []
        
        for i, agent in enumerate(self.agents):
            # 获取原始观测
            obs = observations[i]
            
            # 添加UAV角色信息（独热编码）
            role_onehot = np.zeros(3)  # [未分配, 基站, 中继]
            if self.uav_roles[i] < 3:
                role_onehot[self.uav_roles[i]] = 1
            
            # 添加到地面基站的连接信息
            bs_connections = self.uav_bs_connections[i]
            
            # 添加跳数信息（归一化）
            if i in self.routing_paths:
                hop_count = len(self.routing_paths[i])
                normalized_hop = min(hop_count / self.max_hops, 1.0)
            else:
                normalized_hop = 1.0  # 无路径时设为最大值
            
            # 组合新的观测
            new_obs = np.concatenate([obs, role_onehot, bs_connections, [normalized_hop]])
            updated_observations.append(new_obs)
        
        return np.array(updated_observations)
    
    def _update_observations_dict(self, observations_dict):
        """
        更新观测，添加UAV角色和连接信息（用于字典格式的观测）
        
        参数:
            observations_dict: 原始观测字典
            
        返回:
            updated_observations_dict: 更新后的观测字典
        """
        updated_observations_dict = {}
        
        for i, agent in enumerate(self.agents):
            # 获取原始观测
            obs_dict = observations_dict[agent]
            obs = obs_dict["obs"]
            
            # 添加UAV角色信息（独热编码）
            role_onehot = np.zeros(3)  # [未分配, 基站, 中继]
            if self.uav_roles[i] < 3:
                role_onehot[self.uav_roles[i]] = 1
            
            # 添加到地面基站的连接信息
            bs_connections = self.uav_bs_connections[i]
            
            # 添加跳数信息（归一化）
            if i in self.routing_paths:
                hop_count = len(self.routing_paths[i])
                normalized_hop = min(hop_count / self.max_hops, 1.0)
            else:
                normalized_hop = 1.0  # 无路径时设为最大值
            
            # 组合新的观测
            new_obs = np.concatenate([obs, role_onehot, bs_connections, [normalized_hop]])
            
            # 更新观测字典
            updated_obs_dict = {
                "obs": new_obs,
                "action_mask": obs_dict["action_mask"]
            }
            updated_observations_dict[agent] = updated_obs_dict
        
        return updated_observations_dict
    
    def step(self, actions):
        """
        执行环境步骤
        
        参数:
            actions: 所有智能体的动作字典 {agent_id: action}
            
        返回:
            observations: 所有智能体的下一个观测字典
            rewards: 所有智能体的奖励字典
            terminations: 所有智能体的终止状态字典
            truncations: 所有智能体的截断状态字典
            infos: 所有智能体的信息字典
        """
        # 执行父类的step
        observations, rewards, terminations, truncations, infos = super().step(actions)
        
        # 更新UAV连接和角色
        self._update_uav_connections()
        self._assign_uav_roles()
        self._compute_routing_paths()
        
        # 更新观测
        observations = self._update_observations_dict(observations)
        
        # 计算新的奖励
        global_reward = self._compute_reward()
        
        # 更新每个智能体的奖励
        for agent in self.agents:
            rewards[agent] = global_reward / self.n_uavs
        
        # 添加场景特定信息到每个智能体的info中
        scenario_info = {
            "scenario": "cooperative_network",
            "reward_info": self.reward_info if hasattr(self, "reward_info") else {},
            "coverage_ratio": np.sum(self.connections) / self.n_users if self.n_users > 0 else 0,  # 避免除零错误
            "connectivity_ratio": self._compute_connectivity_ratio(),
            "uav_roles": self.uav_roles.copy(),
            "routing_paths": {k: v.copy() for k, v in self.routing_paths.items()},
        }
        
        for agent in self.agents:
            infos[agent].update(scenario_info)
        
        return observations, rewards, terminations, truncations, infos
    
    def _update_uav_connections(self):
        """更新UAV之间的连接和UAV到地面基站的连接"""
        # 更新UAV之间的连接
        for i in range(self.n_uavs):
            for j in range(i+1, self.n_uavs):
                # 计算UAV之间的距离
                distance = self._compute_distance(self.uav_positions[i], self.uav_positions[j])
                
                # 计算UAV之间的SINR
                # 确保距离不为零，避免log10(0)错误
                safe_distance = max(distance, 1e-6)  # 使用一个很小的正数代替零
                path_loss = 20 * np.log10(safe_distance) + 20 * np.log10(4 * np.pi * self.carrier_frequency / 3e8)
                rx_power = self.tx_power - path_loss
                sinr = rx_power - self.noise_power
                
                # 如果SINR大于阈值，则建立连接
                if sinr >= self.min_sinr:
                    self.uav_connections[i, j] = True
                    self.uav_connections[j, i] = True
                else:
                    self.uav_connections[i, j] = False
                    self.uav_connections[j, i] = False
        
        # 更新UAV到地面基站的连接
        for i in range(self.n_uavs):
            for j in range(self.n_ground_bs):
                # 计算UAV到地面基站的距离
                distance = self._compute_distance(self.uav_positions[i], self.ground_bs_positions[j])
                
                # 计算UAV到地面基站的SINR
                # 确保距离不为零，避免log10(0)错误
                safe_distance = max(distance, 1e-6)  # 使用一个很小的正数代替零
                path_loss = 20 * np.log10(safe_distance) + 20 * np.log10(4 * np.pi * self.carrier_frequency / 3e8)
                rx_power = self.ground_bs_tx_power - path_loss  # 地面基站发射功率更高
                sinr = rx_power - self.noise_power
                
                # 如果SINR大于阈值，则建立连接
                if sinr >= self.min_sinr:
                    self.uav_bs_connections[i, j] = True
                else:
                    self.uav_bs_connections[i, j] = False
    
    def _assign_uav_roles(self):
        """
        分配UAV角色（基站或中继）
        
        策略：
        1. 直接连接到地面基站的UAV可以作为基站或中继
        2. 不直接连接到地面基站但能通过其他UAV连接的UAV作为基站
        3. 其余UAV作为未分配
        """
        # 重置角色
        self.uav_roles = np.zeros(self.n_uavs, dtype=int)
        
        # 计算每个UAV连接的用户数
        uav_user_counts = np.sum(self.connections, axis=1)
        
        # 首先标记直接连接到地面基站的UAV
        direct_bs_connected = np.any(self.uav_bs_connections, axis=1)
        
        # 根据连接的用户数和到地面基站的连接情况分配角色
        for i in range(self.n_uavs):
            if direct_bs_connected[i]:
                # 直接连接到地面基站的UAV
                if uav_user_counts[i] > 0:
                    # 如果连接了用户，则作为基站
                    self.uav_roles[i] = 1  # 基站
                else:
                    # 如果没有连接用户，则作为中继
                    self.uav_roles[i] = 2  # 中继
            else:
                # 不直接连接到地面基站的UAV
                if uav_user_counts[i] > 0:
                    # 如果连接了用户，则作为基站
                    self.uav_roles[i] = 1  # 基站
                else:
                    # 如果没有连接用户，则作为未分配
                    self.uav_roles[i] = 0  # 未分配
    
    def _compute_routing_paths(self):
        """
        计算每个UAV到地面基站的路由路径
        
        使用广度优先搜索找到最短路径
        """
        self.routing_paths = {}
        
        # 对每个UAV计算到地面基站的路径
        for i in range(self.n_uavs):
            # 如果UAV直接连接到地面基站
            if np.any(self.uav_bs_connections[i]):
                # 找到连接的地面基站索引
                bs_idx = np.where(self.uav_bs_connections[i])[0][0]
                self.routing_paths[i] = [("ground_bs", bs_idx)]
                continue
            
            # 否则，使用BFS寻找到地面基站的路径
            path = self._bfs_shortest_path(i)
            if path:
                self.routing_paths[i] = path
    
    def _bfs_shortest_path(self, start_uav):
        """
        使用BFS寻找从UAV到地面基站的最短路径
        
        参数:
            start_uav: 起始UAV索引
            
        返回:
            path: 路径列表 [(node_type, node_idx), ...]，如果没有路径则返回None
        """
        # 初始化队列和访问标记
        queue = [(start_uav, [])]  # (当前节点, 路径)
        visited = set([start_uav])
        
        while queue:
            current, path = queue.pop(0)
            
            # 检查是否直接连接到地面基站
            for bs_idx in range(self.n_ground_bs):
                if self.uav_bs_connections[current, bs_idx]:
                    return path + [("uav", current), ("ground_bs", bs_idx)]
            
            # 检查连接到的其他UAV
            for next_uav in range(self.n_uavs):
                if self.uav_connections[current, next_uav] and next_uav not in visited:
                    if len(path) >= self.max_hops - 1:
                        continue  # 超过最大跳数限制
                    
                    visited.add(next_uav)
                    queue.append((next_uav, path + [("uav", current)]))
        
        return None  # 没有找到路径
    
    def _compute_connectivity_ratio(self):
        """
        计算网络连通性比率
        
        返回:
            connectivity_ratio: 连通性比率 [0,1]
        """
        # 计算有效路由的UAV数量
        connected_uavs = len(self.routing_paths)
        
        # 计算连通性比率
        connectivity_ratio = connected_uavs / self.n_uavs
        
        return connectivity_ratio
    
    def _compute_throughput(self, uav_idx, user_idx):
        """
        计算UAV-用户连接的吞吐量 (bps)
        
        参数:
            uav_idx: 无人机索引
            user_idx: 用户索引
            
        返回:
            throughput: 吞吐量 (bps)
        """
        if not self.connections[uav_idx, user_idx]:
            return 0
        
        # 获取SINR (dB)
        sinr_db = self.sinr_matrix[uav_idx, user_idx]
        
        # 转换为线性单位
        sinr_linear = 10 ** (sinr_db / 10)
        
        # 香农定理计算吞吐量: C = B * log2(1 + SINR_linear)
        throughput = self.bandwidth * np.log2(1 + sinr_linear)
        
        return throughput  # 单位：bps
    
    def _compute_backhaul_capacity(self, uav_idx):
        """
        计算UAV的回程容量
        
        参数:
            uav_idx: 无人机索引
            
        返回:
            backhaul_capacity: 回程容量 (bps)
        """
        if uav_idx not in self.routing_paths:
            return 0
        
        path = self.routing_paths[uav_idx]
        if not path:
            return 0
        
        # 如果直接连接到地面基站
        if len(path) == 1 and path[0][0] == "ground_bs":
            # 使用UAV到地面基站的链路容量
            bs_idx = path[0][1]
            distance = self._compute_distance(self.uav_positions[uav_idx], self.ground_bs_positions[bs_idx])
            safe_distance = max(distance, 1e-6)
            path_loss = 20 * np.log10(safe_distance) + 20 * np.log10(4 * np.pi * self.carrier_frequency / 3e8)
            rx_power = self.ground_bs_tx_power - path_loss
            sinr_db = rx_power - self.noise_power
            sinr_linear = 10 ** (sinr_db / 10)
            backhaul_capacity = self.bandwidth * np.log2(1 + sinr_linear)
        else:
            # 多跳路径，需要找到瓶颈链路
            min_capacity = float('inf')
            
            # 计算路径上每一跳的容量
            for i in range(len(path) - 1):
                current_node = path[i]
                next_node = path[i + 1]
                
                if current_node[0] == "uav" and next_node[0] == "uav":
                    # UAV到UAV的链路
                    uav1_idx = current_node[1]
                    uav2_idx = next_node[1]
                    
                    distance = self._compute_distance(self.uav_positions[uav1_idx], self.uav_positions[uav2_idx])
                    safe_distance = max(distance, 1e-6)
                    path_loss = 20 * np.log10(safe_distance) + 20 * np.log10(4 * np.pi * self.carrier_frequency / 3e8)
                    rx_power = self.tx_power - path_loss
                    sinr_db = rx_power - self.noise_power
                    sinr_linear = 10 ** (sinr_db / 10)
                    link_capacity = self.bandwidth * np.log2(1 + sinr_linear)
                    
                elif current_node[0] == "uav" and next_node[0] == "ground_bs":
                    # UAV到地面基站的链路
                    uav_idx_link = current_node[1]
                    bs_idx = next_node[1]
                    
                    distance = self._compute_distance(self.uav_positions[uav_idx_link], self.ground_bs_positions[bs_idx])
                    safe_distance = max(distance, 1e-6)
                    path_loss = 20 * np.log10(safe_distance) + 20 * np.log10(4 * np.pi * self.carrier_frequency / 3e8)
                    rx_power = self.ground_bs_tx_power - path_loss
                    sinr_db = rx_power - self.noise_power
                    sinr_linear = 10 ** (sinr_db / 10)
                    link_capacity = self.bandwidth * np.log2(1 + sinr_linear)
                else:
                    link_capacity = self.bandwidth  # 默认值
                
                min_capacity = min(min_capacity, link_capacity)
            
            backhaul_capacity = min_capacity if min_capacity != float('inf') else 0
        
        return backhaul_capacity
    
    def _compute_effective_throughput(self, uav_idx, user_idx):
        """
        计算考虑多跳路径的有效吞吐量
        
        参数:
            uav_idx: 无人机索引
            user_idx: 用户索引
            
        返回:
            effective_throughput: 有效吞吐量 (bps)
        """
        if not self.connections[uav_idx, user_idx]:
            return 0
        
        # 基础吞吐量（UAV到用户的链路容量）
        base_throughput = self._compute_throughput(uav_idx, user_idx)
        
        # 如果UAV有到地面基站的路径，考虑回程瓶颈
        if uav_idx in self.routing_paths:
            path = self.routing_paths[uav_idx]
            hop_count = len(path)
            
            # 多跳效率损失：每一跳都会降低有效吞吐量
            hop_efficiency = 1.0 / hop_count if hop_count > 0 else 0
            
            # 考虑回程链路容量限制
            backhaul_capacity = self._compute_backhaul_capacity(uav_idx)
            
            # 有效吞吐量取最小值（瓶颈原则）
            adjusted_throughput = base_throughput * hop_efficiency
            effective_throughput = min(adjusted_throughput, backhaul_capacity)
        else:
            # 无路径到地面基站，吞吐量为0
            effective_throughput = 0
        
        return effective_throughput
    
    def _compute_reward(self):
        """
        计算奖励
        
        返回:
            reward: 全局奖励
        """
        # 基本奖励：已连接用户数（覆盖率）
        connected_users = np.sum(self.connections)
        coverage_ratio = connected_users / self.n_users
        coverage_reward = coverage_ratio
        
        # 服务质量奖励：SINR质量
        total_sinr = 0
        for i in range(self.n_uavs):
            for j in range(self.n_users):
                if self.connections[i, j]:
                    # 归一化SINR到[0,1]范围
                    normalized_sinr = np.clip((self.sinr_matrix[i, j] - self.min_sinr) / 30, 0, 1)
                    total_sinr += normalized_sinr
        
        # 平均SINR质量
        quality_reward = total_sinr / max(connected_users, 1)
        
        # 网络连通性奖励
        connectivity_ratio = self._compute_connectivity_ratio()
        connectivity_reward = connectivity_ratio
        
        # 【修正】考虑回程瓶颈的系统吞吐量计算
        system_throughput = 0
        max_possible_system_throughput = 0
        
        # 按UAV计算有效吞吐量（考虑回程瓶颈）
        for i in range(self.n_uavs):
            # 计算该UAV服务的所有用户的总需求吞吐量
            uav_user_throughput = 0
            uav_max_possible_throughput = 0
            
            for j in range(self.n_users):
                if self.connections[i, j]:
                    # 计算前端链路吞吐量（UAV到用户）
                    user_throughput = self._compute_throughput(i, j)
                    uav_user_throughput += user_throughput
                
                # 计算理论最大前端链路吞吐量
                max_sinr_linear = 10 ** (30 / 10)  # 30dB转换为线性
                max_link_throughput = self.bandwidth * np.log2(1 + max_sinr_linear)
                uav_max_possible_throughput += max_link_throughput
            
            # 获取该UAV的回程容量限制
            if i in self.routing_paths:
                backhaul_capacity = self._compute_backhaul_capacity(i)
                
                # 考虑多跳效率损失
                path = self.routing_paths[i]
                hop_count = len(path)
                hop_efficiency = 1.0 / hop_count if hop_count > 0 else 0
                
                # 有效回程容量
                effective_backhaul = backhaul_capacity * hop_efficiency
                
                # 实际有效吞吐量 = min(前端总需求, 有效回程容量)
                uav_effective_throughput = min(uav_user_throughput, effective_backhaul)
            else:
                # 无回程路径，吞吐量为0
                uav_effective_throughput = 0
            
            # 累加到系统总吞吐量
            system_throughput += uav_effective_throughput
            max_possible_system_throughput += min(uav_max_possible_throughput, 
                                                  self.bandwidth * np.log2(1 + 10**(30/10)))  # 单UAV理论最大回程容量
        
        # 归一化吞吐量奖励
        throughput_reward = system_throughput / max_possible_system_throughput if max_possible_system_throughput > 0 else 0
        
        # 跳数惩罚：路径越长，惩罚越大
        total_hops = 0
        for path in self.routing_paths.values():
            total_hops += len(path)
        
        avg_hops = total_hops / max(len(self.routing_paths), 1)
        hop_penalty = avg_hops / self.max_hops * 0.1  # 归一化并缩放
        
        # 组合奖励
        reward = (
            self.coverage_weight * coverage_reward + 
            self.quality_weight * quality_reward + 
            self.connectivity_weight * connectivity_reward +
            self.throughput_weight * throughput_reward -
            hop_penalty
        )
        
        # 记录奖励组成（修正后的数据）
        self.reward_info = {
            "coverage_reward": coverage_reward,
            "quality_reward": quality_reward,
            "connectivity_reward": connectivity_reward,
            "throughput_reward": throughput_reward,
            "hop_penalty": hop_penalty,
            "total_reward": reward,
            "system_throughput_mbps": system_throughput / 1e6,  # 系统实际吞吐量（考虑回程瓶颈）
            "avg_throughput_per_user_mbps": (system_throughput / max(connected_users, 1)) / 1e6,
            # 额外的调试信息
            "connected_users": connected_users,
            "connectivity_ratio": connectivity_ratio,
            "avg_hops": avg_hops
        }
        
        return reward
    
    def _render_frame(self):
        """渲染单帧"""
        try:
            import matplotlib.pyplot as plt
            from matplotlib.patches import Circle
            from mpl_toolkits.mplot3d import Axes3D
        except ImportError:
            print("渲染需要matplotlib库")
            return None
        
        # 调用父类的渲染方法
        frame = super()._render_frame()
        
        # 添加UAV之间的连接和UAV到地面基站的连接
        for i in range(self.n_uavs):
            uav_pos_i = self.uav_positions[i]
            
            # 绘制UAV之间的连接
            for j in range(i+1, self.n_uavs):
                if self.uav_connections[i, j]:
                    uav_pos_j = self.uav_positions[j]
                    self.ax.plot([uav_pos_i[0], uav_pos_j[0]], 
                                [uav_pos_i[1], uav_pos_j[1]], 
                                [uav_pos_i[2], uav_pos_j[2]], 
                                'y-', alpha=0.5, linewidth=1.5)
            
            # 绘制UAV到地面基站的连接
            for j in range(self.n_ground_bs):
                if self.uav_bs_connections[i, j]:
                    bs_pos = self.ground_bs_positions[j]
                    self.ax.plot([uav_pos_i[0], bs_pos[0]], 
                                [uav_pos_i[1], bs_pos[1]], 
                                [uav_pos_i[2], bs_pos[2]], 
                                'c-', alpha=0.7, linewidth=2.0)
        
        # 根据角色为UAV添加不同的颜色
        for i in range(self.n_uavs):
            uav_pos = self.uav_positions[i]
            role = self.uav_roles[i]
            
            # 清除之前的UAV标记
            if hasattr(self, 'uav_markers'):
                for marker in self.uav_markers:
                    if marker in self.ax.collections:
                        marker.remove()
            
            # 根据角色设置颜色
            if role == 0:  # 未分配
                color = 'gray'
            elif role == 1:  # 基站
                color = 'red'
            elif role == 2:  # 中继
                color = 'orange'
            
            # 重新绘制UAV
            self.ax.scatter(uav_pos[0], uav_pos[1], uav_pos[2], 
                           c=color, marker='^', s=100, 
                           label=f'UAV {i} ({["未分配", "基站", "中继"][role]})' if i == 0 else "")
        
        # 添加连通性信息
        connectivity_ratio = self._compute_connectivity_ratio()
        self.ax.text2D(0.02, 0.90, f'网络连通性: {connectivity_ratio:.2%}', transform=self.ax.transAxes)
        
        # 添加角色统计
        role_counts = np.bincount(self.uav_roles, minlength=3)
        self.ax.text2D(0.02, 0.85, f'角色: 基站={role_counts[1]}, 中继={role_counts[2]}, 未分配={role_counts[0]}', 
                      transform=self.ax.transAxes)
        
        # 添加吞吐量信息
        if hasattr(self, 'reward_info') and 'total_throughput_mbps' in self.reward_info:
            total_throughput_mbps = self.reward_info['total_throughput_mbps']
            avg_throughput_mbps = self.reward_info['avg_throughput_per_user_mbps']
            self.ax.text2D(0.02, 0.80, f'总吞吐量: {total_throughput_mbps:.1f} Mbps', 
                          transform=self.ax.transAxes)
            self.ax.text2D(0.02, 0.75, f'平均用户吞吐量: {avg_throughput_mbps:.1f} Mbps', 
                          transform=self.ax.transAxes)
        
        self.fig.canvas.draw()
        
        return frame
