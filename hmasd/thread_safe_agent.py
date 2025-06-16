#!/usr/bin/env python3
"""
【阶段4】线程安全的HMASD Agent实现
解决多线程训练中的数据丢失问题

核心修复：
1. Agent buffer线程安全加固
2. 原子性存储操作
3. 数据完整性验证
4. 分布式buffer架构支持
"""

import torch
import torch.nn.functional as F
import numpy as np
import threading
import time
from collections import defaultdict, deque
from threading import Lock, RLock
from logger import get_logger

class ThreadSafeBuffer:
    """【阶段4核心】线程安全的经验缓冲区"""
    
    def __init__(self, maxsize=10000, buffer_type="generic"):
        self.maxsize = maxsize
        self.buffer_type = buffer_type
        self.buffer = deque(maxlen=maxsize)
        
        # 【关键】使用可重入锁确保线程安全
        self.lock = RLock()
        
        # 原子性操作计数器
        self.total_added = 0
        self.total_consumed = 0
        self.operation_id_counter = 0
        
        # 【阶段4新增】数据完整性验证
        self.pending_operations = {}
        self.failed_operations = 0
        self.checksum_errors = 0
        
        # 性能监控
        self.last_size_check = 0
        self.size_check_interval = 100
        
        self.logger = get_logger(f"ThreadSafeBuffer-{buffer_type}")
    
    def push(self, item):
        """【原子性】线程安全的数据插入"""
        operation_id = None
        try:
            with self.lock:
                # 开始原子性操作
                operation_id = self._begin_operation("push", item)
                
                # 验证数据完整性
                if not self._validate_item(item):
                    self.checksum_errors += 1
                    self._rollback_operation(operation_id)
                    return False
                
                # 执行插入操作
                old_size = len(self.buffer)
                self.buffer.append(item)
                new_size = len(self.buffer)
                
                # 验证插入成功
                if new_size <= old_size and old_size < self.maxsize:
                    # 插入失败（不应该发生）
                    self._rollback_operation(operation_id)
                    self.failed_operations += 1
                    return False
                
                # 提交操作
                self.total_added += 1
                self._commit_operation(operation_id)
                
                # 【调试】定期记录缓冲区状态
                if self.total_added % self.size_check_interval == 0:
                    self.logger.debug(f"[{self.buffer_type}] 缓冲区状态: 大小={new_size}/{self.maxsize}, "
                                    f"总添加={self.total_added}, 总消费={self.total_consumed}")
                
                return True
                
        except Exception as e:
            if operation_id:
                self._rollback_operation(operation_id)
            self.failed_operations += 1
            self.logger.error(f"[{self.buffer_type}] 插入操作失败: {e}")
            return False
    
    def sample(self, batch_size):
        """【原子性】线程安全的批量采样"""
        with self.lock:
            if len(self.buffer) < batch_size:
                return []
            
            # 随机采样
            indices = np.random.choice(len(self.buffer), batch_size, replace=False)
            samples = [self.buffer[i] for i in indices]
            
            self.total_consumed += batch_size
            return samples
    
    def clear(self):
        """【原子性】线程安全的缓冲区清空"""
        with self.lock:
            old_size = len(self.buffer)
            self.buffer.clear()
            new_size = len(self.buffer)
            
            if new_size == 0:
                self.logger.debug(f"[{self.buffer_type}] 缓冲区清空成功: {old_size}→{new_size}")
                return True
            else:
                self.logger.error(f"[{self.buffer_type}] 缓冲区清空失败: {old_size}→{new_size}")
                return False
    
    def __len__(self):
        """线程安全的长度获取"""
        with self.lock:
            return len(self.buffer)
    
    def get_stats(self):
        """获取详细统计信息"""
        with self.lock:
            return {
                'buffer_type': self.buffer_type,
                'current_size': len(self.buffer),
                'max_size': self.maxsize,
                'total_added': self.total_added,
                'total_consumed': self.total_consumed,
                'failed_operations': self.failed_operations,
                'checksum_errors': self.checksum_errors,
                'pending_operations': len(self.pending_operations),
                'utilization': len(self.buffer) / self.maxsize if self.maxsize > 0 else 0.0
            }
    
    def _begin_operation(self, op_type, data):
        """开始原子性操作"""
        self.operation_id_counter += 1
        operation_id = self.operation_id_counter
        
        self.pending_operations[operation_id] = {
            'type': op_type,
            'timestamp': time.time(),
            'data_hash': hash(str(data)) if data else None
        }
        
        return operation_id
    
    def _commit_operation(self, operation_id):
        """提交操作"""
        if operation_id in self.pending_operations:
            del self.pending_operations[operation_id]
    
    def _rollback_operation(self, operation_id):
        """回滚操作"""
        if operation_id in self.pending_operations:
            del self.pending_operations[operation_id]
    
    def _validate_item(self, item):
        """验证数据项的完整性"""
        try:
            # 基本验证
            if item is None:
                return False
            
            # 对于tuple类型的经验数据，验证长度
            if isinstance(item, tuple):
                if self.buffer_type == "high_level" and len(item) != 5:
                    return False
                elif self.buffer_type == "low_level" and len(item) != 11:
                    return False
            
            return True
            
        except Exception:
            return False

class ThreadSafeStateSkillDataset:
    """【阶段4】线程安全的状态-技能数据集"""
    
    def __init__(self, maxsize=10000):
        self.maxsize = maxsize
        self.data = deque(maxlen=maxsize)
        self.lock = RLock()
        self.total_added = 0
        self.logger = get_logger("ThreadSafeStateSkillDataset")
    
    def push(self, state, team_skill, observations, agent_skills):
        """线程安全的数据插入"""
        try:
            with self.lock:
                item = (state, team_skill, observations, agent_skills)
                self.data.append(item)
                self.total_added += 1
                return True
        except Exception as e:
            self.logger.error(f"StateSkillDataset插入失败: {e}")
            return False
    
    def sample(self, batch_size):
        """线程安全的批量采样"""
        with self.lock:
            if len(self.data) < batch_size:
                return []
            
            indices = np.random.choice(len(self.data), batch_size, replace=False)
            return [self.data[i] for i in indices]
    
    def clear(self):
        """线程安全的清空"""
        with self.lock:
            self.data.clear()
    
    def __len__(self):
        with self.lock:
            return len(self.data)

class ThreadSafeAgentMixin:
    """【阶段4核心】线程安全Agent混入类
    
    为现有的HMASDAgent添加线程安全功能，无需重写整个Agent
    """
    
    def __init_thread_safety__(self):
        """初始化线程安全组件"""
        # 【关键】为每个buffer添加独立的锁
        self.high_level_buffer_lock = RLock()
        self.low_level_buffer_lock = RLock()
        self.state_skill_dataset_lock = RLock()
        
        # 【阶段4】替换原有buffer为线程安全版本
        original_high_level_size = len(self.high_level_buffer) if hasattr(self, 'high_level_buffer') else 0
        original_low_level_size = len(self.low_level_buffer) if hasattr(self, 'low_level_buffer') else 0
        original_state_skill_size = len(self.state_skill_dataset) if hasattr(self, 'state_skill_dataset') else 0
        
        # 创建线程安全的buffer
        self.high_level_buffer = ThreadSafeBuffer(
            maxsize=getattr(self.config, 'buffer_size', 10000),
            buffer_type="high_level"
        )
        self.low_level_buffer = ThreadSafeBuffer(
            maxsize=getattr(self.config, 'buffer_size', 10000),
            buffer_type="low_level"
        )
        self.state_skill_dataset = ThreadSafeStateSkillDataset(
            maxsize=getattr(self.config, 'buffer_size', 10000)
        )
        
        # 【阶段4新增】数据完整性验证机制
        self.integrity_check_enabled = True
        self.integrity_failures = defaultdict(int)
        
        # 【阶段4新增】存储操作统计
        self.storage_stats = {
            'high_level_attempts': 0,
            'high_level_successes': 0,
            'low_level_attempts': 0,
            'low_level_successes': 0,
            'integrity_failures': 0,
            'concurrent_operations': 0
        }
        self.storage_stats_lock = Lock()
        
        # 【阶段4新增】性能监控
        self.performance_monitor = {
            'avg_storage_time': deque(maxlen=1000),
            'max_storage_time': 0.0,
            'total_storage_operations': 0
        }
        self.performance_lock = Lock()
        
        self.logger = get_logger("ThreadSafeAgent")
        self.logger.info(f"线程安全Agent初始化完成 - "
                        f"原buffer大小: H={original_high_level_size}, L={original_low_level_size}, S={original_state_skill_size}")
    
    def store_high_level_transition_safe(self, state, team_skill, observations, agent_skills, 
                                       accumulated_reward, skill_log_probs=None, worker_id=0):
        """【阶段4核心】线程安全的高层经验存储"""
        storage_start = time.time()
        
        try:
            with self.storage_stats_lock:
                self.storage_stats['high_level_attempts'] += 1
            
            # 【数据完整性验证】
            if self.integrity_check_enabled:
                if not self._validate_high_level_data(state, team_skill, observations, agent_skills, accumulated_reward):
                    with self.storage_stats_lock:
                        self.storage_stats['integrity_failures'] += 1
                    self.logger.warning(f"Worker {worker_id}: 高层经验数据完整性验证失败")
                    return False
            
            # 【原子性存储操作】
            with self.high_level_buffer_lock:
                # 转换为tensor格式
                state_tensor = torch.FloatTensor(state).to(self.device)
                team_skill_tensor = torch.tensor(team_skill, device=self.device)
                observations_tensor = torch.FloatTensor(observations).to(self.device)
                agent_skills_tensor = torch.tensor(agent_skills, device=self.device)
                
                # 创建高层经验元组
                high_level_experience = (
                    state_tensor,
                    team_skill_tensor,
                    observations_tensor,
                    agent_skills_tensor,
                    torch.tensor(accumulated_reward, device=self.device)
                )
                
                # 【关键】使用线程安全的push方法
                success = self.high_level_buffer.push(high_level_experience)
                
                if success:
                    # 更新统计信息
                    with self.storage_stats_lock:
                        self.storage_stats['high_level_successes'] += 1
                    
                    # 存储带log probabilities的经验
                    if skill_log_probs is not None:
                        self._store_high_level_with_logprobs(
                            state_tensor, team_skill, observations_tensor, 
                            agent_skills_tensor, accumulated_reward, skill_log_probs
                        )
                    
                    # 【性能监控】
                    storage_time = time.time() - storage_start
                    with self.performance_lock:
                        self.performance_monitor['avg_storage_time'].append(storage_time)
                        self.performance_monitor['max_storage_time'] = max(
                            self.performance_monitor['max_storage_time'], storage_time
                        )
                        self.performance_monitor['total_storage_operations'] += 1
                    
                    self.logger.debug(f"Worker {worker_id}: 高层经验存储成功 - "
                                    f"缓冲区大小={len(self.high_level_buffer)}, "
                                    f"存储耗时={storage_time*1000:.2f}ms")
                    return True
                else:
                    self.logger.error(f"Worker {worker_id}: 高层经验存储失败 - buffer.push()返回False")
                    return False
                    
        except Exception as e:
            self.logger.error(f"Worker {worker_id}: 高层经验存储异常: {e}")
            return False
    
    def store_low_level_transition_safe(self, state, next_state, observations, next_observations,
                                      actions, rewards, dones, team_skill, agent_skills, 
                                      action_logprobs, skill_log_probs=None, worker_id=0):
        """【阶段4核心】线程安全的低层经验存储"""
        storage_start = time.time()
        
        try:
            with self.storage_stats_lock:
                self.storage_stats['low_level_attempts'] += 1
            
            # 【数据完整性验证】
            if self.integrity_check_enabled:
                if not self._validate_low_level_data(state, next_state, observations, next_observations,
                                                   actions, rewards, dones, team_skill, agent_skills):
                    with self.storage_stats_lock:
                        self.storage_stats['integrity_failures'] += 1
                    self.logger.warning(f"Worker {worker_id}: 低层经验数据完整性验证失败")
                    return False
            
            # 【原子性存储操作】
            with self.low_level_buffer_lock:
                # 计算内在奖励组件
                current_reward = rewards if isinstance(rewards, (int, float)) else rewards.item()
                
                # 转换为tensor
                state_tensor = torch.FloatTensor(state).to(self.device)
                next_state_tensor = torch.FloatTensor(next_state).to(self.device)
                team_skill_tensor = torch.tensor(team_skill, device=self.device)
                observations_tensor = torch.FloatTensor(observations).to(self.device)
                next_observations_tensor = torch.FloatTensor(next_observations).to(self.device)
                actions_tensor = torch.FloatTensor(actions).to(self.device)
                action_logprobs_tensor = torch.FloatTensor(action_logprobs).to(self.device)
                agent_skills_tensor = torch.tensor(agent_skills, device=self.device)
                
                # 计算判别器输出
                with torch.no_grad():
                    team_disc_logits = self.team_discriminator(next_state_tensor.unsqueeze(0))
                    team_disc_log_probs = F.log_softmax(team_disc_logits, dim=-1)
                    team_skill_log_prob = team_disc_log_probs[0, team_skill]
                
                # 计算个体判别器输出（团队级别）
                n_agents = len(agent_skills)
                agent_disc_log_probs = []
                for i in range(n_agents):
                    next_obs_i = next_observations_tensor[i].unsqueeze(0)
                    with torch.no_grad():
                        agent_disc_logits = self.individual_discriminator(next_obs_i, team_skill_tensor)
                        agent_disc_log_prob = F.log_softmax(agent_disc_logits, dim=-1)[0, agent_skills[i]]
                        agent_disc_log_probs.append(agent_disc_log_prob.item())
                
                # 计算内在奖励组件
                env_reward_component = self.config.lambda_e * current_reward
                team_disc_component = self.config.lambda_D * team_skill_log_prob.item()
                avg_ind_disc_component = self.config.lambda_d * np.mean(agent_disc_log_probs)
                
                # 团队总内在奖励
                team_intrinsic_reward = env_reward_component + team_disc_component + avg_ind_disc_component
                
                # 创建低层经验元组（团队级别）
                low_level_experience = (
                    state_tensor,
                    team_skill_tensor,
                    observations_tensor,
                    agent_skills_tensor,
                    actions_tensor,
                    torch.tensor(team_intrinsic_reward, device=self.device),
                    torch.tensor(dones, dtype=torch.float, device=self.device),
                    action_logprobs_tensor,
                    torch.tensor(env_reward_component, device=self.device),
                    torch.tensor(team_disc_component, device=self.device),
                    torch.tensor(avg_ind_disc_component, device=self.device)
                )
                
                # 【关键】使用线程安全的push方法
                success = self.low_level_buffer.push(low_level_experience)
                
                if success:
                    with self.storage_stats_lock:
                        self.storage_stats['low_level_successes'] += 1
                    
                    # 【性能监控】
                    storage_time = time.time() - storage_start
                    with self.performance_lock:
                        self.performance_monitor['avg_storage_time'].append(storage_time)
                        self.performance_monitor['max_storage_time'] = max(
                            self.performance_monitor['max_storage_time'], storage_time
                        )
                        self.performance_monitor['total_storage_operations'] += 1
                    
                    self.logger.debug(f"Worker {worker_id}: 低层经验存储成功 - "
                                    f"缓冲区大小={len(self.low_level_buffer)}, "
                                    f"存储耗时={storage_time*1000:.2f}ms")
                    return True
                else:
                    self.logger.error(f"Worker {worker_id}: 低层经验存储失败 - buffer.push()返回False")
                    return False
                    
        except Exception as e:
            self.logger.error(f"Worker {worker_id}: 低层经验存储异常: {e}")
            return False
    
    def _validate_high_level_data(self, state, team_skill, observations, agent_skills, accumulated_reward):
        """验证高层经验数据的完整性"""
        try:
            # 基本类型检查
            if state is None or observations is None or agent_skills is None:
                return False
            
            # 维度检查
            if hasattr(state, '__len__') and len(state) != self.config.state_dim:
                return False
            
            if hasattr(observations, 'shape') and observations.shape[-1] != self.config.obs_dim:
                return False
            
            # 技能范围检查
            if not (0 <= team_skill < self.config.n_Z):
                return False
            
            for skill in agent_skills:
                if not (0 <= skill < self.config.n_z):
                    return False
            
            # 奖励有效性检查
            if not isinstance(accumulated_reward, (int, float)) or np.isnan(accumulated_reward) or np.isinf(accumulated_reward):
                return False
            
            return True
            
        except Exception:
            return False
    
    def _validate_low_level_data(self, state, next_state, observations, next_observations,
                                actions, rewards, dones, team_skill, agent_skills):
        """验证低层经验数据的完整性"""
        try:
            # 基本存在性检查
            required_data = [state, next_state, observations, next_observations, actions, agent_skills]
            if any(data is None for data in required_data):
                return False
            
            # 维度一致性检查
            if hasattr(observations, 'shape') and hasattr(next_observations, 'shape'):
                if observations.shape != next_observations.shape:
                    return False
            
            if hasattr(actions, '__len__') and hasattr(agent_skills, '__len__'):
                if len(actions) != len(agent_skills):
                    return False
            
            # 奖励有效性检查
            reward_val = rewards if isinstance(rewards, (int, float)) else rewards.item()
            if np.isnan(reward_val) or np.isinf(reward_val):
                return False
            
            return True
            
        except Exception:
            return False
    
    def _store_high_level_with_logprobs(self, state_tensor, team_skill, observations_tensor, 
                                      agent_skills_tensor, accumulated_reward, skill_log_probs):
        """存储带log probabilities的高层经验"""
        try:
            if not hasattr(self, 'high_level_buffer_with_logprobs'):
                self.high_level_buffer_with_logprobs = deque(maxlen=self.config.buffer_size)
            
            experience_with_logprobs = {
                'state': state_tensor.clone(),
                'team_skill': team_skill,
                'observations': observations_tensor.clone(),
                'agent_skills': agent_skills_tensor.clone(),
                'reward': accumulated_reward,
                'team_log_prob': skill_log_probs.get('team_log_prob', 0.0),
                'agent_log_probs': skill_log_probs.get('agent_log_probs', [0.0] * len(agent_skills_tensor))
            }
            
            self.high_level_buffer_with_logprobs.append(experience_with_logprobs)
            
        except Exception as e:
            self.logger.error(f"存储带log probabilities的高层经验失败: {e}")
    
    def get_thread_safety_stats(self):
        """获取线程安全统计信息"""
        with self.storage_stats_lock:
            storage_stats = self.storage_stats.copy()
        
        with self.performance_lock:
            avg_time = np.mean(self.performance_monitor['avg_storage_time']) if self.performance_monitor['avg_storage_time'] else 0.0
            performance_stats = {
                'avg_storage_time_ms': avg_time * 1000,
                'max_storage_time_ms': self.performance_monitor['max_storage_time'] * 1000,
                'total_operations': self.performance_monitor['total_storage_operations']
            }
        
        # 获取buffer统计
        high_level_stats = self.high_level_buffer.get_stats()
        low_level_stats = self.low_level_buffer.get_stats()
        
        return {
            'storage_stats': storage_stats,
            'performance_stats': performance_stats,
            'high_level_buffer': high_level_stats,
            'low_level_buffer': low_level_stats,
            'integrity_failures': dict(self.integrity_failures)
        }
    
    def clear_buffers_safe(self):
        """【阶段4】线程安全的缓冲区清空"""
        results = {}
        
        # 清空高层缓冲区
        with self.high_level_buffer_lock:
            results['high_level'] = self.high_level_buffer.clear()
        
        # 清空低层缓冲区
        with self.low_level_buffer_lock:
            results['low_level'] = self.low_level_buffer.clear()
        
        # 清空状态技能数据集
        with self.state_skill_dataset_lock:
            self.state_skill_dataset.clear()
            results['state_skill'] = True
        
        # 清空带log probabilities的缓冲区
        if hasattr(self, 'high_level_buffer_with_logprobs'):
            self.high_level_buffer_with_logprobs.clear()
        
        self.logger.info(f"线程安全缓冲区清空结果: {results}")
        return all(results.values())
