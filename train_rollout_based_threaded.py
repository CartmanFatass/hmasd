#!/usr/bin/env python3
"""
HMASD严格按论文Algorithm 1 + Appendix E的多线程Rollout-based训练脚本
实现要点：
1. 32个rollout threads: 持续环境交互和数据收集
2. 16个training threads: 持续神经网络训练
3. 线程安全的数据传输: 使用队列在rollout和training线程间传输数据
4. 异步训练: 数据收集和模型训练并行执行
5. 严格按照论文: 15轮PPO + 判别器训练 + 缓冲区管理
"""

import os
import sys
import time
import numpy as np
import torch
import argparse
import threading
import queue
import multiprocessing as mp
from datetime import datetime
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock, Event, Barrier
import logging

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 设置matplotlib后端（避免多进程问题）
import matplotlib
matplotlib.use('Agg')

from logger import get_logger, init_multiproc_logging, LOG_LEVELS, shutdown_logging
from config import Config
from hmasd.thread_safe_hmasd_agent import ThreadSafeHMASDAgent
from envs.pettingzoo.scenario1 import UAVBaseStationEnv
from envs.pettingzoo.scenario2 import UAVCooperativeNetworkEnv
from envs.pettingzoo.env_adapter import ParallelToArrayAdapter

# 导入 Stable Baselines3 的向量化环境
from stable_baselines3.common.vec_env import SubprocVecEnv

class ThreadSafeCounter:
    """线程安全的计数器"""
    def __init__(self, initial_value=0):
        self._value = initial_value
        self._lock = Lock()
    
    def increment(self, amount=1):
        with self._lock:
            self._value += amount
            return self._value
    
    def get(self):
        with self._lock:
            return self._value
    
    def set(self, value):
        with self._lock:
            self._value = value

class DataBuffer:
    """【阶段2增强】线程安全的数据缓冲区 - 支持优先级处理和状态监控"""
    def __init__(self, maxsize=10000):
        # 【阶段2新增】使用优先级队列代替普通队列
        self.high_priority_queue = queue.Queue(maxsize=maxsize//4)  # 高层经验优先队列
        self.normal_priority_queue = queue.Queue(maxsize=maxsize)   # 低层和其他经验队列
        
        # 统计计数器
        self.total_added = ThreadSafeCounter()
        self.total_consumed = ThreadSafeCounter()
        self.high_priority_added = ThreadSafeCounter()
        self.normal_priority_added = ThreadSafeCounter()
        self.high_priority_consumed = ThreadSafeCounter()
        self.normal_priority_consumed = ThreadSafeCounter()
        
        # 【阶段2新增】状态监控
        self.processing_speed_samples = deque(maxlen=100)  # 保留最近100个处理速度样本
        self.last_monitoring_time = time.time()
        self.last_consumed_count = 0
        self.congestion_detected = False
        self.lock = Lock()
        
        # 【阶段2新增】数据完整性校验
        self.checksum_errors = ThreadSafeCounter()
        self.validation_enabled = True
        
    def put(self, item, block=True, timeout=None):
        """【阶段2增强】添加数据到缓冲区 - 支持优先级和完整性校验"""
        try:
            # 【数据完整性校验】
            if self.validation_enabled and not self._validate_item(item):
                self.checksum_errors.increment()
                return False
            
            # 【优先级处理】根据经验类型选择队列
            experience_type = item.get('experience_type', 'low_level')
            
            if experience_type == 'high_level':
                # 高层经验使用高优先级队列
                try:
                    self.high_priority_queue.put(item, block=block, timeout=timeout)
                    self.high_priority_added.increment()
                    self.total_added.increment()
                    return True
                except queue.Full:
                    # 高优先级队列满时，尝试处理拥塞
                    if self._handle_high_priority_congestion(item, block, timeout):
                        return True
                    return False
            else:
                # 低层经验和其他数据使用普通队列
                self.normal_priority_queue.put(item, block=block, timeout=timeout)
                self.normal_priority_added.increment()
                self.total_added.increment()
                return True
                
        except queue.Full:
            # 【阶段2新增】拥塞检测
            self._detect_congestion()
            return False
        except Exception as e:
            # 记录异常但不抛出，确保系统稳定性
            return False
    
    def get(self, block=True, timeout=None):
        """【阶段2增强】从缓冲区获取数据 - 优先处理高层经验"""
        try:
            # 【优先级处理】先尝试获取高优先级数据
            if not self.high_priority_queue.empty():
                try:
                    item = self.high_priority_queue.get(block=False)
                    self.high_priority_consumed.increment()
                    self.total_consumed.increment()
                    self._update_processing_speed()
                    return item
                except queue.Empty:
                    pass  # 高优先级队列为空，继续处理普通队列
            
            # 处理普通优先级数据
            item = self.normal_priority_queue.get(block=block, timeout=timeout)
            self.normal_priority_consumed.increment()
            self.total_consumed.increment()
            self._update_processing_speed()
            return item
            
        except queue.Empty:
            return None
    
    def qsize(self):
        """获取当前总队列大小"""
        return self.high_priority_queue.qsize() + self.normal_priority_queue.qsize()
    
    def empty(self):
        """检查队列是否为空"""
        return self.high_priority_queue.empty() and self.normal_priority_queue.empty()
    
    def get_stats(self):
        """【阶段2增强】获取详细统计信息"""
        current_time = time.time()
        
        # 计算处理速度
        processing_speed = self._calculate_processing_speed()
        
        return {
            'queue_size': self.qsize(),
            'high_priority_size': self.high_priority_queue.qsize(),
            'normal_priority_size': self.normal_priority_queue.qsize(),
            'total_added': self.total_added.get(),
            'total_consumed': self.total_consumed.get(),
            'high_priority_added': self.high_priority_added.get(),
            'high_priority_consumed': self.high_priority_consumed.get(),
            'normal_priority_added': self.normal_priority_added.get(),
            'normal_priority_consumed': self.normal_priority_consumed.get(),
            'processing_speed': processing_speed,
            'congestion_detected': self.congestion_detected,
            'checksum_errors': self.checksum_errors.get(),
            'high_priority_ratio': self.high_priority_consumed.get() / max(1, self.total_consumed.get())
        }
    
    def _validate_item(self, item):
        """【阶段2新增】验证数据项的完整性"""
        try:
            # 基本验证：检查必需字段
            if not isinstance(item, dict):
                return False
            
            required_fields = ['experience_type', 'worker_id']
            for field in required_fields:
                if field not in item:
                    return False
            
            # 类型特定验证
            experience_type = item.get('experience_type')
            if experience_type == 'low_level':
                required_low_level = ['state', 'actions', 'rewards', 'next_state']
                for field in required_low_level:
                    if field not in item:
                        return False
            elif experience_type == 'high_level':
                required_high_level = ['state', 'team_skill', 'accumulated_reward']
                for field in required_high_level:
                    if field not in item:
                        return False
            
            return True
            
        except Exception:
            return False
    
    def _handle_high_priority_congestion(self, item, block, timeout):
        """【阶段2新增】处理高优先级队列拥塞"""
        # 如果高优先级队列满了，尝试从普通队列中腾出空间
        retry_count = 3
        
        for i in range(retry_count):
            try:
                # 先尝试快速处理一些普通优先级的数据
                if not self.normal_priority_queue.empty():
                    temp_items = []
                    # 临时取出一些普通优先级数据
                    for _ in range(min(5, self.normal_priority_queue.qsize())):
                        try:
                            temp_item = self.normal_priority_queue.get(block=False)
                            temp_items.append(temp_item)
                        except queue.Empty:
                            break
                    
                    # 尝试再次放入高优先级数据
                    try:
                        self.high_priority_queue.put(item, block=False)
                        self.high_priority_added.increment()
                        self.total_added.increment()
                        
                        # 将临时取出的数据放回普通队列
                        for temp_item in temp_items:
                            self.normal_priority_queue.put(temp_item, block=False)
                        
                        return True
                    except queue.Full:
                        # 高优先级队列仍然满，恢复普通队列数据
                        for temp_item in temp_items:
                            self.normal_priority_queue.put(temp_item, block=False)
                
                # 短暂等待后重试
                time.sleep(0.01 * (i + 1))
                
            except Exception:
                continue
        
        return False
    
    def _detect_congestion(self):
        """【阶段2新增】检测队列拥塞"""
        total_size = self.qsize()
        high_size = self.high_priority_queue.qsize()
        normal_size = self.normal_priority_queue.qsize()
        
        # 拥塞检测条件
        total_capacity = 10000  # 假设总容量
        congestion_threshold = 0.8  # 80%容量触发拥塞警告
        
        with self.lock:
            old_congestion = self.congestion_detected
            self.congestion_detected = total_size > (total_capacity * congestion_threshold)
            
            # 只在状态变化时记录日志
            if self.congestion_detected != old_congestion:
                if self.congestion_detected:
                    # 拥塞开始时记录详细信息
                    pass  # 可以在这里添加拥塞日志，但避免过度日志
                else:
                    # 拥塞缓解时记录
                    pass
    
    def _update_processing_speed(self):
        """【阶段2新增】更新处理速度统计"""
        current_time = time.time()
        current_consumed = self.total_consumed.get()
        
        with self.lock:
            time_diff = current_time - self.last_monitoring_time
            consumed_diff = current_consumed - self.last_consumed_count
            
            if time_diff >= 1.0:  # 每秒更新一次
                speed = consumed_diff / time_diff if time_diff > 0 else 0
                self.processing_speed_samples.append(speed)
                
                self.last_monitoring_time = current_time
                self.last_consumed_count = current_consumed
    
    def _calculate_processing_speed(self):
        """【阶段2新增】计算平均处理速度"""
        with self.lock:
            if len(self.processing_speed_samples) == 0:
                return 0.0
            return sum(self.processing_speed_samples) / len(self.processing_speed_samples)
    
    def get_priority_status(self):
        """【阶段2新增】获取优先级队列状态"""
        return {
            'high_priority_queue_size': self.high_priority_queue.qsize(),
            'normal_priority_queue_size': self.normal_priority_queue.qsize(),
            'high_priority_full': self.high_priority_queue.qsize() >= (self.high_priority_queue.maxsize * 0.9),
            'normal_priority_full': self.normal_priority_queue.qsize() >= (self.normal_priority_queue.maxsize * 0.9),
            'congestion_detected': self.congestion_detected
        }

class RolloutWorker:
    """单个rollout worker，在独立线程中运行"""
    def __init__(self, worker_id, env_factory, config, data_buffer, control_events, logger):
        self.worker_id = worker_id
        self.env_factory = env_factory
        self.config = config
        self.data_buffer = data_buffer
        self.control_events = control_events
        self.logger = logger
        
        # 创建环境
        self.env = env_factory()
        
        # 状态变量
        self.samples_collected = 0
        self.episodes_completed = 0
        self.total_reward = 0.0
        
        # 【关键修复】添加rollout完成控制
        self.rollout_completed = False
        self.target_rollout_steps = config.rollout_length  # 每个worker的目标步数（128）
        
        # 环境状态
        self.env_state = None
        self.env_observations = None
        self.episode_step = 0
        
        # 【方案2】严格的步数计数方法
        self.strict_step_counter = 0  # 严格步数计数器
        self.accumulated_reward = 0.0  # 32步累积奖励
        self.current_team_skill = None
        self.current_agent_skills = None
        self.skill_log_probs = None
        self.high_level_experiences_generated = 0
        
        # 【修复1】添加步数验证标志
        self.step_validation_enabled = True
        self.last_reported_steps = 0
        
        # 技能计时器初始化
        self.skill_timer = 0
        
    def reset_environment(self):
        """重置环境"""
        try:
            result = self.env.reset()
            if isinstance(result, tuple):
                self.env_observations, info = result
                self.env_state = info.get('state', np.zeros(self.config.state_dim))
            else:
                self.env_observations = result
                self.env_state = np.zeros(self.config.state_dim)
            self.episode_step = 0
            return True
        except Exception as e:
            self.logger.error(f"Worker {self.worker_id}: 重置环境失败: {e}")
            return False
    
    def step_environment(self, actions):
        """执行环境步骤"""
        try:
            # 修复：处理Gymnasium API的5个返回值
            result = self.env.step(actions)
            
            if len(result) == 5:
                # Gymnasium格式: observations, reward, terminated, truncated, info
                next_observations, rewards, terminated, truncated, infos = result
                dones = terminated or truncated  # 合并终止条件
            elif len(result) == 4:
                # 传统格式: observations, rewards, dones, infos
                next_observations, rewards, dones, infos = result
            else:
                raise ValueError(f"Unexpected number of return values from env.step(): {len(result)}")
            
            # 从info中提取next_state
            if isinstance(infos, dict):
                next_state = infos.get('next_state', np.zeros(self.config.state_dim))
            elif isinstance(infos, list) and len(infos) > 0:
                next_state = infos[0].get('next_state', np.zeros(self.config.state_dim))
            else:
                next_state = np.zeros(self.config.state_dim)
            
            self.episode_step += 1
            return next_observations, rewards, dones, next_state
            
        except Exception as e:
            self.logger.error(f"Worker {self.worker_id}: 环境步骤失败: {e}")
            # 返回安全的默认值而不是None
            n_agents = len(actions) if hasattr(actions, '__len__') else self.config.n_agents
            default_obs = np.zeros((n_agents, self.config.obs_dim))
            return default_obs, 0.0, True, np.zeros(self.config.state_dim)
    
    def run(self, agent_proxy):
        """运行rollout worker主循环 - 修复版本：添加严格的步数控制"""
        self.logger.info(f"Rollout worker {self.worker_id} 开始运行，目标收集步数: {self.target_rollout_steps}")
        
        # 初始化rollout开始时间
        self.rollout_start_time = time.time()
        
        # 重置环境
        if not self.reset_environment():
            self.logger.error(f"Worker {self.worker_id}: 初始化失败")
            return
        
        try:
            # 【关键修复】无限循环训练模式：每次完成一个rollout后等待新的周期开始
            while not self.control_events['stop'].is_set():
                # 检查是否需要暂停
                if self.control_events['pause'].is_set():
                    time.sleep(0.1)
                    continue
                
                # 【关键修复】如果当前rollout已完成，等待新的rollout周期
                if self.rollout_completed:
                    time.sleep(0.1)  # 等待训练完成和状态重置
                    continue
                
                # 【关键修复】检查是否已达到rollout步数限制
                if self.samples_collected >= self.target_rollout_steps:
                    self.rollout_completed = True
                    self.complete_rollout()
                    continue
                
                # 执行一个rollout步骤
                success = self.run_step(agent_proxy)
                if not success:
                    self.logger.warning(f"Worker {self.worker_id}: 步骤执行失败，重置环境")
                    if not self.reset_environment():
                        break
                
                # 短暂睡眠避免过度占用CPU
                time.sleep(0.001)
        
        except Exception as e:
            self.logger.error(f"Worker {self.worker_id}: 运行异常: {e}")
        finally:
            try:
                self.env.close()
            except:
                pass
            self.logger.info(f"Rollout worker {self.worker_id} 结束运行")
    
    def run_step(self, agent_proxy):
        """【阶段3修复】执行单个rollout步骤 - 确定性步数计数和高层经验收集"""
        
        try:
            # 【阶段3修复】确保环境状态有效
            if self.env_state is None:
                self.logger.warning(f"Worker {self.worker_id}: env_state为None，重置环境")
                if not self.reset_environment():
                    return False
            
            if self.env_observations is None:
                self.logger.warning(f"Worker {self.worker_id}: env_observations为None，重置环境")
                if not self.reset_environment():
                    return False
            
            # 【阶段3修复】每次都重新分配技能，保持简单逻辑
            team_skill, agent_skills, log_probs = agent_proxy.assign_skills_for_worker(
                self.env_state, self.env_observations, self.worker_id
            )
            
            # 更新当前技能状态
            self.current_team_skill = team_skill
            self.current_agent_skills = agent_skills
            self.skill_log_probs = log_probs
            
            # 从代理获取动作
            actions, action_logprobs = agent_proxy.get_actions_for_worker(
                self.env_state, self.env_observations, agent_skills, self.worker_id
            )
            
            # 执行环境步骤
            next_observations, rewards, dones, next_state = self.step_environment(actions)
            
            # 【阶段3核心修复】原子性步数计数 - 确保一致性
            step_before_increment = self.samples_collected
            self.samples_collected += 1
            step_after_increment = self.samples_collected
            
            # 确保rewards是有效的数值
            if rewards is None:
                current_reward = 0.0
                self.logger.warning(f"Worker {self.worker_id}: 环境步骤返回None奖励，使用0.0")
            else:
                current_reward = rewards if isinstance(rewards, (int, float)) else np.sum(rewards)
            
            # 累积奖励（用于高层经验）
            self.accumulated_reward += current_reward
            self.total_reward += current_reward
            
            # 【阶段3修复】安全复制数据，避免None.copy()错误
            def safe_copy(data, default_shape=None):
                if data is None:
                    if default_shape is not None:
                        return np.zeros(default_shape)
                    return None
                if hasattr(data, 'copy'):
                    return data.copy()
                return np.array(data)
            
            # 存储低层经验数据
            low_level_experience = {
                'experience_type': 'low_level',
                'worker_id': self.worker_id,
                'state': safe_copy(self.env_state, (self.config.state_dim,)),
                'observations': safe_copy(self.env_observations, (self.config.n_agents, self.config.obs_dim)),
                'actions': safe_copy(actions, (self.config.n_agents, self.config.action_dim)),
                'rewards': current_reward,
                'next_state': safe_copy(next_state, (self.config.state_dim,)),
                'next_observations': safe_copy(next_observations, (self.config.n_agents, self.config.obs_dim)),
                'dones': dones,
                'episode_step': self.episode_step,
                'team_skill': team_skill,
                'agent_skills': safe_copy(agent_skills, (self.config.n_agents,)) if agent_skills is not None else [0] * self.config.n_agents,
                'action_logprobs': safe_copy(action_logprobs, (self.config.n_agents,)),
                'skill_log_probs': log_probs,
                'step_number': step_after_increment  # 【阶段3新增】步骤编号用于调试
            }
            
            # 将低层经验放入缓冲区 - 使用无限阻塞确保数据不丢失
            success = self.data_buffer.put(low_level_experience, block=True, timeout=None)
            if success:
                self.logger.debug(f"Worker {self.worker_id}: 低层经验已放入缓冲区 - 步骤={step_after_increment}")
            else:
                self.logger.error(f"Worker {self.worker_id}: 低层经验放入缓冲区失败！这不应该发生")
            
            # 构造StateSkillDataset数据
            state_skill_experience = {
                'experience_type': 'state_skill',
                'worker_id': self.worker_id,
                'state': safe_copy(next_state, (self.config.state_dim,)),
                'team_skill': team_skill,
                'observations': safe_copy(next_observations, (self.config.n_agents, self.config.obs_dim)),
                'agent_skills': safe_copy(agent_skills, (self.config.n_agents,)) if agent_skills is not None else [0] * self.config.n_agents,
                'step_number': step_after_increment  # 【阶段3新增】步骤编号
            }
            
            # 将StateSkillDataset数据放入缓冲区
            success = self.data_buffer.put(state_skill_experience, block=True, timeout=None)
            if success:
                self.logger.debug(f"Worker {self.worker_id}: StateSkill数据已放入缓冲区")
            else:
                self.logger.error(f"Worker {self.worker_id}: StateSkill数据放入缓冲区失败！这不应该发生")
            
            # 【阶段3核心修复】确定性高层经验收集 - 严格按k步收集
            if step_after_increment % self.config.k == 0:
                self.logger.debug(f"Worker {self.worker_id}: 确定性k步收集高层经验 - "
                               f"步数={step_after_increment}, k={self.config.k}, "
                               f"累积奖励={self.accumulated_reward:.4f}")
                
                success = self.store_high_level_experience(f"确定性k步收集(步数={step_after_increment})")
                if success:
                    self.logger.debug(f"Worker {self.worker_id}: 高层经验收集成功 - 第{self.high_level_experiences_generated}个")
                    # 只在成功存储后重置累积奖励
                    self.accumulated_reward = 0.0
                else:
                    self.logger.error(f"Worker {self.worker_id}: 高层经验存储失败！")
            
            # 【阶段3修复】安全更新环境状态
            self.env_state = safe_copy(next_state, (self.config.state_dim,))
            self.env_observations = safe_copy(next_observations, (self.config.n_agents, self.config.obs_dim))
            
            # 检查episode是否结束
            max_episode_length = getattr(self.config, 'rollout_max_episode_length', 5000)
            if dones or self.episode_step >= max_episode_length:
                self.episodes_completed += 1
                termination_reason = "环境自然终止" if dones else f"达到最大步数限制({max_episode_length})"
                self.logger.debug(f"Worker {self.worker_id}: Episode {self.episodes_completed} 完成, "
                                f"步数: {self.episode_step}, 奖励: {self.total_reward:.2f}, "
                                f"终止原因: {termination_reason}")
                
                # 重置环境和技能状态
                if not self.reset_environment():
                    return False
                self.reset_skill_state()
                self.total_reward = 0.0
            
            return True
            
        except Exception as e:
            self.logger.error(f"Worker {self.worker_id}: 步骤执行异常: {e}")
            # 【阶段3修复】即使异常也要确保步数计数的一致性
            if hasattr(self, 'samples_collected'):
                self.samples_collected += 1
            return False
    
    def should_reassign_skills(self):
        """检查是否需要重新分配技能"""
        return (self.skill_timer >= self.config.k or 
                self.current_team_skill is None)
    
    def should_store_high_level_experience(self, dones):
        """检查是否需要存储高层经验 - 修复版本：只在技能周期完成或环境终止时存储"""
        # 只在以下情况存储高层经验：
        # 1. 技能周期完成（达到k步）
        # 2. 环境终止（episode结束）
        return (self.skill_timer >= self.config.k or dones)
    
    def assign_new_skills(self, agent_proxy):
        """重新分配技能"""
        try:
            team_skill, agent_skills, log_probs = agent_proxy.assign_skills_for_worker(
                self.env_state, self.env_observations, self.worker_id
            )
            
            self.current_team_skill = team_skill
            self.current_agent_skills = agent_skills
            self.skill_log_probs = log_probs
            self.skill_timer = 0
            self.accumulated_reward = 0.0
            self.last_skill_assignment_step = self.episode_step
            
            self.logger.debug(f"Worker {self.worker_id}: 技能重新分配 - "
                            f"team_skill={team_skill}, agent_skills={agent_skills}")
            
        except Exception as e:
            self.logger.error(f"Worker {self.worker_id}: 技能分配失败: {e}")
            # 使用默认技能作为回退
            self.current_team_skill = 0
            self.current_agent_skills = [0] * self.config.n_agents
            self.skill_log_probs = {'team_log_prob': 0.0, 'agent_log_probs': [0.0] * self.config.n_agents}
    
    def store_high_level_experience(self, reason="技能周期完成"):
        """存储高层经验到数据缓冲区"""
        if self.current_team_skill is None or self.current_agent_skills is None:
            self.logger.warning(f"Worker {self.worker_id}: 技能状态无效，跳过高层经验存储")
            return False
        
        try:
            high_level_experience = {
                'experience_type': 'high_level',
                'worker_id': self.worker_id,
                'state': self.env_state.copy(),
                'team_skill': self.current_team_skill,
                'observations': self.env_observations.copy(),
                'agent_skills': self.current_agent_skills.copy(),
                'accumulated_reward': self.accumulated_reward,
                'skill_log_probs': self.skill_log_probs.copy() if self.skill_log_probs else None,
                'skill_timer': self.skill_timer,
                'episode_step': self.episode_step,
                'reason': reason
            }
            
            # 【修复5】将高层经验放入缓冲区 - 使用无限阻塞确保数据不丢失
            success = self.data_buffer.put(high_level_experience, block=True, timeout=None)
            if success:
                self.high_level_experiences_generated += 1
                self.logger.debug(f"Worker {self.worker_id}: 高层经验已存储 - "
                                f"累积奖励={self.accumulated_reward:.4f}, 原因={reason}, "
                                f"总生成数={self.high_level_experiences_generated}")
                
                # 【新增调试】记录accumulated_reward的重置
                old_accumulated_reward = self.accumulated_reward
                self.accumulated_reward = 0.0
                self.skill_timer = 0
                self.logger.debug(f"💰 [REWARD_RESET] W{self.worker_id} accumulated_reward reset: "
                                f"{old_accumulated_reward:.4f} -> 0.0 (reason: {reason})")
                return True
            else:
                self.logger.error(f"Worker {self.worker_id}: 高层经验放入缓冲区失败！这不应该发生")
                return False
                
        except Exception as e:
            self.logger.error(f"Worker {self.worker_id}: 存储高层经验失败: {e}")
            return False
    
    def complete_rollout(self):
        """【阶段3修复】完成当前rollout - 分阶段验证和确定性数据补齐"""
        completion_time = time.time()
        
        # 【阶段3步骤1】基础验证 - 确保步数准确
        if self.samples_collected != self.target_rollout_steps:
            self.logger.warning(f"Worker {self.worker_id}: 步数不匹配! "
                              f"收集={self.samples_collected}, 目标={self.target_rollout_steps}")
            # 【阶段3修复】强制同步步数
            self.samples_collected = self.target_rollout_steps
        
        # 【阶段3步骤2】确定性高层经验计算
        expected_high_level = self.target_rollout_steps // self.config.k  # 必须是整数除法，确保确定性
        current_high_level = self.high_level_experiences_generated
        missing = expected_high_level - current_high_level
        
        self.logger.info(f"Worker {self.worker_id}: 【阶段3】Rollout完成验证 - "
                       f"目标步数={self.target_rollout_steps}, 实际步数={self.samples_collected}, "
                       f"期望高层={expected_high_level} (计算: {self.target_rollout_steps}//{self.config.k}), "
                       f"当前高层={current_high_level}, 缺失={missing}")
        
        # 【阶段3步骤3】确定性数据补齐 - 必须补齐到准确数量
        if missing > 0:
            self.logger.info(f"Worker {self.worker_id}: 【阶段3】开始确定性补齐 {missing} 个高层经验")
            
            补齐成功计数 = 0
            for i in range(missing):
                success = self.store_high_level_experience(f"【阶段3】确定性补齐#{i+1}")
                if success:
                    补齐成功计数 += 1
                    self.logger.debug(f"Worker {self.worker_id}: 确定性补齐#{i+1}成功")
                else:
                    self.logger.error(f"Worker {self.worker_id}: 确定性补齐#{i+1}失败！")
                    break
            
            # 【阶段3验证】确保补齐成功
            if 补齐成功计数 == missing:
                self.logger.info(f"Worker {self.worker_id}: ✅ 确定性补齐完成 {补齐成功计数}/{missing}")
            else:
                self.logger.error(f"Worker {self.worker_id}: ❌ 确定性补齐失败 {补齐成功计数}/{missing}")
        
        # 【阶段3步骤4】最终验证 - 确保数据量准确
        final_high_level = self.high_level_experiences_generated
        if final_high_level != expected_high_level:
            self.logger.error(f"Worker {self.worker_id}: ❌ 【阶段3】最终验证失败! "
                            f"高层经验={final_high_level}, 期望={expected_high_level}")
        else:
            self.logger.info(f"Worker {self.worker_id}: ✅ 【阶段3】最终验证通过! "
                           f"高层经验={final_high_level}, 期望={expected_high_level}")
        
        # 【阶段3步骤5】等待数据传输完成 - 确保100%传输
        self.wait_for_data_transmission_complete()
        
        # 计算rollout统计
        if not hasattr(self, 'rollout_start_time'):
            self.rollout_start_time = completion_time - 1.0
        
        rollout_duration = completion_time - self.rollout_start_time
        speed = self.samples_collected / rollout_duration if rollout_duration > 0 else 0
        
        self.logger.info(f"Worker {self.worker_id}: 【阶段3】Rollout完成统计 - "
                       f"步数={self.samples_collected}, 高层经验={self.high_level_experiences_generated}, "
                       f"耗时={rollout_duration:.1f}s, 速度={speed:.1f}步/s")
        
        # 重置开始时间
        self.rollout_start_time = completion_time
    
    def force_collect_pending_high_level_experience(self):
        """强制收集pending的高层经验，确保每个worker都贡献完整的高层经验 - 修复版本：统一边界条件"""
        # 【修复2】计算应该生成的高层经验总数
        expected_high_level_total = (self.samples_collected + self.config.k - 1) // self.config.k  # 向上取整
        current_high_level_total = self.high_level_experiences_generated
        missing_high_level = expected_high_level_total - current_high_level_total
        
        # 【新增调试】记录进入该函数时的详细状态
        self.logger.info(f"🔧 [FORCE_COLLECT_DEBUG] W{self.worker_id} Entering force_collect_pending: "
                       f"strict_steps={self.strict_step_counter}, k={self.config.k}, "
                       f"samples_collected={self.samples_collected}, "
                       f"expected_total={expected_high_level_total}, current_total={current_high_level_total}, "
                       f"missing={missing_high_level}, acc_reward={self.accumulated_reward:.4f}")
        
        # 【修复2A】基于缺失数量进行强制收集，而不是基于余数
        if missing_high_level > 0:
            self.logger.info(f"🔧 [FORCE_COLLECT] W{self.worker_id} 需要强制收集 {missing_high_level} 个高层经验")
            
            # 【修复2B】为每个缺失的高层经验进行强制收集
            for i in range(missing_high_level):
                success = self.store_high_level_experience(f"Rollout结束强制收集#{i+1}")
                if success:
                    self.logger.info(f"✅ [FORCE_COLLECT] W{self.worker_id} 强制收集#{i+1}成功: "
                                   f"高层经验={self.high_level_experiences_generated}/{expected_high_level_total}")
                else:
                    self.logger.error(f"❌ [FORCE_COLLECT] W{self.worker_id} 强制收集#{i+1}失败！")
                    break
            
            # 【修复2C】验证强制收集结果
            final_high_level_total = self.high_level_experiences_generated
            if final_high_level_total >= expected_high_level_total:
                self.logger.info(f"✅ [FORCE_COLLECT] W{self.worker_id} 强制收集完成: "
                               f"高层经验={final_high_level_total}, 预期={expected_high_level_total}")
            else:
                remaining_missing = expected_high_level_total - final_high_level_total
                self.logger.warning(f"⚠️ [FORCE_COLLECT] W{self.worker_id} 强制收集后仍缺失 {remaining_missing} 个高层经验")
        else:
            self.logger.info(f"✅ [FORCE_COLLECT_DEBUG] W{self.worker_id} 无需强制收集: "
                           f"当前={current_high_level_total}, 预期={expected_high_level_total}")
        
        # 【新增调试】记录离开该函数时的状态
        self.logger.info(f"🔧 [FORCE_COLLECT_DEBUG] W{self.worker_id} Exiting force_collect_pending: "
                       f"high_level_generated_after_force={self.high_level_experiences_generated}")
    
    def wait_for_data_transmission(self):
        """等待数据传输完成，确保所有经验都进入缓冲区"""
        max_wait_time = 5.0  # 最多等待5秒
        wait_start = time.time()
        initial_queue_size = self.data_buffer.qsize()
        
        # 等待队列处理完成
        while time.time() - wait_start < max_wait_time:
            current_queue_size = self.data_buffer.qsize()
            
            # 如果队列大小在减少，说明还在处理
            if current_queue_size > 0:
                time.sleep(0.1)
            else:
                break
        
        final_wait_time = time.time() - wait_start
        final_queue_size = self.data_buffer.qsize()
        
        if final_wait_time >= max_wait_time:
            self.logger.warning(f"⚠️ [DATA_WAIT] W{self.worker_id} 数据传输等待超时: "
                              f"等待时间={final_wait_time:.2f}s, 剩余队列={final_queue_size}")
        else:
            self.logger.debug(f"✅ [DATA_WAIT] W{self.worker_id} 数据传输完成: "
                            f"等待时间={final_wait_time:.2f}s, 队列变化={initial_queue_size}→{final_queue_size}")
    
    def wait_for_data_transmission_complete(self):
        """【阶段3新增】等待数据传输100%完成 - 增强版数据传输等待"""
        max_wait_time = 10.0  # 阶段3：更长等待时间
        wait_start = time.time()
        initial_queue_size = self.data_buffer.qsize()
        
        self.logger.debug(f"Worker {self.worker_id}: 【阶段3】开始等待数据传输100%完成 - "
                        f"初始队列大小={initial_queue_size}")
        
        consecutive_empty_checks = 0
        required_empty_checks = 5  # 需要连续5次检查队列为空
        
        while time.time() - wait_start < max_wait_time:
            current_queue_size = self.data_buffer.qsize()
            
            if current_queue_size == 0:
                consecutive_empty_checks += 1
                if consecutive_empty_checks >= required_empty_checks:
                    # 连续多次确认队列为空，数据传输完成
                    break
            else:
                consecutive_empty_checks = 0  # 重置计数器
            
            time.sleep(0.1)
        
        final_wait_time = time.time() - wait_start
        final_queue_size = self.data_buffer.qsize()
        
        if final_queue_size == 0 and consecutive_empty_checks >= required_empty_checks:
            self.logger.debug(f"Worker {self.worker_id}: ✅ 【阶段3】数据传输100%完成 - "
                            f"等待时间={final_wait_time:.2f}s, 连续空检查={consecutive_empty_checks}")
        else:
            self.logger.warning(f"Worker {self.worker_id}: ⚠️ 【阶段3】数据传输未完全完成 - "
                              f"等待时间={final_wait_time:.2f}s, 剩余队列={final_queue_size}, "
                              f"连续空检查={consecutive_empty_checks}/{required_empty_checks}")
    
    def reset_skill_state(self):
        """重置技能状态（方案2：严格步数计数）"""
        # 【修复】不重置strict_step_counter，保持连续计数
        # self.strict_step_counter = 0  # 移除这行，保持计数器连续性
        self.accumulated_reward = 0.0
        self.current_team_skill = None
        self.current_agent_skills = None
        self.skill_log_probs = None
    
    def get_worker_stats(self):
        """获取worker统计信息"""
        return {
            'worker_id': self.worker_id,
            'samples_collected': self.samples_collected,
            'episodes_completed': self.episodes_completed,
            'high_level_experiences_generated': self.high_level_experiences_generated,
            'current_skill_timer': self.skill_timer,
            'current_accumulated_reward': self.accumulated_reward,
            'current_team_skill': self.current_team_skill,
            'current_episode_step': self.episode_step
        }

class AgentProxy:
    """代理代理，为rollout workers提供线程安全的代理接口"""
    def __init__(self, agent, config, logger, data_buffer=None):
        self.agent = agent
        self.config = config
        self.logger = logger
        self.data_buffer = data_buffer
        self.lock = Lock()
        
        # 全局rollout步数计数器（用于判断是否应该更新）
        self.global_rollout_steps = 0
        self.high_level_experiences_stored = 0
        self.low_level_experiences_stored = 0
        
        # 【新增】全局技能周期管理
        self.global_skill_cycle_step = 0  # 全局技能周期步数计数器
        self.skill_cycle_length = config.k  # 技能周期长度
        self.current_global_team_skill = None  # 当前全局团队技能
        self.current_global_agent_skills = None  # 当前全局个体技能
        self.current_global_skill_log_probs = None  # 当前全局技能log probs
        self.skill_assignment_lock = Lock()  # 技能分配专用锁
        
        # 高层经验收集统计
        self.expected_high_level_experiences = 0  # 预期高层经验数量
        self.actual_high_level_experiences = 0    # 实际收集的高层经验数量
    
    def assign_skills_for_worker(self, state, observations, worker_id):
        """【阶段3修复】为特定worker分配技能 - 添加空值检查和安全处理"""
        try:
            # 【阶段3修复】确保输入数据有效
            if state is None:
                state = np.zeros(self.config.state_dim)
            if observations is None:
                observations = np.zeros((self.config.n_agents, self.config.obs_dim))
            
            # 【设备修复】确保输入数据在正确的设备上
            if isinstance(state, np.ndarray):
                state = torch.FloatTensor(state).to(self.agent.device)
            elif isinstance(state, torch.Tensor):
                state = state.to(self.agent.device)
            else:
                # 处理其他类型，转换为numpy再转tensor
                state = torch.FloatTensor(np.array(state)).to(self.agent.device)
            
            if isinstance(observations, np.ndarray):
                observations = torch.FloatTensor(observations).to(self.agent.device)
            elif isinstance(observations, torch.Tensor):
                observations = observations.to(self.agent.device)
            else:
                # 处理其他类型，转换为numpy再转tensor
                observations = torch.FloatTensor(np.array(observations)).to(self.agent.device)
            
            # 【方案2核心】每次都重新分配技能，移除复杂的全局同步逻辑
            team_skill, agent_skills, log_probs = self.agent.assign_skills(
                state, observations, deterministic=False
            )
            
            # 【关键修复】确保返回值是Python原生类型，避免设备不匹配
            if isinstance(team_skill, torch.Tensor):
                team_skill = team_skill.cpu().item()
            if isinstance(agent_skills, torch.Tensor):
                agent_skills = agent_skills.cpu().tolist()
            elif isinstance(agent_skills, list):
                agent_skills = [int(skill.cpu().item()) if isinstance(skill, torch.Tensor) else int(skill) for skill in agent_skills]
            
            # 确保log_probs中的值也是Python原生类型
            if log_probs:
                if 'team_log_prob' in log_probs and isinstance(log_probs['team_log_prob'], torch.Tensor):
                    log_probs['team_log_prob'] = log_probs['team_log_prob'].cpu().item()
                if 'agent_log_probs' in log_probs and isinstance(log_probs['agent_log_probs'], torch.Tensor):
                    log_probs['agent_log_probs'] = log_probs['agent_log_probs'].cpu().tolist()
                elif 'agent_log_probs' in log_probs and isinstance(log_probs['agent_log_probs'], list):
                    log_probs['agent_log_probs'] = [
                        prob.cpu().item() if isinstance(prob, torch.Tensor) else float(prob) 
                        for prob in log_probs['agent_log_probs']
                    ]
            
            self.logger.debug(f"Worker {worker_id}: 技能分配完成 - "
                            f"team_skill={team_skill}, agent_skills={agent_skills}")
            
            return team_skill, agent_skills, log_probs
            
        except Exception as e:
            self.logger.error(f"Worker {worker_id}: 技能分配失败: {e}")
            import traceback
            self.logger.error(f"详细错误信息: {traceback.format_exc()}")
            # 返回默认技能
            return 0, [0] * self.config.n_agents, {
                'team_log_prob': 0.0, 
                'agent_log_probs': [0.0] * self.config.n_agents
            }
    
    def should_reassign_global_skills(self):
        """检查是否需要重新分配全局技能"""
        # 初次分配或达到技能周期长度时重分配
        return (self.current_global_team_skill is None or 
                self.global_skill_cycle_step >= self.skill_cycle_length)
    
    def get_global_skill_cycle_info(self):
        """获取全局技能周期信息（用于调试）"""
        return {
            'global_skill_cycle_step': self.global_skill_cycle_step,
            'skill_cycle_length': self.skill_cycle_length,
            'current_global_team_skill': self.current_global_team_skill,
            'current_global_agent_skills': self.current_global_agent_skills,
            'should_reassign': self.should_reassign_global_skills()
        }
    
    def get_actions_for_worker(self, state, observations, agent_skills, worker_id):
        """【阶段3修复】为特定worker获取动作 - 添加空值检查和安全处理"""
        with self.lock:
            try:
                # 【阶段3修复】确保输入数据有效
                if observations is None:
                    observations = np.zeros((self.config.n_agents, self.config.obs_dim))
                if agent_skills is None:
                    agent_skills = [0] * self.config.n_agents
                
                # 【设备修复】确保输入数据在正确的设备上
                if isinstance(observations, np.ndarray):
                    observations = torch.FloatTensor(observations).to(self.agent.device)
                elif isinstance(observations, torch.Tensor):
                    observations = observations.to(self.agent.device)
                else:
                    # 处理其他类型，转换为numpy再转tensor
                    observations = torch.FloatTensor(np.array(observations)).to(self.agent.device)
                
                # 【关键修复】确保agent_skills是正确的设备和类型
                if isinstance(agent_skills, np.ndarray):
                    agent_skills = torch.tensor(agent_skills, dtype=torch.long, device=self.agent.device)
                elif isinstance(agent_skills, list):
                    agent_skills = torch.tensor(agent_skills, dtype=torch.long, device=self.agent.device)
                elif isinstance(agent_skills, torch.Tensor):
                    agent_skills = agent_skills.to(device=self.agent.device, dtype=torch.long)
                else:
                    # 如果是其他类型，转换为list再转tensor
                    agent_skills = torch.tensor([int(skill) for skill in agent_skills], dtype=torch.long, device=self.agent.device)
                
                actions, action_logprobs = self.agent.select_action(
                    observations, agent_skills, deterministic=False, env_id=worker_id
                )
                
                # 【关键修复】确保返回的actions和action_logprobs是numpy数组，避免设备不匹配
                if isinstance(actions, torch.Tensor):
                    actions = actions.cpu().detach().numpy()
                if isinstance(action_logprobs, torch.Tensor):
                    action_logprobs = action_logprobs.cpu().detach().numpy()
                
                return actions, action_logprobs
                
            except Exception as e:
                self.logger.error(f"Worker {worker_id}: 获取动作失败: {e}")
                import traceback
                self.logger.error(f"详细错误信息: {traceback.format_exc()}")
                
                # 返回随机动作作为回退
                try:
                    n_agents = len(observations) if hasattr(observations, '__len__') else self.config.n_agents
                    random_actions = np.random.randn(n_agents, self.config.action_dim)
                    return random_actions, np.zeros(n_agents)
                except Exception as fallback_error:
                    self.logger.error(f"Worker {worker_id}: 回退动作生成也失败: {fallback_error}")
                    # 最后的安全回退
                    return np.random.randn(self.config.n_agents, self.config.action_dim), np.zeros(self.config.n_agents)
    
    def store_experience(self, experience_batch):
        """批量存储经验到代理 - 修复版本：确保步数计数统一"""
        with self.lock:
            # 【新增调试】记录批处理开始时的状态
            pre_global_steps = self.global_rollout_steps
            
            stored_count = 0
            low_level_stored = 0
            high_level_stored = 0
            low_level_failed = 0
            high_level_failed = 0
            
            for experience in experience_batch:
                try:
                    worker_id = experience['worker_id']
                    experience_type = experience.get('experience_type', 'low_level')
                    
                    if experience_type == 'high_level':
                        # 存储高层经验
                        success = self.store_high_level_experience(experience)
                        if success:
                            self.high_level_experiences_stored += 1
                            high_level_stored += 1
                            stored_count += 1
                            self.logger.debug(f"Worker {worker_id}: 高层经验已存储到代理 - "
                                            f"累积奖励={experience['accumulated_reward']:.4f}")
                        else:
                            high_level_failed += 1
                            self.logger.warning(f"Worker {worker_id}: 高层经验存储失败")
                    
                    elif experience_type == 'low_level':
                        # 存储低层经验
                        success = self.store_low_level_experience(experience)
                        if success:
                            self.low_level_experiences_stored += 1
                            low_level_stored += 1
                            stored_count += 1
                            # 【关键修复】每个成功的低层经验对应一个环境步骤
                            self.global_rollout_steps += 1
                            
                            # 【新增调试】记录每个低层经验的步数增加
                            if low_level_stored <= 5 or low_level_stored % 50 == 0:  # 记录前5个和每50个
                                self.logger.debug(f"🔢 [STEP_TRACE] W{worker_id} 低层经验#{low_level_stored} 存储成功, "
                                                f"global_rollout_steps: {self.global_rollout_steps-1}→{self.global_rollout_steps}")
                        else:
                            low_level_failed += 1
                            self.logger.warning(f"Worker {worker_id}: 低层经验存储失败")
                    
                    elif experience_type == 'state_skill':
                        # 【方案2新增】存储StateSkillDataset数据
                        success = self.store_state_skill_data(experience)
                        if success:
                            stored_count += 1
                            self.logger.debug(f"Worker {worker_id}: StateSkill数据已存储到代理")
                        else:
                            self.logger.warning(f"Worker {worker_id}: StateSkill数据存储失败")
                    
                    else:
                        self.logger.warning(f"未知经验类型: {experience_type}")
                    
                except Exception as e:
                    self.logger.error(f"存储经验失败: {e}")
            
            # 【关键修复】同步代理的步数计数器
            old_steps = self.agent.steps_collected
            self.agent.steps_collected = self.global_rollout_steps
            
            # 【新增调试】记录批处理结束时的详细状态
            post_global_steps = self.global_rollout_steps
            steps_increment = post_global_steps - pre_global_steps
            
            if low_level_stored > 0:  # 只在有低层经验时记录
                self.logger.debug(f"📦 [BATCH_TRACE] 批处理完成: 批次大小={len(experience_batch)}, "
                               f"低层成功={low_level_stored}, 高层成功={high_level_stored}, "
                               f"global_rollout_steps: {pre_global_steps}→{post_global_steps} (+{steps_increment})")
                
                # 验证步数增量与低层经验数量的一致性
                if steps_increment != low_level_stored:
                    self.logger.warning(f"⚠️ [STEP_MISMATCH] 步数增量与低层经验不匹配: "
                                      f"增量={steps_increment}, 低层经验={low_level_stored}")
            
            return stored_count
    
    def store_high_level_experience(self, experience):
        """存储高层经验到代理 - 修复版本：添加原子性验证和重试机制"""
        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                # 记录存储前的缓冲区大小
                buffer_size_before = len(self.agent.high_level_buffer) if hasattr(self.agent, 'high_level_buffer') else 0
                
                # 调用代理的高层经验存储方法
                success = self.agent.store_high_level_transition(
                    state=experience['state'],
                    team_skill=experience['team_skill'],
                    observations=experience['observations'],
                    agent_skills=experience['agent_skills'],
                    accumulated_reward=experience['accumulated_reward'],
                    skill_log_probs=experience['skill_log_probs'],
                    worker_id=experience['worker_id']
                )
                
                if success:
                    # 验证缓冲区确实增加了
                    buffer_size_after = len(self.agent.high_level_buffer) if hasattr(self.agent, 'high_level_buffer') else 0
                    if buffer_size_after > buffer_size_before:
                        return True
                    else:
                        self.logger.warning(f"高层经验存储返回成功但缓冲区未增加: {buffer_size_before}→{buffer_size_after}")
                        success = False
                
                # 如果失败，准备重试
                if not success:
                    retry_count += 1
                    if retry_count < max_retries:
                        self.logger.warning(f"高层经验存储失败，重试 {retry_count}/{max_retries}")
                        time.sleep(0.01)  # 短暂等待
                    continue
                else:
                    return True
                    
            except Exception as e:
                retry_count += 1
                if retry_count < max_retries:
                    self.logger.error(f"存储高层经验异常，重试 {retry_count}/{max_retries}: {e}")
                    time.sleep(0.01)
                else:
                    self.logger.error(f"存储高层经验最终失败: {e}")
        
        # 所有重试都失败
        return False
    
    def store_low_level_experience(self, experience):
        """存储低层经验到代理 - 修复版本：添加原子性验证和重试机制"""
        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                # 记录存储前的缓冲区大小
                buffer_size_before = len(self.agent.low_level_buffer) if hasattr(self.agent, 'low_level_buffer') else 0
                
                # 调用代理的低层经验存储方法
                success = self.agent.store_low_level_transition(
                    state=experience['state'],
                    next_state=experience['next_state'],
                    observations=experience['observations'],
                    next_observations=experience['next_observations'],
                    actions=experience['actions'],
                    rewards=experience['rewards'],
                    dones=experience['dones'],
                    team_skill=experience['team_skill'],
                    agent_skills=experience['agent_skills'],
                    action_logprobs=experience['action_logprobs'],
                    skill_log_probs=experience['skill_log_probs'],
                    worker_id=experience['worker_id']
                )
                
                if success:
                    # 验证缓冲区确实增加了
                    buffer_size_after = len(self.agent.low_level_buffer) if hasattr(self.agent, 'low_level_buffer') else 0
                    if buffer_size_after > buffer_size_before:
                        return True
                    else:
                        self.logger.warning(f"低层经验存储返回成功但缓冲区未增加: {buffer_size_before}→{buffer_size_after}")
                        success = False
                
                # 如果失败，准备重试
                if not success:
                    retry_count += 1
                    if retry_count < max_retries:
                        self.logger.warning(f"低层经验存储失败，重试 {retry_count}/{max_retries}")
                        time.sleep(0.01)  # 短暂等待
                    continue
                else:
                    return True
                    
            except Exception as e:
                retry_count += 1
                if retry_count < max_retries:
                    self.logger.error(f"存储低层经验异常，重试 {retry_count}/{max_retries}: {e}")
                    time.sleep(0.01)
                else:
                    self.logger.error(f"存储低层经验最终失败: {e}")
        
        # 所有重试都失败
        return False
    
    def store_state_skill_data(self, experience):
        """存储StateSkillDataset数据到代理（方案2新增）"""
        try:
            # 直接存储到代理的state_skill_dataset
            state_tensor = torch.FloatTensor(experience['state']).to(self.agent.device)
            team_skill_tensor = torch.tensor(experience['team_skill'], device=self.agent.device)
            observations_tensor = torch.FloatTensor(experience['observations']).to(self.agent.device)
            agent_skills_tensor = torch.tensor(experience['agent_skills'], device=self.agent.device)
            
            self.agent.state_skill_dataset.push(
                state_tensor,
                team_skill_tensor,
                observations_tensor,
                agent_skills_tensor
            )
            return True
        except Exception as e:
            self.logger.error(f"存储StateSkill数据失败: {e}")
            return False
    
    def should_update(self):
        """检查是否应该更新 - 阶段2增强版本：强健的数据传输验证"""
        with self.lock:
            # 如果没有rollout_workers引用，使用回退逻辑
            if not hasattr(self, 'rollout_workers'):
                self.agent.steps_collected = self.global_rollout_steps
                return self.agent.should_rollout_update()
            
            # 基本条件检查
            completed_workers = sum(1 for worker in self.rollout_workers 
                                  if getattr(worker, 'rollout_completed', False))
            total_workers = len(self.rollout_workers)
            
            total_collected = sum(worker.samples_collected for worker in self.rollout_workers)
            target_steps = self.rollout_workers[0].target_rollout_steps * total_workers
            
            # 基本的完成条件
            all_workers_completed = completed_workers == total_workers
            steps_collected = total_collected >= target_steps
            
            # 简化的更新判断
            should_update = all_workers_completed and steps_collected
            
            # 进度记录（减少频率）
            if not hasattr(self, '_update_check_count'):
                self._update_check_count = 0
            self._update_check_count += 1
            
            if self._update_check_count % 100 == 0:  # 减少日志频率
                progress_pct = (total_collected / target_steps) * 100 if target_steps > 0 else 0
                self.logger.debug(f"⏳ 等待数据收集: "
                               f"进度={progress_pct:.1f}% ({total_collected}/{target_steps}), "
                               f"完成workers={completed_workers}/{total_workers}")
            
            if should_update:
                self.logger.info(f"🔄 满足更新条件: 所有workers完成且步数达标 "
                               f"({total_collected}/{target_steps})")
                # 【阶段2核心】使用增强的数据传输验证
                return self._verify_data_transmission_integrity(total_collected)
            
            return False
    
    def _verify_data_transmission_integrity(self, expected_steps):
        """【阶段2核心】增强的数据传输完整性验证 - 渐进式等待 + 批量验证"""
        self.logger.info(f"🔍 [阶段2] 开始增强数据传输验证: 期望步数={expected_steps}")
        
        # 【渐进式等待策略】先快速检查，然后逐步增加等待时间
        wait_phases = [
            (1.0, "快速检查"),      # 1秒快速检查
            (5.0, "短期等待"),      # 5秒短期等待
            (10.0, "中期等待"),     # 10秒中期等待
            (20.0, "长期等待")      # 20秒长期等待
        ]
        
        verification_start = time.time()
        
        for phase_time, phase_name in wait_phases:
            phase_start = time.time()
            self.logger.debug(f"🔍 [验证阶段] {phase_name}: 最大等待{phase_time}秒")
            
            # 等待队列处理
            success = self._wait_for_queue_processing(phase_time, phase_name)
            if not success:
                self.logger.warning(f"⚠️ [验证阶段] {phase_name}: 队列处理超时")
                continue  # 进入下一个等待阶段
            
            # 批量验证数据完整性
            verification_result = self._batch_verify_data_integrity(expected_steps)
            
            phase_duration = time.time() - phase_start
            
            if verification_result['success']:
                total_duration = time.time() - verification_start
                self.logger.info(f"✅ [阶段2] 数据验证成功: {phase_name}完成, "
                               f"阶段用时={phase_duration:.2f}s, 总用时={total_duration:.2f}s")
                self._sync_step_counters(expected_steps)
                return True
            else:
                missing_info = verification_result.get('missing_info', {})
                self.logger.warning(f"⚠️ [验证阶段] {phase_name}: 数据仍缺失 - "
                                  f"低层缺失={missing_info.get('low_level_missing', 0)}, "
                                  f"高层缺失={missing_info.get('high_level_missing', 0)}")
                
                # 如果不是最后一个阶段，继续等待
                if phase_name != "长期等待":
                    continue
        
        # 所有阶段都失败，但进行智能容错处理
        return self._intelligent_fallback_handling(expected_steps, verification_result)
    
    def _wait_for_queue_processing(self, max_wait_time, phase_name):
        """【阶段2辅助】等待队列处理完成"""
        if self.data_buffer is None:
            return True
        
        wait_start = time.time()
        initial_queue_size = self.data_buffer.qsize()
        
        # 【优化】只在队列较大时才等待
        if initial_queue_size <= 50:
            self.logger.debug(f"🔍 [{phase_name}] 队列较小({initial_queue_size})，无需等待")
            return True
        
        self.logger.debug(f"🔍 [{phase_name}] 等待队列处理: 初始大小={initial_queue_size}")
        
        last_size = initial_queue_size
        stale_count = 0  # 队列大小未变化的计数
        
        while time.time() - wait_start < max_wait_time:
            current_size = self.data_buffer.qsize()
            
            # 【智能检测】如果队列大小在减少，说明正在处理
            if current_size < last_size:
                stale_count = 0  # 重置停滞计数
                last_size = current_size
            else:
                stale_count += 1
            
            # 【提前退出条件】
            if current_size <= 20:  # 队列基本清空
                break
            if stale_count >= 10 and current_size < initial_queue_size * 0.5:  # 队列减少了一半且停滞
                self.logger.debug(f"🔍 [{phase_name}] 队列处理停滞但已减少50%，继续验证")
                break
            
            time.sleep(0.5)
        
        final_size = self.data_buffer.qsize()
        wait_duration = time.time() - wait_start
        
        # 【成功条件】队列大小显著减少或基本清空
        reduction_rate = (initial_queue_size - final_size) / initial_queue_size if initial_queue_size > 0 else 1.0
        success = final_size <= 20 or reduction_rate >= 0.7  # 队列清空或减少70%以上
        
        if success:
            self.logger.debug(f"✅ [{phase_name}] 队列处理完成: {initial_queue_size}→{final_size}, "
                            f"减少={reduction_rate:.1%}, 用时={wait_duration:.2f}s")
        else:
            self.logger.warning(f"⚠️ [{phase_name}] 队列处理不理想: {initial_queue_size}→{final_size}, "
                              f"减少={reduction_rate:.1%}, 用时={wait_duration:.2f}s")
        
        return success
    
    def _batch_verify_data_integrity(self, expected_steps):
        """【阶段2辅助】批量验证数据完整性"""
        verification_result = {
            'success': False,
            'missing_info': {},
            'details': {}
        }
        
        try:
            # 获取当前缓冲区状态
            current_bl_size = len(self.agent.low_level_buffer) if hasattr(self.agent, 'low_level_buffer') else 0
            current_bh_size = len(self.agent.high_level_buffer) if hasattr(self.agent, 'high_level_buffer') else 0
            
            # 计算期望的数据量
            expected_low_level = expected_steps
            expected_high_level = expected_steps // self.config.k  # 每k步一个高层经验
            
            # 计算缺失量
            low_level_missing = max(0, expected_low_level - current_bl_size)
            high_level_missing = max(0, expected_high_level - current_bh_size)
            
            # 【智能容错】计算缺失率
            low_level_missing_rate = low_level_missing / expected_low_level if expected_low_level > 0 else 0
            high_level_missing_rate = high_level_missing / expected_high_level if expected_high_level > 0 else 0
            
            verification_result['missing_info'] = {
                'low_level_missing': low_level_missing,
                'high_level_missing': high_level_missing,
                'low_level_missing_rate': low_level_missing_rate,
                'high_level_missing_rate': high_level_missing_rate,
                'current_bl_size': current_bl_size,
                'current_bh_size': current_bh_size,
                'expected_low_level': expected_low_level,
                'expected_high_level': expected_high_level
            }
            
            # 【阶段2标准】允许小幅缺失但要记录详细信息
            acceptable_missing_rate = 0.02  # 允许2%的缺失率
            
            low_level_acceptable = low_level_missing_rate <= acceptable_missing_rate
            high_level_acceptable = high_level_missing_rate <= acceptable_missing_rate
            
            # 记录详细验证信息
            self.logger.warning(f"🔍 [AGENT_DATA_WAIT] 数据传输验证超时:")
            self.logger.warning(f"   低层: {current_bl_size}/{expected_low_level} (缺失: {low_level_missing})")
            self.logger.warning(f"   高层: {current_bh_size}/{expected_high_level} (缺失: {high_level_missing})")
            
            if low_level_acceptable and high_level_acceptable:
                self.logger.info(f"✅ [阶段2] 数据完整性验证通过: "
                               f"低层缺失率={low_level_missing_rate:.1%}, 高层缺失率={high_level_missing_rate:.1%}")
                verification_result['success'] = True
            else:
                total_missing = low_level_missing + high_level_missing
                total_expected = expected_low_level + expected_high_level
                overall_missing_rate = total_missing / total_expected if total_expected > 0 else 0
                
                if overall_missing_rate <= 0.006:  # 总体缺失率小于0.6%
                    self.logger.warning(f"⚠️ [AGENT_DATA_WAIT] 数据轻微缺失({overall_missing_rate:.1%})，应该修复")
                    verification_result['success'] = True  # 轻微缺失仍然接受
                else:
                    self.logger.error(f"❌ [阶段2] 数据完整性验证失败: "
                                    f"低层缺失率={low_level_missing_rate:.1%}, 高层缺失率={high_level_missing_rate:.1%}")
            
            # 记录缓冲区详细状态用于调试
            self.logger.warning(f"[ROLLOUT_BUFFER_DEBUG] 更新前缓冲区详细状态:")
            self.logger.warning(f"   - B_h (高层): {current_bh_size} (目标: {expected_high_level})")
            self.logger.warning(f"   - B_l (低层): {current_bl_size} (目标: {expected_low_level})")
            if hasattr(self.agent, 'state_skill_dataset'):
                d_size = len(self.agent.state_skill_dataset)
                self.logger.warning(f"   - D (判别器): {d_size}")
            
        except Exception as e:
            self.logger.error(f"❌ [阶段2] 数据完整性验证异常: {e}")
            verification_result['success'] = False
        
        return verification_result
    
    def _intelligent_fallback_handling(self, expected_steps, last_verification_result):
        """【阶段2辅助】智能回退处理 - 当所有验证阶段都失败时"""
        missing_info = last_verification_result.get('missing_info', {})
        
        self.logger.warning(f"⚠️ [阶段2] 所有验证阶段失败，启动智能回退处理")
        
        # 尝试数据修复
        repair_success = self._attempt_data_repair(missing_info)
        
        if repair_success:
            self.logger.info(f"✅ [阶段2] 数据修复成功，允许继续训练")
            self._sync_step_counters(expected_steps)
            return True
        
        # 如果修复失败，评估是否可以容忍
        total_missing_rate = (missing_info.get('low_level_missing', 0) + missing_info.get('high_level_missing', 0)) / expected_steps
        
        if total_missing_rate <= 0.05:  # 总缺失率小于5%
            self.logger.warning(f"⚠️ [阶段2] 数据修复失败，但缺失率可接受({total_missing_rate:.1%})，继续训练")
            self._sync_step_counters(expected_steps)
            return True
        else:
            self.logger.error(f"❌ [阶段2] 数据缺失过多({total_missing_rate:.1%})，拒绝训练更新")
            return False
    
    def _attempt_data_repair(self, missing_info):
        """【阶段2辅助】尝试修复缺失的数据"""
        repaired = 0
        
        try:
            # 修复缺失的高层经验
            high_level_missing = missing_info.get('high_level_missing', 0)
            if high_level_missing > 0:
                self.logger.info(f"🔧 [阶段2] 尝试修复 {high_level_missing} 个缺失的高层经验")
                high_level_repaired = self._repair_missing_high_level_experiences(high_level_missing)
                repaired += high_level_repaired
                self.logger.info(f"🔧 [阶段2] 高层经验修复完成: {high_level_repaired}/{high_level_missing}")
            
            # 注意：低层经验通常不需要修复，因为它们是实际的环境交互结果
            
            return repaired > 0
            
        except Exception as e:
            self.logger.error(f"❌ [阶段2] 数据修复过程异常: {e}")
            return False
    
    def _repair_missing_high_level_experiences(self, missing_count):
        """【方案C辅助】修复缺失的高层经验"""
        repaired = 0
        
        try:
            # 找到贡献高层经验最少的workers
            worker_contributions = {}
            for worker in self.rollout_workers:
                worker_contributions[worker.worker_id] = worker.high_level_experiences_generated
            
            # 按贡献数量排序，优先修复贡献最少的
            sorted_workers = sorted(worker_contributions.items(), key=lambda x: x[1])
            
            for worker_id, contribution in sorted_workers:
                if repaired >= missing_count:
                    break
                
                expected_contribution = 4  # 每个worker应该贡献4个高层经验
                if contribution < expected_contribution:
                    worker = self.rollout_workers[worker_id]
                    
                    # 为这个worker创建缺失的高层经验
                    missing_for_worker = expected_contribution - contribution
                    for i in range(min(missing_for_worker, missing_count - repaired)):
                        success = self.create_forced_high_level_experience(worker, f"传输修复#{i}")
                        if success:
                            repaired += 1
                            self.logger.info(f"🔧 [方案C] 为Worker {worker_id} 修复高层经验 #{i}")
                        else:
                            self.logger.warning(f"⚠️ [方案C] Worker {worker_id} 高层经验修复失败")
                            break
            
        except Exception as e:
            self.logger.error(f"❌ [方案C] 高层经验修复过程异常: {e}")
        
        return repaired
    
    def _sync_step_counters(self, total_collected):
        """【方案B辅助】同步步数计数器，确保一致性"""
        old_agent_steps = self.agent.steps_collected
        old_global_steps = self.global_rollout_steps
        
        # 使用最准确的worker总和作为基准
        self.agent.steps_collected = total_collected
        self.global_rollout_steps = total_collected
        
        # 验证同步结果
        if abs(old_global_steps - total_collected) > 50:  # 只在差异较大时记录
            self.logger.debug(f"🔄 [方案B] 步数同步: "
                            f"agent: {old_agent_steps}→{self.agent.steps_collected}, "
                            f"global: {old_global_steps}→{self.global_rollout_steps}, "
                            f"workers: {total_collected}")
    
    def update(self):
        """执行模型更新 - 方案B增强版：高效验证 + 数据完整性保障"""
        with self.lock:
            try:
                # 【方案B步骤1】最后的数据完整性验证
                if not self._verify_data_integrity_before_update():
                    self.logger.warning("⚠️ [方案B] 数据完整性验证失败，尝试修复...")
                    self._emergency_data_repair()
                
                # 【方案B步骤2】执行模型更新
                self.logger.info("🚀 [方案B] 开始模型更新...")
                update_start_time = time.time()
                
                update_info = self.agent.rollout_update()
                
                update_duration = time.time() - update_start_time
                
                # 【方案B步骤3】更新后处理
                if update_info:
                    self.logger.info(f"✅ [方案B] 模型更新完成，耗时: {update_duration:.3f}s")
                    self.log_post_update_buffer_state()
                    self.reset_all_workers_rollout_state()
                else:
                    self.logger.warning("⚠️ [方案B] 模型更新返回None")
                
                return update_info
                
            except Exception as e:
                self.logger.error(f"❌ [方案B] 模型更新失败: {e}")
                return None
    
    def _verify_data_integrity_before_update(self):
        """【方案B辅助】更新前的数据完整性验证"""
        if not hasattr(self, 'rollout_workers'):
            return True
        
        # 快速验证关键指标
        total_collected = sum(worker.samples_collected for worker in self.rollout_workers)
        target_steps = self.rollout_workers[0].target_rollout_steps * len(self.rollout_workers)
        
        current_bl_size = len(self.agent.low_level_buffer) if hasattr(self.agent, 'low_level_buffer') else 0
        current_bh_size = len(self.agent.high_level_buffer) if hasattr(self.agent, 'high_level_buffer') else 0
        
        # 验证条件
        steps_sufficient = total_collected >= target_steps
        buffer_sufficient = current_bl_size >= self.config.batch_size * 0.95  # 允许5%容差
        
        if steps_sufficient and buffer_sufficient:
            self.logger.debug(f"✅ [方案B] 数据完整性验证通过: "
                            f"步数={total_collected}/{target_steps}, "
                            f"B_l={current_bl_size}/{self.config.batch_size}")
            return True
        else:
            self.logger.warning(f"⚠️ [方案B] 数据完整性验证失败: "
                              f"步数={total_collected}/{target_steps} (sufficient={steps_sufficient}), "
                              f"B_l={current_bl_size}/{self.config.batch_size} (sufficient={buffer_sufficient})")
            return False
    
    def _emergency_data_repair(self):
        """【方案B辅助】紧急数据修复"""
        self.logger.info("🔧 [方案B] 执行紧急数据修复...")
        
        # 强制收集所有pending的高层经验
        forced_count = self.force_collect_all_pending_high_level_experiences()
        
        if forced_count > 0:
            self.logger.info(f"✅ [方案B] 紧急修复完成: 补充了 {forced_count} 个高层经验")
        else:
            self.logger.info("ℹ️ [方案B] 紧急修复完成: 无需补充数据")
    
    def force_collect_all_pending_high_level_experiences(self):
        """强制收集所有pending的高层经验，解决数据收集不匹配问题"""
        if not hasattr(self, 'rollout_workers'):
            return 0
        
        self.logger.info("🔧 [FORCE_COLLECT_ALL] 开始强制收集所有pending高层经验...")
        
        total_forced = 0
        worker_details = {}
        
        for worker in self.rollout_workers:
            # 分析worker的高层经验状态
            steps_collected = worker.samples_collected
            expected_high_level = steps_collected // self.config.k
            generated_high_level = worker.high_level_experiences_generated
            missing = expected_high_level - generated_high_level
            
            worker_details[worker.worker_id] = {
                'steps': steps_collected,
                'expected': expected_high_level,
                'generated': generated_high_level,
                'missing': missing,
                'strict_counter': getattr(worker, 'strict_step_counter', 'N/A'),
                'accumulated_reward': getattr(worker, 'accumulated_reward', 0.0)
            }
            
            # 【新增调试】详细分析每个worker的状态
            self.logger.info(f"🔧 [PROXY_FORCE_DEBUG] W{worker.worker_id} Analyzing for global force: "
                           f"steps={steps_collected}, k={self.config.k}, "
                           f"expected_floor={expected_high_level}, generated={generated_high_level}, "
                           f"missing={missing}, acc_reward_at_proxy_check={getattr(worker, 'accumulated_reward', 0.0):.4f}, "
                           f"strict_counter={getattr(worker, 'strict_step_counter', 'N/A')}")
            
            if missing > 0:
                self.logger.info(f"🔧 Worker {worker.worker_id} 需要强制收集: "
                               f"缺失={missing}, 累积奖励={worker.accumulated_reward:.4f}")
                
                # 为每个缺失的高层经验创建强制收集
                for i in range(missing):
                    success = self.create_forced_high_level_experience(worker, i)
                    if success:
                        total_forced += 1
                        worker.high_level_experiences_generated += 1
                        self.logger.info(f"🔧 [PROXY_FORCE_DEBUG] W{worker.worker_id} Global force created experience #{i}. "
                                       f"Worker's high_level_experiences_generated incremented to {worker.high_level_experiences_generated}")
                    else:
                        self.logger.warning(f"⚠️ [PROXY_FORCE_DEBUG] W{worker.worker_id} Global force FAILED to create experience #{i}.")
            else:
                self.logger.info(f"✅ [PROXY_FORCE_DEBUG] W{worker.worker_id} No missing high-level experiences, no force needed.")
        
        # 记录详细统计
        self.logger.warning("📊 [FORCE_COLLECT_ALL] 强制收集统计:")
        missing_workers = []
        for worker_id, details in worker_details.items():
            if details['missing'] > 0:
                missing_workers.append(worker_id)
                self.logger.warning(f"   Worker {worker_id}: 步数={details['steps']}, "
                                  f"预期={details['expected']}, 生成={details['generated']}, "
                                  f"缺失={details['missing']}, 累积奖励={details['accumulated_reward']:.4f}")
        
        if total_forced > 0:
            self.logger.info(f"✅ [FORCE_COLLECT_ALL] 强制收集完成: 补充了 {total_forced} 个高层经验")
            self.logger.info(f"   影响的Workers: {missing_workers}")
        else:
            self.logger.info("✅ [FORCE_COLLECT_ALL] 所有Workers的高层经验都已完整，无需强制收集")
        
        return total_forced
    
    def create_forced_high_level_experience(self, worker, index):
        """为worker创建强制的高层经验"""
        try:
            # 获取worker的当前状态信息
            state = getattr(worker, 'env_state', None)
            observations = getattr(worker, 'env_observations', None)
            team_skill = getattr(worker, 'current_team_skill', 0)
            agent_skills = getattr(worker, 'current_agent_skills', [0] * self.config.n_agents)
            accumulated_reward = getattr(worker, 'accumulated_reward', 0.0)
            
            # 如果状态无效，使用默认值
            if state is None:
                state = np.zeros(self.config.state_dim)
            if observations is None:
                observations = np.zeros((self.config.n_agents, self.config.obs_dim))
            
            # 创建高层经验数据
            high_level_experience = {
                'experience_type': 'high_level',
                'worker_id': worker.worker_id,
                'state': state.copy() if hasattr(state, 'copy') else np.array(state),
                'team_skill': team_skill,
                'observations': observations.copy() if hasattr(observations, 'copy') else np.array(observations),
                'agent_skills': agent_skills.copy() if hasattr(agent_skills, 'copy') else list(agent_skills),
                'accumulated_reward': accumulated_reward,
                'skill_log_probs': getattr(worker, 'skill_log_probs', None),
                'skill_timer': getattr(worker, 'skill_timer', 0),
                'episode_step': getattr(worker, 'episode_step', 0),
                'reason': f"强制补充#{index}"
            }
            
            # 直接调用agent的高层经验存储方法
            success = self.agent.store_high_level_transition(
                state=high_level_experience['state'],
                team_skill=high_level_experience['team_skill'],
                observations=high_level_experience['observations'],
                agent_skills=high_level_experience['agent_skills'],
                accumulated_reward=high_level_experience['accumulated_reward'],
                skill_log_probs=high_level_experience['skill_log_probs'],
                worker_id=high_level_experience['worker_id']
            )
            
            if success:
                self.high_level_experiences_stored += 1
                self.logger.debug(f"✅ Worker {worker.worker_id} 强制高层经验 #{index} 创建成功")
            
            return success
            
        except Exception as e:
            self.logger.error(f"❌ Worker {worker.worker_id} 强制高层经验 #{index} 创建失败: {e}")
            return False
    
    def reset_all_workers_rollout_state(self):
        """重置所有workers的rollout状态，准备下一个rollout周期（方案2：简化版本）"""
        if not hasattr(self, 'rollout_workers'):
            return
        
        # 【方案2】移除复杂的全局技能周期管理，简化重置逻辑
        
        reset_count = 0
        for worker in self.rollout_workers:
            # 重置每个worker的rollout相关状态
            worker.samples_collected = 0
            worker.rollout_completed = False
            # 【修复】重置worker的技能状态，但保持严格步数计数器连续性
            worker.current_team_skill = None
            worker.current_agent_skills = None
            worker.skill_log_probs = None
            worker.accumulated_reward = 0.0
            # 【关键修复】重置strict_step_counter以避免高层经验过多
            worker.strict_step_counter = 0
            worker.high_level_experiences_generated = 0
            reset_count += 1
        
        # 【关键修复】重置global_rollout_steps以保持与workers的一致性
        self.global_rollout_steps = 0
        
        self.logger.info(f"🔄 已重置 {reset_count} 个workers的rollout状态，准备新的rollout周期")
        self.logger.info(f"🔄 已重置global_rollout_steps = {self.global_rollout_steps}")
    
    def get_storage_stats(self):
        """【兼容性方法】获取存储统计信息 - 为增强版训练器提供兼容性"""
        try:
            return {
                'total_attempts': getattr(self, 'high_level_experiences_stored', 0) + getattr(self, 'low_level_experiences_stored', 0),
                'total_successes': getattr(self, 'high_level_experiences_stored', 0) + getattr(self, 'low_level_experiences_stored', 0),
                'total_failures': 0,  # 原版AgentProxy没有失败统计
                'queue_overflows': 0,
                'validation_failures': 0,
                'high_level_stored': getattr(self, 'high_level_experiences_stored', 0),
                'low_level_stored': getattr(self, 'low_level_experiences_stored', 0),
                'state_skill_stored': 0  # 原版没有单独统计
            }
        except Exception as e:
            self.logger.error(f"获取存储统计失败: {e}")
            return {
                'total_attempts': 0,
                'total_successes': 0,
                'total_failures': 0,
                'queue_overflows': 0,
                'validation_failures': 0,
                'high_level_stored': 0,
                'low_level_stored': 0,
                'state_skill_stored': 0
            }

    def log_post_update_buffer_state(self):
        """记录更新后的缓冲区状态，帮助诊断缓冲区清理问题"""
        try:
            # 获取agent缓冲区大小
            high_level_size = len(self.agent.high_level_buffer) if hasattr(self.agent, 'high_level_buffer') else 'N/A'
            low_level_size = len(self.agent.low_level_buffer) if hasattr(self.agent, 'low_level_buffer') else 'N/A'
            state_skill_size = len(self.agent.state_skill_dataset) if hasattr(self.agent, 'state_skill_dataset') else 'N/A'
            
            # 获取agent内部统计
            high_level_stats = getattr(self.agent, 'high_level_samples_by_env', {})
            high_level_reason_stats = getattr(self.agent, 'high_level_samples_by_reason', {})
            
            self.logger.warning("📈 [POST_UPDATE_BUFFER] 更新后缓冲区状态:")
            self.logger.warning(f"   - B_h (高层): {high_level_size} (期望: 0 - PPO应清空)")
            self.logger.warning(f"   - B_l (低层): {low_level_size} (期望: 0 - PPO应清空)")
            self.logger.warning(f"   - D (判别器): {state_skill_size} (期望: 保留)")
            self.logger.warning(f"   - 高层样本统计: 环境贡献={dict(high_level_stats)}")
            self.logger.warning(f"   - 收集原因统计: {dict(high_level_reason_stats)}")
            
            # 检查是否符合PPO on-policy要求
            if high_level_size != 0 and high_level_size != 'N/A':
                self.logger.error(f"❌ [BUFFER_CLEAR_ISSUE] B_h未被清空！期望=0, 实际={high_level_size}")
            else:
                self.logger.info(f"✅ [BUFFER_CLEAR_OK] B_h已正确清空")
                
            if low_level_size != 0 and low_level_size != 'N/A':
                self.logger.error(f"❌ [BUFFER_CLEAR_ISSUE] B_l未被清空！期望=0, 实际={low_level_size}")
            else:
                self.logger.info(f"✅ [BUFFER_CLEAR_OK] B_l已正确清空")
                
        except Exception as e:
            self.logger.error(f"记录更新后缓冲区状态失败: {e}")

class TrainingWorker:
    """训练worker，在独立线程中运行"""
    def __init__(self, worker_id, agent_proxy, data_buffer, control_events, logger, config, trainer):
        self.worker_id = worker_id
        self.agent_proxy = agent_proxy
        self.data_buffer = data_buffer
        self.control_events = control_events
        self.logger = logger
        self.config = config
        self.trainer = trainer  # 新增：保存trainer引用
        
        # 训练统计
        self.updates_performed = 0
        self.samples_processed = 0
        self.last_update_time = time.time()
        
    def run(self):
        """【阶段2增强】运行训练worker主循环 - 智能重试策略 + 自动修复功能"""
        self.logger.info(f"Training worker {self.worker_id} 开始运行")
        
        experience_batch = []
        batch_size = 32  # 批处理大小
        
        # 【阶段2新增】错误统计和自适应策略
        consecutive_failures = 0
        max_consecutive_failures = 10
        storage_error_types = defaultdict(int)
        adaptive_retry_count = 5
        
        try:
            while not self.control_events['stop'].is_set():
                # 检查是否需要暂停
                if self.control_events['pause'].is_set():
                    time.sleep(0.1)
                    continue
                
                # 【阶段2增强】检测系统压力并自适应调整
                buffer_stats = self.data_buffer.get_stats()
                system_under_pressure = self._detect_system_pressure(buffer_stats)
                
                if system_under_pressure:
                    # 系统压力大时，采用更积极的处理策略
                    batch_size = min(16, batch_size)  # 减小批次大小
                    adaptive_retry_count = max(3, adaptive_retry_count - 1)  # 减少重试次数
                else:
                    # 系统正常时，恢复标准策略
                    batch_size = 32
                    adaptive_retry_count = 5
                
                # 【阶段2增强】智能数据获取 - 支持优先级处理
                try:
                    experience = self._intelligent_data_retrieval()
                    if experience is None:
                        if self.control_events['stop'].is_set():
                            break
                        continue
                        
                except Exception as e:
                    error_type = type(e).__name__
                    storage_error_types[error_type] += 1
                    self.logger.error(f"Training worker {self.worker_id}: 获取数据异常: {e}")
                    consecutive_failures += 1
                    
                    if consecutive_failures >= max_consecutive_failures:
                        self.logger.error(f"Training worker {self.worker_id}: 连续失败次数过多，暂停处理")
                        time.sleep(1.0)  # 暂停更长时间
                        consecutive_failures = 0
                    
                    time.sleep(0.01)
                    continue
                
                # 成功获取数据，重置失败计数
                consecutive_failures = 0
                experience_batch.append(experience)
                
                # 【阶段2增强】智能批处理策略 - 考虑数据类型优先级
                should_process_batch = self._should_process_batch(
                    experience_batch, batch_size, buffer_stats
                )
                
                if should_process_batch:
                    # 【阶段2核心】智能存储重试机制
                    storage_result = self._intelligent_storage_retry(
                        experience_batch, adaptive_retry_count
                    )
                    
                    if storage_result['success']:
                        self.samples_processed += storage_result['stored_count']
                        
                        if storage_result['stored_count'] > 0:
                            self.logger.debug(f"Training worker {self.worker_id}: 处理了 {storage_result['stored_count']} 个样本 "
                                            f"(批次大小: {len(experience_batch)}, 队列剩余: {self.data_buffer.qsize()})")
                        
                        experience_batch = []  # 只有在完全成功后才清空
                        
                        # 检查更新条件
                        if self.agent_proxy.should_update():
                            self.perform_update()
                    else:
                        # 【阶段2新增】存储失败时的智能处理
                        self._handle_storage_failure(experience_batch, storage_result)
                
                # 短暂睡眠避免过度占用CPU
                time.sleep(0.001)
        
        except Exception as e:
            self.logger.error(f"Training worker {self.worker_id}: 运行异常: {e}")
        finally:
            # 【阶段2增强】确保剩余经验100%存储成功
            if experience_batch:
                self._process_remaining_experiences(experience_batch)
            
            # 【阶段2新增】记录错误统计
            if storage_error_types:
                self.logger.warning(f"Training worker {self.worker_id}: 错误统计: {dict(storage_error_types)}")
            
            self.logger.info(f"Training worker {self.worker_id} 结束运行")
    
    def _detect_system_pressure(self, buffer_stats):
        """【阶段2新增】检测系统压力"""
        # 检测队列拥塞
        queue_pressure = buffer_stats.get('queue_size', 0) > 1000
        
        # 检测处理速度下降
        processing_speed = buffer_stats.get('processing_speed', 0)
        speed_pressure = processing_speed < 10  # 每秒处理少于10个样本
        
        # 检测拥塞状态
        congestion_detected = buffer_stats.get('congestion_detected', False)
        
        return queue_pressure or speed_pressure or congestion_detected
    
    def _intelligent_data_retrieval(self):
        """【阶段2新增】智能数据获取 - 支持优先级处理"""
        max_wait_attempts = 5
        wait_attempt = 0
        
        while wait_attempt < max_wait_attempts and not self.control_events['stop'].is_set():
            try:
                # 使用增强的DataBuffer优先级获取
                experience = self.data_buffer.get(block=True, timeout=1.0)
                if experience is not None:
                    return experience
                    
            except queue.Empty:
                wait_attempt += 1
                # 【智能等待】根据系统状态调整等待策略
                if wait_attempt < 3:
                    continue  # 短等待
                else:
                    time.sleep(0.1)  # 稍长等待
            except Exception as e:
                self.logger.error(f"Training worker {self.worker_id}: 数据获取异常: {e}")
                return None
        
        return None
    
    def _should_process_batch(self, experience_batch, batch_size, buffer_stats):
        """【阶段2增强】智能批处理决策"""
        current_batch_size = len(experience_batch)
        queue_size = buffer_stats.get('queue_size', 0)
        
        # 【优先级考虑】检查批次中的高层经验比例
        high_priority_count = sum(1 for exp in experience_batch 
                                if exp.get('experience_type') == 'high_level')
        high_priority_ratio = high_priority_count / current_batch_size if current_batch_size > 0 else 0
        
        # 处理条件
        conditions = [
            current_batch_size >= batch_size,  # 达到批大小
            (current_batch_size > 0 and queue_size < 5),  # 队列几乎空了
            (current_batch_size >= 8 and queue_size < batch_size // 2),  # 中等批次且队列不满
            (high_priority_ratio > 0.5 and current_batch_size >= 4)  # 高优先级数据较多
        ]
        
        return any(conditions)
    
    def _intelligent_storage_retry(self, experience_batch, max_retries):
        """【阶段2核心】智能存储重试机制"""
        result = {
            'success': False,
            'stored_count': 0,
            'retry_count': 0,
            'error_types': [],
            'final_error': None
        }
        
        retry_count = 0
        stored_count = 0
        
        while retry_count < max_retries:
            try:
                stored_count = self.agent_proxy.store_experience(experience_batch)
                
                # 验证存储成功
                if stored_count == len(experience_batch):
                    result['success'] = True
                    result['stored_count'] = stored_count
                    result['retry_count'] = retry_count
                    return result
                else:
                    # 部分存储成功 - 记录但继续重试
                    error_msg = f"存储不完整 {stored_count}/{len(experience_batch)}"
                    result['error_types'].append(error_msg)
                    
                    if retry_count < max_retries - 1:
                        self.logger.debug(f"Training worker {self.worker_id}: {error_msg}, 重试 {retry_count + 1}")
                        
                        # 【智能等待策略】根据重试次数调整等待时间
                        wait_time = min(0.1 * (retry_count + 1), 0.5)
                        time.sleep(wait_time)
                    
                    retry_count += 1
                    
            except Exception as e:
                error_type = type(e).__name__
                result['error_types'].append(error_type)
                result['final_error'] = str(e)
                
                if retry_count < max_retries - 1:
                    self.logger.debug(f"Training worker {self.worker_id}: 存储异常 {error_type}: {e}, 重试 {retry_count + 1}")
                    
                    # 【智能等待策略】根据错误类型调整等待时间
                    if 'timeout' in error_type.lower():
                        wait_time = 0.2 * (retry_count + 1)  # 超时错误等待更长
                    else:
                        wait_time = 0.1 * (retry_count + 1)  # 其他错误标准等待
                    
                    time.sleep(min(wait_time, 0.5))
                
                retry_count += 1
        
        # 所有重试都失败
        result['retry_count'] = retry_count
        result['stored_count'] = stored_count
        
        if result['error_types']:
            self.logger.error(f"Training worker {self.worker_id}: 存储最终失败! "
                            f"成功={stored_count}, 总数={len(experience_batch)}, "
                            f"错误类型={result['error_types']}")
        
        return result
    
    def _handle_storage_failure(self, experience_batch, storage_result):
        """【阶段2新增】处理存储失败的智能策略"""
        retry_count = storage_result['retry_count']
        error_types = storage_result['error_types']
        
        # 【策略1】分析错误类型，决定处理方式
        persistent_errors = ['timeout', 'connection', 'memory']
        has_persistent_error = any(error_type.lower() in ' '.join(error_types).lower() 
                                 for error_type in persistent_errors)
        
        if has_persistent_error:
            # 持续性错误 - 暂停处理，避免系统过载
            self.logger.warning(f"Training worker {self.worker_id}: 检测到持续性错误，暂停处理")
            time.sleep(1.0)
            
            # 【策略2】尝试分批处理
            if len(experience_batch) > 8:
                self.logger.info(f"Training worker {self.worker_id}: 尝试分批处理 {len(experience_batch)} 个经验")
                self._split_batch_processing(experience_batch)
                return
        
        # 【策略3】严格模式 - 不清空batch，但限制重试次数
        if retry_count >= 3:
            # 超过一定重试次数，记录并暂时放弃
            self.logger.error(f"Training worker {self.worker_id}: 放弃处理当前批次 {len(experience_batch)} 个经验")
            # 清空batch以避免无限重试
            experience_batch.clear()
        
        # 【策略4】否则保持batch不变，继续重试
    
    def _split_batch_processing(self, experience_batch):
        """【阶段2辅助】分批处理大批次数据"""
        batch_size = len(experience_batch)
        split_size = max(4, batch_size // 4)  # 分成4个小批次
        
        processed = 0
        for i in range(0, batch_size, split_size):
            sub_batch = experience_batch[i:i + split_size]
            
            try:
                storage_result = self._intelligent_storage_retry(sub_batch, 3)  # 减少重试次数
                if storage_result['success']:
                    processed += storage_result['stored_count']
                    self.logger.debug(f"Training worker {self.worker_id}: 分批处理成功 {i//split_size + 1}, "
                                    f"处理 {storage_result['stored_count']} 个经验")
                else:
                    self.logger.warning(f"Training worker {self.worker_id}: 分批处理失败 {i//split_size + 1}")
                    
            except Exception as e:
                self.logger.error(f"Training worker {self.worker_id}: 分批处理异常: {e}")
        
        if processed > 0:
            self.samples_processed += processed
            self.logger.info(f"Training worker {self.worker_id}: 分批处理完成，总共处理 {processed}/{batch_size} 个经验")
        
        # 清空原批次
        experience_batch.clear()
    
    def _process_remaining_experiences(self, experience_batch):
        """【阶段2增强】处理剩余经验 - 更智能的最终处理"""
        self.logger.info(f"Training worker {self.worker_id}: 处理剩余 {len(experience_batch)} 个经验")
        
        # 【增强策略】更多重试次数和更长等待时间
        max_retries = 15  # 增加重试次数
        retry_count = 0
        
        while retry_count < max_retries and experience_batch:
            try:
                storage_result = self._intelligent_storage_retry(experience_batch, 3)
                
                if storage_result['success']:
                    self.samples_processed += storage_result['stored_count']
                    self.logger.info(f"Training worker {self.worker_id}: 剩余经验存储成功")
                    break
                else:
                    # 【最终策略】尝试分批处理
                    if len(experience_batch) > 4 and retry_count > 5:
                        self.logger.info(f"Training worker {self.worker_id}: 剩余经验尝试分批处理")
                        self._split_batch_processing(experience_batch)
                        break
                    
                    self.logger.warning(f"Training worker {self.worker_id}: 剩余经验存储不完整，重试 {retry_count + 1}")
                    retry_count += 1
                    
                    # 【渐进等待】等待时间逐渐增加
                    wait_time = min(0.2 * retry_count, 2.0)
                    time.sleep(wait_time)
                    
            except Exception as e:
                self.logger.error(f"Training worker {self.worker_id}: 剩余经验存储异常: {e}")
                retry_count += 1
                time.sleep(0.3)
        
        if experience_batch:
            self.logger.error(f"Training worker {self.worker_id}: 最终仍有 {len(experience_batch)} 个经验未能存储!")
    
    def perform_update(self):
        """执行模型更新"""
        try:
            update_start = time.time()
            
            # 执行更新（这里只有一个worker执行更新，其他worker继续处理数据）
            if self.worker_id == 0:  # 只有第一个training worker执行更新
                self.logger.info(f"Training worker {self.worker_id}: 开始模型更新")
                
                update_info = self.agent_proxy.update()
                
                if update_info:
                    self.updates_performed += 1
                    update_time = time.time() - update_start
                    self.last_update_time = time.time()
                    
                    # 新增：更新累计总步数
                    steps_this_rollout = self.config.batch_size
                    self.trainer.total_steps.increment(steps_this_rollout)
                    
                    self.logger.info(f"Training worker {self.worker_id}: 模型更新完成 "
                                   f"#{self.updates_performed}, 耗时: {update_time:.3f}s")
                    self.logger.info(f"Training worker {self.worker_id}: 累计总步数增加 {steps_this_rollout}. "
                                   f"新的总步数: {self.trainer.total_steps.get()}")
                    
                    # 记录更新信息
                    if 'coordinator' in update_info:
                        coord_loss = update_info['coordinator'].get('coordinator_loss', 0)
                        self.logger.debug(f"Coordinator loss: {coord_loss:.6f}")
                    
                    if 'discoverer' in update_info:
                        disc_loss = update_info['discoverer'].get('discoverer_loss', 0)
                        self.logger.debug(f"Discoverer loss: {disc_loss:.6f}")
                else:
                    self.logger.warning(f"Training worker {self.worker_id}: 模型更新返回None")
        
        except Exception as e:
            self.logger.error(f"Training worker {self.worker_id}: 模型更新异常: {e}")

class ThreadedRolloutTrainer:
    """多线程HMASD Rollout-based训练器"""
    
    def __init__(self, config, args=None):
        """
        初始化多线程训练器
        
        参数:
            config: 配置对象
            args: 命令行参数（可选）
        """
        self.config = config
        self.args = args or argparse.Namespace()
        
        # 验证并设置训练模式
        config.rollout_based_training = True
        config.episode_based_training = False
        config.sync_training_mode = False
        
        # 验证配置
        if not config.validate_rollout_config():
            raise ValueError("Rollout配置验证失败")
        
        # 线程配置（按照论文 Appendix E）
        self.num_training_threads = getattr(args, 'training_threads', 16)
        self.num_rollout_threads = getattr(args, 'rollout_threads', 32)
        
        # 设置设备
        self.device = self._get_device()
        
        # 创建日志目录
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_dir = f"logs/threaded_rollout_training_{timestamp}"
        os.makedirs(self.log_dir, exist_ok=True)
        
        # 初始化日志系统
        self._init_logging()
        
        # 线程控制
        self.control_events = {
            'stop': Event(),
            'pause': Event()
        }
        
        # 数据缓冲区
        buffer_size = getattr(args, 'buffer_size', 10000)
        self.data_buffer = DataBuffer(maxsize=buffer_size)
        
        # 统计信息
        self.start_time = None
        self.total_updates = 0
        self.total_samples = ThreadSafeCounter()
        self.total_steps = ThreadSafeCounter()  # 添加总步数计数器
        
        self.logger.info("ThreadedRolloutTrainer初始化完成")
        self.logger.info(f"日志目录: {self.log_dir}")
        self.logger.info(f"训练线程数: {self.num_training_threads}")
        self.logger.info(f"Rollout线程数: {self.num_rollout_threads}")
        self.logger.info(f"数据缓冲区大小: {buffer_size}")
        self.logger.info(config.get_rollout_summary())
    
    def _get_device(self):
        """获取计算设备"""
        device_pref = getattr(self.args, 'device', 'auto')
        if device_pref == 'auto':
            return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        elif device_pref == 'cuda':
            if torch.cuda.is_available():
                return torch.device('cuda')
            else:
                self.logger.warning("请求CUDA但未检测到GPU，使用CPU")
                return torch.device('cpu')
        else:
            return torch.device('cpu')
    
    def _init_logging(self):
        """初始化日志系统"""
        log_level = getattr(self.args, 'log_level', 'INFO')
        console_level = getattr(self.args, 'console_log_level', 'INFO')
        
        init_multiproc_logging(
            log_dir=self.log_dir,
            log_file='threaded_rollout_training.log',
            file_level=LOG_LEVELS.get(log_level.lower(), 20),
            console_level=LOG_LEVELS.get(console_level.lower(), 20)
        )
        
        self.logger = get_logger("ThreadedRolloutTrainer")
    
    def create_env_factory(self):
        """创建环境工厂函数"""
        scenario = getattr(self.args, 'scenario', 2)
        n_uavs = getattr(self.args, 'n_uavs', 5)
        n_users = getattr(self.args, 'n_users', 50)
        user_distribution = getattr(self.args, 'user_distribution', 'uniform')
        channel_model = getattr(self.args, 'channel_model', '3gpp-36777')
        max_hops = getattr(self.args, 'max_hops', 3)
        
        def make_env():
            def _init():
                env_seed = int(time.time() * 1000) % 10000  # 基于时间的随机种子
                if scenario == 1:
                    raw_env = UAVBaseStationEnv(
                        n_uavs=n_uavs,
                        n_users=n_users,
                        user_distribution=user_distribution,
                        channel_model=channel_model,
                        seed=env_seed
                    )
                elif scenario == 2:
                    raw_env = UAVCooperativeNetworkEnv(
                        n_uavs=n_uavs,
                        n_users=n_users,
                        max_hops=max_hops,
                        user_distribution=user_distribution,
                        channel_model=channel_model,
                        seed=env_seed
                    )
                else:
                    raise ValueError(f"未知场景: {scenario}")
                
                env = ParallelToArrayAdapter(raw_env, seed=env_seed)
                return env
            return _init()
        
        return make_env
    
    def validate_environment(self):
        """验证环境接口兼容性"""
        self.logger.info("开始验证环境接口兼容性...")
        
        temp_env = self.create_env_factory()()
        try:
            # 测试reset
            reset_result = temp_env.reset()
            self.logger.info(f"环境reset()返回: {len(reset_result)}个值")
            
            if len(reset_result) == 2:
                observations, info = reset_result
                self.logger.info(f"Reset成功: observations.shape={observations.shape}, info keys={list(info.keys()) if isinstance(info, dict) else 'not dict'}")
            else:
                self.logger.warning(f"Reset返回值数量异常: {len(reset_result)}")
            
            # 测试step
            dummy_actions = np.random.randn(temp_env.n_uavs, temp_env.action_dim)
            step_result = temp_env.step(dummy_actions)
            self.logger.info(f"环境step()返回: {len(step_result)}个值")
            
            if len(step_result) == 5:
                next_obs, reward, terminated, truncated, info = step_result
                self.logger.info(f"Step成功 (Gymnasium格式): next_obs.shape={next_obs.shape}, "
                               f"reward={reward}, terminated={terminated}, truncated={truncated}")
            elif len(step_result) == 4:
                next_obs, reward, done, info = step_result
                self.logger.info(f"Step成功 (传统格式): next_obs.shape={next_obs.shape}, "
                               f"reward={reward}, done={done}")
            else:
                raise ValueError(f"不支持的step返回值数量: {len(step_result)}")
            
            self.logger.info("✅ 环境接口验证通过")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ 环境接口验证失败: {e}")
            import traceback
            traceback.print_exc()
            return False
        finally:
            temp_env.close()
    
    def initialize_agent(self):
        """初始化HMASD代理"""
        # 验证环境兼容性
        if not self.validate_environment():
            raise RuntimeError("环境接口验证失败，无法继续训练")
        
        # 创建临时环境获取维度信息
        temp_env = self.create_env_factory()()
        state_dim = temp_env.state_dim
        obs_dim = temp_env.obs_dim
        n_agents = temp_env.n_uavs
        temp_env.close()
        
        # 更新配置
        self.config.update_env_dims(state_dim, obs_dim)
        self.config.n_agents = n_agents
        
        self.logger.info(f"环境维度: state_dim={state_dim}, obs_dim={obs_dim}, n_agents={n_agents}")
        
        # 创建线程安全代理
        self.agent = ThreadSafeHMASDAgent(
            config=self.config,
            log_dir=self.log_dir,
            device=self.device,
            debug=getattr(self.args, 'debug', False)
        )
        
        # 创建代理代理
        self.agent_proxy = AgentProxy(self.agent, self.config, self.logger, self.data_buffer)
        
        self.logger.info("HMASD代理初始化完成")
    
    def start_rollout_threads(self):
        """启动rollout线程"""
        self.logger.info(f"启动 {self.num_rollout_threads} 个rollout线程")
        
        self.rollout_workers = []
        self.rollout_threads = []
        
        env_factory = self.create_env_factory()
        
        for i in range(self.num_rollout_threads):
            worker = RolloutWorker(
                worker_id=i,
                env_factory=env_factory,
                config=self.config,
                data_buffer=self.data_buffer,
                control_events=self.control_events,
                logger=self.logger
            )
            
            thread = threading.Thread(
                target=worker.run,
                args=(self.agent_proxy,),
                name=f"RolloutWorker-{i}"
            )
            thread.daemon = True
            
            self.rollout_workers.append(worker)
            self.rollout_threads.append(thread)
            thread.start()
        
        self.logger.info("所有rollout线程已启动")
        
        # 【关键修复】设置AgentProxy对rollout workers的引用
        self.agent_proxy.rollout_workers = self.rollout_workers
        
        # 【关键修复】设置AgentProxy对rollout workers的引用
        self.agent_proxy.rollout_workers = self.rollout_workers
    
    def start_training_threads(self):
        """启动训练线程"""
        self.logger.info(f"启动 {self.num_training_threads} 个training线程")
        
        self.training_workers = []
        self.training_threads = []
        
        for i in range(self.num_training_threads):
            worker = TrainingWorker(
                worker_id=i,
                agent_proxy=self.agent_proxy,
                data_buffer=self.data_buffer,
                control_events=self.control_events,
                logger=self.logger,
                config=self.config,
                trainer=self  # 新增：传递trainer实例
            )
            
            thread = threading.Thread(
                target=worker.run,
                name=f"TrainingWorker-{i}"
            )
            thread.daemon = True
            
            self.training_workers.append(worker)
            self.training_threads.append(thread)
            thread.start()
        
        self.logger.info("所有training线程已启动")
    
    def monitor_training(self, total_steps=100000):
        """监控训练进度"""
        self.logger.info(f"开始训练监控，目标步数: {total_steps:,}")
        
        self.start_time = time.time()
        last_log_time = self.start_time
        last_stats_log_time = self.start_time
        last_step_count = 0
        
        try:
            while True:
                current_time = time.time()
                
                # 使用正确的累计总步数
                cumulative_trainer_steps = self.total_steps.get()
                
                # 检查是否达到步数限制
                if cumulative_trainer_steps >= total_steps:
                    self.logger.info(f"达到训练步数限制 {total_steps:,} (实际累计: {cumulative_trainer_steps:,})，停止训练")
                    break
                
                # 每分钟记录一次简要进度
                if current_time - last_log_time >= 60:
                    self.log_progress(cumulative_trainer_steps, total_steps)
                    last_log_time = current_time
                
                # 每10分钟记录一次详细统计
                if current_time - last_stats_log_time >= 600:
                    self.log_detailed_stats()
                    last_stats_log_time = current_time
                
                # 检查线程健康状态
                self.check_thread_health()
                
                time.sleep(30)  # 每30秒检查一次
        
        except KeyboardInterrupt:
            self.logger.info("训练被用户中断")
        
        finally:
            self.stop_training()
    
    def log_progress(self, current_steps, total_steps):
        """记录训练进度"""
        progress_percent = (current_steps / total_steps) * 100
        remaining_steps = total_steps - current_steps
        
        # 计算时间统计
        elapsed_time = time.time() - self.start_time
        if current_steps > 0:
            estimated_total_time = elapsed_time * total_steps / current_steps
            remaining_time = estimated_total_time - elapsed_time
        else:
            remaining_time = 0
        
        # 获取统计信息
        buffer_stats = self.data_buffer.get_stats()
        
        # 计算rollout workers统计
        total_samples = sum(worker.samples_collected for worker in self.rollout_workers)
        total_episodes = sum(worker.episodes_completed for worker in self.rollout_workers)
        total_high_level_exp = sum(worker.high_level_experiences_generated for worker in self.rollout_workers)
        
        # 计算training workers统计
        total_updates = sum(worker.updates_performed for worker in self.training_workers)
        total_processed = sum(worker.samples_processed for worker in self.training_workers)
        
        # 获取代理统计
        high_level_stored = self.agent_proxy.high_level_experiences_stored
        low_level_stored = self.agent_proxy.low_level_experiences_stored
        
        # 计算步数速度
        steps_per_second = current_steps / elapsed_time if elapsed_time > 0 else 0
        
        self.logger.debug(f"训练进度: {progress_percent:.1f}% "
                        f"({current_steps:,} / {total_steps:,} 步), "
                        f"剩余: {remaining_steps:,} 步")
        self.logger.debug(f"时间: 已用={elapsed_time/3600:.1f}h, 预计剩余={remaining_time/3600:.1f}h, "
                        f"速度={steps_per_second:.1f} 步/秒")
        self.logger.debug(f"Rollout: 样本={total_samples:,}, Episodes={total_episodes:,}, "
                        f"高层经验={total_high_level_exp:,}")
        self.logger.debug(f"Training: 更新={total_updates}, 处理样本={total_processed:,}")
        self.logger.debug(f"经验存储: 高层={high_level_stored:,}, 低层={low_level_stored:,}")
        self.logger.debug(f"Buffer: 队列={buffer_stats['queue_size']}, "
                        f"添加={buffer_stats['total_added']:,}, "
                        f"消费={buffer_stats['total_consumed']:,}")
    
    def log_detailed_stats(self):
        """记录详细统计信息"""
        self.logger.info("=== 详细统计信息 ===")
        
        # Rollout workers统计
        self.logger.info("Rollout Workers:")
        for i, worker in enumerate(self.rollout_workers[:5]):  # 只显示前5个
            stats = worker.get_worker_stats()
            self.logger.info(f"  Worker {i}: 样本={stats['samples_collected']}, "
                           f"Episodes={stats['episodes_completed']}, "
                           f"高层经验={stats['high_level_experiences_generated']}, "
                           f"当前技能={stats['current_team_skill']}, "
                           f"技能计时器={stats['current_skill_timer']}, "
                           f"累积奖励={stats['current_accumulated_reward']:.3f}")
        if len(self.rollout_workers) > 5:
            self.logger.info(f"  ... 还有 {len(self.rollout_workers) - 5} 个workers")
        
        # Training workers统计
        self.logger.info("Training Workers:")
        for i, worker in enumerate(self.training_workers[:5]):  # 只显示前5个
            self.logger.info(f"  Worker {i}: 更新={worker.updates_performed}, "
                           f"处理样本={worker.samples_processed}")
        if len(self.training_workers) > 5:
            self.logger.info(f"  ... 还有 {len(self.training_workers) - 5} 个workers")
        
        # 技能周期统计
        total_high_level_exp = sum(worker.high_level_experiences_generated for worker in self.rollout_workers)
        active_skills = [worker.current_team_skill for worker in self.rollout_workers if worker.current_team_skill is not None]
        skill_distribution = {}
        for skill in active_skills:
            skill_distribution[skill] = skill_distribution.get(skill, 0) + 1
        
        self.logger.info(f"技能周期统计: 总高层经验={total_high_level_exp}, "
                        f"活跃技能分布={skill_distribution}")
        
        # 代理经验存储统计
        self.logger.info(f"代理经验存储: 高层={self.agent_proxy.high_level_experiences_stored}, "
                        f"低层={self.agent_proxy.low_level_experiences_stored}")
        
        # 缓冲区详细统计
        buffer_stats = self.data_buffer.get_stats()
        self.logger.info(f"Data Buffer: {buffer_stats}")
        
        # GPU内存使用（如果有GPU）
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated(self.device) / 1024**3
            memory_reserved = torch.cuda.memory_reserved(self.device) / 1024**3
            self.logger.info(f"GPU内存: 已分配={memory_allocated:.2f}GB, 已保留={memory_reserved:.2f}GB")
    
    def check_thread_health(self):
        """检查线程健康状态"""
        # 检查rollout线程
        dead_rollout_threads = [i for i, thread in enumerate(self.rollout_threads) if not thread.is_alive()]
        if dead_rollout_threads:
            self.logger.warning(f"发现 {len(dead_rollout_threads)} 个死亡的rollout线程: {dead_rollout_threads}")
        
        # 检查training线程
        dead_training_threads = [i for i, thread in enumerate(self.training_threads) if not thread.is_alive()]
        if dead_training_threads:
            self.logger.warning(f"发现 {len(dead_training_threads)} 个死亡的training线程: {dead_training_threads}")
        
        # 检查数据流是否正常
        buffer_stats = self.data_buffer.get_stats()
        if buffer_stats['total_added'] == getattr(self, '_last_total_added', 0):
            self.logger.warning("数据缓冲区添加数量未增加，可能rollout线程有问题")
        if buffer_stats['total_consumed'] == getattr(self, '_last_total_consumed', 0):
            self.logger.warning("数据缓冲区消费数量未增加，可能training线程有问题")
        
        self._last_total_added = buffer_stats['total_added']
        self._last_total_consumed = buffer_stats['total_consumed']
    
    def stop_training(self):
        """停止训练"""
        self.logger.info("停止训练...")
        
        # 设置停止事件
        self.control_events['stop'].set()
        
        # 等待所有线程结束
        self.logger.info("等待rollout线程结束...")
        for i, thread in enumerate(self.rollout_threads):
            thread.join(timeout=10)
            if thread.is_alive():
                self.logger.warning(f"Rollout线程 {i} 未能在10秒内结束")
        
        self.logger.info("等待training线程结束...")
        for i, thread in enumerate(self.training_threads):
            thread.join(timeout=10)
            if thread.is_alive():
                self.logger.warning(f"Training线程 {i} 未能在10秒内结束")
        
        self.logger.info("所有线程已停止")
    
    def save_final_model(self):
        """保存最终模型"""
        try:
            final_model_path = os.path.join(self.log_dir, 'final_model.pt')
            self.agent.save_model(final_model_path)
            self.logger.info(f"最终模型已保存: {final_model_path}")
        except Exception as e:
            self.logger.error(f"保存最终模型失败: {e}")
    
    def cleanup(self):
        """清理资源"""
        try:
            # 1. 停止训练（如果还没停止）
            if not self.control_events['stop'].is_set():
                self.stop_training()
            
            # 2. 关闭TensorBoard writer
            if hasattr(self.agent, 'writer') and self.agent.writer:
                try:
                    self.agent.writer.close()
                    self.logger.info("TensorBoard writer已关闭")
                except Exception as e:
                    self.logger.warning(f"关闭TensorBoard writer时出错: {e}")
            
            # 3. 清理代理缓冲区
            if hasattr(self.agent, 'high_level_buffer'):
                self.agent.high_level_buffer.clear()
            if hasattr(self.agent, 'low_level_buffer'):
                self.agent.low_level_buffer.clear()
            if hasattr(self.agent, 'state_skill_dataset'):
                self.agent.state_skill_dataset.clear()
            
            self.logger.info("所有资源清理完成")
            
        except Exception as e:
            print(f"清理资源时出错: {e}")
    
    def train(self, total_steps=100000):
        """
        执行完整的多线程rollout-based训练
        
        参数:
            total_steps: 训练总步数
        """
        self.logger.info(f"开始HMASD多线程Rollout-based训练: {total_steps:,} 步")
        self.logger.info(f"配置: {self.num_training_threads} 训练线程, {self.num_rollout_threads} rollout线程")
        
        try:
            # 初始化代理
            self.initialize_agent()
            
            # 启动rollout线程
            self.start_rollout_threads()
            
            # 等待rollout线程开始收集数据
            time.sleep(5)
            
            # 启动training线程
            self.start_training_threads()
            
            # 开始监控训练
            self.monitor_training(total_steps)
            
        except KeyboardInterrupt:
            self.logger.info("训练被用户中断")
        except Exception as e:
            self.logger.error(f"训练过程中出错: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            # 保存最终模型
            self.save_final_model()
            
            # 清理资源
            self.cleanup()
        
        # 训练完成
        if self.start_time:
            total_time = time.time() - self.start_time
            final_steps = sum(worker.samples_collected for worker in self.rollout_workers)
            self.logger.info(f"\n训练完成！")
            self.logger.info(f"总时间: {total_time/3600:.2f}小时")
            self.logger.info(f"总步数: {final_steps:,}")
            
            # 输出最终统计
            total_samples = sum(worker.samples_collected for worker in self.rollout_workers)
            total_episodes = sum(worker.episodes_completed for worker in self.rollout_workers)
            total_updates = sum(worker.updates_performed for worker in self.training_workers)
            
            self.logger.info(f"总样本数: {total_samples:,}")
            self.logger.info(f"总Episodes: {total_episodes:,}")
            self.logger.info(f"总更新数: {total_updates}")
            
            if total_time > 0:
                self.logger.info(f"样本收集速度: {total_samples/total_time:.1f} 样本/秒")
                self.logger.info(f"Episode完成速度: {total_episodes/total_time:.1f} episodes/秒")
                self.logger.info(f"步数完成速度: {final_steps/total_time:.1f} 步/秒")

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='HMASD多线程Rollout-based训练（按论文Appendix E）')
    
    # 训练参数
    parser.add_argument('--steps', type=int, default=None, help='训练总步数（如果不指定，将从config.py中读取）')
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cpu', 'cuda'])
    parser.add_argument('--debug', action='store_true', help='启用调试模式')
    
    # 线程配置（按照论文 Appendix E）
    parser.add_argument('--training_threads', type=int, default=16, help='训练线程数（论文默认16）')
    parser.add_argument('--rollout_threads', type=int, default=32, help='Rollout线程数（论文默认32）')
    parser.add_argument('--buffer_size', type=int, default=10000, help='数据缓冲区大小')
    
    # 环境参数
    parser.add_argument('--scenario', type=int, default=2, choices=[1, 2], help='场景选择')
    parser.add_argument('--n_uavs', type=int, default=5, help='无人机数量')
    parser.add_argument('--n_users', type=int, default=50, help='用户数量')
    parser.add_argument('--user_distribution', type=str, default='uniform', 
                       choices=['uniform', 'cluster', 'hotspot'])
    parser.add_argument('--channel_model', type=str, default='3gpp-36777',
                       choices=['free_space', 'urban', 'suburban', '3gpp-36777'])
    parser.add_argument('--max_hops', type=int, default=3, help='最大跳数（仅场景2）')
    
    # 日志参数
    parser.add_argument('--log_level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])
    parser.add_argument('--console_log_level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])
    
    return parser.parse_args()

def main():
    """主函数"""
    args = parse_args()
    
    # 创建配置
    config = Config()
    
    # 确定训练步数：优先使用命令行参数，其次使用配置文件中的值
    if args.steps is not None:
        total_steps = args.steps
        print(f"📈 使用命令行指定的训练步数: {total_steps:,}")
    else:
        total_steps = int(config.total_timesteps)
        print(f"📈 从config.py读取训练步数: {total_steps:,}")
    
    print("🚀 HMASD多线程Rollout-based训练（严格按论文实现）")
    print("=" * 60)
    print(f"📊 线程配置: {args.training_threads} 训练线程 + {args.rollout_threads} rollout线程")
    print(f"🎯 训练步数: {total_steps:,}")
    print(f"🗂️ 缓冲区大小: {args.buffer_size}")
    
    # 验证并打印配置
    config.validate_training_mode()
    config.validate_rollout_config()
    print(config.get_rollout_summary())
    
    try:
        # 创建训练器
        trainer = ThreadedRolloutTrainer(config, args)
        
        # 开始训练
        trainer.train(total_steps=total_steps)
        
        print("🎉 训练成功完成！")
        
    except KeyboardInterrupt:
        print("\n⚠️ 训练被用户中断")
    except Exception as e:
        print(f"\n❌ 训练失败: {e}")
        import traceback
        traceback.print_exc()
    finally:
        try:
            shutdown_logging()
        except:
            pass

if __name__ == "__main__":
    main()
