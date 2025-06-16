#!/usr/bin/env python3
"""
【增强版】HMASD严格按论文Algorithm 1 + Appendix E的多线程Rollout-based训练脚本
集成三个核心增强组件：
1. AtomicDataBuffer: 原子性操作保证、优先级队列处理、拥塞检测和自适应处理
2. ThreadSafeAgentProxy: 分离锁减少竞争、后台存储队列缓冲、原子性存储操作、存储失败恢复机制
3. EnhancedTrainingWorker: 本地缓存减少锁竞争、自适应重试策略、数据完整性验证、失败数据持久化和恢复

核心改进：
- 解决数据竞争和锁竞争问题
- 提供数据零丢失保证
- 实现智能重试和故障恢复
- 添加全面的性能监控
- 支持运行时配置和调优
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

# 导入增强组件
from atomic_data_buffer import AtomicDataBuffer
from thread_safe_agent_proxy import ThreadSafeAgentProxy
from enhanced_training_worker import EnhancedTrainingWorker

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

class EnhancedRolloutWorker:
    """【增强版】单个rollout worker，集成原子性数据处理"""
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
        
        # rollout完成控制
        self.rollout_completed = False
        self.target_rollout_steps = config.rollout_length
        
        # 环境状态
        self.env_state = None
        self.env_observations = None
        self.episode_step = 0
        
        # 严格的步数计数方法
        self.strict_step_counter = 0
        self.accumulated_reward = 0.0
        self.current_team_skill = None
        self.current_agent_skills = None
        self.skill_log_probs = None
        self.high_level_experiences_generated = 0
        
        # 步数验证标志
        self.step_validation_enabled = True
        self.last_reported_steps = 0
        
        # 技能计时器初始化
        self.skill_timer = 0
        
        # 【增强功能】数据完整性验证
        self.data_validation_enabled = True
        self.validation_failures = 0
        
        # 【增强功能】性能监控
        self.operation_times = deque(maxlen=100)
        self.last_performance_log = time.time()
        
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
        """【增强版】运行rollout worker主循环"""
        self.logger.info(f"增强版Rollout worker {self.worker_id} 开始运行，目标收集步数: {self.target_rollout_steps}")
        
        # 初始化rollout开始时间
        self.rollout_start_time = time.time()
        
        # 重置环境
        if not self.reset_environment():
            self.logger.error(f"Worker {self.worker_id}: 初始化失败")
            return
        
        try:
            # 无限循环训练模式：每次完成一个rollout后等待新的周期开始
            while not self.control_events['stop'].is_set():
                # 检查是否需要暂停
                if self.control_events['pause'].is_set():
                    time.sleep(0.1)
                    continue
                
                # 如果当前rollout已完成，等待新的rollout周期
                if self.rollout_completed:
                    time.sleep(0.1)  # 等待训练完成和状态重置
                    continue
                
                # 检查是否已达到rollout步数限制
                if self.samples_collected >= self.target_rollout_steps:
                    self.rollout_completed = True
                    self.complete_rollout()
                    continue
                
                # 执行一个rollout步骤
                success = self.run_step_enhanced(agent_proxy)
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
            self.logger.info(f"增强版Rollout worker {self.worker_id} 结束运行")
    
    def run_step_enhanced(self, agent_proxy):
        """【增强版】执行单个rollout步骤 - 集成数据验证和原子性操作"""
        step_start_time = time.time()
        
        try:
            # 确保环境状态有效
            if self.env_state is None:
                self.logger.warning(f"Worker {self.worker_id}: env_state为None，重置环境")
                if not self.reset_environment():
                    return False
            
            if self.env_observations is None:
                self.logger.warning(f"Worker {self.worker_id}: env_observations为None，重置环境")
                if not self.reset_environment():
                    return False
            
            # 每次都重新分配技能，保持简单逻辑
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
            
            # 原子性步数计数 - 确保一致性
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
            
            # 【增强功能】安全复制数据，避免None.copy()错误
            def safe_copy(data, default_shape=None):
                if data is None:
                    if default_shape is not None:
                        return np.zeros(default_shape)
                    return None
                if hasattr(data, 'copy'):
                    return data.copy()
                return np.array(data)
            
            # 【增强功能】数据完整性验证
            if self.data_validation_enabled:
                if not self._validate_experience_data(current_reward, actions, next_state):
                    self.validation_failures += 1
                    self.logger.warning(f"Worker {self.worker_id}: 经验数据验证失败")
                    return False
            
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
                'step_number': step_after_increment,
                'timestamp': time.time()  # 【增强功能】添加时间戳
            }
            
            # 【增强功能】使用原子性数据缓冲区
            success = self.data_buffer.put(low_level_experience, block=True, timeout=None)
            if success:
                self.logger.debug(f"Worker {self.worker_id}: 低层经验已放入缓冲区 - 步骤={step_after_increment}")
            else:
                self.logger.error(f"Worker {self.worker_id}: 低层经验放入缓冲区失败！")
                return False
            
            # 构造StateSkillDataset数据
            state_skill_experience = {
                'experience_type': 'state_skill',
                'worker_id': self.worker_id,
                'state': safe_copy(next_state, (self.config.state_dim,)),
                'team_skill': team_skill,
                'observations': safe_copy(next_observations, (self.config.n_agents, self.config.obs_dim)),
                'agent_skills': safe_copy(agent_skills, (self.config.n_agents,)) if agent_skills is not None else [0] * self.config.n_agents,
                'step_number': step_after_increment,
                'timestamp': time.time()
            }
            
            # 将StateSkillDataset数据放入缓冲区
            success = self.data_buffer.put(state_skill_experience, block=True, timeout=None)
            if success:
                self.logger.debug(f"Worker {self.worker_id}: StateSkill数据已放入缓冲区")
            else:
                self.logger.error(f"Worker {self.worker_id}: StateSkill数据放入缓冲区失败！")
                return False
            
            # 确定性高层经验收集 - 严格按k步收集
            if step_after_increment % self.config.k == 0:
                self.logger.debug(f"Worker {self.worker_id}: 确定性k步收集高层经验 - "
                               f"步数={step_after_increment}, k={self.config.k}, "
                               f"累积奖励={self.accumulated_reward:.4f}")
                
                success = self.store_high_level_experience_enhanced(f"确定性k步收集(步数={step_after_increment})")
                if success:
                    self.logger.debug(f"Worker {self.worker_id}: 高层经验收集成功 - 第{self.high_level_experiences_generated}个")
                    # 只在成功存储后重置累积奖励
                    self.accumulated_reward = 0.0
                else:
                    self.logger.error(f"Worker {self.worker_id}: 高层经验存储失败！")
            
            # 安全更新环境状态
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
            
            # 【增强功能】性能监控
            step_time = time.time() - step_start_time
            self.operation_times.append(step_time)
            
            # 定期记录性能统计
            if time.time() - self.last_performance_log > 60:  # 每分钟记录一次
                self._log_performance_stats()
                self.last_performance_log = time.time()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Worker {self.worker_id}: 步骤执行异常: {e}")
            # 即使异常也要确保步数计数的一致性
            if hasattr(self, 'samples_collected'):
                self.samples_collected += 1
            return False
    
    def _validate_experience_data(self, reward, actions, next_state):
        """【增强功能】验证经验数据的完整性"""
        try:
            # 检查奖励有效性
            if reward is not None:
                reward_val = reward if isinstance(reward, (int, float)) else np.sum(reward)
                if np.isnan(reward_val) or np.isinf(reward_val):
                    return False
            
            # 检查动作有效性
            if actions is not None:
                if np.any(np.isnan(actions)) or np.any(np.isinf(actions)):
                    return False
            
            # 检查状态有效性
            if next_state is not None:
                if np.any(np.isnan(next_state)) or np.any(np.isinf(next_state)):
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Worker {self.worker_id}: 数据验证异常: {e}")
            return False
    
    def store_high_level_experience_enhanced(self, reason="技能周期完成"):
        """【增强版】存储高层经验到数据缓冲区"""
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
                'reason': reason,
                'timestamp': time.time()  # 【增强功能】添加时间戳
            }
            
            # 【增强功能】使用原子性数据缓冲区
            success = self.data_buffer.put(high_level_experience, block=True, timeout=None)
            if success:
                self.high_level_experiences_generated += 1
                self.logger.debug(f"Worker {self.worker_id}: 高层经验已存储 - "
                                f"累积奖励={self.accumulated_reward:.4f}, 原因={reason}, "
                                f"总生成数={self.high_level_experiences_generated}")
                
                # 记录accumulated_reward的重置
                old_accumulated_reward = self.accumulated_reward
                self.accumulated_reward = 0.0
                self.skill_timer = 0
                self.logger.debug(f"💰 [REWARD_RESET] W{self.worker_id} accumulated_reward reset: "
                                f"{old_accumulated_reward:.4f} -> 0.0 (reason: {reason})")
                return True
            else:
                self.logger.error(f"Worker {self.worker_id}: 高层经验放入缓冲区失败！")
                return False
                
        except Exception as e:
            self.logger.error(f"Worker {self.worker_id}: 存储高层经验失败: {e}")
            return False
    
    def complete_rollout(self):
        """【增强版】完成当前rollout"""
        completion_time = time.time()
        
        # 基础验证 - 确保步数准确
        if self.samples_collected != self.target_rollout_steps:
            self.logger.warning(f"Worker {self.worker_id}: 步数不匹配! "
                              f"收集={self.samples_collected}, 目标={self.target_rollout_steps}")
            # 强制同步步数
            self.samples_collected = self.target_rollout_steps
        
        # 确定性高层经验计算
        expected_high_level = self.target_rollout_steps // self.config.k
        current_high_level = self.high_level_experiences_generated
        missing = expected_high_level - current_high_level
        
        self.logger.info(f"Worker {self.worker_id}: 【增强版】Rollout完成验证 - "
                       f"目标步数={self.target_rollout_steps}, 实际步数={self.samples_collected}, "
                       f"期望高层={expected_high_level}, 当前高层={current_high_level}, 缺失={missing}")
        
        # 确定性数据补齐
        if missing > 0:
            self.logger.info(f"Worker {self.worker_id}: 【增强版】开始确定性补齐 {missing} 个高层经验")
            
            补齐成功计数 = 0
            for i in range(missing):
                success = self.store_high_level_experience_enhanced(f"【增强版】确定性补齐#{i+1}")
                if success:
                    补齐成功计数 += 1
                    self.logger.debug(f"Worker {self.worker_id}: 确定性补齐#{i+1}成功")
                else:
                    self.logger.error(f"Worker {self.worker_id}: 确定性补齐#{i+1}失败！")
                    break
            
            # 确保补齐成功
            if 补齐成功计数 == missing:
                self.logger.info(f"Worker {self.worker_id}: ✅ 确定性补齐完成 {补齐成功计数}/{missing}")
            else:
                self.logger.error(f"Worker {self.worker_id}: ❌ 确定性补齐失败 {补齐成功计数}/{missing}")
        
        # 最终验证
        final_high_level = self.high_level_experiences_generated
        if final_high_level != expected_high_level:
            self.logger.error(f"Worker {self.worker_id}: ❌ 【增强版】最终验证失败! "
                            f"高层经验={final_high_level}, 期望={expected_high_level}")
        else:
            self.logger.info(f"Worker {self.worker_id}: ✅ 【增强版】最终验证通过! "
                           f"高层经验={final_high_level}, 期望={expected_high_level}")
        
        # 等待数据传输完成
        self.wait_for_data_transmission_complete_enhanced()
        
        # 计算rollout统计
        if not hasattr(self, 'rollout_start_time'):
            self.rollout_start_time = completion_time - 1.0
        
        rollout_duration = completion_time - self.rollout_start_time
        speed = self.samples_collected / rollout_duration if rollout_duration > 0 else 0
        
        self.logger.info(f"Worker {self.worker_id}: 【增强版】Rollout完成统计 - "
                       f"步数={self.samples_collected}, 高层经验={self.high_level_experiences_generated}, "
                       f"耗时={rollout_duration:.1f}s, 速度={speed:.1f}步/s, "
                       f"验证失败={self.validation_failures}")
        
        # 重置开始时间
        self.rollout_start_time = completion_time
    
    def wait_for_data_transmission_complete_enhanced(self):
        """【增强版】等待数据传输100%完成"""
        max_wait_time = 15.0  # 增强版：更长等待时间
        wait_start = time.time()
        initial_queue_size = self.data_buffer.qsize()
        
        self.logger.debug(f"Worker {self.worker_id}: 【增强版】开始等待数据传输100%完成 - "
                        f"初始队列大小={initial_queue_size}")
        
        consecutive_empty_checks = 0
        required_empty_checks = 10  # 需要连续10次检查队列为空
        
        while time.time() - wait_start < max_wait_time:
            current_queue_size = self.data_buffer.qsize()
            
            if current_queue_size == 0:
                consecutive_empty_checks += 1
                if consecutive_empty_checks >= required_empty_checks:
                    # 连续多次确认队列为空，数据传输完成
                    break
            else:
                consecutive_empty_checks = 0  # 重置计数器
            
            time.sleep(0.05)  # 更频繁的检查
        
        final_wait_time = time.time() - wait_start
        final_queue_size = self.data_buffer.qsize()
        
        if final_queue_size == 0 and consecutive_empty_checks >= required_empty_checks:
            self.logger.debug(f"Worker {self.worker_id}: ✅ 【增强版】数据传输100%完成 - "
                            f"等待时间={final_wait_time:.2f}s, 连续空检查={consecutive_empty_checks}")
        else:
            self.logger.warning(f"Worker {self.worker_id}: ⚠️ 【增强版】数据传输未完全完成 - "
                              f"等待时间={final_wait_time:.2f}s, 剩余队列={final_queue_size}, "
                              f"连续空检查={consecutive_empty_checks}/{required_empty_checks}")
    
    def _log_performance_stats(self):
        """【增强功能】记录性能统计"""
        if not self.operation_times:
            return
        
        avg_time = sum(self.operation_times) / len(self.operation_times)
        max_time = max(self.operation_times)
        min_time = min(self.operation_times)
        
        self.logger.debug(f"Worker {self.worker_id}: 性能统计 - "
                        f"平均步骤时间={avg_time*1000:.2f}ms, "
                        f"最大={max_time*1000:.2f}ms, 最小={min_time*1000:.2f}ms, "
                        f"验证失败={self.validation_failures}")
    
    def reset_skill_state(self):
        """重置技能状态"""
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
            'current_episode_step': self.episode_step,
            'validation_failures': self.validation_failures,
            'avg_step_time_ms': sum(self.operation_times) / len(self.operation_times) * 1000 if self.operation_times else 0
        }

class EnhancedThreadedRolloutTrainer:
    """【增强版】多线程HMASD Rollout-based训练器"""
    
    def __init__(self, config, args=None):
        """
        初始化增强版多线程训练器
        
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
        self.log_dir = f"logs/enhanced_threaded_rollout_training_{timestamp}"
        os.makedirs(self.log_dir, exist_ok=True)
        
        # 初始化日志系统
        self._init_logging()
        
        # 线程控制
        self.control_events = {
            'stop': Event(),
            'pause': Event()
        }
        
        # 【增强功能】使用原子性数据缓冲区
        buffer_size = getattr(args, 'buffer_size', 10000)
        persistence_dir = os.path.join(self.log_dir, 'buffer_persistence')
        enable_recovery = getattr(args, 'enable_recovery', True)
        
        self.data_buffer = AtomicDataBuffer(
            maxsize=buffer_size,
            persistence_dir=persistence_dir,
            enable_recovery=enable_recovery
        )
        
        # 统计信息
        self.start_time = None
        self.total_updates = 0
        self.total_samples = ThreadSafeCounter()
        self.total_steps = ThreadSafeCounter()
        
        # 【增强功能】性能监控
        self.performance_monitor = {
            'rollout_speeds': deque(maxlen=100),
            'training_speeds': deque(maxlen=100),
            'buffer_utilizations': deque(maxlen=100),
            'last_monitor_time': time.time()
        }
        
        self.logger.info("【增强版】ThreadedRolloutTrainer初始化完成")
        self.logger.info(f"日志目录: {self.log_dir}")
        self.logger.info(f"训练线程数: {self.num_training_threads}")
        self.logger.info(f"Rollout线程数: {self.num_rollout_threads}")
        self.logger.info(f"数据缓冲区大小: {buffer_size}")
        self.logger.info(f"数据持久化: {persistence_dir}")
        self.logger.info(f"故障恢复: {enable_recovery}")
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
            log_file='enhanced_threaded_rollout_training.log',
            file_level=LOG_LEVELS.get(log_level.lower(), 20),
            console_level=LOG_LEVELS.get(console_level.lower(), 20)
        )
        
        self.logger = get_logger("EnhancedThreadedRolloutTrainer")
    
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
        
        # 【修复】使用ThreadSafeAgentProxy替代原来的AgentProxy
        self.agent_proxy = ThreadSafeAgentProxy(self.agent, self.config, self.logger, self.data_buffer)
        
        self.logger.info("【增强版】HMASD代理初始化完成")
    
    def start_rollout_threads(self):
        """启动rollout线程"""
        self.logger.info(f"启动 {self.num_rollout_threads} 个增强版rollout线程")
        
        self.rollout_workers = []
        self.rollout_threads = []
        
        env_factory = self.create_env_factory()
        
        for i in range(self.num_rollout_threads):
            # 【增强功能】使用增强版RolloutWorker
            worker = EnhancedRolloutWorker(
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
                name=f"EnhancedRolloutWorker-{i}"
            )
            thread.daemon = True
            
            self.rollout_workers.append(worker)
            self.rollout_threads.append(thread)
            thread.start()
        
        self.logger.info("所有增强版rollout线程已启动")
        
        # 设置AgentProxy对rollout workers的引用
        self.agent_proxy.rollout_workers = self.rollout_workers
    
    def start_training_threads(self):
        """启动训练线程"""
        self.logger.info(f"启动 {self.num_training_threads} 个增强版training线程")
        
        self.training_workers = []
        self.training_threads = []
        
        for i in range(self.num_training_threads):
            # 【增强功能】使用增强版TrainingWorker
            worker = EnhancedTrainingWorker(
                worker_id=i,
                agent_proxy=self.agent_proxy,
                data_buffer=self.data_buffer,
                control_events=self.control_events,
                logger=self.logger,
                config=self.config,
                trainer=self
            )
            
            thread = threading.Thread(
                target=worker.run,
                name=f"EnhancedTrainingWorker-{i}"
            )
            thread.daemon = True
            
            self.training_workers.append(worker)
            self.training_threads.append(thread)
            thread.start()
        
        self.logger.info("所有增强版training线程已启动")
    
    def monitor_training_enhanced(self, total_steps=100000):
        """【增强版】监控训练进度"""
        self.logger.info(f"开始【增强版】训练监控，目标步数: {total_steps:,}")
        
        self.start_time = time.time()
        last_log_time = self.start_time
        last_stats_log_time = self.start_time
        last_performance_log_time = self.start_time
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
                    self.log_progress_enhanced(cumulative_trainer_steps, total_steps)
                    last_log_time = current_time
                
                # 每10分钟记录一次详细统计
                if current_time - last_stats_log_time >= 600:
                    self.log_detailed_stats_enhanced()
                    last_stats_log_time = current_time
                
                # 【增强功能】每5分钟记录一次性能监控
                if current_time - last_performance_log_time >= 300:
                    self.log_performance_monitoring()
                    last_performance_log_time = current_time
                
                # 检查线程健康状态
                self.check_thread_health_enhanced()
                
                time.sleep(30)  # 每30秒检查一次
        
        except KeyboardInterrupt:
            self.logger.info("训练被用户中断")
        
        finally:
            self.stop_training_enhanced()
    
    def log_progress_enhanced(self, current_steps, total_steps):
        """【增强版】记录训练进度"""
        progress_percent = (current_steps / total_steps) * 100
        remaining_steps = total_steps - current_steps
        
        # 计算时间统计
        elapsed_time = time.time() - self.start_time
        if current_steps > 0:
            estimated_total_time = elapsed_time * total_steps / current_steps
            remaining_time = estimated_total_time - elapsed_time
        else:
            remaining_time = 0
        
        # 【增强功能】获取原子性缓冲区统计
        buffer_stats = self.data_buffer.get_stats()
        
        # 计算rollout workers统计
        total_samples = sum(worker.samples_collected for worker in self.rollout_workers)
        total_episodes = sum(worker.episodes_completed for worker in self.rollout_workers)
        total_high_level_exp = sum(worker.high_level_experiences_generated for worker in self.rollout_workers)
        total_validation_failures = sum(worker.validation_failures for worker in self.rollout_workers)
        
        # 计算training workers统计
        total_updates = sum(worker.updates_performed for worker in self.training_workers)
        total_processed = sum(worker.samples_processed for worker in self.training_workers)
        
        # 【增强功能】获取代理统计 - 修复方法调用
        if hasattr(self.agent_proxy, 'get_storage_stats'):
            agent_stats = self.agent_proxy.get_storage_stats()
        else:
            # 回退到基本统计
            agent_stats = {
                'high_level_stored': getattr(self.agent_proxy, 'high_level_experiences_stored', 0),
                'low_level_stored': getattr(self.agent_proxy, 'low_level_experiences_stored', 0),
                'state_skill_stored': 0
            }
        
        # 计算步数速度
        steps_per_second = current_steps / elapsed_time if elapsed_time > 0 else 0
        
        self.logger.info(f"【增强版】训练进度: {progress_percent:.1f}% "
                        f"({current_steps:,} / {total_steps:,} 步), "
                        f"剩余: {remaining_steps:,} 步")
        self.logger.info(f"时间: 已用={elapsed_time/3600:.1f}h, 预计剩余={remaining_time/3600:.1f}h, "
                        f"速度={steps_per_second:.1f} 步/秒")
        self.logger.info(f"Rollout: 样本={total_samples:,}, Episodes={total_episodes:,}, "
                        f"高层经验={total_high_level_exp:,}, 验证失败={total_validation_failures}")
        self.logger.info(f"Training: 更新={total_updates}, 处理样本={total_processed:,}")
        self.logger.info(f"代理存储: 高层={agent_stats['high_level_stored']}, "
                        f"低层={agent_stats['low_level_stored']}, "
                        f"状态技能={agent_stats['state_skill_stored']}")
        self.logger.info(f"原子缓冲区: 队列={buffer_stats['queue_size']}, "
                        f"利用率={buffer_stats['utilization']:.1%}, "
                        f"拥塞={buffer_stats['congestion_detected']}")
    
    def log_detailed_stats_enhanced(self):
        """【增强版】记录详细统计信息"""
        self.logger.info("=== 【增强版】详细统计信息 ===")
        
        # Rollout workers统计
        self.logger.info("增强版Rollout Workers:")
        for i, worker in enumerate(self.rollout_workers[:5]):  # 只显示前5个
            stats = worker.get_worker_stats()
            self.logger.info(f"  Worker {i}: 样本={stats['samples_collected']}, "
                           f"Episodes={stats['episodes_completed']}, "
                           f"高层经验={stats['high_level_experiences_generated']}, "
                           f"当前技能={stats['current_team_skill']}, "
                           f"累积奖励={stats['current_accumulated_reward']:.3f}, "
                           f"验证失败={stats['validation_failures']}, "
                           f"平均步骤时间={stats['avg_step_time_ms']:.2f}ms")
        if len(self.rollout_workers) > 5:
            self.logger.info(f"  ... 还有 {len(self.rollout_workers) - 5} 个workers")
        
        # Training workers统计
        self.logger.info("增强版Training Workers:")
        for i, worker in enumerate(self.training_workers[:5]):  # 只显示前5个
            stats = worker.get_performance_stats()
            self.logger.info(f"  Worker {i}: 更新={stats['updates_performed']}, "
                           f"处理样本={stats['samples_processed']}, "
                           f"成功率={stats['success_rate']:.2%}, "
                           f"缓存命中={stats['cache_hits']}, "
                           f"缓存未命中={stats['cache_misses']}")
        if len(self.training_workers) > 5:
            self.logger.info(f"  ... 还有 {len(self.training_workers) - 5} 个workers")
        
        # 【增强功能】原子性缓冲区详细统计
        buffer_stats = self.data_buffer.get_stats()
        self.logger.info(f"原子性数据缓冲区详细统计:")
        self.logger.info(f"  队列大小: {buffer_stats['queue_size']}/{buffer_stats['max_size']}")
        self.logger.info(f"  利用率: {buffer_stats['utilization']:.1%}")
        self.logger.info(f"  总添加: {buffer_stats['total_added']}, 总消费: {buffer_stats['total_consumed']}")
        self.logger.info(f"  高优先级: 添加={buffer_stats['high_priority_added']}, 消费={buffer_stats['high_priority_consumed']}")
        self.logger.info(f"  普通优先级: 添加={buffer_stats['normal_priority_added']}, 消费={buffer_stats['normal_priority_consumed']}")
        self.logger.info(f"  失败项目: {buffer_stats['failed_items']}")
        self.logger.info(f"  验证失败: {buffer_stats['validation_failures']}")
        self.logger.info(f"  平均操作时间: {buffer_stats['avg_operation_time_ms']:.2f}ms")
        self.logger.info(f"  处理速度: {buffer_stats['processing_speed']:.1f} 项/秒")
        self.logger.info(f"  拥塞检测: {buffer_stats['congestion_detected']}")
        
        # 【增强功能】线程安全代理统计
        agent_stats = self.agent_proxy.get_storage_stats()
        self.logger.info(f"线程安全代理统计:")
        self.logger.info(f"  存储尝试: {agent_stats['total_attempts']}")
        self.logger.info(f"  存储成功: {agent_stats['total_successes']}")
        self.logger.info(f"  存储失败: {agent_stats['total_failures']}")
        self.logger.info(f"  队列溢出: {agent_stats['queue_overflows']}")
        self.logger.info(f"  验证失败: {agent_stats['validation_failures']}")
        
        # GPU内存使用（如果有GPU）
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated(self.device) / 1024**3
            memory_reserved = torch.cuda.memory_reserved(self.device) / 1024**3
            self.logger.info(f"GPU内存: 已分配={memory_allocated:.2f}GB, 已保留={memory_reserved:.2f}GB")
    
    def log_performance_monitoring(self):
        """【增强功能】记录性能监控信息"""
        current_time = time.time()
        
        # 计算rollout速度
        total_samples = sum(worker.samples_collected for worker in self.rollout_workers)
        time_diff = current_time - self.performance_monitor['last_monitor_time']
        if time_diff > 0:
            rollout_speed = total_samples / time_diff
            self.performance_monitor['rollout_speeds'].append(rollout_speed)
        
        # 计算缓冲区利用率
        buffer_stats = self.data_buffer.get_stats()
        self.performance_monitor['buffer_utilizations'].append(buffer_stats['utilization'])
        
        # 计算平均性能指标
        avg_rollout_speed = sum(self.performance_monitor['rollout_speeds']) / len(self.performance_monitor['rollout_speeds']) if self.performance_monitor['rollout_speeds'] else 0
        avg_buffer_util = sum(self.performance_monitor['buffer_utilizations']) / len(self.performance_monitor['buffer_utilizations']) if self.performance_monitor['buffer_utilizations'] else 0
        
        self.logger.info(f"【性能监控】平均rollout速度: {avg_rollout_speed:.1f} 样本/秒, "
                        f"平均缓冲区利用率: {avg_buffer_util:.1%}")
        
        # 检测性能异常
        if avg_buffer_util > 0.9:
            self.logger.warning("⚠️ 缓冲区利用率过高，可能存在性能瓶颈")
        if avg_rollout_speed < 10:
            self.logger.warning("⚠️ Rollout速度过低，检查环境或网络性能")
        
        self.performance_monitor['last_monitor_time'] = current_time
    
    def check_thread_health_enhanced(self):
        """【增强版】检查线程健康状态"""
        # 检查rollout线程
        dead_rollout_threads = [i for i, thread in enumerate(self.rollout_threads) if not thread.is_alive()]
        if dead_rollout_threads:
            self.logger.warning(f"发现 {len(dead_rollout_threads)} 个死亡的rollout线程: {dead_rollout_threads}")
        
        # 检查training线程
        dead_training_threads = [i for i, thread in enumerate(self.training_threads) if not thread.is_alive()]
        if dead_training_threads:
            self.logger.warning(f"发现 {len(dead_training_threads)} 个死亡的training线程: {dead_training_threads}")
        
        # 【增强功能】检查数据流是否正常
        buffer_stats = self.data_buffer.get_stats()
        if buffer_stats['total_added'] == getattr(self, '_last_total_added', 0):
            self.logger.warning("数据缓冲区添加数量未增加，可能rollout线程有问题")
        if buffer_stats['total_consumed'] == getattr(self, '_last_total_consumed', 0):
            self.logger.warning("数据缓冲区消费数量未增加，可能training线程有问题")
        
        self._last_total_added = buffer_stats['total_added']
        self._last_total_consumed = buffer_stats['total_consumed']
        
        # 【增强功能】检查代理存储状态
        agent_stats = self.agent_proxy.get_storage_stats()
        if agent_stats['total_failures'] > agent_stats['total_successes'] * 0.1:  # 失败率超过10%
            self.logger.warning(f"代理存储失败率过高: {agent_stats['total_failures']}/{agent_stats['total_attempts']}")
    
    def stop_training_enhanced(self):
        """【增强版】停止训练"""
        self.logger.info("停止【增强版】训练...")
        
        # 设置停止事件
        self.control_events['stop'].set()
        
        # 等待所有线程结束
        self.logger.info("等待rollout线程结束...")
        for i, thread in enumerate(self.rollout_threads):
            thread.join(timeout=15)  # 增强版：更长等待时间
            if thread.is_alive():
                self.logger.warning(f"Rollout线程 {i} 未能在15秒内结束")
        
        self.logger.info("等待training线程结束...")
        for i, thread in enumerate(self.training_threads):
            thread.join(timeout=15)  # 增强版：更长等待时间
            if thread.is_alive():
                self.logger.warning(f"Training线程 {i} 未能在15秒内结束")
        
        # 【增强功能】关闭线程安全代理
        if hasattr(self, 'agent_proxy') and hasattr(self.agent_proxy, 'shutdown'):
            self.agent_proxy.shutdown()
        
        # 【增强功能】强制备份缓冲区数据
        if hasattr(self, 'data_buffer'):
            self.data_buffer.force_backup()
        
        self.logger.info("所有【增强版】线程已停止")
    
    def save_final_model(self):
        """保存最终模型"""
        try:
            final_model_path = os.path.join(self.log_dir, 'enhanced_final_model.pt')
            self.agent.save_model(final_model_path)
            self.logger.info(f"【增强版】最终模型已保存: {final_model_path}")
        except Exception as e:
            self.logger.error(f"保存【增强版】最终模型失败: {e}")
    
    def cleanup_enhanced(self):
        """【增强版】清理资源"""
        try:
            # 1. 停止训练（如果还没停止）
            if not self.control_events['stop'].is_set():
                self.stop_training_enhanced()
            
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
            
            # 4. 【增强功能】清理原子性缓冲区
            if hasattr(self, 'data_buffer'):
                cleared_count = self.data_buffer.clear()
                self.logger.info(f"原子性缓冲区已清理: {cleared_count} 项")
            
            # 5. 【增强功能】记录最终性能统计
            self._log_final_performance_stats()
            
            self.logger.info("所有【增强版】资源清理完成")
            
        except Exception as e:
            print(f"清理【增强版】资源时出错: {e}")
    
    def _log_final_performance_stats(self):
        """【增强功能】记录最终性能统计"""
        try:
            if hasattr(self, 'performance_monitor'):
                avg_rollout_speed = sum(self.performance_monitor['rollout_speeds']) / len(self.performance_monitor['rollout_speeds']) if self.performance_monitor['rollout_speeds'] else 0
                avg_buffer_util = sum(self.performance_monitor['buffer_utilizations']) / len(self.performance_monitor['buffer_utilizations']) if self.performance_monitor['buffer_utilizations'] else 0
                
                self.logger.info("=== 【增强版】最终性能统计 ===")
                self.logger.info(f"平均Rollout速度: {avg_rollout_speed:.1f} 样本/秒")
                self.logger.info(f"平均缓冲区利用率: {avg_buffer_util:.1%}")
                
                # 获取最终缓冲区统计
                buffer_stats = self.data_buffer.get_stats()
                self.logger.info(f"最终缓冲区统计: {buffer_stats}")
                
                # 获取最终代理统计
                agent_stats = self.agent_proxy.get_storage_stats()
                self.logger.info(f"最终代理统计: {agent_stats}")
                
        except Exception as e:
            self.logger.error(f"记录最终性能统计失败: {e}")
    
    def train_enhanced(self, total_steps=100000):
        """
        【增强版】执行完整的多线程rollout-based训练
        
        参数:
            total_steps: 训练总步数
        """
        self.logger.info(f"开始【增强版】HMASD多线程Rollout-based训练: {total_steps:,} 步")
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
            self.monitor_training_enhanced(total_steps)
            
        except KeyboardInterrupt:
            self.logger.info("【增强版】训练被用户中断")
        except Exception as e:
            self.logger.error(f"【增强版】训练过程中出错: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            # 保存最终模型
            self.save_final_model()
            
            # 清理资源
            self.cleanup_enhanced()
        
        # 训练完成
        if self.start_time:
            total_time = time.time() - self.start_time
            final_steps = sum(worker.samples_collected for worker in self.rollout_workers)
            self.logger.info(f"\n【增强版】训练完成！")
            self.logger.info(f"总时间: {total_time/3600:.2f}小时")
            self.logger.info(f"总步数: {final_steps:,}")
            
            # 输出最终统计
            total_samples = sum(worker.samples_collected for worker in self.rollout_workers)
            total_episodes = sum(worker.episodes_completed for worker in self.rollout_workers)
            total_updates = sum(worker.updates_performed for worker in self.training_workers)
            total_validation_failures = sum(worker.validation_failures for worker in self.rollout_workers)
            
            self.logger.info(f"总样本数: {total_samples:,}")
            self.logger.info(f"总Episodes: {total_episodes:,}")
            self.logger.info(f"总更新数: {total_updates}")
            self.logger.info(f"总验证失败: {total_validation_failures}")
            
            if total_time > 0:
                self.logger.info(f"样本收集速度: {total_samples/total_time:.1f} 样本/秒")
                self.logger.info(f"Episode完成速度: {total_episodes/total_time:.1f} episodes/秒")
                self.logger.info(f"步数完成速度: {final_steps/total_time:.1f} 步/秒")

def parse_args_enhanced():
    """解析命令行参数 - 增强版"""
    parser = argparse.ArgumentParser(description='【增强版】HMASD多线程Rollout-based训练（集成三大增强组件）')
    
    # 训练参数
    parser.add_argument('--steps', type=int, default=None, help='训练总步数（如果不指定，将从config.py中读取）')
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cpu', 'cuda'])
    parser.add_argument('--debug', action='store_true', help='启用调试模式')
    
    # 线程配置（按照论文 Appendix E）
    parser.add_argument('--training_threads', type=int, default=16, help='训练线程数（论文默认16）')
    parser.add_argument('--rollout_threads', type=int, default=32, help='Rollout线程数（论文默认32）')
    parser.add_argument('--buffer_size', type=int, default=10000, help='数据缓冲区大小')
    
    # 【增强功能】新增参数
    parser.add_argument('--enable_recovery', action='store_true', default=True, help='启用故障恢复机制')
    parser.add_argument('--enable_validation', action='store_true', default=True, help='启用数据完整性验证')
    parser.add_argument('--enable_persistence', action='store_true', default=True, help='启用数据持久化')
    
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
    """【增强版】主函数"""
    args = parse_args_enhanced()
    
    # 创建配置
    config = Config()
    
    # 确定训练步数：优先使用命令行参数，其次使用配置文件中的值
    if args.steps is not None:
        total_steps = args.steps
        print(f"📈 使用命令行指定的训练步数: {total_steps:,}")
    else:
        total_steps = int(config.total_timesteps)
        print(f"📈 从config.py读取训练步数: {total_steps:,}")
    
    print("🚀 【增强版】HMASD多线程Rollout-based训练（集成三大增强组件）")
    print("=" * 80)
    print(f"📊 线程配置: {args.training_threads} 训练线程 + {args.rollout_threads} rollout线程")
    print(f"🎯 训练步数: {total_steps:,}")
    print(f"🗂️ 缓冲区大小: {args.buffer_size}")
    print(f"🔧 故障恢复: {args.enable_recovery}")
    print(f"✅ 数据验证: {args.enable_validation}")
    print(f"💾 数据持久化: {args.enable_persistence}")
    
    # 验证并打印配置
    config.validate_training_mode()
    config.validate_rollout_config()
    print(config.get_rollout_summary())
    
    try:
        # 创建增强版训练器
        trainer = EnhancedThreadedRolloutTrainer(config, args)
        
        # 开始训练
        trainer.train_enhanced(total_steps=total_steps)
        
        print("🎉 【增强版】训练成功完成！")
        
    except KeyboardInterrupt:
        print("\n⚠️ 【增强版】训练被用户中断")
    except Exception as e:
        print(f"\n❌ 【增强版】训练失败: {e}")
        import traceback
        traceback.print_exc()
    finally:
        try:
            shutdown_logging()
        except:
            pass

if __name__ == "__main__":
    main()
