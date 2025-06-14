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
from hmasd.agent import HMASDAgent
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
    """线程安全的数据缓冲区"""
    def __init__(self, maxsize=10000):
        self.queue = queue.Queue(maxsize=maxsize)
        self.total_added = ThreadSafeCounter()
        self.total_consumed = ThreadSafeCounter()
        
    def put(self, item, block=True, timeout=None):
        """添加数据到缓冲区"""
        try:
            self.queue.put(item, block=block, timeout=timeout)
            self.total_added.increment()
            return True
        except queue.Full:
            return False
    
    def get(self, block=True, timeout=None):
        """从缓冲区获取数据"""
        try:
            item = self.queue.get(block=block, timeout=timeout)
            self.total_consumed.increment()
            return item
        except queue.Empty:
            return None
    
    def qsize(self):
        """获取当前队列大小"""
        return self.queue.qsize()
    
    def empty(self):
        """检查队列是否为空"""
        return self.queue.empty()
    
    def get_stats(self):
        """获取统计信息"""
        return {
            'queue_size': self.qsize(),
            'total_added': self.total_added.get(),
            'total_consumed': self.total_consumed.get()
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
        
        # 【新增】技能周期管理
        self.skill_timer = 0
        self.accumulated_reward = 0.0
        self.current_team_skill = None
        self.current_agent_skills = None
        self.skill_log_probs = None
        self.high_level_experiences_generated = 0
        self.last_skill_assignment_step = 0
        
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
                    self.logger.info(f"🔄 Worker {self.worker_id} 完成rollout: "
                                   f"{self.samples_collected}/{self.target_rollout_steps} 步")
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
        """执行单个rollout步骤 - 修复版本：确保每次都正确计数"""
        
        try:
            # 【修复】检查是否需要重新分配技能
            if self.should_reassign_skills():
                # 在重分配前，先存储上一周期的高层经验（如果有的话）
                if (self.current_team_skill is not None and 
                    self.skill_timer >= self.config.k and 
                    self.accumulated_reward != 0):
                    self.store_high_level_experience("技能周期完成")
                
                # 重新分配技能
                self.assign_new_skills(agent_proxy)
            
            # 从代理获取动作（使用worker自己的技能状态）
            actions, action_logprobs = agent_proxy.get_actions_for_worker(
                self.env_state, self.env_observations, self.current_agent_skills, self.worker_id
            )
            
            # 执行环境步骤
            next_observations, rewards, dones, next_state = self.step_environment(actions)
            
            # 【关键修复】无论后续处理如何，环境步骤已执行，必须计数
            self.samples_collected += 1
            
            # 【修复A2】确保rewards是有效的数值
            if rewards is None:
                current_reward = 0.0
                self.logger.warning(f"Worker {self.worker_id}: 环境步骤返回None奖励，使用0.0")
            else:
                current_reward = rewards if isinstance(rewards, (int, float)) else np.sum(rewards)
            
            # 【修复A3】累积奖励和更新技能计时器
            self.accumulated_reward += current_reward
            self.skill_timer += 1
            self.total_reward += current_reward
            
            # 构造低层经验数据
            low_level_experience = {
                'experience_type': 'low_level',
                'worker_id': self.worker_id,
                'state': self.env_state.copy(),
                'observations': self.env_observations.copy(),
                'actions': actions.copy(),
                'rewards': current_reward,
                'next_state': next_state.copy(),
                'next_observations': next_observations.copy(),
                'dones': dones,
                'episode_step': self.episode_step,
                'team_skill': self.current_team_skill,
                'agent_skills': self.current_agent_skills.copy() if self.current_agent_skills is not None else None,
                'action_logprobs': action_logprobs,
                'skill_log_probs': self.skill_log_probs
            }
            
            # 【修复A4】将低层经验放入缓冲区，记录是否成功
            buffer_success = self.data_buffer.put(low_level_experience, block=False)
            if not buffer_success:
                self.logger.warning(f"Worker {self.worker_id}: 数据缓冲区满，丢弃低层样本")
            
            # 【修复】只在环境终止时存储高层经验（技能周期完成的情况已在重分配时处理）
            if dones:
                self.store_high_level_experience("环境终止")
            
            # 更新环境状态
            self.env_state = next_state
            self.env_observations = next_observations
            
            # 检查episode是否结束
            if dones or self.episode_step >= 1000:  # 最大步数限制
                self.episodes_completed += 1
                self.logger.debug(f"Worker {self.worker_id}: Episode {self.episodes_completed} 完成, "
                                f"步数: {self.episode_step}, 奖励: {self.total_reward:.2f}")
                
                # 强制存储任何pending的高层经验
                if self.skill_timer > 0 and self.accumulated_reward != 0:
                    self.store_high_level_experience("Episode结束")
                
                # 重置环境和技能状态
                if not self.reset_environment():
                    return False
                self.reset_skill_state()
                self.total_reward = 0.0
            
            return True
            
        except Exception as e:
            self.logger.error(f"Worker {self.worker_id}: 步骤执行异常: {e}")
            # 【关键修复】即使异常，环境步骤也已执行，必须计数
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
            
            # 将高层经验放入缓冲区
            if self.data_buffer.put(high_level_experience, block=False):
                self.high_level_experiences_generated += 1
                self.logger.debug(f"Worker {self.worker_id}: 高层经验已存储 - "
                                f"累积奖励={self.accumulated_reward:.4f}, 原因={reason}, "
                                f"总生成数={self.high_level_experiences_generated}")
                
                # 重置累积状态
                self.accumulated_reward = 0.0
                self.skill_timer = 0
                return True
            else:
                self.logger.warning(f"Worker {self.worker_id}: 数据缓冲区满，丢弃高层经验")
                return False
                
        except Exception as e:
            self.logger.error(f"Worker {self.worker_id}: 存储高层经验失败: {e}")
            return False
    
    def reset_skill_state(self):
        """重置技能状态"""
        self.skill_timer = 0
        self.accumulated_reward = 0.0
        self.current_team_skill = None
        self.current_agent_skills = None
        self.skill_log_probs = None
        self.last_skill_assignment_step = 0
    
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
    def __init__(self, agent, config, logger):
        self.agent = agent
        self.config = config
        self.logger = logger
        self.lock = Lock()
        
        # 全局rollout步数计数器（用于判断是否应该更新）
        self.global_rollout_steps = 0
        self.high_level_experiences_stored = 0
        self.low_level_experiences_stored = 0
    
    def assign_skills_for_worker(self, state, observations, worker_id):
        """为特定worker分配技能"""
        with self.lock:
            try:
                team_skill, agent_skills, log_probs = self.agent.assign_skills(
                    state, observations, deterministic=False
                )
                
                self.logger.debug(f"Worker {worker_id}: 技能分配 - "
                                f"team_skill={team_skill}, agent_skills={agent_skills}")
                
                return team_skill, agent_skills, log_probs
                
            except Exception as e:
                self.logger.error(f"Worker {worker_id}: 技能分配失败: {e}")
                # 返回默认技能
                return 0, [0] * self.config.n_agents, {
                    'team_log_prob': 0.0, 
                    'agent_log_probs': [0.0] * self.config.n_agents
                }
    
    def get_actions_for_worker(self, state, observations, agent_skills, worker_id):
        """为特定worker获取动作（使用给定的技能）"""
        with self.lock:
            try:
                actions, action_logprobs = self.agent.select_action(
                    observations, agent_skills, deterministic=False, env_id=worker_id
                )
                
                return actions, action_logprobs
                
            except Exception as e:
                self.logger.error(f"Worker {worker_id}: 获取动作失败: {e}")
                # 返回随机动作作为回退
                n_agents = len(observations)
                random_actions = np.random.randn(n_agents, self.config.action_dim)
                return random_actions, np.zeros(n_agents)
    
    def store_experience(self, experience_batch):
        """批量存储经验到代理 - 修复版本：确保步数计数统一"""
        with self.lock:
            stored_count = 0
            low_level_stored = 0
            high_level_stored = 0
            
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
                    
                    elif experience_type == 'low_level':
                        # 存储低层经验
                        success = self.store_low_level_experience(experience)
                        if success:
                            self.low_level_experiences_stored += 1
                            low_level_stored += 1
                            stored_count += 1
                            # 【关键修复】每个成功的低层经验对应一个环境步骤
                            self.global_rollout_steps += 1
                    
                    else:
                        self.logger.warning(f"未知经验类型: {experience_type}")
                    
                except Exception as e:
                    self.logger.error(f"存储经验失败: {e}")
            
            # 【关键修复】同步代理的步数计数器
            old_steps = self.agent.steps_collected
            self.agent.steps_collected = self.global_rollout_steps
            
            # 【详细调试】记录步数同步情况
            if self.global_rollout_steps % 100 == 0 and low_level_stored > 0:
                self.logger.info(f"[STEP_SYNC] 步数同步: agent.steps_collected {old_steps}→{self.agent.steps_collected}, "
                                f"global_rollout_steps={self.global_rollout_steps}, "
                                f"本批次低层样本={low_level_stored}")
            
            return stored_count
    
    def store_high_level_experience(self, experience):
        """存储高层经验到代理"""
        try:
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
            return success
        except Exception as e:
            self.logger.error(f"存储高层经验失败: {e}")
            return False
    
    def store_low_level_experience(self, experience):
        """存储低层经验到代理"""
        try:
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
            return success
        except Exception as e:
            self.logger.error(f"存储低层经验失败: {e}")
            return False
    
    def should_update(self):
        """检查是否应该更新 - 修复版本：基于所有workers完成状态判断"""
        with self.lock:
            # 【关键修复】检查所有rollout workers是否都完成了rollout
            if not hasattr(self, 'rollout_workers'):
                # 如果还没有rollout_workers引用，使用旧逻辑作为回退
                self.agent.steps_collected = self.global_rollout_steps
                return self.agent.should_rollout_update()
            
            # 统计完成的workers数量
            completed_workers = sum(1 for worker in self.rollout_workers 
                                  if getattr(worker, 'rollout_completed', False))
            total_workers = len(self.rollout_workers)
            
            # 所有workers都完成rollout时才开始更新
            all_completed = completed_workers == total_workers
            
            if all_completed:
                # 计算总收集步数
                total_collected = sum(worker.samples_collected for worker in self.rollout_workers)
                target_steps = self.rollout_workers[0].target_rollout_steps * total_workers
                
                self.logger.info(f"🎯 所有workers完成rollout: {completed_workers}/{total_workers}, "
                               f"总收集步数: {total_collected}/{target_steps}")
                
                # 同步代理步数计数器
                self.agent.steps_collected = self.global_rollout_steps
                return True
            
            # 记录进度（每50次检查记录一次）
            if not hasattr(self, '_update_check_count'):
                self._update_check_count = 0
            self._update_check_count += 1
            
            if self._update_check_count % 50 == 0:
                self.logger.debug(f"⏳ 等待rollout完成: {completed_workers}/{total_workers} workers完成")
            
            return False
    
    def update(self):
        """执行模型更新"""
        with self.lock:
            try:
                update_info = self.agent.rollout_update()
                
                # 【关键修复】训练完成后重置所有workers的rollout状态
                if update_info:
                    self.reset_all_workers_rollout_state()
                
                return update_info
            except Exception as e:
                self.logger.error(f"模型更新失败: {e}")
                return None
    
    def reset_all_workers_rollout_state(self):
        """重置所有workers的rollout状态，准备下一个rollout周期"""
        if not hasattr(self, 'rollout_workers'):
            return
        
        reset_count = 0
        for worker in self.rollout_workers:
            # 重置每个worker的rollout相关状态
            worker.samples_collected = 0
            worker.rollout_completed = False
            reset_count += 1
        
        self.logger.info(f"🔄 已重置 {reset_count} 个workers的rollout状态，准备新的rollout周期")

class TrainingWorker:
    """训练worker，在独立线程中运行"""
    def __init__(self, worker_id, agent_proxy, data_buffer, control_events, logger, config):
        self.worker_id = worker_id
        self.agent_proxy = agent_proxy
        self.data_buffer = data_buffer
        self.control_events = control_events
        self.logger = logger
        self.config = config
        
        # 训练统计
        self.updates_performed = 0
        self.samples_processed = 0
        self.last_update_time = time.time()
        
    def run(self):
        """运行训练worker主循环"""
        self.logger.info(f"Training worker {self.worker_id} 开始运行")
        
        experience_batch = []
        batch_size = 32  # 批处理大小
        
        try:
            while not self.control_events['stop'].is_set():
                # 检查是否需要暂停
                if self.control_events['pause'].is_set():
                    time.sleep(0.1)
                    continue
                
                # 收集经验数据
                experience = self.data_buffer.get(block=True, timeout=1.0)
                if experience is None:
                    continue  # 超时，继续循环
                
                experience_batch.append(experience)
                
                # 达到批大小或数据充足时处理
                if (len(experience_batch) >= batch_size or 
                    (len(experience_batch) > 0 and self.data_buffer.qsize() < batch_size)):
                    
                    # 存储经验到代理
                    stored_count = self.agent_proxy.store_experience(experience_batch)
                    self.samples_processed += stored_count
                    
                    if stored_count > 0:
                        self.logger.debug(f"Training worker {self.worker_id}: 处理了 {stored_count} 个样本")
                    
                    experience_batch = []
                    
                    # 【关键修复】确保每个worker都检查更新条件
                    if self.agent_proxy.should_update():
                        self.perform_update()
                
                # 短暂睡眠避免过度占用CPU
                time.sleep(0.001)
        
        except Exception as e:
            self.logger.error(f"Training worker {self.worker_id}: 运行异常: {e}")
        finally:
            # 处理剩余的经验
            if experience_batch:
                stored_count = self.agent_proxy.store_experience(experience_batch)
                self.samples_processed += stored_count
            
            self.logger.info(f"Training worker {self.worker_id} 结束运行")
    
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
                    
                    self.logger.info(f"Training worker {self.worker_id}: 模型更新完成 "
                                   f"#{self.updates_performed}, 耗时: {update_time:.3f}s")
                    
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
        
        # 创建代理
        self.agent = HMASDAgent(
            config=self.config,
            log_dir=self.log_dir,
            device=self.device,
            debug=getattr(self.args, 'debug', False)
        )
        
        # 创建代理代理
        self.agent_proxy = AgentProxy(self.agent, self.config, self.logger)
        
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
                config=self.config
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
                
                # 计算当前总步数
                current_steps = sum(worker.samples_collected for worker in self.rollout_workers)
                self.total_steps.set(current_steps)
                
                # 检查是否达到步数限制
                if current_steps >= total_steps:
                    self.logger.info(f"达到训练步数限制 {total_steps:,}，停止训练")
                    break
                
                # 每分钟记录一次简要进度
                if current_time - last_log_time >= 60:
                    self.log_progress(current_steps, total_steps)
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
        
        self.logger.info(f"训练进度: {progress_percent:.1f}% "
                        f"({current_steps:,} / {total_steps:,} 步), "
                        f"剩余: {remaining_steps:,} 步")
        self.logger.info(f"时间: 已用={elapsed_time/3600:.1f}h, 预计剩余={remaining_time/3600:.1f}h, "
                        f"速度={steps_per_second:.1f} 步/秒")
        self.logger.info(f"Rollout: 样本={total_samples:,}, Episodes={total_episodes:,}, "
                        f"高层经验={total_high_level_exp:,}")
        self.logger.info(f"Training: 更新={total_updates}, 处理样本={total_processed:,}")
        self.logger.info(f"经验存储: 高层={high_level_stored:,}, 低层={low_level_stored:,}")
        self.logger.info(f"Buffer: 队列={buffer_stats['queue_size']}, "
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
