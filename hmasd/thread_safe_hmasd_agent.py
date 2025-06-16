#!/usr/bin/env python3
"""
【阶段4】线程安全的HMASD Agent实现
继承原有HMASDAgent并添加线程安全功能

核心特性：
1. 继承所有原有功能
2. 线程安全的buffer操作
3. 原子性存储验证
4. 数据完整性保障
"""

import time
import numpy as np
from hmasd.agent import HMASDAgent
from hmasd.thread_safe_agent import ThreadSafeAgentMixin
from logger import get_logger

class ThreadSafeHMASDAgent(HMASDAgent, ThreadSafeAgentMixin):
    """【阶段4】线程安全的HMASD Agent
    
    继承HMASDAgent的所有功能，并添加线程安全支持
    """
    
    def __init__(self, config, log_dir='logs', device=None, debug=False):
        """初始化线程安全的HMASD Agent"""
        # 首先初始化父类
        super().__init__(config, log_dir, device, debug)
        
        # 然后初始化线程安全组件
        self.__init_thread_safety__()
        
        self.logger = get_logger("ThreadSafeHMASDAgent")
        self.logger.info("线程安全HMASD Agent初始化完成")
    
    def store_high_level_transition(self, state, team_skill, observations, agent_skills, 
                                   accumulated_reward, skill_log_probs=None, worker_id=0):
        """【阶段4重写】使用线程安全的高层经验存储"""
        return self.store_high_level_transition_safe(
            state, team_skill, observations, agent_skills, 
            accumulated_reward, skill_log_probs, worker_id
        )
    
    def store_low_level_transition(self, state, next_state, observations, next_observations,
                                 actions, rewards, dones, team_skill, agent_skills, 
                                 action_logprobs, skill_log_probs=None, worker_id=0):
        """【阶段4重写】使用线程安全的低层经验存储"""
        return self.store_low_level_transition_safe(
            state, next_state, observations, next_observations,
            actions, rewards, dones, team_skill, agent_skills, 
            action_logprobs, skill_log_probs, worker_id
        )
    
    def rollout_update(self):
        """【阶段4增强】线程安全的rollout更新"""
        if not self.rollout_based_training:
            raise ValueError("rollout_update只能在rollout_based_training模式下使用")
        
        update_start_time = time.time()
        steps_for_update = self.steps_collected
        target_samples = self.rollout_length * self.num_parallel_envs
        
        self.logger.info(f"🔄 开始线程安全Rollout更新 #{self.rollout_count + 1}")
        self.logger.info(f"📊 数据统计: 收集步数={steps_for_update}, 目标样本={target_samples}, "
                        f"并行环境={self.num_parallel_envs}")
        
        # 【阶段4增强】线程安全的数据完整性验证
        data_integrity_verified = self._thread_safe_data_verification(target_samples)
        
        if not data_integrity_verified:
            self.logger.error("❌ 线程安全数据验证失败，跳过此次更新")
            return None
        
        # 记录更新前的缓冲区状态（线程安全）
        high_level_size_before = len(self.high_level_buffer)
        low_level_size_before = len(self.low_level_buffer)
        state_skill_size_before = len(self.state_skill_dataset)
        
        # 【线程安全统计】
        thread_safety_stats = self.get_thread_safety_stats()
        self.logger.info(f"📈 线程安全统计: {thread_safety_stats['storage_stats']}")
        
        # 执行15轮PPO训练（使用线程安全的buffer）
        self.logger.info(f"🎯 开始{self.ppo_epochs}轮线程安全PPO训练")
        
        coordinator_losses = []
        discoverer_losses = []
        discriminator_losses = []
        
        for epoch in range(self.ppo_epochs):
            epoch_start_time = time.time()
            self.logger.debug(f"   轮次 {epoch + 1}/{self.ppo_epochs}")
            
            # 1. 更新高层策略（使用线程安全buffer）
            coordinator_info = self._thread_safe_update_coordinator()
            coordinator_losses.append(coordinator_info)
            
            # 2. 更新低层策略（使用线程安全buffer）
            discoverer_info = self._thread_safe_update_discoverer()
            discoverer_losses.append(discoverer_info)
            
            # 3. 更新判别器（使用线程安全dataset）
            discriminator_loss = self._thread_safe_update_discriminators()
            discriminator_losses.append(discriminator_loss)
            
            epoch_time = time.time() - epoch_start_time
            
            if epoch % 5 == 0 or epoch == self.ppo_epochs - 1:
                self.logger.debug(f"   轮次 {epoch + 1} 完成，耗时: {epoch_time:.3f}s")
        
        # 【阶段4关键】线程安全的缓冲区清空
        self.logger.info("🧹 线程安全清空PPO缓冲区")
        clear_success = self.clear_buffers_safe()
        
        if not clear_success:
            self.logger.error("❌ 线程安全缓冲区清空失败！")
        else:
            self.logger.info("✅ 线程安全缓冲区清空成功")
        
        # 验证清空结果
        high_level_size_after = len(self.high_level_buffer)
        low_level_size_after = len(self.low_level_buffer)
        state_skill_size_after = len(self.state_skill_dataset)
        
        # 重置rollout状态
        steps_before_reset = self.steps_collected
        self.steps_collected = 0
        self.rollout_count += 1
        self.total_steps_collected += steps_for_update
        update_duration = time.time() - update_start_time
        
        # 计算平均损失
        avg_coordinator_info = self._average_update_info(coordinator_losses)
        avg_discoverer_info = self._average_update_info(discoverer_losses)
        avg_discriminator_loss = np.mean(discriminator_losses) if discriminator_losses else 0.0
        
        # 计算样本使用效率
        samples_per_second = target_samples / update_duration if update_duration > 0 else 0
        
        self.logger.info(f"🎉 线程安全Rollout更新 #{self.rollout_count} 完成")
        self.logger.info(f"⏱️ 耗时: {update_duration:.2f}s, 效率: {samples_per_second:.0f} 样本/秒")
        self.logger.info(f"📈 累计: rollouts={self.rollout_count}, 总步数={self.total_steps_collected:,}")
        
        # 构建详细的更新信息
        update_info = {
            'update_type': 'thread_safe_rollout_batch',
            'rollout_count': self.rollout_count,
            'steps_used': steps_for_update,
            'target_samples': target_samples,
            'total_steps': self.total_steps_collected,
            'ppo_epochs': self.ppo_epochs,
            'num_parallel_envs': self.num_parallel_envs,
            'update_duration': update_duration,
            'samples_per_second': samples_per_second,
            'buffer_changes': {
                'high_level': (high_level_size_before, high_level_size_after),
                'low_level': (low_level_size_before, low_level_size_after),
                'state_skill': (state_skill_size_before, state_skill_size_after)
            },
            'coordinator': avg_coordinator_info,
            'discoverer': avg_discoverer_info,
            'discriminator': {'discriminator_loss': avg_discriminator_loss},
            'thread_safety': {
                'buffer_cleared_safely': clear_success,
                'data_integrity_verified': data_integrity_verified,
                'thread_safety_stats': thread_safety_stats
            }
        }
        
        return update_info
    
    def _thread_safe_data_verification(self, target_samples):
        """【阶段4新增】线程安全的数据完整性验证"""
        max_wait_time = 15.0
        wait_start = time.time()
        
        self.logger.info(f"🔍 [线程安全] 开始数据完整性验证: 期望步数={target_samples}")
        
        expected_high_level = target_samples // self.config.k
        
        verification_attempts = 0
        max_verification_attempts = 10
        
        while verification_attempts < max_verification_attempts and time.time() - wait_start < max_wait_time:
            # 线程安全地获取缓冲区状态
            current_low_level = len(self.low_level_buffer)
            current_high_level = len(self.high_level_buffer)
            
            self.logger.debug(f"🔍 [线程安全] 验证#{verification_attempts + 1}: "
                           f"低层={current_low_level}/{target_samples}, "
                           f"高层={current_high_level}/{expected_high_level}")
            
            # 检查数据完整性
            low_level_sufficient = current_low_level >= target_samples * 0.98
            high_level_sufficient = current_high_level >= expected_high_level * 0.95
            
            if low_level_sufficient and high_level_sufficient:
                wait_time = time.time() - wait_start
                self.logger.info(f"✅ [线程安全] 数据验证通过，等待时间: {wait_time:.2f}s")
                return True
            
            verification_attempts += 1
            if verification_attempts < max_verification_attempts:
                time.sleep(1.0)
        
        # 最终检查
        final_low_level = len(self.low_level_buffer)
        final_high_level = len(self.high_level_buffer)
        
        low_level_missing = target_samples - final_low_level
        high_level_missing = expected_high_level - final_high_level
        
        self.logger.warning(f"⚠️ [线程安全] 数据验证超时:")
        self.logger.warning(f"   低层: {final_low_level}/{target_samples} (缺失: {low_level_missing})")
        self.logger.warning(f"   高层: {final_high_level}/{expected_high_level} (缺失: {high_level_missing})")
        
        # 计算缺失百分比
        total_missing = low_level_missing + high_level_missing
        total_expected = target_samples + expected_high_level
        missing_pct = (total_missing / total_expected) * 100 if total_expected > 0 else 0
        
        if missing_pct <= 3.0:  # 允许3%的缺失
            self.logger.warning(f"⚠️ [线程安全] 数据轻微缺失({missing_pct:.1%})，允许继续训练")
            return True
        else:
            self.logger.error(f"❌ [线程安全] 数据严重缺失({missing_pct:.1%})，拒绝此次更新")
            return False
    
    def _thread_safe_update_coordinator(self):
        """【阶段4新增】线程安全的协调器更新"""
        # 使用线程安全的buffer进行更新
        required_batch_size = self.config.high_level_batch_size
        current_buffer_size = len(self.high_level_buffer)
        
        if current_buffer_size == 0:
            self.logger.debug(f"[线程安全协调器] 高层缓冲区为空，跳过更新")
            return self._get_default_coordinator_info()
        
        if current_buffer_size < required_batch_size:
            self.logger.warning(f"[线程安全协调器] 高层缓冲区不足，需要{required_batch_size}个样本，"
                              f"但只有{current_buffer_size}个。跳过此轮更新。")
            return self._get_default_coordinator_info()
        
        # 调用原有的协调器更新逻辑（现在使用线程安全的buffer）
        return self._update_coordinator_with_all_buffer()
    
    def _thread_safe_update_discoverer(self):
        """【阶段4新增】线程安全的发现器更新"""
        if len(self.low_level_buffer) == 0:
            return self._get_default_discoverer_info()
        
        # 调用原有的发现器更新逻辑（现在使用线程安全的buffer）
        return self._update_discoverer_with_all_buffer()
    
    def _thread_safe_update_discriminators(self):
        """【阶段4新增】线程安全的判别器更新"""
        if len(self.state_skill_dataset) < self.config.batch_size:
            self.logger.warning(f"[线程安全判别器] 数据集不足，需要{self.config.batch_size}个样本，"
                               f"但只有{len(self.state_skill_dataset)}个。跳过更新。")
            return 0.0
        
        # 调用原有的判别器更新逻辑（现在使用线程安全的dataset）
        return self.update_discriminators()
