#!/usr/bin/env python3
"""
【线程安全增强版】训练工作器
解决数据获取和存储中的线程安全问题

核心改进：
1. 本地缓存减少锁竞争
2. 批量数据获取提高效率
3. 自适应重试策略
4. 数据完整性验证
5. 失败数据持久化和恢复
"""

import threading
import time
import os
import pickle
import logging
from collections import deque, defaultdict
from threading import Lock
import numpy as np

class AdaptiveRetryStrategy:
    """自适应重试策略"""
    
    def __init__(self):
        self.base_wait = 0.1
        self.max_wait = 2.0
        self.backoff_factor = 1.5
        self.success_history = deque(maxlen=100)
        self.failure_history = deque(maxlen=100)
    
    def get_wait_time(self, retry_count, operation_type='default'):
        """获取自适应等待时间"""
        # 基础指数退避
        wait_time = self.base_wait * (self.backoff_factor ** retry_count)
        
        # 根据历史成功率调整
        if len(self.success_history) > 10:
            success_rate = sum(self.success_history) / len(self.success_history)
            if success_rate < 0.5:  # 成功率低于50%
                wait_time *= 2.0  # 增加等待时间
            elif success_rate > 0.8:  # 成功率高于80%
                wait_time *= 0.5  # 减少等待时间
        
        # 根据操作类型调整
        if operation_type == 'storage':
            wait_time *= 1.5  # 存储操作需要更多时间
        elif operation_type == 'retrieval':
            wait_time *= 0.8  # 获取操作可以更快重试
        
        return min(wait_time, self.max_wait)
    
    def record_success(self):
        """记录成功操作"""
        self.success_history.append(1)
    
    def record_failure(self):
        """记录失败操作"""
        self.success_history.append(0)
        self.failure_history.append(time.time())
    
    def get_success_rate(self):
        """获取成功率"""
        if not self.success_history:
            return 0.0
        return sum(self.success_history) / len(self.success_history)

class DataIntegrityChecker:
    """数据完整性检查器"""
    
    def __init__(self):
        self.validation_errors = defaultdict(int)
        self.logger = logging.getLogger("DataIntegrityChecker")
    
    def validate_batch(self, experience_batch):
        """验证批次数据完整性"""
        if not experience_batch:
            self.validation_errors['empty_batch'] += 1
            return False
        
        valid_count = 0
        for i, experience in enumerate(experience_batch):
            if self._validate_single_experience(experience):
                valid_count += 1
            else:
                self.logger.warning(f"批次中第{i}个经验验证失败")
        
        # 要求至少90%的数据有效
        validity_ratio = valid_count / len(experience_batch)
        if validity_ratio < 0.9:
            self.validation_errors['low_validity_ratio'] += 1
            self.logger.error(f"批次有效性过低: {validity_ratio:.2%}")
            return False
        
        return True
    
    def _validate_single_experience(self, experience):
        """验证单个经验数据"""
        try:
            # 基本结构检查
            if not isinstance(experience, dict):
                self.validation_errors['not_dict'] += 1
                return False
            
            # 必需字段检查
            required_fields = ['experience_type', 'worker_id']
            for field in required_fields:
                if field not in experience:
                    self.validation_errors[f'missing_{field}'] += 1
                    return False
            
            # 类型特定验证
            experience_type = experience.get('experience_type')
            if experience_type == 'low_level':
                return self._validate_low_level_experience(experience)
            elif experience_type == 'high_level':
                return self._validate_high_level_experience(experience)
            elif experience_type == 'state_skill':
                return self._validate_state_skill_experience(experience)
            else:
                self.validation_errors['unknown_type'] += 1
                return False
                
        except Exception as e:
            self.validation_errors['validation_exception'] += 1
            self.logger.error(f"验证异常: {e}")
            return False
    
    def _validate_low_level_experience(self, experience):
        """验证低层经验"""
        required_fields = ['state', 'actions', 'rewards', 'next_state']
        for field in required_fields:
            if field not in experience:
                self.validation_errors[f'low_level_missing_{field}'] += 1
                return False
        
        # 数值有效性检查
        rewards = experience.get('rewards')
        if rewards is not None:
            reward_val = rewards if isinstance(rewards, (int, float)) else np.sum(rewards)
            if np.isnan(reward_val) or np.isinf(reward_val):
                self.validation_errors['invalid_reward'] += 1
                return False
        
        return True
    
    def _validate_high_level_experience(self, experience):
        """验证高层经验"""
        required_fields = ['state', 'team_skill', 'accumulated_reward']
        for field in required_fields:
            if field not in experience:
                self.validation_errors[f'high_level_missing_{field}'] += 1
                return False
        
        # 累积奖励有效性检查
        accumulated_reward = experience.get('accumulated_reward')
        if accumulated_reward is not None:
            if np.isnan(accumulated_reward) or np.isinf(accumulated_reward):
                self.validation_errors['invalid_accumulated_reward'] += 1
                return False
        
        return True
    
    def _validate_state_skill_experience(self, experience):
        """验证状态技能经验"""
        required_fields = ['state', 'team_skill']
        for field in required_fields:
            if field not in experience:
                self.validation_errors[f'state_skill_missing_{field}'] += 1
                return False
        
        return True
    
    def get_validation_stats(self):
        """获取验证统计"""
        return dict(self.validation_errors)

class EnhancedTrainingWorker:
    """【线程安全增强版】训练工作器"""
    
    def __init__(self, worker_id, agent_proxy, data_buffer, control_events, logger, config, trainer=None):
        self.worker_id = worker_id
        self.agent_proxy = agent_proxy
        self.data_buffer = data_buffer
        self.control_events = control_events
        self.logger = logger
        self.config = config
        self.trainer = trainer
        
        # 【改进1】本地缓存减少竞争
        self.local_cache = deque(maxlen=200)
        self.cache_lock = Lock()
        
        # 【改进2】自适应重试策略
        self.retry_strategy = AdaptiveRetryStrategy()
        
        # 【改进3】数据完整性验证
        self.integrity_checker = DataIntegrityChecker()
        
        # 【改进4】性能监控
        self.performance_stats = {
            'total_processed': 0,
            'successful_batches': 0,
            'failed_batches': 0,
            'cache_hits': 0,
            'cache_misses': 0
        }
        self.stats_lock = Lock()
        
        # 【改进5】失败数据管理
        self.failed_data_dir = f"failed_data/worker_{worker_id}"
        os.makedirs(self.failed_data_dir, exist_ok=True)
        
        # 训练统计
        self.updates_performed = 0
        self.samples_processed = 0
        self.last_update_time = time.time()
        
        self.logger.info(f"增强版训练工作器 {worker_id} 初始化完成")
    
    def run(self):
        """【增强版】运行训练worker主循环"""
        self.logger.info(f"增强版训练工作器 {self.worker_id} 开始运行")
        
        experience_batch = []
        batch_size = 32
        
        try:
            while not self.control_events['stop'].is_set():
                if self.control_events['pause'].is_set():
                    time.sleep(0.1)
                    continue
                
                # 【增强版数据获取】
                experience = self._enhanced_data_retrieval()
                if experience is None:
                    continue
                
                experience_batch.append(experience)
                
                # 批处理
                if len(experience_batch) >= batch_size:
                    success = self._reliable_storage_with_recovery(experience_batch)
                    if success:
                        self.samples_processed += len(experience_batch)
                        experience_batch = []
                        
                        # 检查更新条件
                        if self.agent_proxy.should_update():
                            self.perform_update()
                
                time.sleep(0.001)
        
        except Exception as e:
            self.logger.error(f"训练工作器 {self.worker_id}: 运行异常: {e}")
        finally:
            if experience_batch:
                self._process_remaining_experiences(experience_batch)
            self.logger.info(f"增强版训练工作器 {self.worker_id} 结束运行")
    
    def _enhanced_data_retrieval(self):
        """【增强版】智能数据获取"""
        # 优先从本地缓存获取
        with self.cache_lock:
            if self.local_cache:
                with self.stats_lock:
                    self.performance_stats['cache_hits'] += 1
                return self.local_cache.popleft()
        
        # 从buffer批量获取
        if hasattr(self.data_buffer, 'batch_get'):
            items = self.data_buffer.batch_get(10, block=False, timeout=0.1)
            if items:
                with self.cache_lock:
                    if len(items) > 1:
                        self.local_cache.extend(items[1:])
                    with self.stats_lock:
                        self.performance_stats['cache_misses'] += 1
                    return items[0]
        else:
            # 回退到单个获取
            item = self.data_buffer.get(block=True, timeout=1.0)
            if item:
                with self.stats_lock:
                    self.performance_stats['cache_misses'] += 1
                return item
        
        return None
    
    def _reliable_storage_with_recovery(self, experience_batch):
        """【可靠存储】带恢复机制的存储"""
        max_retries = 10
        retry_count = 0
        
        # 数据完整性预检查
        if not self.integrity_checker.validate_batch(experience_batch):
            self.logger.error("批次数据完整性验证失败")
            return False
        
        while retry_count < max_retries:
            try:
                stored_count = self.agent_proxy.store_experience(experience_batch)
                
                if stored_count == len(experience_batch):
                    self.retry_strategy.record_success()
                    return True
                
                # 部分失败处理
                if stored_count > 0:
                    experience_batch = experience_batch[stored_count:]
                
                retry_count += 1
                wait_time = self.retry_strategy.get_wait_time(retry_count, 'storage')
                time.sleep(wait_time)
                
            except Exception as e:
                self.logger.error(f"存储异常: {e}")
                retry_count += 1
                time.sleep(0.1 * retry_count)
        
        # 最终失败处理
        self.retry_strategy.record_failure()
        if experience_batch:
            self._persist_failed_batch(experience_batch)
        
        return False
    
    def _persist_failed_batch(self, experience_batch):
        """持久化失败的批次数据"""
        try:
            timestamp = int(time.time())
            filename = f"failed_batch_{self.worker_id}_{timestamp}.pkl"
            filepath = os.path.join(self.failed_data_dir, filename)
            
            with open(filepath, 'wb') as f:
                pickle.dump(experience_batch, f)
            
            self.logger.warning(f"失败批次已持久化: {filepath}")
            
        except Exception as e:
            self.logger.error(f"持久化失败批次时出错: {e}")
    
    def _process_remaining_experiences(self, experience_batch):
        """处理剩余经验"""
        self.logger.info(f"训练工作器 {self.worker_id}: 处理剩余 {len(experience_batch)} 个经验")
        
        max_retries = 15
        retry_count = 0
        
        while retry_count < max_retries and experience_batch:
            try:
                success = self._reliable_storage_with_recovery(experience_batch)
                if success:
                    self.samples_processed += len(experience_batch)
                    self.logger.info(f"训练工作器 {self.worker_id}: 剩余经验存储成功")
                    break
                else:
                    retry_count += 1
                    wait_time = min(0.2 * retry_count, 2.0)
                    time.sleep(wait_time)
                    
            except Exception as e:
                self.logger.error(f"训练工作器 {self.worker_id}: 剩余经验存储异常: {e}")
                retry_count += 1
                time.sleep(0.3)
        
        if experience_batch:
            self.logger.error(f"训练工作器 {self.worker_id}: 最终仍有 {len(experience_batch)} 个经验未能存储!")
    
    def perform_update(self):
        """执行模型更新"""
        try:
            update_start = time.time()
            
            if self.worker_id == 0:  # 只有第一个training worker执行更新
                self.logger.info(f"训练工作器 {self.worker_id}: 开始模型更新")
                
                update_info = self.agent_proxy.update()
                
                if update_info:
                    self.updates_performed += 1
                    update_time = time.time() - update_start
                    self.last_update_time = time.time()
                    
                    # 更新累计总步数
                    if self.trainer:
                        steps_this_rollout = self.config.batch_size
                        self.trainer.total_steps.increment(steps_this_rollout)
                    
                    self.logger.info(f"训练工作器 {self.worker_id}: 模型更新完成 "
                                   f"#{self.updates_performed}, 耗时: {update_time:.3f}s")
                else:
                    self.logger.warning(f"训练工作器 {self.worker_id}: 模型更新返回None")
        
        except Exception as e:
            self.logger.error(f"训练工作器 {self.worker_id}: 模型更新异常: {e}")
    
    def get_performance_stats(self):
        """获取性能统计"""
        with self.stats_lock:
            stats = self.performance_stats.copy()
        
        stats.update({
            'worker_id': self.worker_id,
            'updates_performed': self.updates_performed,
            'samples_processed': self.samples_processed,
            'success_rate': self.retry_strategy.get_success_rate(),
            'validation_stats': self.integrity_checker.get_validation_stats()
        })
        
        return stats
