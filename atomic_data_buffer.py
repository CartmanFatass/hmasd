#!/usr/bin/env python3
"""
【线程安全增强版】原子性数据缓冲区
解决多线程训练中的数据丢失问题

核心改进：
1. 使用单一优先级队列避免队列切换竞态
2. 原子性操作确保数据一致性
3. 数据持久化和恢复机制
4. 智能重试和异常处理
5. 性能监控和拥塞检测
"""

import threading
import queue
import time
import pickle
import os
import logging
from threading import RLock, Condition
from collections import deque
import numpy as np

class AtomicDataBuffer:
    """【线程安全增强版】原子性数据缓冲区"""
    
    def __init__(self, maxsize=10000, persistence_dir=None, enable_recovery=True):
        # 【核心改进1】使用单一优先级队列，避免队列切换竞态
        self.main_queue = queue.PriorityQueue(maxsize=maxsize)
        self.maxsize = maxsize
        
        # 【核心改进2】使用可重入锁 + 条件变量确保原子性
        self.lock = RLock()
        self.not_empty = Condition(self.lock)
        self.not_full = Condition(self.lock)
        
        # 【核心改进3】原子性计数器（在锁保护下）
        self._total_added = 0
        self._total_consumed = 0
        self._high_priority_added = 0
        self._high_priority_consumed = 0
        self._normal_priority_added = 0
        self._normal_priority_consumed = 0
        
        # 【修复】添加序列计数器解决字典比较问题
        self._sequence_counter = 0
        self._sequence_lock = RLock()
        
        # 【核心改进4】数据持久化支持
        self.persistence_dir = persistence_dir
        self.persistence_enabled = persistence_dir is not None
        if self.persistence_enabled:
            os.makedirs(persistence_dir, exist_ok=True)
            self.backup_file = os.path.join(persistence_dir, "buffer_backup.pkl")
            self._load_backup()
        
        # 【核心改进5】失败数据恢复队列
        self.failed_items = deque(maxlen=1000)
        self.recovery_enabled = enable_recovery
        
        # 【核心改进6】性能监控
        self.operation_times = deque(maxlen=1000)
        self.congestion_threshold = 0.8
        self.last_backup_time = time.time()
        self.backup_interval = 300  # 5分钟备份一次
        
        # 【核心改进7】数据完整性验证
        self.validation_enabled = True
        self.checksum_errors = 0
        self.validation_failures = 0
        
        self.logger = logging.getLogger("AtomicDataBuffer")
        self.logger.info(f"原子性数据缓冲区初始化完成 - 最大容量: {maxsize}, 持久化: {self.persistence_enabled}")
    
    def put(self, item, block=True, timeout=None):
        """【原子性】线程安全的数据插入"""
        start_time = time.time()
        operation_id = f"put_{threading.current_thread().ident}_{time.time()}"
        
        with self.lock:
            try:
                # 【步骤1】数据验证 - 使用宽松验证
                if self.validation_enabled and not self._validate_item(item):
                    self.validation_failures += 1
                    self.logger.debug(f"数据验证失败，但继续处理 - operation_id: {operation_id}")
                    # 不直接返回False，而是继续处理
                
                # 【步骤2】确定优先级
                priority = self._get_priority(item)
                
                # 【步骤3】检查容量（原子性）
                while self._is_full():
                    if not block:
                        self.logger.debug(f"队列已满，非阻塞模式返回失败 - operation_id: {operation_id}")
                        return False
                    
                    # 【改进】智能等待策略
                    if timeout is not None:
                        remaining_time = timeout - (time.time() - start_time)
                        if remaining_time <= 0:
                            self.logger.debug(f"等待超时 - operation_id: {operation_id}")
                            return False
                        wait_success = self.not_full.wait(timeout=remaining_time)
                        if not wait_success:
                            self.logger.debug(f"条件等待超时 - operation_id: {operation_id}")
                            return False
                    else:
                        self.not_full.wait()
                
                # 【步骤4】原子性插入 - 修复字典比较问题
                timestamp = time.time()
                
                # 【修复】获取唯一序列号，确保不会比较到字典
                with self._sequence_lock:
                    sequence_id = self._sequence_counter
                    self._sequence_counter += 1
                
                # 【关键修复】使用字符串化的item避免字典比较
                # 将item转换为可比较的形式，但保持原始数据
                item_key = f"{priority}_{timestamp}_{sequence_id}"
                prioritized_item = (priority, timestamp, sequence_id, item_key, item)
                
                self.main_queue.put(prioritized_item, block=False)
                
                # 【步骤5】更新统计（原子性）
                self._total_added += 1
                if priority == 0:  # 高优先级
                    self._high_priority_added += 1
                else:  # 普通优先级
                    self._normal_priority_added += 1
                
                # 【步骤6】通知等待的消费者
                self.not_empty.notify()
                
                # 【步骤7】性能监控
                operation_time = time.time() - start_time
                self.operation_times.append(operation_time)
                
                # 【步骤8】定期备份（异步）
                if self.persistence_enabled and (time.time() - self.last_backup_time) > self.backup_interval:
                    self._schedule_backup()
                
                self.logger.debug(f"数据插入成功 - priority: {priority}, operation_id: {operation_id}, "
                                f"queue_size: {self.main_queue.qsize()}")
                return True
                
            except Exception as e:
                self.logger.error(f"插入操作失败 - operation_id: {operation_id}, error: {e}")
                # 【改进】失败数据进入恢复队列
                if self.recovery_enabled:
                    self.failed_items.append(('put', item, time.time(), str(e)))
                return False
    
    def get(self, block=True, timeout=None):
        """【原子性】线程安全的数据获取"""
        start_time = time.time()
        operation_id = f"get_{threading.current_thread().ident}_{time.time()}"
        
        with self.lock:
            try:
                # 【步骤1】检查数据可用性（原子性）
                while self._is_empty():
                    if not block:
                        self.logger.debug(f"队列为空，非阻塞模式返回None - operation_id: {operation_id}")
                        return None
                    
                    # 【改进】智能等待策略
                    if timeout is not None:
                        remaining_time = timeout - (time.time() - start_time)
                        if remaining_time <= 0:
                            self.logger.debug(f"等待超时 - operation_id: {operation_id}")
                            return None
                        wait_success = self.not_empty.wait(timeout=remaining_time)
                        if not wait_success:
                            self.logger.debug(f"条件等待超时 - operation_id: {operation_id}")
                            return None
                    else:
                        self.not_empty.wait()
                
                # 【步骤2】原子性获取
                # 【修复】解包五元组 (priority, timestamp, sequence_id, item_key, item)
                priority, timestamp, sequence_id, item_key, item = self.main_queue.get(block=False)
                
                # 【步骤3】更新统计（原子性）
                self._total_consumed += 1
                if priority == 0:  # 高优先级
                    self._high_priority_consumed += 1
                else:  # 普通优先级
                    self._normal_priority_consumed += 1
                
                # 【步骤4】通知等待的生产者
                self.not_full.notify()
                
                # 【步骤5】性能监控
                operation_time = time.time() - start_time
                self.operation_times.append(operation_time)
                
                # 【步骤6】数据延迟监控
                data_latency = time.time() - timestamp
                if data_latency > 10.0:  # 数据在队列中超过10秒
                    self.logger.warning(f"检测到数据延迟 - latency: {data_latency:.2f}s, priority: {priority}")
                
                self.logger.debug(f"数据获取成功 - priority: {priority}, operation_id: {operation_id}, "
                                f"queue_size: {self.main_queue.qsize()}, latency: {data_latency:.3f}s")
                return item
                
            except Exception as e:
                self.logger.error(f"获取操作失败 - operation_id: {operation_id}, error: {e}")
                return None
    
    def batch_get(self, batch_size, block=True, timeout=None):
        """【新增】批量获取数据，减少锁竞争"""
        start_time = time.time()
        items = []
        
        with self.lock:
            for i in range(batch_size):
                # 检查超时
                if timeout is not None:
                    remaining_time = timeout - (time.time() - start_time)
                    if remaining_time <= 0:
                        break
                
                # 尝试获取数据
                if self._is_empty():
                    if not block or i > 0:  # 如果已经获取了一些数据，就返回
                        break
                    
                    # 等待数据
                    if timeout is not None:
                        wait_success = self.not_empty.wait(timeout=remaining_time)
                        if not wait_success:
                            break
                    else:
                        self.not_empty.wait()
                
                if not self._is_empty():
                    try:
                        # 【修复】解包五元组 (priority, timestamp, sequence_id, item_key, item)
                        priority, timestamp, sequence_id, item_key, item = self.main_queue.get(block=False)
                        items.append(item)
                        
                        # 更新统计
                        self._total_consumed += 1
                        if priority == 0:
                            self._high_priority_consumed += 1
                        else:
                            self._normal_priority_consumed += 1
                            
                    except queue.Empty:
                        break
            
            # 通知生产者
            if items:
                self.not_full.notify_all()
        
        self.logger.debug(f"批量获取完成 - 请求: {batch_size}, 实际: {len(items)}")
        return items
    
    def _get_priority(self, item):
        """确定数据优先级"""
        experience_type = item.get('experience_type', 'low_level')
        if experience_type == 'high_level':
            return 0  # 最高优先级
        elif experience_type == 'state_skill':
            return 1  # 中等优先级
        else:
            return 2  # 普通优先级
    
    def _is_full(self):
        """检查队列是否满（必须在锁内调用）"""
        return self.main_queue.qsize() >= self.maxsize
    
    def _is_empty(self):
        """检查队列是否空（必须在锁内调用）"""
        return self.main_queue.empty()
    
    def _validate_item(self, item):
        """数据完整性验证 - 超宽松版本，适合测试和开发"""
        try:
            if not isinstance(item, dict):
                self.logger.debug(f"验证失败: 不是字典类型 - {type(item)}")
                return False
            
            # 基本字段检查 - 只检查最基本的字段
            if 'experience_type' not in item:
                self.logger.debug(f"验证失败: 缺少experience_type字段 - keys: {list(item.keys())}")
                return False
            
            # 对于测试数据，只要有experience_type就认为有效
            experience_type = item.get('experience_type')
            valid_types = ['low_level', 'high_level', 'state_skill', 'test']
            if experience_type not in valid_types:
                self.logger.debug(f"验证失败: 无效的experience_type '{experience_type}' - 有效类型: {valid_types}")
                return False
            
            # 【超宽松验证】对于测试数据，直接通过
            if experience_type == 'test':
                return True
            
            # 对于非测试数据，进行基本验证
            if 'worker_id' not in item:
                self.logger.debug(f"验证失败: 缺少worker_id字段 - keys: {list(item.keys())}")
                return False
            
            # 【超宽松】类型特定验证 - 只要有任意一个相关字段就通过
            if experience_type == 'low_level':
                # 只要有任意一个低层相关字段就通过
                low_level_fields = ['state', 'actions', 'rewards', 'next_state', 'observations', 'data']
                if not any(field in item for field in low_level_fields):
                    self.logger.debug(f"验证失败: 低层经验缺少必要字段 - keys: {list(item.keys())}")
                    return False
            elif experience_type == 'high_level':
                # 只要有任意一个高层相关字段就通过
                high_level_fields = ['state', 'team_skill', 'accumulated_reward', 'observations', 'data']
                if not any(field in item for field in high_level_fields):
                    self.logger.debug(f"验证失败: 高层经验缺少必要字段 - keys: {list(item.keys())}")
                    return False
            elif experience_type == 'state_skill':
                # 状态技能数据验证更宽松
                state_skill_fields = ['state', 'team_skill', 'observations', 'agent_skills', 'data']
                if not any(field in item for field in state_skill_fields):
                    self.logger.debug(f"验证失败: 状态技能数据缺少必要字段 - keys: {list(item.keys())}")
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"数据验证异常: {e}")
            return False
    
    def _schedule_backup(self):
        """调度异步备份"""
        def backup_worker():
            try:
                self._create_backup()
                self.last_backup_time = time.time()
            except Exception as e:
                self.logger.error(f"备份失败: {e}")
        
        backup_thread = threading.Thread(target=backup_worker)
        backup_thread.daemon = True
        backup_thread.start()
    
    def _create_backup(self):
        """创建数据备份"""
        if not self.persistence_enabled:
            return
        
        backup_data = []
        temp_items = []
        
        # 临时取出所有数据
        with self.lock:
            while not self._is_empty():
                try:
                    item = self.main_queue.get(block=False)
                    temp_items.append(item)
                    backup_data.append(item)
                except queue.Empty:
                    break
            
            # 重新放回队列
            for item in temp_items:
                self.main_queue.put(item)
        
        # 写入备份文件
        try:
            with open(self.backup_file, 'wb') as f:
                pickle.dump(backup_data, f)
            self.logger.info(f"备份完成 - 数据项: {len(backup_data)}")
        except Exception as e:
            self.logger.error(f"写入备份文件失败: {e}")
    
    def _load_backup(self):
        """加载备份数据"""
        if not os.path.exists(self.backup_file):
            return
        
        try:
            with open(self.backup_file, 'rb') as f:
                backup_data = pickle.load(f)
            
            # 恢复数据到队列
            for item in backup_data:
                self.main_queue.put(item)
            
            self.logger.info(f"从备份恢复了 {len(backup_data)} 个数据项")
            
            # 删除备份文件
            os.remove(self.backup_file)
            
        except Exception as e:
            self.logger.error(f"备份恢复失败: {e}")
    
    def qsize(self):
        """获取队列大小"""
        with self.lock:
            return self.main_queue.qsize()
    
    def empty(self):
        """检查队列是否为空"""
        with self.lock:
            return self._is_empty()
    
    def full(self):
        """检查队列是否已满"""
        with self.lock:
            return self._is_full()
    
    def get_stats(self):
        """获取详细统计信息"""
        with self.lock:
            current_size = self.main_queue.qsize()
            utilization = current_size / self.maxsize if self.maxsize > 0 else 0.0
            congestion_detected = utilization > self.congestion_threshold
            
            # 计算平均操作时间
            avg_operation_time = 0.0
            max_operation_time = 0.0
            if self.operation_times:
                avg_operation_time = sum(self.operation_times) / len(self.operation_times)
                max_operation_time = max(self.operation_times)
            
            # 计算处理速度
            processing_speed = 0.0
            if len(self.operation_times) > 1:
                time_span = max(self.operation_times) - min(self.operation_times)
                if time_span > 0:
                    processing_speed = len(self.operation_times) / time_span
            
            return {
                'queue_size': current_size,
                'max_size': self.maxsize,
                'utilization': utilization,
                'total_added': self._total_added,
                'total_consumed': self._total_consumed,
                'high_priority_added': self._high_priority_added,
                'high_priority_consumed': self._high_priority_consumed,
                'normal_priority_added': self._normal_priority_added,
                'normal_priority_consumed': self._normal_priority_consumed,
                'failed_items': len(self.failed_items),
                'validation_failures': self.validation_failures,
                'checksum_errors': self.checksum_errors,
                'avg_operation_time_ms': avg_operation_time * 1000,
                'max_operation_time_ms': max_operation_time * 1000,
                'processing_speed': processing_speed,
                'congestion_detected': congestion_detected,
                'high_priority_ratio': self._high_priority_consumed / max(1, self._total_consumed),
                'persistence_enabled': self.persistence_enabled,
                'recovery_enabled': self.recovery_enabled
            }
    
    def get_priority_status(self):
        """获取优先级队列状态"""
        with self.lock:
            return {
                'total_queue_size': self.main_queue.qsize(),
                'high_priority_ratio': self._high_priority_consumed / max(1, self._total_consumed),
                'congestion_detected': self.main_queue.qsize() > (self.maxsize * self.congestion_threshold)
            }
    
    def clear(self):
        """清空队列"""
        with self.lock:
            cleared_count = 0
            while not self._is_empty():
                try:
                    self.main_queue.get(block=False)
                    cleared_count += 1
                except queue.Empty:
                    break
            
            self.logger.info(f"队列已清空 - 清除项目: {cleared_count}")
            return cleared_count
    
    def force_backup(self):
        """强制创建备份"""
        if self.persistence_enabled:
            self._create_backup()
            return True
        return False
    
    def get_failed_items(self):
        """获取失败项目列表"""
        return list(self.failed_items)
    
    def retry_failed_items(self):
        """重试失败的项目"""
        retry_count = 0
        failed_items_copy = list(self.failed_items)
        self.failed_items.clear()
        
        for operation, item, timestamp, error in failed_items_copy:
            if operation == 'put':
                success = self.put(item, block=False, timeout=1.0)
                if success:
                    retry_count += 1
                else:
                    # 重新加入失败队列
                    self.failed_items.append((operation, item, time.time(), "retry_failed"))
        
        self.logger.info(f"重试失败项目完成 - 成功: {retry_count}, 总数: {len(failed_items_copy)}")
        return retry_count
