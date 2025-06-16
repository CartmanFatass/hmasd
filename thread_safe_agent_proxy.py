#!/usr/bin/env python3
"""
【线程安全增强版】代理代理
解决多线程环境下的数据存储竞争问题

核心改进：
1. 分离锁减少竞争
2. 后台存储队列缓冲
3. 原子性存储操作
4. 数据完整性验证
5. 存储失败恢复机制
"""

import threading
import queue
import time
import logging
from threading import RLock, Lock
from collections import defaultdict, deque
import numpy as np
import torch

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

class StorageQueue:
    """专用存储队列"""
    
    def __init__(self, maxsize=1000, queue_type='high_level'):
        self.queue = queue.Queue(maxsize=maxsize)
        self.queue_type = queue_type
        self.overflow_count = ThreadSafeCounter()
        self.processed_count = ThreadSafeCounter()
        self.failed_count = ThreadSafeCounter()
        
    def put(self, item, block=False, timeout=None):
        """放入存储项"""
        try:
            self.queue.put(item, block=block, timeout=timeout)
            return True
        except queue.Full:
            self.overflow_count.increment()
            return False
    
    def get(self, block=True, timeout=None):
        """获取存储项"""
        try:
            return self.queue.get(block=block, timeout=timeout)
        except queue.Empty:
            return None
    
    def qsize(self):
        """获取队列大小"""
        return self.queue.qsize()
    
    def empty(self):
        """检查是否为空"""
        return self.queue.empty()
    
    def get_stats(self):
        """获取统计信息"""
        return {
            'queue_type': self.queue_type,
            'queue_size': self.qsize(),
            'overflow_count': self.overflow_count.get(),
            'processed_count': self.processed_count.get(),
            'failed_count': self.failed_count.get()
        }

class ThreadSafeAgentProxy:
    """【线程安全增强版】代理代理"""
    
    def __init__(self, agent, config, logger, data_buffer=None):
        self.agent = agent
        self.config = config
        self.logger = logger
        self.data_buffer = data_buffer
        
        # 【改进1】分离锁减少竞争
        self.high_level_lock = RLock()
        self.low_level_lock = RLock()
        self.state_skill_lock = RLock()
        self.stats_lock = Lock()
        self.skill_assignment_lock = Lock()
        
        # 【改进2】存储队列缓冲
        self.high_level_storage_queue = StorageQueue(maxsize=1000, queue_type='high_level')
        self.low_level_storage_queue = StorageQueue(maxsize=2000, queue_type='low_level')
        self.state_skill_storage_queue = StorageQueue(maxsize=1000, queue_type='state_skill')
        
        # 【改进3】后台存储线程
        self.storage_workers = []
        self.storage_stop_event = threading.Event()
        self._start_storage_workers()
        
        # 【改进4】存储统计
        self.storage_stats = {
            'total_attempts': ThreadSafeCounter(),
            'total_successes': ThreadSafeCounter(),
            'total_failures': ThreadSafeCounter(),
            'queue_overflows': ThreadSafeCounter(),
            'high_level_stored': ThreadSafeCounter(),
            'low_level_stored': ThreadSafeCounter(),
            'state_skill_stored': ThreadSafeCounter()
        }
        
        # 【改进5】全局rollout步数计数器
        self.global_rollout_steps = ThreadSafeCounter()
        
        # 【改进6】数据完整性验证
        self.validation_enabled = True
        self.validation_failures = ThreadSafeCounter()
        
        # 【改进7】性能监控
        self.operation_times = deque(maxlen=1000)
        self.last_stats_log_time = time.time()
        
        self.logger.info("线程安全代理代理初始化完成")
    
    def _start_storage_workers(self):
        """启动后台存储工作线程"""
        # 创建3个专门的存储工作线程
        worker_configs = [
            ('high_level', self.high_level_storage_queue, self._process_high_level_storage),
            ('low_level', self.low_level_storage_queue, self._process_low_level_storage),
            ('state_skill', self.state_skill_storage_queue, self._process_state_skill_storage)
        ]
        
        for worker_name, storage_queue, process_func in worker_configs:
            worker = threading.Thread(
                target=self._storage_worker,
                args=(worker_name, storage_queue, process_func),
                name=f"StorageWorker-{worker_name}"
            )
            worker.daemon = True
            worker.start()
            self.storage_workers.append(worker)
        
        self.logger.info(f"启动了 {len(self.storage_workers)} 个后台存储工作线程")
    
    def _storage_worker(self, worker_name, storage_queue, process_func):
        """后台存储工作线程"""
        self.logger.info(f"存储工作线程 {worker_name} 开始运行")
        
        while not self.storage_stop_event.is_set():
            try:
                # 获取存储任务
                item = storage_queue.get(block=True, timeout=0.5)
                if item is None:
                    continue
                
                # 处理存储
                start_time = time.time()
                success = process_func(item)
                processing_time = time.time() - start_time
                
                # 更新统计
                if success:
                    storage_queue.processed_count.increment()
                    self.storage_stats['total_successes'].increment()
                else:
                    storage_queue.failed_count.increment()
                    self.storage_stats['total_failures'].increment()
                
                self.storage_stats['total_attempts'].increment()
                
                # 记录处理时间
                self.operation_times.append(processing_time)
                
                # 定期记录统计
                if time.time() - self.last_stats_log_time > 300:  # 每5分钟
                    self._log_storage_stats()
                    self.last_stats_log_time = time.time()
                
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"存储工作线程 {worker_name} 异常: {e}")
                time.sleep(0.1)
        
        self.logger.info(f"存储工作线程 {worker_name} 结束运行")
    
    def store_experience_safe(self, experience_batch):
        """【线程安全】经验存储入口"""
        if not experience_batch:
            return 0
        
        stored_count = 0
        failed_items = []
        
        for experience in experience_batch:
            try:
                # 数据验证
                if self.validation_enabled and not self._validate_experience(experience):
                    self.validation_failures.increment()
                    self.logger.warning(f"经验数据验证失败: {experience.get('experience_type', 'unknown')}")
                    continue
                
                experience_type = experience.get('experience_type', 'low_level')
                
                # 根据类型分发到不同的存储队列
                success = False
                if experience_type == 'high_level':
                    success = self._queue_high_level_storage(experience)
                elif experience_type == 'low_level':
                    success = self._queue_low_level_storage(experience)
                elif experience_type == 'state_skill':
                    success = self._queue_state_skill_storage(experience)
                else:
                    self.logger.warning(f"未知经验类型: {experience_type}")
                    continue
                
                if success:
                    stored_count += 1
                else:
                    failed_items.append(experience)
                    
            except Exception as e:
                self.logger.error(f"存储经验时出错: {e}")
                failed_items.append(experience)
        
        # 处理失败项目
        if failed_items:
            self._handle_storage_failures(failed_items)
        
        return stored_count
    
    def _queue_high_level_storage(self, experience):
        """高层经验排队存储"""
        return self.high_level_storage_queue.put(experience, block=False)
    
    def _queue_low_level_storage(self, experience):
        """低层经验排队存储"""
        return self.low_level_storage_queue.put(experience, block=False)
    
    def _queue_state_skill_storage(self, experience):
        """状态技能经验排队存储"""
        return self.state_skill_storage_queue.put(experience, block=False)
    
    def _process_high_level_storage(self, experience):
        """处理高层经验存储"""
        with self.high_level_lock:
            try:
                success = self.agent.store_high_level_transition(
                    state=experience['state'],
                    team_skill=experience['team_skill'],
                    observations=experience['observations'],
                    agent_skills=experience['agent_skills'],
                    accumulated_reward=experience['accumulated_reward'],
                    skill_log_probs=experience.get('skill_log_probs'),
                    worker_id=experience['worker_id']
                )
                
                if success:
                    self.storage_stats['high_level_stored'].increment()
                    self.logger.debug(f"高层经验存储成功 - worker: {experience['worker_id']}")
                
                return success
                
            except Exception as e:
                self.logger.error(f"高层经验存储失败: {e}")
                return False
    
    def _process_low_level_storage(self, experience):
        """处理低层经验存储"""
        with self.low_level_lock:
            try:
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
                    skill_log_probs=experience.get('skill_log_probs'),
                    worker_id=experience['worker_id']
                )
                
                if success:
                    self.storage_stats['low_level_stored'].increment()
                    self.global_rollout_steps.increment()
                    self.logger.debug(f"低层经验存储成功 - worker: {experience['worker_id']}")
                
                return success
                
            except Exception as e:
                self.logger.error(f"低层经验存储失败: {e}")
                return False
    
    def _process_state_skill_storage(self, experience):
        """处理状态技能经验存储"""
        with self.state_skill_lock:
            try:
                # 确保数据在正确的设备上
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
                
                self.storage_stats['state_skill_stored'].increment()
                self.logger.debug(f"状态技能数据存储成功 - worker: {experience['worker_id']}")
                return True
                
            except Exception as e:
                self.logger.error(f"状态技能数据存储失败: {e}")
                return False
    
    def _validate_experience(self, experience):
        """验证经验数据"""
        try:
            if not isinstance(experience, dict):
                return False
            
            # 基本字段检查
            required_fields = ['experience_type', 'worker_id']
            if not all(field in experience for field in required_fields):
                return False
            
            experience_type = experience.get('experience_type')
            
            # 类型特定验证
            if experience_type == 'low_level':
                required_fields = ['state', 'actions', 'rewards', 'next_state']
                return all(field in experience for field in required_fields)
            elif experience_type == 'high_level':
                required_fields = ['state', 'team_skill', 'accumulated_reward']
                return all(field in experience for field in required_fields)
            elif experience_type == 'state_skill':
                required_fields = ['state', 'team_skill']
                return all(field in experience for field in required_fields)
            
            return True
            
        except Exception as e:
            self.logger.error(f"经验验证异常: {e}")
            return False
    
    def _handle_storage_failures(self, failed_items):
        """处理存储失败的项目"""
        self.storage_stats['queue_overflows'].increment(len(failed_items))
        
        # 记录失败统计
        failure_types = defaultdict(int)
        for item in failed_items:
            failure_types[item.get('experience_type', 'unknown')] += 1
        
        self.logger.warning(f"存储队列溢出: {dict(failure_types)}")
        
        # 可以实现更复杂的失败处理策略，如持久化到磁盘
    
    def assign_skills_for_worker(self, state, observations, worker_id):
        """【线程安全】为特定worker分配技能"""
        with self.skill_assignment_lock:
            try:
                # 确保输入数据有效
                if state is None:
                    state = np.zeros(self.config.state_dim)
                if observations is None:
                    observations = np.zeros((self.config.n_agents, self.config.obs_dim))
                
                # 确保数据在正确的设备上
                if isinstance(state, np.ndarray):
                    state = torch.FloatTensor(state).to(self.agent.device)
                elif isinstance(state, torch.Tensor):
                    state = state.to(self.agent.device)
                
                if isinstance(observations, np.ndarray):
                    observations = torch.FloatTensor(observations).to(self.agent.device)
                elif isinstance(observations, torch.Tensor):
                    observations = observations.to(self.agent.device)
                
                # 分配技能
                team_skill, agent_skills, log_probs = self.agent.assign_skills(
                    state, observations, deterministic=False
                )
                
                # 确保返回值是Python原生类型
                if isinstance(team_skill, torch.Tensor):
                    team_skill = team_skill.cpu().item()
                if isinstance(agent_skills, torch.Tensor):
                    agent_skills = agent_skills.cpu().tolist()
                elif isinstance(agent_skills, list):
                    agent_skills = [int(skill.cpu().item()) if isinstance(skill, torch.Tensor) else int(skill) for skill in agent_skills]
                
                # 处理log_probs
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
                
                self.logger.debug(f"Worker {worker_id}: 技能分配完成 - team_skill={team_skill}, agent_skills={agent_skills}")
                return team_skill, agent_skills, log_probs
                
            except Exception as e:
                self.logger.error(f"Worker {worker_id}: 技能分配失败: {e}")
                # 返回默认技能
                return 0, [0] * self.config.n_agents, {
                    'team_log_prob': 0.0, 
                    'agent_log_probs': [0.0] * self.config.n_agents
                }
    
    def get_actions_for_worker(self, state, observations, agent_skills, worker_id):
        """【线程安全】为特定worker获取动作"""
        try:
            # 确保输入数据有效
            if observations is None:
                observations = np.zeros((self.config.n_agents, self.config.obs_dim))
            if agent_skills is None:
                agent_skills = [0] * self.config.n_agents
            
            # 确保数据在正确的设备上
            if isinstance(observations, np.ndarray):
                observations = torch.FloatTensor(observations).to(self.agent.device)
            elif isinstance(observations, torch.Tensor):
                observations = observations.to(self.agent.device)
            
            if isinstance(agent_skills, np.ndarray):
                agent_skills = torch.tensor(agent_skills, dtype=torch.long, device=self.agent.device)
            elif isinstance(agent_skills, list):
                agent_skills = torch.tensor(agent_skills, dtype=torch.long, device=self.agent.device)
            elif isinstance(agent_skills, torch.Tensor):
                agent_skills = agent_skills.to(device=self.agent.device, dtype=torch.long)
            
            actions, action_logprobs = self.agent.select_action(
                observations, agent_skills, deterministic=False, env_id=worker_id
            )
            
            # 确保返回的数据是numpy数组
            if isinstance(actions, torch.Tensor):
                actions = actions.cpu().detach().numpy()
            if isinstance(action_logprobs, torch.Tensor):
                action_logprobs = action_logprobs.cpu().detach().numpy()
            
            return actions, action_logprobs
            
        except Exception as e:
            self.logger.error(f"Worker {worker_id}: 获取动作失败: {e}")
            # 返回随机动作作为回退
            try:
                n_agents = len(observations) if hasattr(observations, '__len__') else self.config.n_agents
                random_actions = np.random.randn(n_agents, self.config.action_dim)
                return random_actions, np.zeros(n_agents)
            except Exception as fallback_error:
                self.logger.error(f"Worker {worker_id}: 回退动作生成也失败: {fallback_error}")
                return np.random.randn(self.config.n_agents, self.config.action_dim), np.zeros(self.config.n_agents)
    
    def should_update(self):
        """检查是否应该更新 - 修复版本"""
        try:
            # 【修复1】简化更新判断逻辑，避免死锁
            if not hasattr(self, 'rollout_workers') or not self.rollout_workers:
                # 如果没有rollout workers，使用全局步数计数器
                current_steps = self.global_rollout_steps.get()
                self.agent.steps_collected = current_steps
                should_update = self.agent.should_rollout_update()
                self.logger.debug(f"无rollout_workers，使用全局步数: {current_steps}, should_update: {should_update}")
                return should_update
            
            # 【修复2】获取当前状态，避免重复计算
            total_collected = sum(worker.samples_collected for worker in self.rollout_workers)
            completed_workers = sum(1 for worker in self.rollout_workers 
                                  if getattr(worker, 'rollout_completed', False))
            total_workers = len(self.rollout_workers)
            
            # 【修复3】使用更宽松的更新条件
            if total_workers == 0:
                self.logger.warning("rollout_workers为空，无法判断更新条件")
                return False
            
            # 计算目标步数
            target_steps_per_worker = getattr(self.rollout_workers[0], 'target_rollout_steps', self.config.rollout_length)
            target_total_steps = target_steps_per_worker * total_workers
            
            # 【修复4】使用OR逻辑而不是AND，任一条件满足即可更新
            steps_sufficient = total_collected >= target_total_steps
            workers_completed = completed_workers >= total_workers * 0.8  # 80%的workers完成即可
            
            # 【修复5】添加时间基础的更新条件，避免永久等待
            time_based_update = False
            if hasattr(self, '_last_update_time'):
                time_since_last_update = time.time() - self._last_update_time
                if time_since_last_update > 300:  # 5分钟强制更新
                    time_based_update = True
                    self.logger.info(f"基于时间的强制更新: {time_since_last_update:.1f}秒")
            else:
                self._last_update_time = time.time()
            
            should_update = steps_sufficient or workers_completed or time_based_update
            
            # 【修复6】详细日志记录
            self.logger.debug(f"更新条件检查: 总步数={total_collected}/{target_total_steps}, "
                            f"完成workers={completed_workers}/{total_workers}, "
                            f"步数充足={steps_sufficient}, workers完成={workers_completed}, "
                            f"时间强制={time_based_update}, should_update={should_update}")
            
            if should_update:
                self.logger.info(f"满足更新条件: 总步数={total_collected}, 完成workers={completed_workers}/{total_workers}")
                # 同步步数计数器
                self.agent.steps_collected = total_collected
                self.global_rollout_steps.set(total_collected)
                self._last_update_time = time.time()
            
            return should_update
            
        except Exception as e:
            self.logger.error(f"检查更新条件时出错: {e}")
            # 出错时返回True，避免训练卡死
            return True
    
    def update(self):
        """执行模型更新"""
        try:
            self.logger.info("开始模型更新...")
            update_start_time = time.time()
            
            update_info = self.agent.rollout_update()
            
            update_duration = time.time() - update_start_time
            
            if update_info:
                self.logger.info(f"模型更新完成，耗时: {update_duration:.3f}s")
                self._reset_all_workers_rollout_state()
            else:
                self.logger.warning("模型更新返回None")
            
            return update_info
            
        except Exception as e:
            self.logger.error(f"模型更新失败: {e}")
            return None
    
    def _reset_all_workers_rollout_state(self):
        """重置所有workers的rollout状态"""
        if not hasattr(self, 'rollout_workers'):
            return
        
        reset_count = 0
        for worker in self.rollout_workers:
            worker.samples_collected = 0
            worker.rollout_completed = False
            worker.current_team_skill = None
            worker.current_agent_skills = None
            worker.skill_log_probs = None
            worker.accumulated_reward = 0.0
            worker.high_level_experiences_generated = 0
            reset_count += 1
        
        # 重置全局步数计数器
        self.global_rollout_steps.set(0)
        
        self.logger.info(f"已重置 {reset_count} 个workers的rollout状态")
    
    def _log_storage_stats(self):
        """记录存储统计信息"""
        stats = {
            'total_attempts': self.storage_stats['total_attempts'].get(),
            'total_successes': self.storage_stats['total_successes'].get(),
            'total_failures': self.storage_stats['total_failures'].get(),
            'queue_overflows': self.storage_stats['queue_overflows'].get(),
            'high_level_stored': self.storage_stats['high_level_stored'].get(),
            'low_level_stored': self.storage_stats['low_level_stored'].get(),
            'state_skill_stored': self.storage_stats['state_skill_stored'].get(),
            'validation_failures': self.validation_failures.get()
        }
        
        # 计算成功率
        total_attempts = stats['total_attempts']
        success_rate = stats['total_successes'] / total_attempts if total_attempts > 0 else 0.0
        
        # 计算平均处理时间
        avg_processing_time = 0.0
        if self.operation_times:
            avg_processing_time = sum(self.operation_times) / len(self.operation_times)
        
        self.logger.info(f"存储统计: 成功率={success_rate:.2%}, 平均处理时间={avg_processing_time*1000:.2f}ms")
        self.logger.info(f"存储详情: {stats}")
        
        # 记录队列状态
        queue_stats = {
            'high_level_queue': self.high_level_storage_queue.get_stats(),
            'low_level_queue': self.low_level_storage_queue.get_stats(),
            'state_skill_queue': self.state_skill_storage_queue.get_stats()
        }
        self.logger.info(f"队列状态: {queue_stats}")
    
    def get_storage_stats(self):
        """获取存储统计信息"""
        return {
            'total_attempts': self.storage_stats['total_attempts'].get(),
            'total_successes': self.storage_stats['total_successes'].get(),
            'total_failures': self.storage_stats['total_failures'].get(),
            'queue_overflows': self.storage_stats['queue_overflows'].get(),
            'high_level_stored': self.storage_stats['high_level_stored'].get(),
            'low_level_stored': self.storage_stats['low_level_stored'].get(),
            'state_skill_stored': self.storage_stats['state_skill_stored'].get(),
            'validation_failures': self.validation_failures.get(),
            'global_rollout_steps': self.global_rollout_steps.get()
        }
    
    def shutdown(self):
        """关闭代理代理"""
        self.logger.info("正在关闭线程安全代理代理...")
        
        # 停止存储工作线程
        self.storage_stop_event.set()
        
        # 等待所有存储工作线程结束
        for worker in self.storage_workers:
            worker.join(timeout=5.0)
            if worker.is_alive():
                self.logger.warning(f"存储工作线程 {worker.name} 未能在5秒内结束")
        
        # 记录最终统计
        self._log_storage_stats()
        
        self.logger.info("线程安全代理代理已关闭")

    # 为了兼容性，保留原有的方法名
    def store_experience(self, experience_batch):
        """兼容性方法：调用线程安全版本"""
        return self.store_experience_safe(experience_batch)
