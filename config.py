# HMASD算法配置参数

class Config:
    # 环境参数
    # 注意：实际环境中应该获取这些值
    n_agents = 10  # 无人机数量上限
    state_dim = None  # 全局状态维度（将在环境初始化时获取）
    obs_dim = None    # 单个智能体观测维度（将在环境初始化时获取）
    action_dim = 3    # 每个智能体输出3D速度向量

    # HMASD参数 - 严格按照论文设置
    n_Z = 3           # 团队技能数量（论文3m场景设置）
    n_z = 3           # 个体技能数量（论文3m场景设置）
    k = 32            # 技能分配间隔（优化为rollout_length的整数因子，确保完整收集高层经验）

    # 网络参数
    hidden_size = 256        # 隐藏层大小
    embedding_dim = 128      # 嵌入维度
    n_encoder_layers = 3     # 编码器层数
    n_decoder_layers = 3     # 解码器层数
    n_heads = 8             # 多头注意力头数
    gru_hidden_size = 256    # GRU隐藏层大小
    lr_coordinator = 3e-4    # 技能协调器学习率
    lr_discoverer = 3e-4     # 技能发现器学习率
    lr_discriminator = 3e-4  # 技能判别器学习率

    # PPO参数
    gamma = 0.99             # 折扣因子
    gae_lambda = 0.95        # GAE参数
    clip_epsilon = 0.2       # PPO裁剪参数
    ppo_epochs = 15          # PPO迭代次数
    value_loss_coef = 0.5    # 价值损失系数
    entropy_coef = 0.01      # 熵损失系数
    max_grad_norm = 0.5      # 最大梯度范数

    # HMASD损失权重 - 严格按照论文Table 3中3m场景设置
    lambda_e = 1.0           # 外部奖励权重（论文3m场景设置）
    lambda_D = 0.1           # 团队技能判别器奖励权重（论文3m场景设置）
    lambda_d = 0.5           # 个体技能判别器奖励权重（论文3m场景设置）
    lambda_h = 0.001         # 高层策略熵权重（论文3m场景设置）
    lambda_l = 0.01          # 低层策略熵权重（论文3m场景设置）

    # 训练参数
    buffer_size = 10000      # 经验回放缓冲区大小（增大以支持更多rollout数据）
    batch_size = 128         # 批处理大小
    high_level_batch_size = 128  # 高层更新的批处理大小（调整为rollout_length大小）
    num_envs = 32            # 并行环境数量（与rollout并行环境一致）
    total_timesteps = 5e6    # 总时间步数
    eval_interval = 1000     # 评估间隔
    
    # =================================================================
    # Rollout-based训练参数（论文标准实现，推荐默认）
    # =================================================================
    rollout_based_training = True   # 启用rollout-based训练模式
    rollout_length = 128            # 每个rollout收集的步数（对应论文Algorithm 1中的rollout周期）
    num_parallel_envs = 32          # 并行环境数量（对应论文rollout_threads）
    ppo_epochs = 15                 # PPO训练轮数（严格对应论文附录E中的ppo_epoch=15）
    num_mini_batch = 1              # 小批次数量（论文设置为1，每轮使用全部rollout数据）
    
    # Rollout计算说明：
    # - 每个rollout收集: num_parallel_envs × rollout_length = 32 × 128 = 4096个样本
    # - 训练阶段: PPO使用这4096个样本训练15轮
    # - 缓冲区管理: 训练后清空B_h和B_l（PPO on-policy要求），保留D（判别器数据）
    # - 技能分配: 每k=50步重新分配技能，rollout内会有多次技能切换
    
    # Rollout-specific配置调整
    rollout_target_samples = 4096   # 目标样本数（num_parallel_envs × rollout_length）
    rollout_high_level_buffer_size = 256  # 高层缓冲区大小（适应rollout模式）
    rollout_discriminator_train_freq = 1   # 判别器训练频率（每个rollout都训练）
    
    # =================================================================
    # Episode-based训练参数（兼容性保留）
    # =================================================================
    episode_based_training = False  # 启用episode-based训练模式
    update_frequency = 10           # 每收集多少个episode后进行一次更新
    min_episodes_for_update = 5     # 开始更新前最少需要收集的episode数
    max_episodes_per_update = 50    # 单次更新最多使用的episode数
    min_high_level_samples = 32     # 高层更新需要的最少样本数
    min_low_level_samples = 128     # 低层更新需要的最少样本数
    
    # =================================================================
    # 同步训练参数（兼容性保留）
    # =================================================================
    sync_training_mode = False      # 同步训练模式（与其他模式互斥）
    
    # =================================================================
    # Rollout训练流程控制参数
    # =================================================================
    # 数据收集阶段
    rollout_skill_reassign_interval = 50  # 技能重新分配间隔（等于k值）
    rollout_max_episode_length = 500      # 单个episode最大长度（避免无限episode）
    rollout_early_termination = True      # 启用early termination（环境完成任务时）
    
    # 训练阶段控制
    rollout_coordinator_first = True      # 优先更新协调器（高层策略）
    rollout_clear_buffers_after_update = True  # 更新后清空PPO缓冲区
    rollout_preserve_discriminator_data = True # 保留判别器训练数据
    
    # 技能多样性和探索
    rollout_skill_entropy_threshold = 0.5    # 技能熵阈值（低于此值增加探索）
    rollout_force_skill_diversity = True     # 强制技能多样性
    rollout_exploration_bonus = 0.01         # 探索奖励系数
    
    # 数值稳定性
    rollout_gradient_clip_enabled = True     # 启用梯度裁剪
    rollout_value_clip_range = 10.0          # 价值函数裁剪范围
    rollout_reward_normalization = False     # 奖励标准化（可选）
    rollout_advantage_normalization = True   # advantage标准化
    
    # Huber Loss配置
    use_huber_loss = True                    # 启用Huber Loss替代MSE Loss
    huber_delta = 1.0                        # Huber Loss的delta参数（控制L1/L2切换点）
    huber_coordinator_delta = 1.0            # 协调器价值函数的Huber delta
    huber_discoverer_delta = 1.0             # 发现器价值函数的Huber delta
    huber_adaptive_delta = False             # 自适应调整delta参数
    huber_delta_decay = 0.999                # delta衰减率（如果启用自适应）
    huber_min_delta = 0.1                    # 最小delta值
    
    # 调试和监控
    rollout_log_interval = 10               # 日志记录间隔（每N个rollout）
    rollout_save_interval = 100             # 模型保存间隔（每N个rollout）
    rollout_eval_interval = 50              # 评估间隔（每N个rollout）
    rollout_detailed_logging = True         # 详细日志记录
    
    # 性能优化
    rollout_vectorized_envs = True          # 使用向量化环境
    rollout_async_collection = False       # 异步数据收集（可选，复杂度较高）
    rollout_gpu_acceleration = True        # GPU加速（如果可用）
    rollout_mixed_precision = False        # 混合精度训练（可选）

    def update_env_dims(self, state_dim, obs_dim):
        """更新环境维度"""
        self.state_dim = state_dim
        self.obs_dim = obs_dim
        print(f"环境维度已更新：state_dim={state_dim}, obs_dim={obs_dim}")
    
    def validate_training_mode(self):
        """验证训练模式配置的一致性"""
        active_modes = []
        if self.rollout_based_training:
            active_modes.append("rollout_based")
        if self.episode_based_training:
            active_modes.append("episode_based")
        if self.sync_training_mode:
            active_modes.append("sync")
        
        if len(active_modes) == 0:
            print("警告：没有启用任何训练模式，将默认使用rollout_based模式")
            self.rollout_based_training = True
            return "rollout_based"
        elif len(active_modes) > 1:
            print(f"警告：启用了多个训练模式{active_modes}，将使用rollout_based作为默认模式")
            self.rollout_based_training = True
            self.episode_based_training = False
            self.sync_training_mode = False
            return "rollout_based"
        else:
            print(f"训练模式验证通过：{active_modes[0]}")
            return active_modes[0]
    
    def validate_rollout_config(self):
        """验证rollout-based训练配置的合理性"""
        if not self.rollout_based_training:
            return True
        
        issues = []
        
        # 检查关键参数
        if self.rollout_length <= 0:
            issues.append(f"rollout_length必须大于0，当前值：{self.rollout_length}")
        
        if self.num_parallel_envs <= 0:
            issues.append(f"num_parallel_envs必须大于0，当前值：{self.num_parallel_envs}")
        
        if self.ppo_epochs <= 0:
            issues.append(f"ppo_epochs必须大于0，当前值：{self.ppo_epochs}")
        
        # 检查计算一致性
        expected_samples = self.num_parallel_envs * self.rollout_length
        if self.rollout_target_samples != expected_samples:
            print(f"自动修正rollout_target_samples：{self.rollout_target_samples} -> {expected_samples}")
            self.rollout_target_samples = expected_samples
        
        # 检查技能分配间隔
        if self.rollout_skill_reassign_interval != self.k:
            print(f"自动同步技能重分配间隔：{self.rollout_skill_reassign_interval} -> {self.k}")
            self.rollout_skill_reassign_interval = self.k
        
        # 检查缓冲区大小
        if self.rollout_high_level_buffer_size < self.rollout_length:
            print(f"警告：高层缓冲区大小({self.rollout_high_level_buffer_size})小于rollout长度({self.rollout_length})")
        
        if issues:
            print("Rollout配置验证失败：")
            for issue in issues:
                print(f"  - {issue}")
            return False
        
        print("Rollout配置验证通过")
        return True
    
    def get_rollout_summary(self):
        """获取rollout训练配置摘要"""
        if not self.rollout_based_training:
            return "Rollout模式未启用"
        
        summary = f"""
=== Rollout-based训练配置摘要 ===
🎯 数据收集阶段：
  - 并行环境数量：{self.num_parallel_envs}
  - 每轮收集步数：{self.rollout_length}
  - 目标样本总数：{self.rollout_target_samples}
  - 技能重分配间隔：{self.rollout_skill_reassign_interval}步

🔄 训练阶段：
  - PPO训练轮数：{self.ppo_epochs}
  - 小批次数量：{self.num_mini_batch}（1=使用全部数据）
  - 高层缓冲区大小：{self.rollout_high_level_buffer_size}
  - 缓冲区清空策略：{'启用' if self.rollout_clear_buffers_after_update else '禁用'}

⚙️ 优化设置：
  - 梯度裁剪：{'启用' if self.rollout_gradient_clip_enabled else '禁用'}
  - advantage标准化：{'启用' if self.rollout_advantage_normalization else '禁用'}
  - 详细日志：{'启用' if self.rollout_detailed_logging else '禁用'}
  - 向量化环境：{'启用' if self.rollout_vectorized_envs else '禁用'}

📊 监控设置：
  - 日志间隔：每{self.rollout_log_interval}个rollout
  - 保存间隔：每{self.rollout_save_interval}个rollout
  - 评估间隔：每{self.rollout_eval_interval}个rollout

🧮 预期性能：
  - 每个rollout产生：{self.rollout_target_samples}个样本
  - 每个rollout训练：{self.ppo_epochs}轮
  - 总计算量：{self.rollout_target_samples * self.ppo_epochs}个样本×轮次
        """
        return summary.strip()
    
    def print_config_summary(self):
        """打印完整配置摘要"""
        mode = self.validate_training_mode()
        print(f"\n{'='*50}")
        print(f"HMASD训练配置摘要 - 模式：{mode.upper()}")
        print(f"{'='*50}")
        
        if mode == "rollout_based":
            print(self.get_rollout_summary())
        elif mode == "episode_based":
            print("Episode-based训练模式已启用")
        elif mode == "sync":
            print("同步训练模式已启用")
        
        print(f"\n🔧 核心参数：")
        print(f"  - 智能体数量：{self.n_agents}")
        print(f"  - 团队技能数：{self.n_Z}")
        print(f"  - 个体技能数：{self.n_z}")
        print(f"  - 技能分配间隔：{self.k}")
        print(f"  - 总训练步数：{int(self.total_timesteps):,}")
        print(f"{'='*50}\n")
