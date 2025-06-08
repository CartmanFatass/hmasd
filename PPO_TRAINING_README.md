# PPO训练脚本使用指南

本文档介绍如何使用基于PPO（Proximal Policy Optimization）的增强训练脚本进行无人机网络优化训练。

## 🚀 快速开始

### 1. 环境要求

确保已安装以下依赖：

```bash
pip install stable-baselines3[extra]
pip install torch
pip install numpy
pip install pandas
pip install matplotlib
pip install tensorboard
```

### 2. 基本训练

运行基本的PPO训练：

```bash
python train_ppo_enhanced_tracking.py --mode train --scenario 2
```

### 3. 测试脚本

在开始正式训练前，建议运行测试脚本验证环境配置：

```bash
python test_ppo_training.py
```

## 📋 命令行参数

### 基本参数

- `--mode`: 运行模式 (`train` 或 `eval`)
- `--scenario`: 场景选择 (1=基站模式, 2=协作组网模式)
- `--model_path`: 模型保存/加载路径 (默认: `models/ppo_enhanced_tracking.zip`)
- `--log_dir`: 日志目录 (默认: `logs`)
- `--device`: 计算设备 (`auto`, `cuda`, `cpu`)

### 环境参数

- `--n_uavs`: 无人机数量 (默认: 5)
- `--n_users`: 用户数量 (默认: 50)
- `--max_hops`: 最大跳数，仅场景2使用 (默认: 3)
- `--user_distribution`: 用户分布 (`uniform`, `cluster`, `hotspot`)
- `--channel_model`: 信道模型 (`free_space`, `urban`, `suburban`, `3gpp-36777`)

### PPO超参数

- `--learning_rate`: 学习率 (默认: 3e-4)
- `--gamma`: 折扣因子 (默认: 0.99)
- `--gae_lambda`: GAE参数 (默认: 0.95)
- `--clip_range`: PPO裁剪参数 (默认: 0.2)
- `--ent_coef`: 熵系数 (默认: 0.01)
- `--vf_coef`: 值函数损失系数 (默认: 0.5)
- `--max_grad_norm`: 最大梯度范数 (默认: 0.5)
- `--n_steps`: 每次更新收集的步数 (默认: 2048)
- `--batch_size`: 小批量大小 (默认: 64)
- `--n_epochs`: 每次更新的优化轮数 (默认: 10)

### 并行化参数

- `--num_envs`: 并行环境数量 (0=使用配置文件值)
- `--eval_rollout_threads`: 评估并行线程数 (0=使用配置文件值)

### 数据收集参数

- `--export_interval`: 数据导出间隔步数 (默认: 1000)
- `--detailed_logging`: 启用详细日志记录

## 💡 使用示例

### 示例1：基本训练

```bash
python train_ppo_enhanced_tracking.py \
  --mode train \
  --scenario 2 \
  --n_uavs 5 \
  --n_users 50 \
  --learning_rate 3e-4 \
  --num_envs 8
```

### 示例2：高性能训练

```bash
python train_ppo_enhanced_tracking.py \
  --mode train \
  --scenario 2 \
  --n_uavs 8 \
  --n_users 100 \
  --learning_rate 5e-4 \
  --num_envs 16 \
  --n_steps 4096 \
  --batch_size 128 \
  --device cuda
```

### 示例3：模型评估

```bash
python train_ppo_enhanced_tracking.py \
  --mode eval \
  --scenario 2 \
  --model_path models/ppo_enhanced_tracking.zip \
  --eval_episodes 50 \
  --render
```

### 示例4：调试模式

```bash
python train_ppo_enhanced_tracking.py \
  --mode train \
  --scenario 1 \
  --n_uavs 3 \
  --n_users 10 \
  --num_envs 2 \
  --export_interval 100 \
  --detailed_logging \
  --log_level debug
```

## 📊 输出文件结构

训练过程中会生成以下文件：

```
logs/ppo_enhanced_tracking_YYYYMMDD-HHMMSS/
├── paper_data/                    # 论文数据
│   ├── episode_rewards_step_*.csv
│   ├── ppo_training_progress_step_*.png
│   └── ...
├── training_summary.json          # 训练摘要
├── ppo_enhanced_tracking_*.log    # 训练日志
└── tensorboard logs/              # TensorBoard日志

models/
├── ppo_enhanced_tracking.zip      # 最终模型
└── best_model.zip                 # 最佳模型
```

## 🔧 自定义网络架构

可以通过修改 `CustomActorCriticPolicy` 类来自定义网络架构：

```python
class CustomActorCriticPolicy(ActorCriticPolicy):
    def __init__(self, observation_space, action_space, lr_schedule, *args, **kwargs):
        # 自定义网络架构
        kwargs['net_arch'] = dict(
            pi=[128, 128, 64],  # Actor网络层
            vf=[128, 128, 64]   # Critic网络层
        )
        kwargs['activation_fn'] = nn.ReLU
        
        super(CustomActorCriticPolicy, self).__init__(
            observation_space, action_space, lr_schedule, *args, **kwargs
        )
```

## 📈 监控训练过程

### TensorBoard

启动TensorBoard监控训练过程：

```bash
tensorboard --logdir=logs/ppo_enhanced_tracking_YYYYMMDD-HHMMSS
```

主要指标：
- `rollout/ep_len_mean`: 平均episode长度
- `rollout/ep_rew_mean`: 平均episode奖励
- `train/policy_gradient_loss`: 策略梯度损失
- `train/value_loss`: 值函数损失
- `train/entropy_loss`: 熵损失

### 实时日志

训练过程中的关键信息会实时输出到控制台和日志文件：

```
PPO模型已创建，设备: cuda
开始训练，总时间步数: 3000000
----------------------------------
| rollout/           |           |
|    ep_len_mean     | 1.24e+03  |
|    ep_rew_mean     | -156      |
| time/              |           |
|    fps             | 1893      |
|    iterations      | 1         |
|    time_elapsed    | 1         |
|    total_timesteps | 2048      |
| train/             |           |
|    approx_kl       | 0.016281  |
|    clip_fraction   | 0.281     |
|    clip_range      | 0.2       |
|    entropy_loss    | -1.41     |
|    explained_var   | -0.0402   |
|    learning_rate   | 0.0003    |
|    loss            | 3.58      |
|    policy_gradient_loss| -0.0185|
|    value_loss      | 7.39      |
----------------------------------
```

## 🎯 与HMASD训练的比较

| 特性 | HMASD训练 | PPO训练 |
|------|-----------|---------|
| 算法类型 | 分层多智能体技能发现 | 近端策略优化 |
| 技能发现 | ✅ 支持 | ❌ 不支持 |
| 实现复杂度 | 高 | 中等 |
| 训练稳定性 | 中等 | 高 |
| 超参数调节 | 复杂 | 相对简单 |
| 多智能体协调 | 显式层次结构 | 隐式学习 |
| 收敛速度 | 较慢 | 较快 |

## 🛠️ 故障排除

### 常见问题

1. **CUDA内存不足**
   ```
   RuntimeError: CUDA out of memory
   ```
   解决方案：
   - 减少 `--num_envs` 或 `--batch_size`
   - 使用 `--device cpu`

2. **环境创建失败**
   ```
   ValueError: 未知的场景: X
   ```
   解决方案：
   - 确保 `--scenario` 参数为1或2
   - 检查环境模块导入

3. **模型加载失败**
   ```
   FileNotFoundError: No such file or directory
   ```
   解决方案：
   - 检查 `--model_path` 路径是否正确
   - 确保模型文件存在

### 性能优化建议

1. **并行环境数量**：通常设置为CPU核心数的1-2倍
2. **批次大小**：根据GPU内存调整，通常32-512之间
3. **学习率**：从3e-4开始，根据收敛情况调整
4. **采样步数**：增大`n_steps`可以提高样本效率但需要更多内存

## 📝 开发注意事项

### 扩展训练脚本

如需添加新的功能：

1. **自定义奖励函数**：修改环境的奖励计算逻辑
2. **新的回调函数**：继承`BaseCallback`实现自定义回调
3. **不同的策略网络**：修改`CustomActorCriticPolicy`类
4. **新的评估指标**：扩展`EnhancedRewardTracker`类

### 调试技巧

1. 使用小的参数值进行快速测试
2. 启用`--detailed_logging`获取更多信息
3. 运行`test_ppo_training.py`验证环境配置
4. 使用TensorBoard监控训练指标

## 📄 许可证

本项目遵循与主项目相同的许可证。

## 🤝 贡献

欢迎提交问题报告和改进建议！
