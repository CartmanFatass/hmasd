# MAPPO训练脚本使用指南

本文档介绍如何使用基于MAPPO（Multi-Agent Proximal Policy Optimization）的增强训练脚本进行无人机网络优化训练。

## 🆚 MAPPO vs IPPO

- **IPPO (Independent PPO)**: `train_ppo_enhanced_tracking.py` - 每个智能体独立训练
- **MAPPO (Multi-Agent PPO)**: `train_mappo_enhanced_tracking.py` - 多智能体协作训练（推荐）

### 主要区别

| 特性 | IPPO | MAPPO |
|------|------|-------|
| 网络结构 | 独立Actor-Critic | 共享Critic，独立Actor |
| 状态信息 | 局部观测 | 全局状态 + 局部观测 |
| 协调机制 | 隐式（通过环境） | 显式（通过共享Critic） |
| 训练稳定性 | 较低 | 较高 |
| 多智能体协作 | 弱 | 强 |

## 🚀 快速开始

### 1. 环境要求

确保已安装以下依赖：

```bash
pip install torch
pip install numpy
pip install pandas
pip install matplotlib
pip install tensorboard
```

### 2. 基本训练

运行基本的MAPPO训练：

```bash
python train_mappo_enhanced_tracking.py --mode train --scenario 2
```

### 3. 测试脚本

创建MAPPO测试脚本验证环境配置：

```bash
python test_mappo_training.py
```

## 📋 命令行参数

### 基本参数

- `--mode`: 运行模式 (`train` 或 `eval`)
- `--scenario`: 场景选择 (1=基站模式, 2=协作组网模式)
- `--model_path`: 模型保存/加载路径 (默认: `models/mappo_enhanced_tracking.pt`)
- `--log_dir`: 日志目录 (默认: `logs`)
- `--device`: 计算设备 (`auto`, `cuda`, `cpu`)

### 环境参数

- `--n_uavs`: 无人机数量 (默认: 5)
- `--n_users`: 用户数量 (默认: 50)
- `--max_hops`: 最大跳数，仅场景2使用 (默认: 3)
- `--user_distribution`: 用户分布 (`uniform`, `cluster`, `hotspot`)
- `--channel_model`: 信道模型 (`free_space`, `urban`, `suburban`, `3gpp-36777`)

### MAPPO超参数

- `--learning_rate`: 学习率 (默认: 3e-4)
- `--gamma`: 折扣因子 (默认: 0.99)
- `--gae_lambda`: GAE参数 (默认: 0.95)
- `--clip_epsilon`: PPO裁剪参数 (默认: 0.2)
- `--entropy_coef`: 熵系数 (默认: 0.01)
- `--max_grad_norm`: 最大梯度范数 (默认: 0.5)
- `--ppo_epochs`: PPO更新轮数 (默认: 10)
- `--buffer_size`: 缓冲区大小 (默认: 2048)

### 并行化参数

- `--num_envs`: 并行环境数量 (默认: 8)

### 数据收集参数

- `--export_interval`: 数据导出间隔步数 (默认: 1000)
- `--detailed_logging`: 启用详细日志记录

## 💡 使用示例

### 示例1：基本MAPPO训练

```bash
python train_mappo_enhanced_tracking.py \
  --mode train \
  --scenario 2 \
  --n_uavs 5 \
  --n_users 50 \
  --learning_rate 3e-4 \
  --num_envs 8
```

### 示例2：高性能MAPPO训练

```bash
python train_mappo_enhanced_tracking.py \
  --mode train \
  --scenario 2 \
  --n_uavs 8 \
  --n_users 100 \
  --learning_rate 5e-4 \
  --num_envs 16 \
  --buffer_size 4096 \
  --ppo_epochs 15 \
  --device cuda
```

### 示例3：MAPPO模型评估

```bash
python train_mappo_enhanced_tracking.py \
  --mode eval \
  --scenario 2 \
  --model_path models/mappo_enhanced_tracking.pt \
  --eval_episodes 50 \
  --render
```

### 示例4：调试模式

```bash
python train_mappo_enhanced_tracking.py \
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
logs/mappo_enhanced_tracking_YYYYMMDD-HHMMSS/
├── paper_data/                        # 论文数据
│   ├── episode_rewards_step_*.csv
│   ├── agent_coordination_step_*.csv  # MAPPO特有
│   ├── mappo_training_progress_step_*.png
│   └── ...
├── training_summary.json              # 训练摘要
├── mappo_enhanced_tracking_*.log      # 训练日志
├── events.out.tfevents.*              # TensorBoard日志
└── runs/                               # TensorBoard运行数据

models/
├── mappo_enhanced_tracking.pt          # 最终模型
└── ...
```

## 🏗️ MAPPO网络架构

### Actor网络
```
观测 (obs_dim) → FC(64) → ReLU → FC(64) → ReLU → FC(action_dim) → 动作均值
                                                 ↓
                                            学习的log_std → 动作标准差
```

### Critic网络
```
全局状态 (state_dim) → FC(64) → ReLU → FC(64) → ReLU → FC(1) → 状态值
```

### 关键特性
- **共享Critic**: 使用全局状态信息，提供更好的值函数估计
- **独立Actor**: 每个智能体有独立的策略网络
- **GAE优势估计**: 使用Generalized Advantage Estimation
- **PPO剪裁**: 防止策略更新过大

## 📈 监控训练过程

### TensorBoard

启动TensorBoard监控MAPPO训练过程：

```bash
tensorboard --logdir=logs/mappo_enhanced_tracking_YYYYMMDD-HHMMSS
```

主要指标：
- `Training/Actor_Loss`: Actor网络损失
- `Training/Critic_Loss`: Critic网络损失
- `Training/Entropy`: 策略熵（探索度）
- `Training/Clip_Fraction`: PPO裁剪比例
- `Training/KL_Divergence`: 策略变化程度

### 实时日志

训练过程中的关键信息：

```
MAPPO模型已创建，设备: cuda
开始训练，总时间步数: 3000000
Episode 1: 环境0, 奖励=85.34, 长度=245
步骤 2048: Actor损失=0.0234, Critic损失=0.1567
智能体协调指标: 0.847
Episode 2: 环境1, 奖励=92.15, 长度=312
...
```

## 🎯 与其他算法的比较

| 特性 | HMASD | IPPO | MAPPO |
|------|-------|------|-------|
| 算法类型 | 分层技能发现 | 独立策略优化 | 多智能体策略优化 |
| 技能发现 | ✅ 支持 | ❌ 不支持 | ❌ 不支持 |
| 多智能体协调 | 显式层次结构 | 隐式学习 | 共享Critic协调 |
| 实现复杂度 | 高 | 低 | 中等 |
| 训练稳定性 | 中等 | 中等 | 高 |
| 收敛速度 | 慢 | 快 | 中等 |
| 协作效果 | 强 | 弱 | 强 |

## 🛠️ 故障排除

### 常见问题

1. **CUDA内存不足**
   ```
   RuntimeError: CUDA out of memory
   ```
   解决方案：
   - 减少 `--num_envs` 或 `--buffer_size`
   - 使用 `--device cpu`

2. **训练不稳定**
   ```
   Actor损失剧烈波动
   ```
   解决方案：
   - 降低学习率
   - 增加 `--ppo_epochs`
   - 调整 `--clip_epsilon`

3. **智能体协调差**
   ```
   协调指标持续很低
   ```
   解决方案：
   - 检查奖励函数设计
   - 增加全局状态信息
   - 调整网络架构

### MAPPO特定优化建议

1. **共享Critic效果**：确保全局状态包含足够的环境信息
2. **策略同步**：使用相同的随机种子确保策略一致性
3. **经验收集**：平衡探索和利用，调整熵系数
4. **网络更新**：使用适当的更新频率和批次大小

## 📝 开发注意事项

### 扩展MAPPO脚本

如需添加新的功能：

1. **自定义网络架构**：修改`MAPPOActor`和`MAPPOCritic`类
2. **新的协调机制**：扩展共享Critic的输入信息
3. **不同的奖励设计**：调整环境的奖励函数
4. **评估指标**：扩展`EnhancedRewardTracker`类

### 调试技巧

1. 使用小的参数值进行快速测试
2. 监控智能体协调指标
3. 检查Actor和Critic损失的变化趋势
4. 使用TensorBoard可视化训练过程

## 🔬 实验建议

### 超参数调优

1. **学习率**: 从1e-4到1e-3范围内调整
2. **裁剪参数**: 0.1-0.3之间尝试不同值
3. **熵系数**: 根据探索需求调整
4. **更新轮数**: 5-15轮之间选择

### 性能基准

建议在以下配置下进行性能测试：
- 5个无人机，50个用户
- 8个并行环境
- 场景2（协作组网）
- 训练300万步

## 📄 许可证

本项目遵循与主项目相同的许可证。

## 🤝 贡献

欢迎提交问题报告和改进建议！特别是关于MAPPO算法优化的建议。
