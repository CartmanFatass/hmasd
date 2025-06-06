# TensorBoard使用指南

本项目已集成TensorBoard支持，可以实时监控训练过程中的各项指标。

## 安装

确保已安装TensorBoard：

```bash
pip install tensorboard>=2.8.0
```

或者直接使用项目的requirements.txt安装所有依赖：

```bash
pip install -r requirements.txt
```

## 记录的指标

TensorBoard会记录以下指标：

### 训练指标
- **Loss/**: 各种损失函数值
  - `high_level`: 高层策略损失
  - `low_level`: 低层策略损失
  - `discriminator`: 判别器损失
  - `high_level_policy`: 高层策略损失详情
  - `high_level_value`: 高层价值损失
  - `low_level_policy`: 低层策略损失详情
  - `low_level_value`: 低层价值损失

- **Entropy/**: 熵相关指标
  - `team_skill`: 团队技能熵
  - `action`: 动作熵

- **Reward/**: 奖励相关指标
  - `episode_reward`: 每个episode的累计奖励
  - `episode_length`: 每个episode的长度
  - `avg_reward_10`: 最近10个episodes的平均奖励

- **Skills/**: 技能分配相关指标
  - `team_skill`: 团队技能选择
  - `agent{i}_skill`: 每个智能体的技能选择
  - `diversity`: 技能多样性指标

### 评估指标
- **Eval/**: 评估相关指标
  - `episode_reward`: 评估episode的奖励
  - `episode_length`: 评估episode的长度
  - `mean_reward`: 评估的平均奖励
  - `reward_std`: 评估奖励的标准差
  - `mean_episode_length`: 评估的平均episode长度

## 启动TensorBoard

训练或评估过程中，日志会保存在`logs`目录下。要查看这些日志，运行：

```bash
tensorboard --logdir=logs
```

然后在浏览器中打开 http://localhost:6006 查看训练进度。

## 多次运行比较

如果要比较多次运行的结果，可以保持所有日志在logs目录下，TensorBoard会自动将它们分组显示：

```bash
tensorboard --logdir=logs
```

## 常见问题

1. **找不到日志文件**：确保训练时指定了正确的`--log_dir`参数。

2. **指标不更新**：TensorBoard有时需要刷新缓存，可以尝试在浏览器中手动刷新页面。

3. **端口被占用**：如果6006端口被占用，可以指定其他端口：
   ```bash
   tensorboard --logdir=logs --port=6007
