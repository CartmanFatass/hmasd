# MAPPO训练监控指南

## 问题诊断

您遇到的问题是：**训练实际上在正常运行，但控制台没有显示进度信息**。

### 原因分析

1. **控制台日志级别设置为 `error`**：您使用了 `--console_log_level error` 参数，这意味着只有错误级别的日志才会显示在终端
2. **训练进度信息是 `INFO` 级别**：步骤更新、损失值等信息都是INFO级别，因此不会在控制台显示
3. **所有信息都写入了日志文件**：从日志文件可以看到训练正常进行，已完成61,440步

## 解决方案

### 方案1：使用新的训练启动器（推荐）

```bash
# 启动训练并自动启用监控
python start_training_with_monitor.py --scenario 2 --num_envs 32

# 自定义参数启动
python start_training_with_monitor.py --scenario 2 --num_envs 32 --console_log_level info --n_uavs 5 --n_users 50
```

### 方案2：单独使用监控工具

```bash
# 监控当前训练进度
python monitor_training.py

# 显示训练摘要
python monitor_training.py --summary

# 实时监控指定日志文件
python monitor_training.py --monitor --log_file logs/mappo_enhanced_tracking_20250607_143916.log
```

### 方案3：修改原始训练命令

将控制台日志级别改为 `info`：

```bash
python train_mappo_enhanced_tracking.py --mode train --scenario 2 --num_envs 32 --console_log_level info
```

## 当前训练状态

根据日志文件分析，您的训练：

- ✅ **正在正常运行**
- ✅ **已完成61,440步** (目标：3,000,000步)
- ✅ **Actor和Critic损失都在正常变化**
- ✅ **使用GPU加速**
- ⏱️ **预计还需要约20小时完成全部训练**

### 训练进度

- **当前进度**: 61,440 / 3,000,000 步 (约2%)
- **最新损失**: Actor损失=0.2447, Critic损失=0.3490
- **训练时间**: 已运行约20分钟

## 监控工具功能

### monitor_training.py 功能

1. **实时监控**: 每10秒更新一次训练进度
2. **训练摘要**: 显示当前状态和统计信息
3. **自动检测**: 自动找到最新的训练日志文件
4. **损失趋势**: 显示近期Actor和Critic损失变化

### start_training_with_monitor.py 功能

1. **一键启动**: 同时启动训练和监控
2. **参数透传**: 支持所有原始训练参数
3. **实时反馈**: 在控制台显示训练和监控信息
4. **错误处理**: 提供友好的错误信息

## 使用示例

### 查看当前训练状态

```bash
# 快速查看训练摘要
python monitor_training.py --summary
```

输出示例：
```
训练摘要 - logs/mappo_enhanced_tracking_20250607_143916.log
============================================================
最新步骤: 61,440
最新Actor损失: 0.2447
最新Critic损失: 0.3490
近10次更新Actor损失均值: 0.1876
近10次更新Critic损失均值: 0.7234
```

### 实时监控训练

```bash
# 开始实时监控
python monitor_training.py --monitor
```

输出示例：
```
开始监控训练进度: logs/mappo_enhanced_tracking_20250607_143916.log
============================================================
[15:58:30] 步骤 62,464: Actor损失=0.1234, Critic损失=0.5678
[15:58:50] 步骤 63,488: Actor损失=0.1456, Critic损失=0.4321
```

## 注意事项

1. **不要终止当前训练**: 您的训练正在正常进行，请不要强制终止
2. **监控不影响训练**: 监控工具只是读取日志文件，不会影响训练性能
3. **日志文件自动更新**: 监控工具会自动检测日志文件的新内容
4. **训练需要时间**: 300万步的训练预计需要约20小时完成

## 故障排除

### 如果监控工具报错

```bash
# 检查日志文件是否存在
ls -la logs/

# 手动指定日志文件
python monitor_training.py --log_file logs/mappo_enhanced_tracking_20250607_143916.log
```

### 如果需要调整训练参数

当前训练无法在运行中修改参数，如需调整：
1. 停止当前训练 (Ctrl+C)
2. 使用新参数重新启动
3. 可以通过 `--model_path` 加载已保存的模型继续训练

## 建议操作

**立即执行**:
```bash
# 在新的终端窗口中运行，查看当前训练状态
python monitor_training.py --summary

# 然后开始实时监控
python monitor_training.py --monitor
```

这样您就可以看到训练的实时进度，而不需要停止当前正在运行的训练过程。
