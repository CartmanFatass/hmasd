# 同步训练HMASD算法指南

## 概述

本项目实现了HMASD（Hierarchical Multi-Agent Skill Discovery）算法的同步训练版本，确保严格的on-policy特性。

## 同步训练的优势

### 1. 严格On-Policy
- **问题**：原训练循环是异步的，环境A在T1时刻存储经验，环境B在T2时刻触发更新，导致环境A的T1经验变为off-policy
- **解决方案**：同步收集1024个样本后统一更新，确保所有数据来自同一策略版本

### 2. 训练稳定性
- 减少更新频率，提高稳定性
- 更大的batch size有助于梯度估计
- 避免频繁的策略变化

### 3. 高效并行
- 保持32个环境并行收集数据
- 每个环境平均贡献32个样本（1024/32=32）
- 最大化利用并行计算资源

## 核心实现

### 1. 同步控制机制

在 `HMASDAgent` 中添加了以下机制：

```python
# 同步训练控制机制
self.sync_mode = True                     # 启用同步训练模式
self.collection_enabled = True            # 数据收集开关
self.policy_version = 0                   # 策略版本号
self.sync_batch_size = config.batch_size  # 同步batch大小（1024）
self.samples_collected_this_round = 0     # 本轮收集的样本数
```

### 2. 同步更新流程

```python
def sync_update(self):
    # 1. 停止数据收集
    self.disable_data_collection()
    
    # 2. 执行网络更新
    update_info = self.update()
    
    # 3. 重置同步状态
    self.policy_version += 1
    self.samples_collected_this_round = 0
    
    # 4. 重新启用数据收集
    self.enable_data_collection()
    
    return update_info
```

### 3. 样本计数机制

```python
def store_transition(self, ...):
    # 同步模式检查：如果数据收集被禁用，则拒绝存储
    if self.sync_mode and not self.collection_enabled:
        return False
    
    # ... 存储逻辑 ...
    
    # 在同步模式下，增加样本计数
    if self.sync_mode:
        self.samples_collected_this_round += 1
    
    return True
```

## 配置调整

### 1. Batch Size
```python
# config_1.py
batch_size = 1024  # 同步训练批处理大小 (32环境 × 32样本/环境)
```

### 2. 并行环境数量
```python
num_envs = 32  # 32个并行环境
```

## 使用方法

### 1. 基本训练

```bash
python train_sync_enhanced.py
```

### 2. 指定日志级别

```bash
python train_sync_enhanced.py --log-level DEBUG
```

### 3. 从检查点继续训练

```bash
python train_sync_enhanced.py --model-path models/hmasd_sync_enhanced_320000.pth
```

### 4. 仅评估模式

```bash
python train_sync_enhanced.py --eval-only --model-path models/hmasd_sync_enhanced_final.pth
```

## 训练流程

### 1. 数据收集阶段
```
while not agent.should_sync_update():
    # 所有32个环境并行执行一步
    # 存储经验到buffer
    # 增加样本计数
```

### 2. 同步更新阶段
```
if agent.should_sync_update():
    # 禁用数据收集
    # 执行网络更新
    # 清空所有buffer
    # 重置样本计数
    # 重新启用数据收集
```

### 3. 循环继续
- 重复数据收集和同步更新
- 直到达到目标训练步数

## 性能监控

### 1. TensorBoard监控
```bash
tensorboard --logdir tf-logs/
```

### 2. 关键指标
- **Sync/BatchSize**: 同步batch大小
- **Sync/PolicyVersion**: 策略版本号
- **Sync/CollectionTime**: 数据收集耗时
- **Sync/UpdateTime**: 网络更新耗时

### 3. Buffer状态
- **Buffer/SamplesCollected**: 已收集样本数
- **Buffer/low_level_buffer_size**: 低层buffer大小
- **Buffer/high_level_buffer_size**: 高层buffer大小

## 预期效果

### 1. 训练稳定性提升
- 更稳定的梯度估计
- 减少策略震荡
- 更好的收敛性

### 2. On-Policy保证
- 所有经验来自同一策略版本
- 符合PPO算法要求
- 理论上更正确的实现

### 3. 性能提升
- 更好的样本效率
- 更快的收敛速度
- 更高的最终性能

## 故障排除

### 1. 内存不足
- 减少 `batch_size`
- 减少 `num_envs`
- 使用更小的网络

### 2. 训练慢
- 检查GPU利用率
- 调整 `batch_size`
- 优化数据加载

### 3. 收敛问题
- 检查学习率
- 调整同步频率
- 检查奖励设计

## 与原版本对比

| 特性 | 原版本 | 同步版本 |
|------|--------|----------|
| 更新模式 | 异步 | 同步 |
| Batch Size | 128 | 1024 |
| On-Policy | 部分违反 | 严格保证 |
| 稳定性 | 一般 | 更好 |
| 内存使用 | 较低 | 较高 |

## 文件结构

```
.
├── train_sync_enhanced.py      # 主训练脚本
├── hmasd/
│   ├── agent.py               # 增强的Agent（含同步机制）
│   └── networks.py            # 网络结构
├── config_1.py                # 调整后的配置
├── SYNC_TRAINING_README.md     # 本文档
└── logs/                      # 日志目录
```

## 下一步优化

1. **动态Batch Size**: 根据训练进度调整batch size
2. **异步评估**: 在后台进行评估，不阻塞训练
3. **梯度累积**: 支持更大的有效batch size
4. **分布式训练**: 支持多机训练

## 总结

同步训练版本通过强制on-policy特性，提供了更稳定和理论正确的HMASD训练实现。虽然内存使用略有增加，但训练质量和稳定性的提升是值得的。
