# HMASD同步训练快速开始指南

## 🚀 快速启动

### 1. 环境准备
```bash
# 安装依赖
pip install torch numpy tensorboardX pettingzoo

# 给启动脚本执行权限
chmod +x start_sync_training.sh
```

### 2. 开始训练
```bash
# 最简单的启动方式
./start_sync_training.sh
```

## 📊 实时监控

### TensorBoard
训练开始后，自动启动TensorBoard：
- 访问地址: http://localhost:6006
- 关键指标:
  - `Losses/`: 各种损失曲线
  - `Rewards/`: 奖励趋势
  - `Sync/`: 同步训练指标

### 关键监控指标
1. **Sync/PolicyVersion**: 策略版本号（每次同步更新递增）
2. **Losses/Coordinator/Total**: 高层协调器总损失
3. **Losses/Discoverer/Total**: 低层发现器总损失
4. **Rewards/HighLevel/K_Step_Accumulated_Mean**: 高层累积奖励
5. **Buffer/SamplesCollected**: 已收集样本数

## 🔧 高级用法

### 从检查点继续训练
```bash
./start_sync_training.sh --model-path models/hmasd_sync_enhanced_320000.pth
```

### 调试模式
```bash
./start_sync_training.sh --log-level DEBUG
```

### 仅评估模式
```bash
./start_sync_training.sh --eval-only --model-path models/hmasd_sync_enhanced_final.pth
```

## 📁 文件结构

```
HMASD/
├── train_sync_enhanced.py      # 主训练脚本
├── start_sync_training.sh      # 启动脚本
├── config_1.py                 # 配置文件
├── hmasd/
│   ├── agent.py               # 同步Agent实现
│   ├── networks.py            # 网络结构
│   └── utils.py               # 工具函数
├── evaltools/
│   ├── __init__.py
│   └── eval_utils.py          # 评估工具
├── envs/pettingzoo/           # 环境定义
├── models/                    # 保存的模型
├── tf-logs/                   # TensorBoard日志
└── logs/                      # 训练日志
```

## ⚡ 同步训练特性

### 1. 严格On-Policy
- 收集1024个样本后统一更新
- 确保所有数据来自同一策略版本
- 避免异步训练的策略不一致问题

### 2. 高效并行
- 32个环境并行收集数据
- 每个环境平均贡献32个样本
- 最大化利用计算资源

### 3. 智能监控
- 实时样本计数和策略版本追踪
- 详细的环境贡献统计
- 性能指标自动记录

## 🎯 预期效果

### 训练稳定性
- 更平滑的损失曲线
- 减少策略震荡
- 更好的收敛性

### 性能提升
- 更高的样本效率
- 更快的收敛速度
- 更强的最终性能

## 🛠️ 故障排除

### 内存不足
```bash
# 减少batch size
# 在config_1.py中修改：
batch_size = 512  # 从1024减少到512
```

### 训练太慢
```bash
# 检查GPU使用情况
nvidia-smi

# 减少环境数量
# 在config_1.py中修改：
num_envs = 16  # 从32减少到16
```

### 无法访问TensorBoard
```bash
# 手动启动
tensorboard --logdir tf-logs --port 6006

# 或使用不同端口
tensorboard --logdir tf-logs --port 6007
```

## 📈 性能基准

### 硬件建议
- **GPU**: NVIDIA GTX 1060 或更好
- **内存**: 8GB RAM 或更多
- **存储**: 5GB 可用空间

### 训练时间预估
- **总步数**: 4M steps
- **更新频率**: 每1024步更新一次
- **预计时间**: 8-12小时（取决于硬件）

### 资源使用
- **GPU内存**: ~4GB
- **系统内存**: ~6GB
- **磁盘I/O**: 中等

## 📝 日志分析

### 关键日志信息
```
同步更新开始 - 收集了 1024 个样本，策略版本: 15
同步更新完成 - 策略版本更新到: 16, 已重置样本计数
训练进度: 32768/4000000 (0.8%), 更新次数: 32, Episodes: 156
```

### 性能指标解读
- **策略版本**: 递增表示正常更新
- **样本计数**: 应该在0-1024之间循环
- **Episodes**: 完成的episode数量
- **步数/秒**: 反映训练速度

## 🎉 成功标志

### 训练正常的迹象
1. 策略版本稳定递增
2. 损失曲线逐渐下降
3. 评估奖励持续提升
4. 无内存泄漏警告

### 预期训练曲线
- **初期(0-1M steps)**: 探索阶段，奖励波动较大
- **中期(1M-3M steps)**: 学习阶段，奖励稳步提升
- **后期(3M-4M steps)**: 收敛阶段，奖励趋于稳定

开始你的HMASD同步训练之旅吧！ 🚀
