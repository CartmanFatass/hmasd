# HMASD 增强训练与论文数据收集系统

## 概述

本系统为您的HMASD（Hierarchical Multi-Agent Skill Discovery）项目提供增强的训练数据收集和分析功能，专门针对学术论文的需求设计。

## 核心功能

### 1. 增强的奖励数据收集
- **训练过程中的详细记录**：实时收集每个训练步骤的奖励信息
- **奖励组成分析**：分别记录环境奖励、团队判别器奖励、个体判别器奖励
- **多环境并行支持**：支持多个并行环境的独立数据收集
- **实时统计**：滑动窗口统计、方差分析、收敛性检测

### 2. 技能使用追踪
- **技能多样性监控**：实时计算技能使用的多样性指标
- **技能切换频率**：记录技能切换的时机和频率
- **技能分布分析**：统计各技能的使用频率和分布
- **均匀性评估**：评估技能使用的均匀程度

### 3. 论文级别的数据导出
- **CSV数据导出**：结构化的数据，便于进一步分析
- **高质量图表**：符合论文标准的可视化图表
- **统计报告**：详细的训练摘要和性能指标
- **JSON格式数据**：便于程序化处理的技能使用数据

## 文件结构

```
├── train_enhanced_reward_tracking.py  # 增强的训练脚本
├── paper_data_analysis.py            # 论文数据分析工具
├── ENHANCED_TRAINING_README.md       # 本说明文档
├── config_1.py                       # 原有配置文件
└── hmasd/agent.py                    # 原有代理文件
```

## 使用方法

### 1. 增强训练

#### 基本训练命令
```bash
python train_enhanced_reward_tracking.py --mode train --detailed_logging
```

#### 完整参数训练
```bash
python train_enhanced_reward_tracking.py \
    --mode train \
    --scenario 2 \
    --n_uavs 5 \
    --n_users 50 \
    --num_envs 32 \
    --detailed_logging \
    --export_interval 500 \
    --log_level info \
    --console_log_level error \
    --device auto
```

#### 关键参数说明
- `--detailed_logging`: 启用详细的奖励组成记录
- `--export_interval`: 数据导出间隔（推荐500-1000步）
- `--num_envs`: 并行环境数量，影响数据收集效率
- `--scenario`: 场景选择（1=基站模式，2=协作组网模式）

### 2. 数据分析

#### 基本分析命令
```bash
python paper_data_analysis.py logs/enhanced_tracking_YYYYMMDD-HHMMSS/
```

#### 指定输出目录
```bash
python paper_data_analysis.py logs/enhanced_tracking_YYYYMMDD-HHMMSS/ \
    --output_dir paper_figures/
```

### 3. 训练数据目录结构

训练完成后，会在日志目录下生成以下结构：
```
logs/enhanced_tracking_YYYYMMDD-HHMMSS/
├── paper_data/                        # 论文数据目录
│   ├── episode_rewards_step_*.csv      # Episode奖励数据
│   ├── reward_components_step_*.csv    # 奖励组成数据
│   ├── skill_usage_step_*.json         # 技能使用统计
│   └── training_progress_step_*.png    # 训练过程图表
├── training_summary.json              # 训练摘要
└── events.out.tfevents.*              # TensorBoard日志
```

## 输出文件说明

### 1. 数据文件

#### episode_rewards_step_*.csv
包含每个episode的详细信息：
- `episode`: Episode编号
- `env_id`: 环境ID
- `total_reward`: 总奖励
- `episode_length`: Episode长度
- `timestamp`: 时间戳
- 其他性能指标（服务用户数、覆盖率等）

#### reward_components_step_*.csv
包含奖励组成的详细分解：
- `step`: 训练步数
- `env_id`: 环境ID
- `component`: 奖励组成部分（env_component、team_disc_component、ind_disc_component）
- `value`: 组成部分的数值

#### skill_usage_step_*.json
技能使用的统计信息：
```json
{
  "team_skills": {"0": 150, "1": 200, "2": 180},
  "skill_switches": 85,
  "total_steps": 10000
}
```

### 2. 分析图表

#### learning_curves.png
- 学习曲线（原始+滑动平均）
- 奖励分布直方图
- Episode长度趋势
- 奖励稳定性分析

#### reward_composition_analysis.png
- 奖励组成随时间变化
- 组成部分比例饼图
- 组成部分分布箱线图
- 组成部分相关性热图

#### skill_analysis.png
- 团队技能使用分布
- 技能切换频率
- 技能多样性趋势
- 技能使用均匀性分析

#### performance_analysis.png
- 训练阶段性能对比
- 收敛性分析
- 关键性能指标
- 训练效率分析

### 3. 摘要报告

#### paper_summary_report.txt
包含以下内容：
1. **训练摘要**：总episode数、总步数、技能切换次数等
2. **学习性能分析**：早期vs后期性能、改进幅度、稳定性
3. **技能使用分析**：技能分布、均匀性、切换频率
4. **奖励组成分析**：各组成部分的平均值和比例
5. **论文建议**：推荐的图表和关键指标

## 论文中的使用建议

### 1. 关键图表
- **Figure 1**: 学习曲线图（learning_curves.png的第一个子图）
- **Figure 2**: 奖励组成分析（reward_composition_analysis.png）
- **Figure 3**: 技能使用分析（skill_analysis.png）
- **Table 1**: 性能对比表（从performance_analysis.png和摘要报告中提取）

### 2. 关键指标
从摘要报告中提取以下指标：
- **最终性能**: `Final performance: X.XX ± Y.YY`
- **学习改进**: `Learning improvement: Z.ZZ (W.W%)`
- **技能多样性指数**: `Skill diversity index: 0.XXX`
- **训练稳定性**: `Training stability (CV): 0.XXX`

### 3. 统计显著性
建议进行多次独立训练运行，收集以下统计：
- 多次运行的平均性能和标准差
- 收敛时间的统计分布
- 技能多样性的一致性

## 与原有代码的兼容性

### 1. 代理增强
增强的训练脚本使用您现有的`HMASDAgent`类，并添加了：
- 更详细的奖励组成记录
- 环境级别的状态追踪
- 增强的TensorBoard日志

### 2. 配置兼容
完全兼容您的`config_1.py`配置文件，所有超参数保持不变。

### 3. 环境兼容
支持您现有的环境实现（scenario1和scenario2）。

## 故障排除

### 1. 常见问题

#### 内存不足
如果遇到内存问题，可以：
- 减少`--num_envs`参数
- 增加`--export_interval`参数
- 关闭`--detailed_logging`

#### 数据导出失败
检查：
- 磁盘空间是否充足
- 日志目录写权限
- pandas和matplotlib版本兼容性

#### TensorBoard显示问题
- 确保TensorBoard版本 >= 2.0
- 检查日志文件路径
- 尝试刷新浏览器

### 2. 性能优化

#### 加速训练
- 使用GPU：`--device cuda`
- 增加并行环境：`--num_envs 64`
- 减少详细日志：移除`--detailed_logging`

#### 提高数据质量
- 增加导出频率：`--export_interval 200`
- 启用详细记录：`--detailed_logging`
- 延长训练时间：修改`config_1.py`中的`total_timesteps`

## 扩展功能

### 1. 自定义分析
您可以扩展`PaperDataAnalyzer`类来添加：
- 特定的性能指标
- 自定义的可视化
- 与基线方法的对比

### 2. 实时监控
可以结合TensorBoard实现实时监控：
```bash
tensorboard --logdir logs/enhanced_tracking_YYYYMMDD-HHMMSS/
```

### 3. 批量实验
可以编写脚本进行批量实验：
```bash
#!/bin/bash
for seed in {1..5}; do
    python train_enhanced_reward_tracking.py \
        --mode train \
        --detailed_logging \
        --log_dir logs/experiment_seed_$seed
done
```

## 技术支持

如果您在使用过程中遇到问题，请检查：

1. **Python环境**：确保所有依赖包已安装
2. **日志文件**：查看详细的错误信息
3. **配置参数**：验证所有参数设置正确
4. **系统资源**：确保有足够的内存和磁盘空间

## 总结

这个增强的训练和分析系统为您提供了：
- 详细的训练过程数据收集
- 专业级的数据分析和可视化
- 符合学术论文标准的图表和报告
- 与现有代码的完全兼容

通过使用这些工具，您可以更好地分析HMASD算法的性能，并为学术论文提供高质量的实验数据和可视化结果。
