#!/usr/bin/env python3
"""
HMASD多线程训练数据收集问题分析和修复方案
基于2025/6/14 23:38:29日志分析结果
"""

def analyze_collection_issue():
    """分析数据收集问题的根本原因"""
    
    print("🔍 HMASD多线程训练数据收集问题分析")
    print("=" * 60)
    
    # 问题1：配置不匹配
    print("\n📊 问题1：配置参数不匹配")
    print("-" * 30)
    actual_k = 32
    expected_high_level_exp_in_log = 256
    calculated_k_from_log = 4096 / expected_high_level_exp_in_log
    actual_expected = 4096 / actual_k
    
    print(f"• 配置中的k值: {actual_k}")
    print(f"• 日志中期望的高层经验数: {expected_high_level_exp_in_log}")
    print(f"• 日志暗示的k值: {calculated_k_from_log}")
    print(f"• 实际应期望的高层经验数: {actual_expected}")
    print(f"• 实际收集的高层经验数: 102")
    print(f"• 差异: {actual_expected - 102} (缺少 {((actual_expected - 102) / actual_expected * 100):.1f}%)")
    
    # 问题2：数据流路径异常
    print("\n🔄 问题2：数据流路径异常")
    print("-" * 30)
    print("• 预期路径: RolloutWorker.run_step() → 基于strict_step_counter % k == 0收集")
    print("• 实际路径: TrainingWorker → AgentProxy → store_high_level_transition")
    print("• 原因统计: {'多线程存储': 102, '严格步数计数': 0, 'Rollout结束强制收集': 0}")
    print("• 说明: RolloutWorker内部的高层经验收集逻辑完全失效")
    
    # 问题3：技能计时器异常
    print("\n⏰ 问题3：技能计时器状态异常")
    print("-" * 30)
    print("• 所有32个Worker的技能计时器都是0")
    print("• 累积奖励都是0.0")
    print("• 说明: 技能周期管理逻辑可能存在问题")
    
    # 问题4：连锁反应
    print("\n⛓️ 问题4：连锁反应分析")
    print("-" * 30)
    print("• 高层经验不足 → should_update()返回False")
    print("• 模型无法更新 → PPO缓冲区无法清空")
    print("• B_l达到10000(满) → DataBuffer阻塞")
    print("• RolloutWorker无法put数据 → 数据收集停止")
    
    return {
        'config_k': actual_k,
        'expected_high_level': actual_expected,
        'actual_high_level': 102,
        'missing_high_level': actual_expected - 102,
        'missing_percentage': (actual_expected - 102) / actual_expected * 100
    }

def propose_fix_strategies():
    """提出修复策略"""
    
    print("\n🛠️ 修复策略")
    print("=" * 60)
    
    print("\n1️⃣ 立即修复：配置验证和同步")
    print("-" * 30)
    print("• 修正HMASDAgent中的高层经验目标计算逻辑")
    print("• 确保使用实际的config.k值而不是硬编码的16")
    print("• 添加配置验证，确保所有组件使用一致的参数")
    
    print("\n2️⃣ 核心修复：RolloutWorker高层经验收集")
    print("-" * 30)
    print("• 修复RolloutWorker.run_step()中的条件判断逻辑")
    print("• 确保strict_step_counter正确递增和重置")
    print("• 添加调试日志跟踪技能周期状态")
    
    print("\n3️⃣ 备选修复：简化数据流")
    print("-" * 30)
    print("• 完全依赖TrainingWorker路径存储经验")
    print("• 在AgentProxy.store_experience中补充高层经验收集逻辑")
    print("• 基于低层经验的步数统计推算高层经验")
    
    print("\n4️⃣ 监控增强：数据验证和告警")
    print("-" * 30)
    print("• 添加实时数据收集监控")
    print("• 高层/低层经验比例验证")
    print("• 自动数据修复机制")
    
    return [
        "config_validation",
        "rollout_worker_fix", 
        "data_flow_simplification",
        "monitoring_enhancement"
    ]

def generate_specific_fixes():
    """生成具体的修复代码建议"""
    
    print("\n💻 具体修复代码建议")
    print("=" * 60)
    
    print("\n修复1：HMASDAgent.rollout_update()中的目标计算")
    print("```python")
    print("# 修复前（错误）：")
    print("# target_high_level = self.rollout_length * self.num_parallel_envs // 16")
    print("")
    print("# 修复后（正确）：")
    print("target_high_level = self.rollout_length * self.num_parallel_envs // self.config.k")
    print("main_logger.info(f'高层经验目标: {target_high_level} (基于k={self.config.k})')")
    print("```")
    
    print("\n修复2：RolloutWorker.run_step()中的高层经验收集")
    print("```python")
    print("# 在run_step方法中添加调试信息：")
    print("if self.strict_step_counter % self.config.k == 0 and self.strict_step_counter > 0:")
    print("    self.logger.debug(f'Worker {self.worker_id}: 触发高层经验收集 - '")
    print("                     f'步数计数器={self.strict_step_counter}, k={self.config.k}')")
    print("    self.store_high_level_experience('严格步数计数')")
    print("```")
    
    print("\n修复3：AgentProxy.should_update()中的条件验证")
    print("```python")
    print("# 使用正确的k值计算预期高层经验数")
    print("expected_high_level_exp = total_collected // self.config.k")
    print("high_level_sufficient = total_high_level_exp >= expected_high_level_exp * 0.9")
    print("```")
    
    print("\n修复4：添加配置一致性检查")
    print("```python")
    print("def validate_training_config(config, agent):")
    print("    assert agent.config.k == config.k, f'k值不一致: agent={agent.config.k}, config={config.k}'")
    print("    expected_target = config.rollout_length * config.num_parallel_envs // config.k")
    print("    assert config.high_level_batch_size >= expected_target, '高层缓冲区大小不足'")
    print("```")

def main():
    """主函数"""
    print("HMASD多线程训练问题诊断报告")
    print("生成时间: 2025-06-14 23:42:00")
    print("基于日志: threaded_rollout_training_20250614_233414")
    
    # 分析问题
    analysis_result = analyze_collection_issue()
    
    # 提出策略
    strategies = propose_fix_strategies()
    
    # 生成具体修复方案
    generate_specific_fixes()
    
    print("\n🎯 总结")
    print("=" * 60)
    print("• 核心问题: RolloutWorker高层经验收集逻辑失效")
    print("• 直接原因: 配置参数不匹配 + 条件判断错误")
    print("• 连锁效应: 缓冲区阻塞 → 训练停滞")
    print("• 修复优先级: 1️⃣配置修正 → 2️⃣收集逻辑 → 3️⃣监控增强")
    print(f"• 预期修复效果: 高层经验从102提升到{analysis_result['expected_high_level']}")

if __name__ == "__main__":
    main()
