import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path
import argparse
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class PaperDataAnalyzer:
    """论文数据分析器，用于生成学术论文所需的图表和统计"""
    
    def __init__(self, data_dir, output_dir=None):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir) if output_dir else self.data_dir / 'analysis_output'
        self.output_dir.mkdir(exist_ok=True)
        
        # 设置matplotlib中文支持和样式
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # 论文质量的图表设置
        self.fig_size = (12, 8)
        self.dpi = 300
        self.font_size = 12
        
        plt.rcParams.update({
            'font.size': self.font_size,
            'axes.titlesize': self.font_size + 2,
            'axes.labelsize': self.font_size,
            'xtick.labelsize': self.font_size - 1,
            'ytick.labelsize': self.font_size - 1,
            'legend.fontsize': self.font_size - 1,
            'figure.titlesize': self.font_size + 4
        })
        
    def load_episode_rewards(self):
        """加载episode奖励数据"""
        reward_files = list(self.data_dir.glob('**/episode_rewards_step_*.csv'))
        if not reward_files:
            print("未找到episode奖励数据文件")
            return None
        
        # 加载最新的文件
        latest_file = max(reward_files, key=lambda x: x.stat().st_mtime)
        print(f"加载episode奖励数据: {latest_file}")
        
        df = pd.read_csv(latest_file)
        return df
    
    def load_reward_components(self):
        """加载奖励组成数据"""
        component_files = list(self.data_dir.glob('**/reward_components_step_*.csv'))
        if not component_files:
            print("未找到奖励组成数据文件")
            return None
        
        # 加载最新的文件
        latest_file = max(component_files, key=lambda x: x.stat().st_mtime)
        print(f"加载奖励组成数据: {latest_file}")
        
        df = pd.read_csv(latest_file)
        return df
    
    def load_skill_usage(self):
        """加载技能使用数据"""
        skill_files = list(self.data_dir.glob('**/skill_usage_step_*.json'))
        if not skill_files:
            print("未找到技能使用数据文件")
            return None
        
        # 加载最新的文件
        latest_file = max(skill_files, key=lambda x: x.stat().st_mtime)
        print(f"加载技能使用数据: {latest_file}")
        
        with open(latest_file, 'r') as f:
            data = json.load(f)
        return data
    
    def load_training_summary(self):
        """加载训练摘要数据"""
        summary_files = list(self.data_dir.glob('**/training_summary.json'))
        if not summary_files:
            print("未找到训练摘要数据文件")
            return None
        
        latest_file = max(summary_files, key=lambda x: x.stat().st_mtime)
        print(f"加载训练摘要数据: {latest_file}")
        
        with open(latest_file, 'r') as f:
            data = json.load(f)
        return data
    
    def generate_learning_curves(self, episode_df):
        """生成学习曲线图（论文Figure 1类型）"""
        if episode_df is None or episode_df.empty:
            print("无法生成学习曲线：数据为空")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Training Performance Analysis', fontsize=16, fontweight='bold')
        
        # 1. Episode奖励随时间变化
        ax1 = axes[0, 0]
        episodes = episode_df['episode'].values
        rewards = episode_df['total_reward'].values
        
        # 原始曲线
        ax1.plot(episodes, rewards, alpha=0.3, color='lightblue', linewidth=0.5, label='Raw rewards')
        
        # 滑动平均
        window_sizes = [50, 100]
        colors = ['red', 'darkred']
        for window, color in zip(window_sizes, colors):
            if len(rewards) >= window:
                smoothed = pd.Series(rewards).rolling(window=window, center=True).mean()
                ax1.plot(episodes, smoothed, color=color, linewidth=2, 
                        label=f'{window}-episode moving average')
        
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Total Reward')
        ax1.set_title('Learning Curve: Episode Rewards')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 奖励分布直方图
        ax2 = axes[0, 1]
        ax2.hist(rewards, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        ax2.axvline(np.mean(rewards), color='red', linestyle='--', linewidth=2, 
                   label=f'Mean: {np.mean(rewards):.2f}')
        ax2.axvline(np.median(rewards), color='orange', linestyle='--', linewidth=2, 
                   label=f'Median: {np.median(rewards):.2f}')
        ax2.set_xlabel('Total Reward')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Reward Distribution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Episode长度趋势
        ax3 = axes[1, 0]
        if 'episode_length' in episode_df.columns:
            lengths = episode_df['episode_length'].values
            ax3.plot(episodes, lengths, alpha=0.6, color='green', linewidth=1)
            
            # 滑动平均
            if len(lengths) >= 50:
                smoothed_lengths = pd.Series(lengths).rolling(window=50, center=True).mean()
                ax3.plot(episodes, smoothed_lengths, color='darkgreen', linewidth=2, 
                        label='50-episode moving average')
            
            ax3.set_xlabel('Episode')
            ax3.set_ylabel('Episode Length')
            ax3.set_title('Episode Length Progression')
            ax3.legend()
        else:
            ax3.text(0.5, 0.5, 'Episode length data\nnot available', 
                    ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('Episode Length Progression')
        ax3.grid(True, alpha=0.3)
        
        # 4. 奖励稳定性分析（方差随时间变化）
        ax4 = axes[1, 1]
        window_size = min(100, len(rewards) // 10)
        if window_size >= 10:
            rolling_mean = pd.Series(rewards).rolling(window=window_size).mean()
            rolling_std = pd.Series(rewards).rolling(window=window_size).std()
            
            valid_indices = ~(rolling_mean.isna() | rolling_std.isna())
            valid_episodes = episodes[valid_indices]
            valid_mean = rolling_mean[valid_indices]
            valid_std = rolling_std[valid_indices]
            
            ax4.fill_between(valid_episodes, valid_mean - valid_std, valid_mean + valid_std, 
                           alpha=0.3, color='purple', label='±1 std')
            ax4.plot(valid_episodes, valid_mean, color='purple', linewidth=2, label='Mean')
            ax4.set_xlabel('Episode')
            ax4.set_ylabel('Reward')
            ax4.set_title(f'Reward Stability ({window_size}-episode window)')
            ax4.legend()
        else:
            ax4.text(0.5, 0.5, 'Insufficient data\nfor stability analysis', 
                    ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Reward Stability Analysis')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_path = self.output_dir / 'learning_curves.png'
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        print(f"学习曲线已保存到: {output_path}")
        
        return str(output_path)
    
    def generate_reward_composition_analysis(self, component_df):
        """生成奖励组成分析图（论文Figure 2类型）"""
        if component_df is None or component_df.empty:
            print("无法生成奖励组成分析：数据为空")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Reward Component Analysis', fontsize=16, fontweight='bold')
        
        # 1. 各组成部分随时间变化
        ax1 = axes[0, 0]
        components = component_df['component'].unique()
        colors = ['blue', 'red', 'green', 'orange', 'purple']
        
        for i, comp in enumerate(components):
            comp_data = component_df[component_df['component'] == comp]
            if not comp_data.empty:
                steps = comp_data['step'].values
                values = comp_data['value'].values
                
                # 滑动平均
                if len(values) >= 50:
                    smoothed = pd.Series(values).rolling(window=50, center=True).mean()
                    ax1.plot(steps, smoothed, color=colors[i % len(colors)], 
                            linewidth=2, label=comp.replace('_', ' ').title())
                else:
                    ax1.plot(steps, values, color=colors[i % len(colors)], 
                            alpha=0.7, label=comp.replace('_', ' ').title())
        
        ax1.set_xlabel('Training Step')
        ax1.set_ylabel('Component Value')
        ax1.set_title('Reward Components Over Time')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 组成部分比例饼图
        ax2 = axes[0, 1]
        component_means = component_df.groupby('component')['value'].mean()
        # 只显示正值组成部分
        positive_components = component_means[component_means > 0]
        
        if not positive_components.empty:
            labels = [comp.replace('_', ' ').title() for comp in positive_components.index]
            ax2.pie(positive_components.values, labels=labels, autopct='%1.1f%%', startangle=90)
            ax2.set_title('Average Reward Component Proportion')
        else:
            ax2.text(0.5, 0.5, 'No positive reward\ncomponents found', 
                    ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('Average Reward Component Proportion')
        
        # 3. 组成部分分布箱线图
        ax3 = axes[1, 0]
        component_values = []
        component_labels = []
        for comp in components:
            comp_data = component_df[component_df['component'] == comp]
            if not comp_data.empty:
                component_values.append(comp_data['value'].values)
                component_labels.append(comp.replace('_', ' ').title())
        
        if component_values:
            ax3.boxplot(component_values, labels=component_labels)
            ax3.set_ylabel('Component Value')
            ax3.set_title('Reward Component Distribution')
            ax3.tick_params(axis='x', rotation=45)
        else:
            ax3.text(0.5, 0.5, 'No component data\navailable', 
                    ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('Reward Component Distribution')
        ax3.grid(True, alpha=0.3)
        
        # 4. 组成部分相关性热图
        ax4 = axes[1, 1]
        if len(components) > 1:
            # 创建组成部分的数据透视表
            pivot_df = component_df.pivot_table(index='step', columns='component', values='value', fill_value=0)
            
            if pivot_df.shape[1] > 1:
                correlation_matrix = pivot_df.corr()
                sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                           square=True, ax=ax4, cbar_kws={'shrink': 0.8})
                ax4.set_title('Reward Component Correlations')
            else:
                ax4.text(0.5, 0.5, 'Insufficient data\nfor correlation analysis', 
                        ha='center', va='center', transform=ax4.transAxes)
                ax4.set_title('Reward Component Correlations')
        else:
            ax4.text(0.5, 0.5, 'Need at least 2 components\nfor correlation analysis', 
                    ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Reward Component Correlations')
        
        plt.tight_layout()
        output_path = self.output_dir / 'reward_composition_analysis.png'
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        print(f"奖励组成分析已保存到: {output_path}")
        
        return str(output_path)
    
    def generate_skill_analysis(self, skill_data, episode_df):
        """生成技能使用分析图（论文Figure 3类型）"""
        if skill_data is None:
            print("无法生成技能分析：数据为空")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Skill Usage Analysis', fontsize=16, fontweight='bold')
        
        # 1. 团队技能使用分布
        ax1 = axes[0, 0]
        if 'team_skills' in skill_data and skill_data['team_skills']:
            team_skills = skill_data['team_skills']
            skills = list(team_skills.keys())
            counts = list(team_skills.values())
            
            bars = ax1.bar(skills, counts, alpha=0.7, color='skyblue', edgecolor='black')
            ax1.set_xlabel('Team Skill ID')
            ax1.set_ylabel('Usage Count')
            ax1.set_title('Team Skill Usage Distribution')
            
            # 添加数值标签
            for bar, count in zip(bars, counts):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + max(counts)*0.01,
                        f'{count}', ha='center', va='bottom')
        else:
            ax1.text(0.5, 0.5, 'No team skill\ndata available', 
                    ha='center', va='center', transform=ax1.transAxes)
            ax1.set_title('Team Skill Usage Distribution')
        ax1.grid(True, alpha=0.3)
        
        # 2. 技能切换频率
        ax2 = axes[0, 1]
        if 'skill_switches' in skill_data and 'total_steps' in skill_data:
            total_switches = skill_data['skill_switches']
            total_steps = skill_data['total_steps']
            switch_rate = total_switches / total_steps * 1000  # 每1000步的切换次数
            
            ax2.bar(['Skill Switches'], [switch_rate], alpha=0.7, color='orange', edgecolor='black')
            ax2.set_ylabel('Switches per 1000 Steps')
            ax2.set_title('Skill Switch Frequency')
            ax2.text(0, switch_rate + switch_rate*0.05, f'{switch_rate:.2f}', 
                    ha='center', va='bottom', fontweight='bold')
        else:
            ax2.text(0.5, 0.5, 'No skill switch\ndata available', 
                    ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('Skill Switch Frequency')
        ax2.grid(True, alpha=0.3)
        
        # 3. 技能多样性随时间变化（如果有episode数据）
        ax3 = axes[1, 0]
        if episode_df is not None and 'team_skill' in episode_df.columns:
            # 计算滑动窗口内的技能多样性
            episodes = episode_df['episode'].values
            window_size = min(50, len(episodes) // 10)
            
            if window_size >= 5:
                diversity_scores = []
                episode_points = []
                
                for i in range(window_size, len(episodes)):
                    window_skills = episode_df['team_skill'].iloc[i-window_size:i]
                    unique_skills = len(window_skills.unique())
                    diversity = unique_skills / len(window_skills) if len(window_skills) > 0 else 0
                    diversity_scores.append(diversity)
                    episode_points.append(episodes[i])
                
                ax3.plot(episode_points, diversity_scores, color='purple', linewidth=2)
                ax3.set_xlabel('Episode')
                ax3.set_ylabel('Skill Diversity')
                ax3.set_title(f'Skill Diversity Over Time ({window_size}-episode window)')
                ax3.set_ylim(0, 1)
            else:
                ax3.text(0.5, 0.5, 'Insufficient data\nfor diversity analysis', 
                        ha='center', va='center', transform=ax3.transAxes)
                ax3.set_title('Skill Diversity Over Time')
        else:
            ax3.text(0.5, 0.5, 'No episode skill\ndata available', 
                    ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('Skill Diversity Over Time')
        ax3.grid(True, alpha=0.3)
        
        # 4. 技能使用均匀性分析
        ax4 = axes[1, 1]
        if 'team_skills' in skill_data and skill_data['team_skills']:
            team_skills = skill_data['team_skills']
            counts = np.array(list(team_skills.values()))
            
            # 计算均匀性指标
            total_usage = np.sum(counts)
            probabilities = counts / total_usage
            entropy = -np.sum(probabilities * np.log(probabilities + 1e-8))
            max_entropy = np.log(len(counts))
            uniformity = entropy / max_entropy if max_entropy > 0 else 0
            
            # 可视化
            expected_uniform = total_usage / len(counts)
            skills = list(team_skills.keys())
            
            x_pos = np.arange(len(skills))
            bars = ax4.bar(x_pos, counts, alpha=0.7, color='lightcoral', edgecolor='black', 
                          label='Actual Usage')
            ax4.axhline(y=expected_uniform, color='blue', linestyle='--', linewidth=2, 
                       label=f'Uniform Distribution ({expected_uniform:.0f})')
            
            ax4.set_xlabel('Team Skill ID')
            ax4.set_ylabel('Usage Count')
            ax4.set_title(f'Skill Usage Uniformity (Score: {uniformity:.3f})')
            ax4.set_xticks(x_pos)
            ax4.set_xticklabels(skills)
            ax4.legend()
            
            # 添加均匀性得分说明
            ax4.text(0.02, 0.98, f'Uniformity Score: {uniformity:.3f}\n(1.0 = perfectly uniform)', 
                    transform=ax4.transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        else:
            ax4.text(0.5, 0.5, 'No team skill\ndata available', 
                    ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Skill Usage Uniformity')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_path = self.output_dir / 'skill_analysis.png'
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        print(f"技能分析已保存到: {output_path}")
        
        return str(output_path)
    
    def generate_performance_comparison(self, episode_df):
        """生成性能对比图（论文Table/Figure类型）"""
        if episode_df is None or episode_df.empty:
            print("无法生成性能对比：数据为空")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Performance Analysis and Metrics', fontsize=16, fontweight='bold')
        
        # 1. 训练阶段性能对比
        ax1 = axes[0, 0]
        episodes = episode_df['episode'].values
        rewards = episode_df['total_reward'].values
        
        # 将训练分成几个阶段
        num_phases = 4
        phase_size = len(episodes) // num_phases
        
        phase_stats = []
        phase_labels = []
        
        for i in range(num_phases):
            start_idx = i * phase_size
            end_idx = (i + 1) * phase_size if i < num_phases - 1 else len(episodes)
            
            phase_rewards = rewards[start_idx:end_idx]
            if len(phase_rewards) > 0:
                phase_stats.append({
                    'mean': np.mean(phase_rewards),
                    'std': np.std(phase_rewards),
                    'min': np.min(phase_rewards),
                    'max': np.max(phase_rewards)
                })
                phase_labels.append(f'Phase {i+1}\n({episodes[start_idx]}-{episodes[end_idx-1]})')
        
        if phase_stats:
            means = [stat['mean'] for stat in phase_stats]
            stds = [stat['std'] for stat in phase_stats]
            
            bars = ax1.bar(range(len(phase_labels)), means, yerr=stds, 
                          alpha=0.7, color='lightgreen', edgecolor='black', capsize=5)
            ax1.set_xlabel('Training Phase')
            ax1.set_ylabel('Average Reward')
            ax1.set_title('Performance Across Training Phases')
            ax1.set_xticks(range(len(phase_labels)))
            ax1.set_xticklabels(phase_labels)
            
            # 添加数值标签
            for bar, mean, std in zip(bars, means, stds):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + std + max(means)*0.02,
                        f'{mean:.1f}±{std:.1f}', ha='center', va='bottom', fontweight='bold')
        else:
            ax1.text(0.5, 0.5, 'Insufficient data\nfor phase analysis', 
                    ha='center', va='center', transform=ax1.transAxes)
            ax1.set_title('Performance Across Training Phases')
        ax1.grid(True, alpha=0.3)
        
        # 2. 收敛性分析
        ax2 = axes[0, 1]
        if len(rewards) >= 100:
            # 计算收敛指标
            window_size = min(100, len(rewards) // 5)
            rolling_mean = pd.Series(rewards).rolling(window=window_size).mean()
            rolling_std = pd.Series(rewards).rolling(window=window_size).std()
            
            # 找到收敛点（标准差稳定的点）
            stable_threshold = np.std(rewards) * 0.1  # 10%的总体标准差作为稳定阈值
            
            convergence_episodes = []
            convergence_means = []
            
            for i in range(window_size, len(rolling_std)):
                if rolling_std.iloc[i] < stable_threshold:
                    convergence_episodes.append(episodes[i])
                    convergence_means.append(rolling_mean.iloc[i])
            
            ax2.plot(episodes, rewards, alpha=0.3, color='lightblue', linewidth=0.5, label='Raw rewards')
            ax2.plot(episodes, rolling_mean, color='red', linewidth=2, label=f'{window_size}-episode MA')
            
            if convergence_episodes:
                ax2.scatter(convergence_episodes, convergence_means, color='green', 
                           s=30, alpha=0.7, label='Convergence points')
                first_convergence = convergence_episodes[0]
                ax2.axvline(first_convergence, color='green', linestyle='--', 
                           label=f'First convergence: Episode {first_convergence}')
            
            ax2.set_xlabel('Episode')
            ax2.set_ylabel('Reward')
            ax2.set_title('Convergence Analysis')
            ax2.legend()
        else:
            ax2.text(0.5, 0.5, 'Insufficient data\nfor convergence analysis', 
                    ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('Convergence Analysis')
        ax2.grid(True, alpha=0.3)
        
        # 3. 性能指标总结
        ax3 = axes[1, 0]
        
        # 计算关键性能指标
        final_performance = np.mean(rewards[-100:]) if len(rewards) >= 100 else np.mean(rewards)
        initial_performance = np.mean(rewards[:100]) if len(rewards) >= 100 else np.mean(rewards)
        improvement = final_performance - initial_performance
        
        best_performance = np.max(rewards)
        worst_performance = np.min(rewards)
        stability = 1.0 / (1.0 + np.std(rewards[-100:]) / np.abs(np.mean(rewards[-100:]))) if len(rewards) >= 100 else 0.5
        
        metrics = {
            'Final Performance': final_performance,
            'Best Performance': best_performance,
            'Improvement': improvement,
            'Stability Score': stability
        }
        
        metric_names = list(metrics.keys())
        metric_values = list(metrics.values())
        
        bars = ax3.barh(metric_names, metric_values, alpha=0.7, color=['blue', 'green', 'orange', 'purple'])
        ax3.set_xlabel('Value')
        ax3.set_title('Key Performance Metrics')
        
        # 添加数值标签
        for i, (bar, value) in enumerate(zip(bars, metric_values)):
            ax3.text(value + max(metric_values)*0.01, bar.get_y() + bar.get_height()/2.,
                    f'{value:.2f}', ha='left', va='center', fontweight='bold')
        
        ax3.grid(True, alpha=0.3)
        
        # 4. 训练效率分析
        ax4 = axes[1, 1]
        if 'episode_length' in episode_df.columns:
            lengths = episode_df['episode_length'].values
            
            # 计算效率指标
            avg_length = np.mean(lengths)
            total_steps = np.sum(lengths)
            episodes_per_1k_steps = 1000 / avg_length if avg_length > 0 else 0
            
            efficiency_data = {
                'Avg Episode Length': avg_length,
                'Total Training Steps': total_steps,
                'Episodes per 1K Steps': episodes_per_1k_steps
            }
            
            # 显示为文本信息
            info_text = '\n'.join([f'{key}: {value:.1f}' for key, value in efficiency_data.items()])
            ax4.text(0.5, 0.5, f'Training Efficiency Metrics:\n\n{info_text}', 
                    ha='center', va='center', transform=ax4.transAxes,
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8),
                    fontsize=12, fontweight='bold')
            ax4.set_title('Training Efficiency')
        else:
            ax4.text(0.5, 0.5, 'Episode length data\nnot available', 
                    ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Training Efficiency')
        
        ax4.set_xticks([])
        ax4.set_yticks([])
        
        plt.tight_layout()
        output_path = self.output_dir / 'performance_analysis.png'
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        print(f"性能分析已保存到: {output_path}")
        
        return str(output_path)
    
    def generate_summary_report(self, episode_df, component_df, skill_data, summary_data):
        """生成论文摘要报告"""
        report_path = self.output_dir / 'paper_summary_report.txt'
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("HMASD Training Analysis Report\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # 1. 训练摘要
            f.write("1. TRAINING SUMMARY\n")
            f.write("-" * 20 + "\n")
            if summary_data:
                f.write(f"Total Episodes: {summary_data.get('total_episodes', 'N/A')}\n")
                f.write(f"Total Steps: {summary_data.get('total_steps', 'N/A')}\n")
                f.write(f"Skill Switches: {summary_data.get('skill_switches', 'N/A')}\n")
                f.write(f"Final Mean Reward: {summary_data.get('reward_mean', 0):.2f} ± {summary_data.get('reward_std', 0):.2f}\n")
                f.write(f"Best Reward: {summary_data.get('reward_max', 'N/A')}\n")
                f.write(f"Worst Reward: {summary_data.get('reward_min', 'N/A')}\n")
            f.write("\n")
            
            # 2. 学习性能分析
            f.write("2. LEARNING PERFORMANCE\n")
            f.write("-" * 25 + "\n")
            if episode_df is not None and not episode_df.empty:
                rewards = episode_df['total_reward'].values
                
                # 计算学习进步
                if len(rewards) >= 200:
                    early_performance = np.mean(rewards[:100])
                    late_performance = np.mean(rewards[-100:])
                    improvement = late_performance - early_performance
                    improvement_pct = (improvement / abs(early_performance)) * 100 if early_performance != 0 else 0
                    
                    f.write(f"Early Training Performance (first 100 episodes): {early_performance:.2f}\n")
                    f.write(f"Late Training Performance (last 100 episodes): {late_performance:.2f}\n")
                    f.write(f"Absolute Improvement: {improvement:.2f}\n")
                    f.write(f"Relative Improvement: {improvement_pct:.1f}%\n")
                
                # 计算稳定性
                if len(rewards) >= 100:
                    recent_std = np.std(rewards[-100:])
                    recent_mean = np.mean(rewards[-100:])
                    cv = recent_std / abs(recent_mean) if recent_mean != 0 else float('inf')
                    f.write(f"Training Stability (CV of last 100 episodes): {cv:.3f}\n")
                
                f.write(f"Overall Mean Reward: {np.mean(rewards):.2f}\n")
                f.write(f"Overall Std Reward: {np.std(rewards):.2f}\n")
            f.write("\n")
            
            # 3. 技能使用分析
            f.write("3. SKILL USAGE ANALYSIS\n")
            f.write("-" * 25 + "\n")
            if skill_data:
                if 'team_skills' in skill_data:
                    f.write("Team Skill Distribution:\n")
                    team_skills = skill_data['team_skills']
                    total_usage = sum(team_skills.values())
                    for skill_id, count in team_skills.items():
                        percentage = (count / total_usage) * 100 if total_usage > 0 else 0
                        f.write(f"  Skill {skill_id}: {count} times ({percentage:.1f}%)\n")
                    
                    # 计算技能使用均匀性
                    if len(team_skills) > 1:
                        counts = np.array(list(team_skills.values()))
                        probabilities = counts / total_usage
                        entropy = -np.sum(probabilities * np.log(probabilities + 1e-8))
                        max_entropy = np.log(len(counts))
                        uniformity = entropy / max_entropy
                        f.write(f"Skill Usage Uniformity: {uniformity:.3f} (1.0 = perfectly uniform)\n")
                
                if 'skill_switches' in skill_data and 'total_steps' in skill_data:
                    switch_rate = skill_data['skill_switches'] / skill_data['total_steps'] * 1000
                    f.write(f"Skill Switch Rate: {switch_rate:.2f} switches per 1000 steps\n")
            f.write("\n")
            
            # 4. 奖励组成分析
            f.write("4. REWARD COMPOSITION\n")
            f.write("-" * 20 + "\n")
            if component_df is not None and not component_df.empty:
                component_means = component_df.groupby('component')['value'].mean()
                total_intrinsic = component_means.sum()
                
                f.write("Average Reward Component Values:\n")
                for comp, mean_val in component_means.items():
                    percentage = (mean_val / total_intrinsic) * 100 if total_intrinsic != 0 else 0
                    f.write(f"  {comp.replace('_', ' ').title()}: {mean_val:.4f} ({percentage:.1f}%)\n")
            f.write("\n")
            
            # 5. 论文建议
            f.write("5. RECOMMENDATIONS FOR PAPER\n")
            f.write("-" * 30 + "\n")
            f.write("Key figures to include in paper:\n")
            f.write("- Learning curves showing training progress\n")
            f.write("- Reward composition analysis demonstrating HMASD components\n")
            f.write("- Skill usage distribution showing learned diversity\n")
            f.write("- Performance comparison across training phases\n")
            f.write("\nKey metrics to report:\n")
            if episode_df is not None and not episode_df.empty:
                rewards = episode_df['total_reward'].values
                f.write(f"- Final performance: {np.mean(rewards[-100:]):.2f} ± {np.std(rewards[-100:]):.2f}\n")
                if len(rewards) >= 200:
                    early_perf = np.mean(rewards[:100])
                    late_perf = np.mean(rewards[-100:])
                    improvement = late_perf - early_perf
                    f.write(f"- Learning improvement: {improvement:.2f} ({(improvement/abs(early_perf)*100):.1f}%)\n")
            
            if skill_data and 'team_skills' in skill_data:
                team_skills = skill_data['team_skills']
                if len(team_skills) > 1:
                    counts = np.array(list(team_skills.values()))
                    total = sum(counts)
                    probs = counts / total
                    entropy = -np.sum(probs * np.log(probs + 1e-8))
                    max_entropy = np.log(len(counts))
                    uniformity = entropy / max_entropy
                    f.write(f"- Skill diversity index: {uniformity:.3f}\n")
            f.write("\n")
        
        print(f"摘要报告已保存到: {report_path}")
        return str(report_path)
    
    def run_full_analysis(self):
        """运行完整的数据分析"""
        print("开始论文数据分析...")
        print(f"数据目录: {self.data_dir}")
        print(f"输出目录: {self.output_dir}")
        
        # 加载数据
        episode_df = self.load_episode_rewards()
        component_df = self.load_reward_components()
        skill_data = self.load_skill_usage()
        summary_data = self.load_training_summary()
        
        generated_files = []
        
        # 生成各种分析图表
        if episode_df is not None:
            learning_curve_path = self.generate_learning_curves(episode_df)
            if learning_curve_path:
                generated_files.append(learning_curve_path)
            
            performance_path = self.generate_performance_comparison(episode_df)
            if performance_path:
                generated_files.append(performance_path)
        
        if component_df is not None:
            composition_path = self.generate_reward_composition_analysis(component_df)
            if composition_path:
                generated_files.append(composition_path)
        
        if skill_data is not None:
            skill_path = self.generate_skill_analysis(skill_data, episode_df)
            if skill_path:
                generated_files.append(skill_path)
        
        # 生成摘要报告
        report_path = self.generate_summary_report(episode_df, component_df, skill_data, summary_data)
        if report_path:
            generated_files.append(report_path)
        
        print("\n" + "="*50)
        print("论文数据分析完成!")
        print(f"共生成 {len(generated_files)} 个文件:")
        for file_path in generated_files:
            print(f"  - {file_path}")
        print(f"\n所有文件保存在: {self.output_dir}")
        
        return generated_files

def main():
    parser = argparse.ArgumentParser(description='HMASD论文数据分析工具')
    parser.add_argument('--data_dir', type=str, help='训练数据目录路径')
    parser.add_argument('--output_dir', type=str, default=None, 
                        help='输出目录路径 (默认为data_dir/analysis_output)')
    parser.add_argument('--figures_only', action='store_true', 
                        help='仅生成图表，不生成报告')
    
    args = parser.parse_args()
    
    # 检查数据目录是否存在
    if not os.path.exists(args.data_dir):
        print(f"错误: 数据目录 {args.data_dir} 不存在")
        return
    
    # 创建分析器
    analyzer = PaperDataAnalyzer(args.data_dir, args.output_dir)
    
    # 运行分析
    try:
        generated_files = analyzer.run_full_analysis()
        print(f"\n分析成功完成，生成了 {len(generated_files)} 个文件。")
    except Exception as e:
        print(f"\n分析过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
