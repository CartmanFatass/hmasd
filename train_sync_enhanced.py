#!/usr/bin/env python3
"""
同步训练HMASD算法 - 强制on-policy训练
基于train_enhanced_reward_tracking.py修改，实现严格的同步训练模式
"""

import os
import sys
import time
import traceback
import datetime
import numpy as np
import torch
import argparse
from tensorboardX import SummaryWriter

# 确保模块路径正确
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from envs.pettingzoo.scenario2 import UAVCooperativeNetworkEnv
from hmasd.agent import HMASDAgent
from config_1 import Config
from logger import main_logger
from evaltools.eval_utils import evaluate_agent

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='同步训练HMASD算法')
    parser.add_argument('--eval-only', action='store_true', 
                        help='仅运行评估，不进行训练')
    parser.add_argument('--model-path', type=str, default=None,
                        help='模型路径，用于加载或保存模型')
    parser.add_argument('--log-level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        help='日志级别')
    return parser.parse_args()

def create_parallel_envs(config):
    """创建并行环境"""
    envs = []
    for i in range(config.num_envs):
        env = UAVCooperativeNetworkEnv()
        envs.append(env)
    return envs

def reset_all_envs(envs):
    """重置所有环境"""
    states = []
    observations = []
    infos = []
    
    for env in envs:
        observation, info = env.reset()
        state = info.get('state', np.zeros(10))  # 获取全局状态
        
        # 处理观测数据：如果是字典格式，提取'obs'字段
        if isinstance(observation, dict):
            # 假设observation是字典格式，提取第一个智能体的观测
            if len(observation) > 0:
                first_agent_key = list(observation.keys())[0]
                if isinstance(observation[first_agent_key], dict) and 'obs' in observation[first_agent_key]:
                    # 从字典格式的观测中提取观测数组
                    obs_array = []
                    for agent_key in sorted(observation.keys()):
                        obs_array.append(observation[agent_key]['obs'])
                    observation = np.array(obs_array)
                else:
                    # 如果不是预期的格式，转换为数组
                    observation = np.array([obs for obs in observation.values()])
        
        states.append(state)
        observations.append(observation)
        infos.append(info)
    
    return states, observations, infos

def train_sync(vec_env, eval_vec_env, config, args, device):
    """同步训练主循环"""
    # 创建agent
    log_dir = f"tf-logs/hmasd_sync_enhanced_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    agent = HMASDAgent(config, log_dir=log_dir, device=device)
    
    # 加载模型（如果指定）
    if args.model_path and os.path.exists(args.model_path):
        agent.load_model(args.model_path)
        main_logger.info(f"已加载模型: {args.model_path}")
    
    # 初始化统计
    total_steps = 0
    update_times = 0
    episode_count = 0
    start_time = time.time()
    
    # 重置所有环境
    states, observations, infos = reset_all_envs(vec_env)
    env_steps = [0] * config.num_envs
    episode_rewards = [0.0] * config.num_envs
    
    main_logger.info(f"开始同步训练，目标步数: {config.total_timesteps}")
    main_logger.info(f"同步batch大小: {agent.sync_batch_size}")
    
    # 训练循环
    while total_steps < config.total_timesteps:
        # 1. 数据收集阶段 - 循环直到收集足够样本
        collection_start_time = time.time()
        samples_collected_this_batch = 0
        
        while not agent.should_sync_update():
            # 所有环境并行执行一步
            all_actions_list = []
            all_info_list = []
            
            for i in range(config.num_envs):
                # 为每个环境获取动作
                actions, agent_info = agent.step(states[i], observations[i], env_steps[i], env_id=i)
                all_actions_list.append(actions)
                all_info_list.append(agent_info)
            
            # 执行环境步骤
            next_states = []
            next_observations = []
            rewards = []
            dones = []
            step_infos = []
            
            for i, env in enumerate(vec_env):
                next_obs, reward, done, truncated, info = env.step(all_actions_list[i])
                next_state = info.get('state', np.zeros(config.state_dim))
                
                # 处理下一个观测：如果是字典格式，提取'obs'字段
                if isinstance(next_obs, dict):
                    if len(next_obs) > 0:
                        first_agent_key = list(next_obs.keys())[0]
                        if isinstance(next_obs[first_agent_key], dict) and 'obs' in next_obs[first_agent_key]:
                            # 从字典格式的观测中提取观测数组
                            obs_array = []
                            for agent_key in sorted(next_obs.keys()):
                                obs_array.append(next_obs[agent_key]['obs'])
                            next_obs = np.array(obs_array)
                        else:
                            # 如果不是预期的格式，转换为数组
                            next_obs = np.array([obs for obs in next_obs.values()])
                
                # 处理done状态：将字典格式转换为布尔值
                if isinstance(done, dict):
                    # 如果是字典格式，检查是否所有智能体都完成
                    done_value = any(done.values()) if done else False
                elif isinstance(done, (list, tuple)):
                    # 如果是列表格式，检查是否有任何智能体完成
                    done_value = any(done) if done else False
                else:
                    # 标量值，直接使用
                    done_value = bool(done)
                
                # 处理truncated状态
                if isinstance(truncated, dict):
                    # 如果是字典格式，检查是否所有智能体都被截断
                    truncated_value = any(truncated.values()) if truncated else False
                elif isinstance(truncated, (list, tuple)):
                    # 如果是列表格式，检查是否有任何智能体被截断
                    truncated_value = any(truncated) if truncated else False
                else:
                    # 标量值，直接使用
                    truncated_value = bool(truncated)
                
                next_states.append(next_state)
                next_observations.append(next_obs)
                rewards.append(reward)
                dones.append(done_value or truncated_value)
                step_infos.append(info)
                
                # 处理奖励类型转换
                if isinstance(reward, dict):
                    # 如果奖励是字典格式，计算所有智能体奖励的总和
                    reward_scalar = sum(reward.values())
                else:
                    # 如果是标量，直接使用
                    reward_scalar = reward if isinstance(reward, (int, float)) else reward.item()
                
                # 更新统计
                episode_rewards[i] += reward_scalar
                env_steps[i] += 1
            
            # 存储所有环境的经验
            stored_count = 0
            for i in range(config.num_envs):
                # 处理存储时的奖励类型转换
                reward_for_storage = rewards[i]
                if isinstance(rewards[i], dict):
                    # 如果奖励是字典格式，计算所有智能体奖励的总和用于存储
                    reward_for_storage = sum(rewards[i].values())
                elif not isinstance(rewards[i], (int, float)):
                    # 如果是张量，转换为标量
                    reward_for_storage = rewards[i].item()
                
                success = agent.store_transition(
                    state=states[i],
                    next_state=next_states[i],
                    observations=observations[i],
                    next_observations=next_observations[i],
                    actions=all_actions_list[i],
                    rewards=reward_for_storage,
                    dones=dones[i],
                    team_skill=all_info_list[i]['team_skill'],
                    agent_skills=all_info_list[i]['agent_skills'],
                    action_logprobs=all_info_list[i]['action_logprobs'],
                    log_probs=all_info_list[i]['log_probs'],
                    skill_timer_for_env=all_info_list[i]['skill_timer'],
                    env_id=i
                )
                if success:
                    stored_count += 1
                
                # 处理环境重置
                if dones[i]:
                    main_logger.info(f"环境{i} episode结束，奖励: {episode_rewards[i]:.4f}, 步数: {env_steps[i]}")
                    
                    # 重置环境
                    next_obs, info = env.reset()
                    next_states[i] = info.get('state', np.zeros(config.state_dim))
                    
                    # 处理重置后的观测：如果是字典格式，提取'obs'字段
                    if isinstance(next_obs, dict):
                        if len(next_obs) > 0:
                            first_agent_key = list(next_obs.keys())[0]
                            if isinstance(next_obs[first_agent_key], dict) and 'obs' in next_obs[first_agent_key]:
                                # 从字典格式的观测中提取观测数组
                                obs_array = []
                                for agent_key in sorted(next_obs.keys()):
                                    obs_array.append(next_obs[agent_key]['obs'])
                                next_obs = np.array(obs_array)
                            else:
                                # 如果不是预期的格式，转换为数组
                                next_obs = np.array([obs for obs in next_obs.values()])
                    
                    next_observations[i] = next_obs
                    
                    # 重置统计
                    episode_rewards[i] = 0.0
                    env_steps[i] = 0
                    episode_count += 1
            
            # 更新状态
            states = next_states
            observations = next_observations
            total_steps += config.num_envs
            samples_collected_this_batch += stored_count
            
            # 检查是否收集被停止（达到同步点）
            if not agent.collection_enabled:
                break
        
        collection_time = time.time() - collection_start_time
        
        # 2. 同步更新阶段
        if agent.should_sync_update():
            sync_start_time = time.time()
            main_logger.info(f"达到同步点 - 收集了 {agent.samples_collected_this_round} 个样本，耗时 {collection_time:.2f}s")
            
            # 执行同步更新
            update_info = agent.sync_update()
            update_times += 1
            sync_time = time.time() - sync_start_time
            
            main_logger.info(f"同步更新完成 - 策略版本: {update_info['policy_version']}, 耗时: {sync_time:.2f}s")
            
            # 记录训练指标
            if update_times % 10 == 0:
                elapsed_time = time.time() - start_time
                steps_per_sec = total_steps / elapsed_time if elapsed_time > 0 else 0
                
                main_logger.info(f"训练进度: {total_steps}/{config.total_timesteps} "
                               f"({100.0 * total_steps / config.total_timesteps:.1f}%), "
                               f"更新次数: {update_times}, Episodes: {episode_count}, "
                               f"步数/秒: {steps_per_sec:.1f}")
        
        # 3. 定期评估
        if total_steps > 0 and total_steps % config.eval_interval == 0:
            main_logger.info(f"开始评估 (步数: {total_steps})")
            eval_start_time = time.time()
            
            try:
                eval_results = evaluate_agent(agent, eval_vec_env, config, num_episodes=config.eval_episodes)
                eval_time = time.time() - eval_start_time
                
                main_logger.info(f"评估完成，耗时: {eval_time:.2f}s")
                main_logger.info(f"评估结果: 平均奖励={eval_results['mean_reward']:.4f}, "
                               f"标准差={eval_results['std_reward']:.4f}")
                
                # 记录到TensorBoard
                agent.writer.add_scalar('Eval/MeanReward', eval_results['mean_reward'], total_steps)
                agent.writer.add_scalar('Eval/StdReward', eval_results['std_reward'], total_steps)
                
            except Exception as e:
                main_logger.error(f"评估失败: {e}")
                main_logger.error(traceback.format_exc())
        
        # 4. 定期保存模型
        if total_steps > 0 and total_steps % (config.eval_interval * 2) == 0:
            model_path = f"models/hmasd_sync_enhanced_{total_steps}.pth"
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            agent.save_model(model_path)
            main_logger.info(f"模型已保存: {model_path}")
    
    # 训练完成
    total_time = time.time() - start_time
    main_logger.info(f"同步训练完成!")
    main_logger.info(f"总时间: {total_time:.2f}s, 总步数: {total_steps}, 总更新: {update_times}")
    main_logger.info(f"平均步数/秒: {total_steps / total_time:.1f}")
    
    # 保存最终模型
    final_model_path = "models/hmasd_sync_enhanced_final.pth"
    os.makedirs(os.path.dirname(final_model_path), exist_ok=True)
    agent.save_model(final_model_path)
    main_logger.info(f"最终模型已保存: {final_model_path}")
    
    return agent

def main():
    """主函数"""
    # 解析参数
    args = parse_args()
    
    # 设置日志级别
    import logging
    from logger import setup_logger
    log_level = getattr(logging, args.log_level.upper())
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    setup_logger(name=f'HMASD_SYNC_TRAINING_{timestamp}', level=log_level)
    
    # 设置设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    main_logger.info(f"使用设备: {device}")
    
    # 初始化配置
    config = Config()
    
    try:
        # 创建环境
        main_logger.info("创建训练环境...")
        vec_env = create_parallel_envs(config)
        main_logger.info(f"创建了 {len(vec_env)} 个并行训练环境")
        
        main_logger.info("创建评估环境...")
        eval_vec_env = create_parallel_envs(Config())  # 使用相同配置创建评估环境
        eval_vec_env = eval_vec_env[:config.eval_rollout_threads]  # 只使用指定数量的评估环境
        main_logger.info(f"创建了 {len(eval_vec_env)} 个并行评估环境")
        
        # 获取环境维度
        sample_env = vec_env[0]
        obs, info = sample_env.reset()
        state = info.get('state', np.zeros(10))
        
        # 处理观测数据来获取正确的维度
        if isinstance(obs, dict):
            if len(obs) > 0:
                first_agent_key = list(obs.keys())[0]
                if isinstance(obs[first_agent_key], dict) and 'obs' in obs[first_agent_key]:
                    # 从字典格式的观测中获取观测维度
                    obs_dim = len(obs[first_agent_key]['obs'])
                    obs_array = []
                    for agent_key in sorted(obs.keys()):
                        obs_array.append(obs[agent_key]['obs'])
                    obs = np.array(obs_array)
                else:
                    # 如果不是预期的格式，转换为数组并获取维度
                    obs_values = [obs_val for obs_val in obs.values()]
                    obs = np.array(obs_values)
                    obs_dim = obs.shape[-1] if len(obs.shape) > 1 else len(obs_values[0])
        else:
            # 如果观测已经是数组格式
            obs_dim = obs.shape[-1] if len(obs.shape) > 1 else len(obs)
        
        config.update_env_dims(len(state), obs_dim)
        main_logger.info(f"环境维度已更新：state_dim={config.state_dim}, obs_dim={config.obs_dim}")
        
        if args.eval_only:
            # 仅评估模式
            if not args.model_path or not os.path.exists(args.model_path):
                main_logger.error("评估模式需要指定有效的模型路径")
                return
            
            main_logger.info("运行评估模式...")
            log_dir = f"tf-logs/hmasd_sync_eval_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
            agent = HMASDAgent(config, log_dir=log_dir, device=device)
            agent.load_model(args.model_path)
            
            eval_results = evaluate_agent(agent, eval_vec_env, config, num_episodes=config.eval_episodes)
            main_logger.info(f"评估结果: 平均奖励={eval_results['mean_reward']:.4f}, "
                           f"标准差={eval_results['std_reward']:.4f}")
        else:
            # 训练模式
            main_logger.info("开始同步训练...")
            agent = train_sync(vec_env, eval_vec_env, config, args, device)
            main_logger.info("训练完成!")
    
    except Exception as e:
        main_logger.error(f"训练过程中发生错误: {e}")
        main_logger.error(traceback.format_exc())
        sys.exit(1)
    
    finally:
        # 清理资源
        main_logger.info("清理环境资源...")
        for env in vec_env:
            try:
                env.close()
            except:
                pass
        for env in eval_vec_env:
            try:
                env.close()
            except:
                pass

if __name__ == "__main__":
    main()
