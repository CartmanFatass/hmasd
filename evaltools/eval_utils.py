"""
HMASD算法评估工具
"""

import numpy as np
import torch
from logger import main_logger

def evaluate_agent(agent, eval_envs, config, num_episodes=4):
    """
    评估agent性能
    
    参数:
        agent: HMASD agent
        eval_envs: 评估环境列表
        config: 配置对象
        num_episodes: 评估episode数量
        
    返回:
        dict: 评估结果
    """
    agent.skill_coordinator.eval()
    agent.skill_discoverer.eval()
    agent.team_discriminator.eval()
    agent.individual_discriminator.eval()
    
    episode_rewards = []
    episode_lengths = []
    
    main_logger.info(f"开始评估，共 {num_episodes} 个episodes")
    
    for episode in range(num_episodes):
        env = eval_envs[episode % len(eval_envs)]
        
        # 重置环境
        obs, info = env.reset()
        state = info.get('state', np.zeros(config.state_dim))
        
        episode_reward = 0.0
        episode_length = 0
        done = False
        
        while not done and episode_length < config.episode_length:
            # 获取动作（确定性策略用于评估）
            actions, agent_info = agent.step(state, obs, episode_length, deterministic=True, env_id=0)
            
            # 执行动作
            next_obs, reward, done, truncated, info = env.step(actions)
            next_state = info.get('state', np.zeros(config.state_dim))
            
            # 更新统计
            episode_reward += reward
            episode_length += 1
            
            # 更新状态
            obs = next_obs
            state = next_state
            
            if done or truncated:
                break
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        main_logger.debug(f"评估episode {episode+1}/{num_episodes}: 奖励={episode_reward:.4f}, 长度={episode_length}")
    
    # 恢复训练模式
    agent.skill_coordinator.train()
    agent.skill_discoverer.train()
    agent.team_discriminator.train()
    agent.individual_discriminator.train()
    
    # 计算统计结果
    results = {
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'min_reward': np.min(episode_rewards),
        'max_reward': np.max(episode_rewards),
        'mean_length': np.mean(episode_lengths),
        'std_length': np.std(episode_lengths),
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths
    }
    
    main_logger.info(f"评估完成: 平均奖励={results['mean_reward']:.4f}±{results['std_reward']:.4f}")
    
    return results
