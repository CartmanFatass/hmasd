import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Normal, Categorical

class MLP(nn.Module):
    """多层感知机"""
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers=2):
        super(MLP, self).__init__()
        
        layers = []
        dims = [input_dim] + [hidden_dim] * n_layers + [output_dim]
        
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims) - 2:  # 不在最后一层应用激活函数
                layers.append(nn.ReLU())
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        # 确保输入是float32类型
        x = x.float()
        return self.model(x)

class PositionalEncoding(nn.Module):
    """位置编码"""
    def __init__(self, d_model, max_len=100):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
        self.d_model = d_model
    
    def forward(self, x):
        # 确保输入是float32类型
        x = x.float()
        x = x + self.pe[:, :x.size(1)]
        return x

class StateEncoder(nn.Module):
    """状态编码器"""
    def __init__(self, state_dim, obs_dim, embedding_dim, n_layers, n_heads):
        super(StateEncoder, self).__init__()
        
        self.state_embedding = nn.Linear(state_dim, embedding_dim)
        self.obs_embedding = nn.Linear(obs_dim, embedding_dim)
        self.positional_encoding = PositionalEncoding(embedding_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=n_heads,
            dim_feedforward=embedding_dim * 4,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, n_layers)
    
    def forward(self, state, observations):
        """
        参数:
            state: 全局状态 [batch_size, state_dim]
            observations: 所有智能体观测 [batch_size, n_agents, obs_dim]
            
        返回:
            encoded_state: 编码后的状态 [batch_size, 1, embedding_dim]
            encoded_observations: 编码后的观测 [batch_size, n_agents, embedding_dim]
        """
        batch_size, n_agents, _ = observations.size()
        
        # 确保输入是float32类型
        state = state.float()
        observations = observations.float()
        
        # 嵌入全局状态和局部观测
        embedded_state = self.state_embedding(state).unsqueeze(1)  # [batch_size, 1, embedding_dim]
        embedded_obs = self.obs_embedding(observations.reshape(-1, observations.size(-1)))
        embedded_obs = embedded_obs.reshape(batch_size, n_agents, -1)  # [batch_size, n_agents, embedding_dim]
        
        # 将状态和观测拼接作为序列
        sequence = torch.cat([embedded_state, embedded_obs], dim=1)  # [batch_size, 1+n_agents, embedding_dim]
        
        # 位置编码
        sequence = self.positional_encoding(sequence)
        
        # Transformer编码器
        encoded_sequence = self.transformer_encoder(sequence)
        
        # 拆分回状态和观测
        encoded_state = encoded_sequence[:, 0:1, :]
        encoded_observations = encoded_sequence[:, 1:, :]
        
        return encoded_state, encoded_observations

class SkillDecoder(nn.Module):
    """技能解码器"""
    def __init__(self, embedding_dim, n_layers, n_heads, n_Z, n_z):
        super(SkillDecoder, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.Z0_embedding = nn.Embedding(1, embedding_dim)
        self.team_skill_embedding = nn.Embedding(n_Z, embedding_dim)
        self.agent_skill_embedding = nn.Embedding(n_z, embedding_dim)
        self.positional_encoding = PositionalEncoding(embedding_dim)
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embedding_dim,
            nhead=n_heads,
            dim_feedforward=embedding_dim * 4,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, n_layers)
        
        # 输出头
        self.team_skill_head = nn.Linear(embedding_dim, n_Z)
        self.agent_skill_head = nn.Linear(embedding_dim, n_z)
    
    def forward(self, encoded_state, encoded_observations, Z=None, z=None, step=0):
        """
        参数:
            encoded_state: 编码后的状态 [batch_size, 1, embedding_dim]
            encoded_observations: 编码后的观测 [batch_size, n_agents, embedding_dim]
            Z: 已选择的团队技能索引 [batch_size]，可选
            z: 已选择的个体技能索引列表 [batch_size, step]，可选
            step: 当前解码步骤
            
        返回:
            output: 技能分布 [batch_size, n_Z/n_z]
        """
        batch_size = encoded_state.size(0)
        device = encoded_state.device
        
        if step == 0:  # 生成团队技能Z
            # 使用特殊起始符Z0
            Z0_idx = torch.zeros(batch_size, 1, dtype=torch.long, device=device)
            decoder_input = self.Z0_embedding(Z0_idx)
            decoder_input = self.positional_encoding(decoder_input)
            
            # Transformer解码
            memory = torch.cat([encoded_state, encoded_observations], dim=1)
            decoded = self.transformer_decoder(decoder_input, memory)
            
            # 输出团队技能分布
            team_skill_logits = self.team_skill_head(decoded).squeeze(1)
            return team_skill_logits
        else:  # 生成第step个智能体的个体技能zi
            # 构建已解码序列
            seq_len = step + 1  # Z0 + Z + z1 + ... + z_{step-1}
            decoder_inputs = []
            
            # 添加Z0
            Z0_idx = torch.zeros(batch_size, 1, dtype=torch.long, device=device)
            decoder_inputs.append(self.Z0_embedding(Z0_idx))
            
            # 添加Z
            Z_embedded = self.team_skill_embedding(Z.unsqueeze(1))
            decoder_inputs.append(Z_embedded)
            
            # 添加z1到z_{step-1}
            for i in range(step - 1):
                zi_embedded = self.agent_skill_embedding(z[:, i].unsqueeze(1))
                decoder_inputs.append(zi_embedded)
            
            # 拼接所有嵌入
            decoder_input = torch.cat(decoder_inputs, dim=1)
            decoder_input = self.positional_encoding(decoder_input)
            
            # Transformer解码
            memory = torch.cat([encoded_state, encoded_observations], dim=1)
            decoded = self.transformer_decoder(decoder_input, memory)
            
            # 输出个体技能分布（仅取最后一步）
            agent_skill_logits = self.agent_skill_head(decoded[:, -1, :])
            return agent_skill_logits

class SkillCoordinator(nn.Module):
    """技能协调器（高层策略）"""
    def __init__(self, config):
        super(SkillCoordinator, self).__init__()
        
        self.config = config
        self.n_Z = config.n_Z
        self.n_z = config.n_z
        
        # 状态编码器
        self.state_encoder = StateEncoder(
            config.state_dim,
            config.obs_dim,
            config.embedding_dim,
            config.n_encoder_layers,
            config.n_heads
        )
        
        # 技能解码器
        self.skill_decoder = SkillDecoder(
            config.embedding_dim,
            config.n_decoder_layers,
            config.n_heads,
            config.n_Z,
            config.n_z
        )
        
        # 高层价值函数
        self.value_head_state = nn.Linear(config.embedding_dim, 1)
        self.value_heads_obs = nn.ModuleList([
            nn.Linear(config.embedding_dim, 1) for _ in range(config.n_agents)
        ])
    
    def get_value(self, state, observations):
        """获取高层价值函数值"""
        encoded_state, encoded_observations = self.state_encoder(state, observations)
        
        # 全局状态价值
        state_value = self.value_head_state(encoded_state.squeeze(1))
        
        # 每个智能体的观测价值
        agent_values = []
        for i in range(min(self.config.n_agents, encoded_observations.size(1))):
            agent_value = self.value_heads_obs[i](encoded_observations[:, i, :])
            agent_values.append(agent_value)
            
        return state_value, agent_values
    
    def forward(self, state, observations, deterministic=False):
        """
        前向传播，按顺序生成技能
        
        参数:
            state: 全局状态 [batch_size, state_dim]
            observations: 所有智能体观测 [batch_size, n_agents, obs_dim]
            deterministic: 是否使用确定性策略
            
        返回:
            Z: 团队技能索引 [batch_size]
            z: 个体技能索引 [batch_size, n_agents]
            Z_logits: 团队技能logits [batch_size, n_Z]
            z_logits: 个体技能logits列表 [n_agents个 [batch_size, n_z]]
        """
        batch_size = state.size(0)
        n_agents = observations.size(1)
        device = state.device
        
        # 确保输入是float32类型
        state = state.float()
        observations = observations.float()
        
        # 编码状态和观测
        encoded_state, encoded_observations = self.state_encoder(state, observations)
        
        # 生成团队技能Z
        Z_logits = self.skill_decoder(encoded_state, encoded_observations)
        Z_dist = Categorical(logits=Z_logits)
        
        if deterministic:
            Z = Z_logits.argmax(dim=-1)
        else:
            Z = Z_dist.sample()
        
        # 依次为每个智能体生成个体技能zi
        z = torch.zeros(batch_size, n_agents, dtype=torch.long, device=device)
        z_logits = []
        
        for i in range(n_agents):
            zi_logits = self.skill_decoder(encoded_state, encoded_observations, Z, z[:, :i], step=i+1)
            z_logits.append(zi_logits)
            zi_dist = Categorical(logits=zi_logits)
            
            if deterministic:
                zi = zi_logits.argmax(dim=-1)
            else:
                zi = zi_dist.sample()
                
            z[:, i] = zi
        
        return Z, z, Z_logits, z_logits

class SkillDiscoverer(nn.Module):
    """技能发现器（低层策略）"""
    def __init__(self, config):
        super(SkillDiscoverer, self).__init__()
        
        self.config = config
        self.obs_dim = config.obs_dim
        self.n_z = config.n_z
        self.action_dim = config.action_dim
        self.hidden_dim = config.hidden_size
        self.gru_hidden_dim = config.gru_hidden_size
        
        # Actor网络（每个智能体共享）
        self.actor_mlp = MLP(config.obs_dim + config.n_z, config.hidden_size, config.hidden_size)
        self.actor_gru = nn.GRU(config.hidden_size, config.gru_hidden_size, batch_first=True)
        
        # 动作均值和标准差
        self.action_mean = nn.Linear(config.gru_hidden_size, config.action_dim)
        self.action_log_std = nn.Linear(config.gru_hidden_size, config.action_dim)
        
        # 重置参数
        self.actor_hidden = None
        
        # Critic网络（中心化价值函数）
        self.critic_mlp = MLP(config.state_dim + config.n_Z, config.hidden_size, config.hidden_size)
        self.critic_gru = nn.GRU(config.hidden_size, config.gru_hidden_size, batch_first=True)
        self.value_head = nn.Linear(config.gru_hidden_size, 1)
        
        # 重置参数
        self.critic_hidden = None
    
    def init_hidden(self, batch_size=1):
        """初始化GRU隐藏状态"""
        device = next(self.parameters()).device
        self.actor_hidden = torch.zeros(1, batch_size, self.gru_hidden_dim, device=device)
        self.critic_hidden = torch.zeros(1, batch_size, self.gru_hidden_dim, device=device)
    
    def get_value(self, state, team_skill, batch_first=True):
        """获取价值函数值"""
        batch_size = state.size(0)
        
        # 确保state是float32类型
        state = state.float()
        
        if isinstance(team_skill, int) or isinstance(team_skill, torch.Tensor):
            # 将技能索引转换为独热编码
            if isinstance(team_skill, int):
                team_skill = torch.tensor([team_skill], device=state.device)
            elif team_skill.dim() == 0:  # 处理标量张量
                team_skill = team_skill.unsqueeze(0)  # 转换为一维张量
            
            # 确保是一维张量后进行独热编码
            if team_skill.dim() == 1:
                team_skill_onehot = F.one_hot(team_skill, self.config.n_Z).float()
            else:
                team_skill_onehot = team_skill.float()  # 已经是独热编码，确保是float32
        else:
            team_skill_onehot = team_skill.float()
        
        # 拼接状态和团队技能
        critic_input = torch.cat([state, team_skill_onehot], dim=-1)
        
        # 前向传播
        critic_features = self.critic_mlp(critic_input)
        
        # 确保critic_features是3D的 [batch_size, seq_len, hidden_dim]
        if critic_features.dim() == 2:
            critic_features = critic_features.unsqueeze(1)  # 添加时序维度
        
        # 初始化隐藏状态（如果需要）
        if self.critic_hidden is None or self.critic_hidden.size(1) != batch_size:
            device = critic_features.device
            self.critic_hidden = torch.zeros(1, batch_size, self.gru_hidden_dim, device=device)
            
        critic_output, self.critic_hidden = self.critic_gru(critic_features, self.critic_hidden)
        
        # 移除时序维度
        critic_output = critic_output.squeeze(1)
            
        value = self.value_head(critic_output)
        
        # 确保返回的值是float32类型
        return value.float()
    
    def forward(self, observation, agent_skill, deterministic=False):
        """
        前向传播，生成动作
        
        参数:
            observation: 智能体观测 [batch_size, obs_dim]
            agent_skill: 个体技能索引 [batch_size] 或独热编码 [batch_size, n_z]
            deterministic: 是否使用确定性策略
            
        返回:
            action: 动作 [batch_size, action_dim]
            action_logprob: 动作对数概率 [batch_size]
            action_distribution: 动作分布
        """
        batch_size = observation.size(0)
        
        # 确保observation是float32类型
        observation = observation.float()
        
        if isinstance(agent_skill, int) or isinstance(agent_skill, torch.Tensor):
            # 将技能索引转换为独热编码
            if isinstance(agent_skill, int):
                agent_skill = torch.tensor([agent_skill], device=observation.device)
            elif agent_skill.dim() == 0:  # 处理标量张量
                agent_skill = agent_skill.unsqueeze(0)  # 转换为一维张量
            
            # 确保是一维张量后进行独热编码
            if agent_skill.dim() == 1:
                agent_skill_onehot = F.one_hot(agent_skill, self.n_z).float()
            else:
                agent_skill_onehot = agent_skill.float()  # 已经是独热编码，确保是float32
        else:
            agent_skill_onehot = agent_skill.float()
        
        # 拼接观测和个体技能
        actor_input = torch.cat([observation, agent_skill_onehot], dim=-1)
        
        # 前向传播
        actor_features = self.actor_mlp(actor_input).unsqueeze(1)  # 添加时序维度
        
        # 初始化隐藏状态（如果需要）
        if self.actor_hidden is None or self.actor_hidden.size(1) != batch_size:
            device = actor_features.device
            self.actor_hidden = torch.zeros(1, batch_size, self.gru_hidden_dim, device=device)
            
        actor_output, self.actor_hidden = self.actor_gru(actor_features, self.actor_hidden)
        actor_output = actor_output.squeeze(1)  # 移除时序维度
        
        # 生成动作分布参数
        action_mean = self.action_mean(actor_output)
        action_log_std = self.action_log_std(actor_output)
        action_std = torch.exp(action_log_std)
        
        # 创建正态分布
        action_distribution = Normal(action_mean, action_std)
        
        # 采样或选择最佳动作
        if deterministic:
            action = action_mean
        else:
            action = action_distribution.sample()
        
        # 计算动作对数概率
        action_logprob = action_distribution.log_prob(action).sum(dim=-1)
        
        return action, action_logprob, action_distribution

class TeamDiscriminator(nn.Module):
    """团队技能判别器"""
    def __init__(self, config):
        super(TeamDiscriminator, self).__init__()
        
        self.model = MLP(
            input_dim=config.state_dim,
            hidden_dim=config.hidden_size,
            output_dim=config.n_Z,
            n_layers=2
        )
    
    def forward(self, state):
        """
        参数:
            state: 全局状态 [batch_size, state_dim]
            
        返回:
            logits: 团队技能logits [batch_size, n_Z]
        """
        # 确保state是float32类型
        state = state.float()
        return self.model(state)

class IndividualDiscriminator(nn.Module):
    """个体技能判别器"""
    def __init__(self, config):
        super(IndividualDiscriminator, self).__init__()
        
        self.config = config
        self.n_Z = config.n_Z
        
        self.model = MLP(
            input_dim=config.obs_dim + config.n_Z,  # 观测 + 团队技能
            hidden_dim=config.hidden_size,
            output_dim=config.n_z,
            n_layers=2
        )
    
    def forward(self, observation, team_skill):
        """
        参数:
            observation: 智能体观测 [batch_size, obs_dim]
            team_skill: 团队技能索引 [batch_size] 或独热编码 [batch_size, n_Z]
            
        返回:
            logits: 个体技能logits [batch_size, n_z]
        """
        # 确保observation是float32类型
        observation = observation.float()
        
        if isinstance(team_skill, int) or isinstance(team_skill, torch.Tensor):
            # 将技能索引转换为独热编码
            if isinstance(team_skill, int):
                team_skill = torch.tensor([team_skill], device=observation.device)
            elif team_skill.dim() == 0:  # 处理标量张量
                team_skill = team_skill.unsqueeze(0)  # 转换为一维张量
            
            # 确保是一维张量后进行独热编码
            if team_skill.dim() == 1:
                team_skill_onehot = F.one_hot(team_skill, self.config.n_Z).float()
            else:
                team_skill_onehot = team_skill.float()  # 已经是独热编码，确保是float32
        else:
            team_skill_onehot = team_skill.float()
        
        # 拼接观测和团队技能
        discriminator_input = torch.cat([observation, team_skill_onehot], dim=-1)
        
        return self.model(discriminator_input)
