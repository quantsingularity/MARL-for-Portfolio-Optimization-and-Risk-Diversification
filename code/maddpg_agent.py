"""
MADDPG Agent Implementation
Implements the complete MADDPG algorithm from the paper
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
import random
from typing import List, Dict


class ReplayBuffer:
    """Experience replay buffer for MADDPG"""

    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """Add experience to buffer"""
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        """Sample a batch of experiences"""
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)


class ActorNetwork(nn.Module):
    """
    Actor network architecture from paper Section 4.2
    [256, 128, 64] with ReLU, BatchNorm, and Softmax output
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int],
        use_batch_norm: bool = True,
    ):
        super(ActorNetwork, self).__init__()

        self.use_batch_norm = use_batch_norm

        # Input layer
        self.fc1 = nn.Linear(state_dim, hidden_dims[0])
        self.bn1 = nn.BatchNorm1d(hidden_dims[0]) if use_batch_norm else None

        # Hidden layers
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.bn2 = nn.BatchNorm1d(hidden_dims[1]) if use_batch_norm else None

        self.fc3 = nn.Linear(hidden_dims[1], hidden_dims[2])

        # Output layer with softmax (ensures weights sum to 1)
        self.fc_out = nn.Linear(hidden_dims[2], action_dim)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize network weights"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.01)

    def forward(self, state):
        """Forward pass"""
        x = self.fc1(state)
        if self.use_batch_norm and x.size(0) > 1:
            x = self.bn1(x)
        x = F.relu(x)

        x = self.fc2(x)
        if self.use_batch_norm and x.size(0) > 1:
            x = self.bn2(x)
        x = F.relu(x)

        x = F.relu(self.fc3(x))

        # Softmax ensures valid portfolio weights
        x = F.softmax(self.fc_out(x), dim=-1)

        return x


class CriticNetwork(nn.Module):
    """
    Centralized critic network architecture from paper Section 4.2
    [512, 256, 128] with ReLU and BatchNorm
    Takes global state and joint actions as input
    """

    def __init__(
        self,
        global_state_dim: int,
        total_action_dim: int,
        hidden_dims: List[int],
        use_batch_norm: bool = True,
    ):
        super(CriticNetwork, self).__init__()

        self.use_batch_norm = use_batch_norm
        input_dim = global_state_dim + total_action_dim

        # Input layer
        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.bn1 = nn.BatchNorm1d(hidden_dims[0]) if use_batch_norm else None

        # Hidden layers
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.bn2 = nn.BatchNorm1d(hidden_dims[1]) if use_batch_norm else None

        self.fc3 = nn.Linear(hidden_dims[1], hidden_dims[2])

        # Output layer (Q-value)
        self.fc_out = nn.Linear(hidden_dims[2], 1)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize network weights"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.01)

    def forward(self, global_state, joint_actions):
        """Forward pass"""
        # Concatenate state and actions
        x = torch.cat([global_state, joint_actions], dim=-1)

        x = self.fc1(x)
        if self.use_batch_norm and x.size(0) > 1:
            x = self.bn1(x)
        x = F.relu(x)

        x = self.fc2(x)
        if self.use_batch_norm and x.size(0) > 1:
            x = self.bn2(x)
        x = F.relu(x)

        x = F.relu(self.fc3(x))

        # Linear output for Q-value
        q_value = self.fc_out(x)

        return q_value


class MADDPGAgent:
    """
    Multi-Agent Deep Deterministic Policy Gradient Agent
    Implements MADDPG with centralized training and decentralized execution
    """

    def __init__(
        self,
        agent_id: int,
        state_dim: int,
        action_dim: int,
        global_state_dim: int,
        total_action_dim: int,
        config,
    ):

        self.agent_id = agent_id
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.global_state_dim = global_state_dim
        self.total_action_dim = total_action_dim
        self.config = config

        self.device = torch.device(
            config.device if torch.cuda.is_available() else "cpu"
        )

        # Actor networks (local observation only)
        self.actor = ActorNetwork(
            state_dim,
            action_dim,
            config.network.actor_hidden_dims,
            config.network.actor_use_batch_norm,
        ).to(self.device)

        self.actor_target = ActorNetwork(
            state_dim,
            action_dim,
            config.network.actor_hidden_dims,
            config.network.actor_use_batch_norm,
        ).to(self.device)

        self.actor_target.load_state_dict(self.actor.state_dict())

        # Critic networks (global state + joint actions)
        self.critic = CriticNetwork(
            global_state_dim,
            total_action_dim,
            config.network.critic_hidden_dims,
            config.network.critic_use_batch_norm,
        ).to(self.device)

        self.critic_target = CriticNetwork(
            global_state_dim,
            total_action_dim,
            config.network.critic_hidden_dims,
            config.network.critic_use_batch_norm,
        ).to(self.device)

        self.critic_target.load_state_dict(self.critic.state_dict())

        # Optimizers
        self.actor_optimizer = optim.Adam(
            self.actor.parameters(), lr=config.training.lr_actor
        )

        self.critic_optimizer = optim.Adam(
            self.critic.parameters(), lr=config.training.lr_critic
        )

        # Exploration noise
        self.noise_std = config.training.noise_std_start
        self.noise_decay = config.training.noise_decay
        self.noise_std_end = config.training.noise_std_end

        # Training parameters
        self.gamma = config.training.gamma
        self.tau = config.training.tau

    def select_action(self, state, add_noise=True):
        """Select action using current policy"""
        self.actor.eval()
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action = self.actor(state_tensor).cpu().numpy()[0]
        self.actor.train()

        # Add exploration noise
        if add_noise:
            noise = np.random.normal(0, self.noise_std, size=action.shape)
            action = action + noise

            # Ensure valid portfolio weights
            action = np.clip(action, 0, 1)
            action = action / (np.sum(action) + 1e-8)

        return action

    def update_critic(self, batch, all_agents):
        """
        Update critic network
        Loss: L(θ_i) = E[(Q_i(s, a_1, ..., a_N) - y)^2]
        where y = r_i + γ * Q_i'(s', a'_1, ..., a'_N)
        """
        states, actions, rewards, next_states, dones = batch

        len(states)

        # Extract this agent's data
        torch.FloatTensor([s[self.agent_id] for s in states]).to(self.device)
        agent_rewards = (
            torch.FloatTensor([r[self.agent_id] for r in rewards])
            .to(self.device)
            .unsqueeze(1)
        )
        agent_dones = torch.FloatTensor([d for d in dones]).to(self.device).unsqueeze(1)

        # Global states
        global_states = torch.FloatTensor([self._flatten_states(s) for s in states]).to(
            self.device
        )
        next_global_states = torch.FloatTensor(
            [self._flatten_states(s) for s in next_states]
        ).to(self.device)

        # Joint actions
        joint_actions = torch.FloatTensor(
            [self._flatten_actions(a) for a in actions]
        ).to(self.device)

        # Target actions from all agents
        next_actions = []
        for i, agent in enumerate(all_agents):
            next_agent_states = torch.FloatTensor([s[i] for s in next_states]).to(
                self.device
            )
            next_action = agent.actor_target(next_agent_states)
            next_actions.append(next_action)

        next_joint_actions = torch.cat(next_actions, dim=1)

        # Calculate target Q-value
        with torch.no_grad():
            target_q = self.critic_target(next_global_states, next_joint_actions)
            y = agent_rewards + self.gamma * target_q * (1 - agent_dones)

        # Current Q-value
        current_q = self.critic(global_states, joint_actions)

        # Critic loss (MSE)
        critic_loss = F.mse_loss(current_q, y)

        # Update critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()

        return critic_loss.item()

    def update_actor(self, batch, all_agents):
        """
        Update actor network using policy gradient
        ∇_θ_i J = E[∇_θ_i μ_i(a_i|o_i) * ∇_a_i Q_i(s, a_1, ..., a_N)]
        """
        states, actions, _, _, _ = batch

        # Extract agent states
        agent_states = torch.FloatTensor([s[self.agent_id] for s in states]).to(
            self.device
        )

        # Global states
        global_states = torch.FloatTensor([self._flatten_states(s) for s in states]).to(
            self.device
        )

        # Get actions from all agents
        current_actions = []
        for i, agent in enumerate(all_agents):
            if i == self.agent_id:
                # Use current actor for this agent
                action = self.actor(agent_states)
            else:
                # Use other agents' current actions
                other_states = torch.FloatTensor([s[i] for s in states]).to(self.device)
                with torch.no_grad():
                    action = agent.actor(other_states)
            current_actions.append(action)

        joint_actions = torch.cat(current_actions, dim=1)

        # Actor loss: negative Q-value (we want to maximize Q)
        actor_loss = -self.critic(global_states, joint_actions).mean()

        # Update actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optimizer.step()

        return actor_loss.item()

    def soft_update(self):
        """Soft update of target networks using Polyak averaging"""
        # Update actor target
        for target_param, param in zip(
            self.actor_target.parameters(), self.actor.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )

        # Update critic target
        for target_param, param in zip(
            self.critic_target.parameters(), self.critic.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )

    def decay_noise(self):
        """Decay exploration noise"""
        self.noise_std = max(self.noise_std * self.noise_decay, self.noise_std_end)

    def _flatten_states(self, states):
        """Flatten list of states into single array"""
        return np.concatenate(states)

    def _flatten_actions(self, actions):
        """Flatten list of actions into single array"""
        return np.concatenate(actions)

    def save(self, path):
        """Save agent networks"""
        torch.save(
            {
                "actor_state_dict": self.actor.state_dict(),
                "critic_state_dict": self.critic.state_dict(),
                "actor_target_state_dict": self.actor_target.state_dict(),
                "critic_target_state_dict": self.critic_target.state_dict(),
                "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
                "critic_optimizer_state_dict": self.critic_optimizer.state_dict(),
                "noise_std": self.noise_std,
            },
            f"{path}/agent_{self.agent_id}.pth",
        )

    def load(self, path):
        """Load agent networks"""
        checkpoint = torch.load(
            f"{path}/agent_{self.agent_id}.pth", map_location=self.device
        )
        self.actor.load_state_dict(checkpoint["actor_state_dict"])
        self.critic.load_state_dict(checkpoint["critic_state_dict"])
        self.actor_target.load_state_dict(checkpoint["actor_target_state_dict"])
        self.critic_target.load_state_dict(checkpoint["critic_target_state_dict"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer_state_dict"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer_state_dict"])
        self.noise_std = checkpoint["noise_std"]


class MADDPGTrainer:
    """MADDPG multi-agent trainer"""

    def __init__(self, env, config):
        self.env = env
        self.config = config
        self.n_agents = env.n_agents

        # Calculate dimensions
        self.state_dims = [
            len(env._get_agent_observation(i)) for i in range(self.n_agents)
        ]
        self.action_dims = env.assets_per_agent
        self.global_state_dim = (
            sum(self.state_dims) + self.n_agents
        )  # Add capital ratios
        self.total_action_dim = sum(self.action_dims)

        # Create agents
        self.agents = []
        for i in range(self.n_agents):
            agent = MADDPGAgent(
                i,
                self.state_dims[i],
                self.action_dims[i],
                self.global_state_dim,
                self.total_action_dim,
                config,
            )
            self.agents.append(agent)

        # Replay buffer
        self.replay_buffer = ReplayBuffer(config.training.buffer_size)

        # Training metrics
        self.episode_rewards = []
        self.episode_metrics = []

    def train_episode(self) -> Dict:
        """Train for one episode"""
        states = self.env.reset()
        episode_reward = np.zeros(self.n_agents)
        step_count = 0

        while not self.env.done:
            # Select actions
            actions = [
                agent.select_action(states[i], add_noise=True)
                for i, agent in enumerate(self.agents)
            ]

            # Execute actions
            next_states, rewards, done, info = self.env.step(actions)

            # Store experience
            self.replay_buffer.push(states, actions, rewards, next_states, done)

            # Update agents
            if len(self.replay_buffer) >= self.config.training.min_buffer_size:
                for _ in range(self.config.training.updates_per_step):
                    batch = self.replay_buffer.sample(self.config.training.batch_size)

                    for agent in self.agents:
                        agent.update_critic(batch, self.agents)
                        agent.update_actor(batch, self.agents)
                        agent.soft_update()

            states = next_states
            episode_reward += rewards
            step_count += 1

        # Decay noise
        for agent in self.agents:
            agent.decay_noise()

        # Get episode metrics
        metrics = self.env.get_episode_metrics()

        return {
            "episode_reward": episode_reward,
            "step_count": step_count,
            "metrics": metrics,
        }


if __name__ == "__main__":
    from config import Config
    from data_loader import MarketDataLoader
    from environment import EnhancedMultiAgentPortfolioEnv

    config = Config()
    loader = MarketDataLoader(config)
    data = loader.prepare_environment_data()
    env = EnhancedMultiAgentPortfolioEnv(config, data)

    trainer = MADDPGTrainer(env, config)

    print("MADDPG Trainer initialized successfully!")
    print(f"Number of agents: {trainer.n_agents}")
    print(f"State dimensions: {trainer.state_dims}")
    print(f"Action dimensions: {trainer.action_dims}")
    print(f"Global state dimension: {trainer.global_state_dim}")
