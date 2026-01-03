"""
Multi-Agent Deep Deterministic Policy Gradient (MADDPG) Implementation
For portfolio optimization with cooperative agents.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import random
from typing import List, Tuple


class ReplayBuffer:
    """Experience replay buffer for MARL."""
    
    def __init__(self, capacity: int = 100000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, states, actions, rewards, next_states, dones):
        """Add experience to buffer."""
        self.buffer.append((states, actions, rewards, next_states, dones))
    
    def sample(self, batch_size: int):
        """Sample a batch of experiences."""
        batch = random.sample(self.buffer, batch_size)
        
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones)
        )
    
    def __len__(self):
        return len(self.buffer)


class Actor(nn.Module):
    """Actor network for MADDPG."""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super(Actor, self).__init__()
        
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        
        self.fc3 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.ln3 = nn.LayerNorm(hidden_dim // 2)
        
        self.fc4 = nn.Linear(hidden_dim // 2, action_dim)
        
    def forward(self, state):
        x = F.relu(self.ln1(self.fc1(state)))
        x = F.relu(self.ln2(self.fc2(x)))
        x = F.relu(self.ln3(self.fc3(x)))
        
        # Softmax to ensure portfolio weights sum to 1
        action = F.softmax(self.fc4(x), dim=-1)
        
        return action


class Critic(nn.Module):
    """Critic network for MADDPG (centralized)."""
    
    def __init__(self, state_dim: int, action_dim: int, n_agents: int, hidden_dim: int = 256):
        super(Critic, self).__init__()
        
        # Critic sees all states and actions
        total_input_dim = (state_dim + action_dim) * n_agents
        
        self.fc1 = nn.Linear(total_input_dim, hidden_dim * 2)
        self.ln1 = nn.LayerNorm(hidden_dim * 2)
        
        self.fc2 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        
        self.fc3 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.ln3 = nn.LayerNorm(hidden_dim // 2)
        
        self.fc4 = nn.Linear(hidden_dim // 2, 1)
        
    def forward(self, states, actions):
        """
        Forward pass.
        
        Args:
            states: Concatenated states of all agents
            actions: Concatenated actions of all agents
        """
        x = torch.cat([states, actions], dim=-1)
        
        x = F.relu(self.ln1(self.fc1(x)))
        x = F.relu(self.ln2(self.fc2(x)))
        x = F.relu(self.ln3(self.fc3(x)))
        
        q_value = self.fc4(x)
        
        return q_value


class MADDPGAgent:
    """Single agent in MADDPG framework."""
    
    def __init__(
        self,
        agent_id: int,
        state_dim: int,
        action_dim: int,
        n_agents: int,
        lr_actor: float = 1e-4,
        lr_critic: float = 1e-3,
        gamma: float = 0.99,
        tau: float = 0.01,
        hidden_dim: int = 256
    ):
        self.agent_id = agent_id
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.n_agents = n_agents
        self.gamma = gamma
        self.tau = tau
        
        # Actor networks
        self.actor = Actor(state_dim, action_dim, hidden_dim)
        self.actor_target = Actor(state_dim, action_dim, hidden_dim)
        self.actor_target.load_state_dict(self.actor.state_dict())
        
        # Critic networks (centralized)
        self.critic = Critic(state_dim, action_dim, n_agents, hidden_dim)
        self.critic_target = Critic(state_dim, action_dim, n_agents, hidden_dim)
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)
        
        # Exploration noise
        self.noise_scale = 0.1
        self.noise_decay = 0.9995
        self.min_noise = 0.01
        
    def select_action(self, state: np.ndarray, explore: bool = True) -> np.ndarray:
        """Select action using actor network."""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
        with torch.no_grad():
            action = self.actor(state_tensor).squeeze(0).numpy()
        
        if explore:
            # Add exploration noise
            noise = np.random.dirichlet(np.ones(self.action_dim) * 0.5) * self.noise_scale
            action = action + noise
            
            # Normalize to ensure valid portfolio weights
            action = np.clip(action, 0, 1)
            action = action / (np.sum(action) + 1e-8)
            
            # Decay noise
            self.noise_scale = max(self.min_noise, self.noise_scale * self.noise_decay)
        
        return action
    
    def update(
        self,
        batch_states: torch.Tensor,
        batch_actions: torch.Tensor,
        batch_rewards: torch.Tensor,
        batch_next_states: torch.Tensor,
        batch_dones: torch.Tensor,
        all_next_actions: torch.Tensor
    ):
        """
        Update actor and critic networks.
        
        Args:
            batch_states: States of all agents [batch_size, n_agents, state_dim]
            batch_actions: Actions of all agents [batch_size, n_agents, action_dim]
            batch_rewards: Rewards for this agent [batch_size, 1]
            batch_next_states: Next states of all agents [batch_size, n_agents, state_dim]
            batch_dones: Done flags [batch_size, 1]
            all_next_actions: Next actions of all agents from target actors [batch_size, n_agents, action_dim]
        """
        # Flatten states and actions for critic
        batch_size = batch_states.shape[0]
        
        flat_states = batch_states.reshape(batch_size, -1)
        flat_actions = batch_actions.reshape(batch_size, -1)
        flat_next_states = batch_next_states.reshape(batch_size, -1)
        flat_next_actions = all_next_actions.reshape(batch_size, -1)
        
        # Update Critic
        with torch.no_grad():
            target_q = self.critic_target(flat_next_states, flat_next_actions)
            target_q = batch_rewards + self.gamma * target_q * (1 - batch_dones)
        
        current_q = self.critic(flat_states, flat_actions)
        critic_loss = F.mse_loss(current_q, target_q)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()
        
        # Update Actor
        # Get current agent's action from actor
        agent_action = self.actor(batch_states[:, self.agent_id, :])
        
        # Replace this agent's action in the action tensor
        updated_actions = batch_actions.clone()
        updated_actions[:, self.agent_id, :] = agent_action
        
        flat_updated_actions = updated_actions.reshape(batch_size, -1)
        
        actor_loss = -self.critic(flat_states, flat_updated_actions).mean()
        
        # Add entropy bonus for exploration
        entropy = -torch.sum(agent_action * torch.log(agent_action + 1e-8), dim=-1).mean()
        actor_loss = actor_loss - 0.01 * entropy
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optimizer.step()
        
        # Soft update target networks
        self._soft_update(self.actor, self.actor_target)
        self._soft_update(self.critic, self.critic_target)
        
        return {
            'critic_loss': critic_loss.item(),
            'actor_loss': actor_loss.item(),
            'q_value': current_q.mean().item()
        }
    
    def _soft_update(self, source: nn.Module, target: nn.Module):
        """Soft update of target network parameters."""
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.tau) + param.data * self.tau
            )


class MADDPG:
    """Multi-Agent DDPG framework."""
    
    def __init__(
        self,
        n_agents: int,
        state_dim: int,
        action_dim: int,
        lr_actor: float = 1e-4,
        lr_critic: float = 1e-3,
        gamma: float = 0.99,
        tau: float = 0.01,
        hidden_dim: int = 256,
        buffer_capacity: int = 100000
    ):
        self.n_agents = n_agents
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Create agents
        self.agents = [
            MADDPGAgent(
                agent_id=i,
                state_dim=state_dim,
                action_dim=action_dim,
                n_agents=n_agents,
                lr_actor=lr_actor,
                lr_critic=lr_critic,
                gamma=gamma,
                tau=tau,
                hidden_dim=hidden_dim
            )
            for i in range(n_agents)
        ]
        
        # Shared replay buffer
        self.replay_buffer = ReplayBuffer(buffer_capacity)
        
    def select_actions(self, states: List[np.ndarray], explore: bool = True) -> List[np.ndarray]:
        """Select actions for all agents."""
        actions = []
        for i, state in enumerate(states):
            action = self.agents[i].select_action(state, explore)
            actions.append(action)
        return actions
    
    def update(self, batch_size: int = 64):
        """Update all agents."""
        if len(self.replay_buffer) < batch_size:
            return None
        
        # Sample batch
        batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones = \
            self.replay_buffer.sample(batch_size)
        
        # Convert to tensors
        batch_states = torch.FloatTensor(batch_states)
        batch_actions = torch.FloatTensor(batch_actions)
        batch_rewards = torch.FloatTensor(batch_rewards)
        batch_next_states = torch.FloatTensor(batch_next_states)
        batch_dones = torch.FloatTensor(batch_dones)
        
        # Get next actions from target actors for all agents
        all_next_actions = []
        for i in range(self.n_agents):
            with torch.no_grad():
                next_action = self.agents[i].actor_target(batch_next_states[:, i, :])
                all_next_actions.append(next_action)
        
        all_next_actions = torch.stack(all_next_actions, dim=1)
        
        # Update each agent
        losses = {}
        for i in range(self.n_agents):
            agent_rewards = batch_rewards[:, i:i+1]
            agent_dones = batch_dones[:, i:i+1]
            
            agent_losses = self.agents[i].update(
                batch_states,
                batch_actions,
                agent_rewards,
                batch_next_states,
                agent_dones,
                all_next_actions
            )
            
            losses[f'agent_{i}'] = agent_losses
        
        return losses
    
    def save(self, path: str):
        """Save all agent models."""
        import os
        os.makedirs(path, exist_ok=True)
        for i, agent in enumerate(self.agents):
            torch.save({
                'actor': agent.actor.state_dict(),
                'critic': agent.critic.state_dict(),
                'actor_target': agent.actor_target.state_dict(),
                'critic_target': agent.critic_target.state_dict(),
            }, f"{path}/agent_{i}.pt")
    
    def load(self, path: str):
        """Load all agent models."""
        for i, agent in enumerate(self.agents):
            checkpoint = torch.load(f"{path}/agent_{i}.pt")
            agent.actor.load_state_dict(checkpoint['actor'])
            agent.critic.load_state_dict(checkpoint['critic'])
            agent.actor_target.load_state_dict(checkpoint['actor_target'])
            agent.critic_target.load_state_dict(checkpoint['critic_target'])
