"""
Multi-Agent Portfolio Environment
Implements a gymnasium-compatible environment for multi-agent portfolio optimization.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Any
import gymnasium as gym
from gymnasium import spaces


class MultiAgentPortfolioEnv(gym.Env):
    """
    Multi-Agent Portfolio Optimization Environment
    
    Each agent manages a sub-portfolio and agents can coordinate to achieve
    better diversification and risk-adjusted returns.
    """
    
    def __init__(
        self, 
        data: pd.DataFrame,
        n_agents: int = 3,
        initial_capital: float = 1000000.0,
        transaction_cost: float = 0.001,
        lookback_window: int = 20,
        diversity_bonus: float = 0.1
    ):
        """
        Initialize the multi-agent portfolio environment.
        
        Args:
            data: DataFrame with columns ['date', 'asset_1', 'asset_2', ..., 'asset_n']
                  containing normalized returns or prices
            n_agents: Number of RL agents
            initial_capital: Initial capital for each agent
            transaction_cost: Transaction cost as fraction of trade value
            lookback_window: Number of past timesteps for state observation
            diversity_bonus: Weight for diversity reward component
        """
        super().__init__()
        
        self.data = data
        self.n_agents = n_agents
        self.n_assets = len(data.columns) - 1  # Exclude date column
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.lookback_window = lookback_window
        self.diversity_bonus = diversity_bonus
        
        # State space: [asset_returns(lookback), current_positions, portfolio_value, other_agents_positions]
        state_dim = (
            self.n_assets * lookback_window +  # Historical returns
            self.n_assets +  # Current positions
            1 +  # Portfolio value
            (n_agents - 1) * self.n_assets  # Other agents' positions
        )
        
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(state_dim,), 
            dtype=np.float32
        )
        
        # Action space: portfolio weights for each asset (continuous)
        self.action_space = spaces.Box(
            low=0.0, 
            high=1.0, 
            shape=(self.n_assets,), 
            dtype=np.float32
        )
        
        self.current_step = 0
        self.max_steps = len(data) - lookback_window - 1
        
        # Agent states
        self.agent_portfolios = None
        self.agent_capitals = None
        self.agent_positions = None
        
        self.reset()
    
    def reset(self, seed=None, options=None):
        """Reset the environment to initial state."""
        super().reset(seed=seed)
        
        self.current_step = self.lookback_window
        
        # Initialize agent portfolios
        self.agent_capitals = np.ones(self.n_agents) * self.initial_capital
        self.agent_positions = np.zeros((self.n_agents, self.n_assets))
        
        # Initialize with equal weights
        self.agent_positions = np.ones((self.n_agents, self.n_assets)) / self.n_assets
        
        # History tracking
        self.capital_history = [self.agent_capitals.copy()]
        self.position_history = [self.agent_positions.copy()]
        
        states = self._get_states()
        
        return states, {}
    
    def _get_states(self) -> List[np.ndarray]:
        """Get current state for all agents."""
        states = []
        
        # Get historical returns
        start_idx = self.current_step - self.lookback_window
        end_idx = self.current_step
        
        historical_returns = self.data.iloc[start_idx:end_idx, 1:].values.flatten()
        
        for agent_id in range(self.n_agents):
            # Agent's current positions
            current_positions = self.agent_positions[agent_id]
            
            # Normalized portfolio value
            normalized_value = self.agent_capitals[agent_id] / self.initial_capital
            
            # Other agents' positions (for coordination)
            other_positions = []
            for other_id in range(self.n_agents):
                if other_id != agent_id:
                    other_positions.extend(self.agent_positions[other_id])
            
            # Concatenate state components
            state = np.concatenate([
                historical_returns,
                current_positions,
                [normalized_value],
                other_positions
            ]).astype(np.float32)
            
            states.append(state)
        
        return states
    
    def _calculate_diversity_score(self) -> float:
        """
        Calculate portfolio diversity score across all agents.
        Higher score means more diverse allocation.
        """
        # Calculate correlation between agent portfolios
        portfolio_matrix = self.agent_positions  # Shape: (n_agents, n_assets)
        
        # Normalize portfolios
        portfolio_matrix = portfolio_matrix / (np.sum(portfolio_matrix, axis=1, keepdims=True) + 1e-8)
        
        # Calculate pairwise cosine similarity
        similarities = []
        for i in range(self.n_agents):
            for j in range(i + 1, self.n_agents):
                similarity = np.dot(portfolio_matrix[i], portfolio_matrix[j])
                similarities.append(similarity)
        
        # Diversity is inverse of average similarity
        avg_similarity = np.mean(similarities) if similarities else 0.0
        diversity_score = 1.0 - avg_similarity
        
        return diversity_score
    
    def step(self, actions: List[np.ndarray]) -> Tuple[List[np.ndarray], List[float], bool, bool, Dict]:
        """
        Execute one time step in the environment.
        
        Args:
            actions: List of action arrays, one per agent
            
        Returns:
            observations: List of next states for each agent
            rewards: List of rewards for each agent
            terminated: Whether episode is finished
            truncated: Whether episode was truncated
            info: Additional information
        """
        # Normalize actions to sum to 1 (portfolio weights)
        normalized_actions = []
        for action in actions:
            action = np.clip(action, 0, 1)
            action = action / (np.sum(action) + 1e-8)
            normalized_actions.append(action)
        
        # Get current and next returns
        current_prices = self.data.iloc[self.current_step, 1:].values
        next_prices = self.data.iloc[self.current_step + 1, 1:].values
        returns = (next_prices - current_prices) / (current_prices + 1e-8)
        
        rewards = []
        transaction_costs = []
        
        for agent_id in range(self.n_agents):
            old_positions = self.agent_positions[agent_id]
            new_positions = normalized_actions[agent_id]
            
            # Calculate transaction cost
            position_change = np.abs(new_positions - old_positions)
            transaction_cost = np.sum(position_change) * self.transaction_cost * self.agent_capitals[agent_id]
            transaction_costs.append(transaction_cost)
            
            # Update positions
            self.agent_positions[agent_id] = new_positions
            
            # Calculate portfolio return
            portfolio_return = np.dot(new_positions, returns)
            
            # Update capital
            old_capital = self.agent_capitals[agent_id]
            self.agent_capitals[agent_id] = old_capital * (1 + portfolio_return) - transaction_cost
            
            # Individual reward: portfolio return - transaction cost
            individual_return = (self.agent_capitals[agent_id] - old_capital) / old_capital
            rewards.append(individual_return)
        
        # Calculate diversity bonus
        diversity_score = self._calculate_diversity_score()
        
        # Add diversity bonus to all agents
        rewards = [r + self.diversity_bonus * diversity_score for r in rewards]
        
        # Move to next step
        self.current_step += 1
        
        # Record history
        self.capital_history.append(self.agent_capitals.copy())
        self.position_history.append(self.agent_positions.copy())
        
        # Check if episode is done
        terminated = self.current_step >= self.max_steps
        truncated = False
        
        # Get next states
        next_states = self._get_states() if not terminated else [np.zeros_like(self.observation_space.sample()) for _ in range(self.n_agents)]
        
        # Info dictionary
        info = {
            'agent_capitals': self.agent_capitals.copy(),
            'diversity_score': diversity_score,
            'transaction_costs': transaction_costs,
            'returns': returns
        }
        
        return next_states, rewards, terminated, truncated, info
    
    def get_portfolio_metrics(self) -> Dict[str, float]:
        """Calculate portfolio performance metrics."""
        capital_history = np.array(self.capital_history)
        
        # Aggregate portfolio (sum of all agents)
        aggregate_capital = np.sum(capital_history, axis=1)
        
        # Returns
        returns = np.diff(aggregate_capital) / aggregate_capital[:-1]
        
        # Cumulative return
        cumulative_return = (aggregate_capital[-1] - aggregate_capital[0]) / aggregate_capital[0]
        
        # Annualized return (assuming daily data, 252 trading days)
        n_periods = len(aggregate_capital)
        annualized_return = (1 + cumulative_return) ** (252 / n_periods) - 1
        
        # Volatility
        volatility = np.std(returns) * np.sqrt(252)
        
        # Sharpe ratio (assuming 0% risk-free rate)
        sharpe_ratio = annualized_return / (volatility + 1e-8)
        
        # Sortino ratio (downside deviation)
        downside_returns = returns[returns < 0]
        downside_deviation = np.std(downside_returns) * np.sqrt(252) if len(downside_returns) > 0 else 1e-8
        sortino_ratio = annualized_return / (downside_deviation + 1e-8)
        
        # Maximum drawdown
        cumulative_returns = aggregate_capital / aggregate_capital[0]
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = np.min(drawdown)
        
        # Calculate average turnover
        position_history = np.array(self.position_history)
        turnovers = []
        for agent_id in range(self.n_agents):
            agent_positions = position_history[:, agent_id, :]
            position_changes = np.sum(np.abs(np.diff(agent_positions, axis=0)), axis=1)
            turnovers.append(np.mean(position_changes))
        avg_turnover = np.mean(turnovers)
        
        return {
            'cumulative_return': cumulative_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown': max_drawdown,
            'avg_turnover': avg_turnover
        }
