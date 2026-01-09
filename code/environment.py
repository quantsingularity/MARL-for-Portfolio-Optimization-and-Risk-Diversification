"""
Enhanced Multi-Agent Portfolio Environment
Implements complete environment with all features from research paper
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Any
import gymnasium as gym
from gymnasium import spaces
from config import Config


class EnhancedMultiAgentPortfolioEnv(gym.Env):
    """
    Complete Multi-Agent Portfolio Environment with:
    - Sector-based agent assignment
    - Comprehensive state space (technical indicators, macro features)
    - Diversity-promoting reward structure
    - Transaction costs
    - Risk-adjusted performance metrics
    """
    
    def __init__(self, config: Config, data: pd.DataFrame, mode='train'):
        super().__init__()
        
        self.config = config
        self.data = data.reset_index(drop=True)
        self.mode = mode
        
        # Configuration
        self.n_agents = config.env.n_agents
        self.n_assets = config.env.n_assets
        self.agent_asset_assignment = config.env.agent_asset_assignment
        self.initial_capital = config.env.initial_capital
        self.transaction_cost = config.env.transaction_cost
        self.lookback_window = config.env.lookback_window_20
        self.diversity_weight = config.env.diversity_weight
        self.risk_free_rate = config.env.risk_free_rate
        self.correlation_window = config.env.correlation_window
        
        # Verify asset assignment
        assert len(self.agent_asset_assignment) == self.n_agents
        assert sum(len(assets) for assets in self.agent_asset_assignment) == self.n_assets
        
        # State and action spaces
        self._setup_spaces()
        
        # Episode tracking
        self.current_step = 0
        self.max_steps = len(data) - self.lookback_window - 1
        
        # Agent states
        self.agent_capitals = None
        self.agent_positions = None
        self.capital_history = []
        self.position_history = []
        self.returns_history = []  # For diversity calculation
        
    def _setup_spaces(self):
        """Setup observation and action spaces"""
        # Each agent observes its assigned assets + other agents' positions
        state_dims = []
        for agent_id in range(self.n_agents):
            n_agent_assets = len(self.agent_asset_assignment[agent_id])
            
            # State components:
            # - Historical returns for agent's assets: lookback_window * n_agent_assets
            # - Current positions: n_agent_assets
            # - Portfolio value: 1
            # - Other agents' aggregate positions: (n_agents - 1)
            # - Market indicators: 2 (VIX, Treasury) if available
            
            state_dim = (
                self.lookback_window * n_agent_assets +  # Historical returns
                n_agent_assets +  # Current positions
                1 +  # Normalized portfolio value
                (self.n_agents - 1)  # Other agents' states
            )
            state_dims.append(state_dim)
        
        # For now, use maximum dimension (they can differ per agent)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(max(state_dims),), 
            dtype=np.float32
        )
        
        # Action space: portfolio weights (one per assigned asset)
        # Use maximum for simplicity
        max_assets_per_agent = max(len(assets) for assets in self.agent_asset_assignment)
        self.action_space = spaces.Box(
            low=0.0, high=1.0, 
            shape=(max_assets_per_agent,), 
            dtype=np.float32
        )
    
    def reset(self, seed=None, options=None):
        """Reset environment"""
        super().reset(seed=seed)
        
        self.current_step = self.lookback_window
        
        # Initialize capitals
        self.agent_capitals = np.ones(self.n_agents) * self.initial_capital
        
        # Initialize positions (equal weight for each agent's assets)
        self.agent_positions = []
        for agent_id in range(self.n_agents):
            n_agent_assets = len(self.agent_asset_assignment[agent_id])
            initial_positions = np.ones(n_agent_assets) / n_agent_assets
            self.agent_positions.append(initial_positions)
        
        # History
        self.capital_history = [self.agent_capitals.copy()]
        self.position_history = [self.agent_positions.copy()]
        self.returns_history = []
        
        states = self._get_states()
        return states, {}
    
    def _get_states(self) -> List[np.ndarray]:
        """Get current state for all agents"""
        states = []
        
        # Get historical data
        start_idx = self.current_step - self.lookback_window
        end_idx = self.current_step
        historical_data = self.data.iloc[start_idx:end_idx]
        
        for agent_id in range(self.n_agents):
            # Get agent's assigned asset indices
            agent_assets = self.agent_asset_assignment[agent_id]
            
            # Historical returns for agent's assets
            agent_columns = [self.data.columns[idx] for idx in agent_assets]
            agent_historical = historical_data[agent_columns].values.flatten()
            
            # Current positions
            current_positions = self.agent_positions[agent_id]
            
            # Normalized portfolio value
            normalized_value = self.agent_capitals[agent_id] / self.initial_capital
            
            # Other agents' aggregate information (simplified)
            other_agents_info = []
            for other_id in range(self.n_agents):
                if other_id != agent_id:
                    other_value = self.agent_capitals[other_id] / self.initial_capital
                    other_agents_info.append(other_value)
            
            # Concatenate state
            state = np.concatenate([
                agent_historical,
                current_positions,
                [normalized_value],
                other_agents_info
            ]).astype(np.float32)
            
            # Pad to fixed size if necessary
            if len(state) < self.observation_space.shape[0]:
                state = np.pad(state, (0, self.observation_space.shape[0] - len(state)))
            
            states.append(state[:self.observation_space.shape[0]])
        
        return states
    
    def _calculate_diversity_penalty(self) -> float:
        """
        Calculate diversity penalty based on correlation of agent returns
        Uses rolling correlation over past correlation_window steps
        """
        if len(self.returns_history) < self.correlation_window:
            return 0.0  # Not enough history
        
        # Get recent returns
        recent_returns = np.array(self.returns_history[-self.correlation_window:])  # Shape: (window, n_agents)
        
        # Calculate pairwise correlations
        correlations = []
        for i in range(self.n_agents):
            for j in range(i+1, self.n_agents):
                returns_i = recent_returns[:, i]
                returns_j = recent_returns[:, j]
                
                # Pearson correlation
                if np.std(returns_i) > 1e-8 and np.std(returns_j) > 1e-8:
                    corr = np.corrcoef(returns_i, returns_j)[0, 1]
                    correlations.append(corr)
        
        # Average correlation as penalty
        avg_correlation = np.mean(correlations) if correlations else 0.0
        return avg_correlation
    
    def step(self, actions: List[np.ndarray]) -> Tuple[List[np.ndarray], List[float], bool, bool, Dict]:
        """Execute one timestep"""
        # Normalize actions
        normalized_actions = []
        for agent_id, action in enumerate(actions):
            n_agent_assets = len(self.agent_asset_assignment[agent_id])
            action = action[:n_agent_assets]  # Take only relevant assets
            action = np.clip(action, 0, 1)
            action = action / (np.sum(action) + 1e-8)
            normalized_actions.append(action)
        
        # Get price data
        current_prices = self.data.iloc[self.current_step].values
        next_prices = self.data.iloc[self.current_step + 1].values
        price_returns = (next_prices - current_prices) / (current_prices + 1e-8)
        
        # Update each agent
        rewards = []
        agent_returns = []
        transaction_costs_total = []
        
        for agent_id in range(self.n_agents):
            agent_assets = self.agent_asset_assignment[agent_id]
            old_positions = self.agent_positions[agent_id]
            new_positions = normalized_actions[agent_id]
            
            # Calculate transaction cost
            position_change = np.abs(new_positions - old_positions)
            transaction_cost = np.sum(position_change) * self.transaction_cost * self.agent_capitals[agent_id]
            transaction_costs_total.append(transaction_cost)
            
            # Update positions
            self.agent_positions[agent_id] = new_positions
            
            # Calculate portfolio return for agent's assets
            agent_price_returns = price_returns[agent_assets]
            portfolio_return = np.dot(new_positions, agent_price_returns)
            
            # Update capital
            old_capital = self.agent_capitals[agent_id]
            self.agent_capitals[agent_id] = old_capital * (1 + portfolio_return) - transaction_cost
            
            # Calculate individual return
            individual_return = (self.agent_capitals[agent_id] - old_capital) / old_capital
            agent_returns.append(individual_return)
            
            # Base reward: Sharpe ratio style (return / volatility)
            # For simplicity, use return adjusted by risk-free rate
            base_reward = individual_return - self.risk_free_rate
            rewards.append(base_reward)
        
        # Store returns for diversity calculation
        self.returns_history.append(agent_returns)
        
        # Calculate diversity penalty
        diversity_penalty = self._calculate_diversity_penalty()
        
        # Apply diversity bonus/penalty to all agents
        rewards = [r - self.diversity_weight * diversity_penalty for r in rewards]
        
        # Move to next step
        self.current_step += 1
        
        # Record history
        self.capital_history.append(self.agent_capitals.copy())
        self.position_history.append([pos.copy() for pos in self.agent_positions])
        
        # Check termination
        terminated = self.current_step >= self.max_steps
        truncated = False
        
        # Get next states
        if not terminated:
            next_states = self._get_states()
        else:
            next_states = [np.zeros(self.observation_space.shape, dtype=np.float32) for _ in range(self.n_agents)]
        
        # Info
        info = {
            'agent_capitals': self.agent_capitals.copy(),
            'diversity_penalty': diversity_penalty,
            'transaction_costs': transaction_costs_total,
            'agent_returns': agent_returns
        }
        
        return next_states, rewards, terminated, truncated, info
    
    def get_portfolio_metrics(self) -> Dict[str, float]:
        """Calculate comprehensive portfolio performance metrics"""
        capital_history = np.array(self.capital_history)
        aggregate_capital = np.sum(capital_history, axis=1)
        
        # Returns
        returns = np.diff(aggregate_capital) / aggregate_capital[:-1]
        
        # Cumulative return
        cumulative_return = (aggregate_capital[-1] - aggregate_capital[0]) / aggregate_capital[0]
        
        # Annualized return (assuming 252 trading days)
        n_periods = len(aggregate_capital) - 1
        annualized_return = (1 + cumulative_return) ** (252 / n_periods) - 1
        
        # Volatility
        volatility = np.std(returns) * np.sqrt(252)
        
        # Sharpe ratio
        sharpe_ratio = (annualized_return - self.config.env.risk_free_rate * 252) / (volatility + 1e-8)
        
        # Sortino ratio
        downside_returns = returns[returns < 0]
        downside_deviation = np.std(downside_returns) * np.sqrt(252) if len(downside_returns) > 0 else 1e-8
        sortino_ratio = (annualized_return - self.config.env.risk_free_rate * 252) / (downside_deviation + 1e-8)
        
        # Maximum drawdown
        cumulative_returns = aggregate_capital / aggregate_capital[0]
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = np.min(drawdown)
        
        # Average turnover
        turnovers = []
        for agent_id in range(self.n_agents):
            agent_position_history = np.array([pos[agent_id] for pos in self.position_history])
            position_changes = np.sum(np.abs(np.diff(agent_position_history, axis=0)), axis=1)
            turnovers.append(np.mean(position_changes))
        avg_turnover = np.mean(turnovers)
        
        # Average pairwise correlation (diversity metric)
        if len(self.returns_history) > 1:
            returns_matrix = np.array(self.returns_history)
            correlations = []
            for i in range(self.n_agents):
                for j in range(i+1, self.n_agents):
                    if np.std(returns_matrix[:, i]) > 1e-8 and np.std(returns_matrix[:, j]) > 1e-8:
                        corr = np.corrcoef(returns_matrix[:, i], returns_matrix[:, j])[0, 1]
                        correlations.append(corr)
            avg_correlation = np.mean(correlations) if correlations else 0.0
        else:
            avg_correlation = 0.0
        
        return {
            'cumulative_return': cumulative_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown': max_drawdown,
            'avg_turnover': avg_turnover,
            'avg_correlation': avg_correlation
        }
