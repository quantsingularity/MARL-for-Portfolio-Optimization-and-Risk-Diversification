"""
Multi-Agent Portfolio Environment
Implements the complete MDP formulation from the research paper
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import deque


class EnhancedMultiAgentPortfolioEnv:
    """
    Multi-Agent Portfolio Environment with diversity-promoting rewards
    Implements Section 3 (Problem Formulation) from the paper
    """

    def __init__(self, config, data: Dict):
        self.config = config
        self.data = data

        # Environment parameters
        self.n_agents = config.env.n_agents
        self.n_assets = config.env.n_assets
        self.initial_capital = config.env.initial_capital
        self.transaction_cost = config.env.transaction_cost
        self.risk_free_rate = config.env.risk_free_rate
        self.diversity_weight = config.env.diversity_weight

        # Data
        self.prices = data["prices"]
        self.returns = data["returns"]
        self.indicators = data["indicators"]
        self.hist_returns = data["hist_returns"]
        self.volatility = data["volatility"]
        self.macro = data["macro"]
        self.tickers = data["tickers"]
        self.sector_allocations = data["sector_allocations"]

        # Agent-asset mapping
        self.agent_assets = self._create_agent_asset_mapping()
        self.assets_per_agent = [len(assets) for assets in self.agent_assets]

        # Episode state
        self.current_step = 0
        self.episode_start = 0
        self.episode_end = 0
        self.done = False

        # Agent portfolios
        self.agent_capitals = np.zeros(self.n_agents)
        self.agent_positions = [np.zeros(n) for n in self.assets_per_agent]
        self.agent_weights = [np.zeros(n) for n in self.assets_per_agent]

        # History for reward calculation
        self.agent_return_history = [
            deque(maxlen=config.env.correlation_window) for _ in range(self.n_agents)
        ]
        self.agent_volatility_history = [
            deque(maxlen=config.env.volatility_window) for _ in range(self.n_agents)
        ]

        # Metrics tracking
        self.episode_rewards = []
        self.portfolio_values = []
        self.agent_sharpe_ratios = []

    def _create_agent_asset_mapping(self) -> List[List[str]]:
        """Create mapping of agents to their assigned assets"""
        agent_assets = []
        sectors = list(self.sector_allocations.keys())

        for i, sector in enumerate(sectors):
            agent_assets.append(self.sector_allocations[sector])

        return agent_assets

    def reset(
        self, start_idx: Optional[int] = None, end_idx: Optional[int] = None
    ) -> List[np.ndarray]:
        """Reset environment for new episode"""
        # Set episode boundaries
        if start_idx is None:
            train_start, train_end = self.data["train_indices"]
            self.episode_start = train_start + max(
                self.config.env.lookback_long, self.config.env.correlation_window
            )
            self.episode_end = train_end
        else:
            self.episode_start = start_idx
            self.episode_end = end_idx

        self.current_step = self.episode_start
        self.done = False

        # Reset agent portfolios
        self.agent_capitals = np.ones(self.n_agents) * self.initial_capital
        self.agent_positions = [np.zeros(n) for n in self.assets_per_agent]
        self.agent_weights = [
            np.ones(n) / n for n in self.assets_per_agent
        ]  # Equal weight initial

        # Reset history
        self.agent_return_history = [
            deque(maxlen=self.config.env.correlation_window)
            for _ in range(self.n_agents)
        ]
        self.agent_volatility_history = [
            deque(maxlen=self.config.env.volatility_window)
            for _ in range(self.n_agents)
        ]

        # Reset metrics
        self.episode_rewards = []
        self.portfolio_values = [self.agent_capitals.copy()]

        return self._get_observations()

    def _get_observations(self) -> List[np.ndarray]:
        """Get observations for all agents"""
        observations = []

        for agent_id in range(self.n_agents):
            obs = self._get_agent_observation(agent_id)
            observations.append(obs)

        return observations

    def _get_agent_observation(self, agent_id: int) -> np.ndarray:
        """
        Get observation for a specific agent
        State space from paper Section 3.1:
        - Historical returns (20-day, 60-day)
        - Technical indicators (RSI, MACD, Bollinger Bands)
        - Volatility measures
        - Macroeconomic variables
        """
        agent_tickers = self.agent_assets[agent_id]
        obs_list = []

        for ticker in agent_tickers:
            # Historical returns
            ret_20d = self.hist_returns[ticker].iloc[self.current_step][
                f"return_{self.config.env.lookback_short}d"
            ]
            ret_60d = self.hist_returns[ticker].iloc[self.current_step][
                f"return_{self.config.env.lookback_long}d"
            ]

            # Technical indicators
            ind = self.indicators[ticker].iloc[self.current_step]
            rsi = ind["rsi"]
            macd = ind["macd"]
            macd_signal = ind["macd_signal"]
            bb_position = ind["bb_position"]
            bb_width = ind["bb_width"]

            # Volatility
            vol = self.volatility[ticker].iloc[self.current_step]

            # Combine features for this asset
            asset_features = [
                ret_20d,
                ret_60d,
                rsi,
                macd,
                macd_signal,
                bb_position,
                bb_width,
                vol,
            ]
            obs_list.extend(asset_features)

        # Add macroeconomic features
        macro = self.macro.iloc[self.current_step]
        obs_list.append(macro["vix"])
        obs_list.append(macro["treasury_yield"])

        # Add current portfolio weights
        obs_list.extend(self.agent_weights[agent_id])

        return np.array(obs_list, dtype=np.float32)

    def _get_global_state(self) -> np.ndarray:
        """Get global state for centralized critic"""
        # Concatenate all agent observations
        obs = self._get_observations()
        global_state = np.concatenate(obs)

        # Add aggregate portfolio info
        total_capital = np.sum(self.agent_capitals)
        agent_capital_ratios = self.agent_capitals / (total_capital + 1e-8)
        global_state = np.concatenate([global_state, agent_capital_ratios])

        return global_state

    def step(
        self, actions: List[np.ndarray]
    ) -> Tuple[List[np.ndarray], List[float], bool, Dict]:
        """
        Execute one step in the environment
        Args:
            actions: List of action arrays (portfolio weights) for each agent
        Returns:
            observations, rewards, done, info
        """
        # Store previous capitals for return calculation
        self.agent_capitals.copy()

        # Execute actions for each agent
        transaction_costs = np.zeros(self.n_agents)

        for agent_id in range(self.n_agents):
            action = actions[agent_id]

            # Ensure valid portfolio weights (should already be from softmax)
            action = np.clip(action, 0, 1)
            action = action / (np.sum(action) + 1e-8)

            # Calculate position changes
            current_weights = self.agent_weights[agent_id]
            weight_changes = np.abs(action - current_weights)
            turnover = np.sum(weight_changes)

            # Transaction costs
            tc = self.transaction_cost * turnover * self.agent_capitals[agent_id]
            transaction_costs[agent_id] = tc

            # Update positions
            self.agent_weights[agent_id] = action
            self.agent_positions[agent_id] = action * self.agent_capitals[agent_id]

        # Move to next time step
        self.current_step += 1

        # Calculate returns and update capitals
        agent_returns = np.zeros(self.n_agents)

        for agent_id in range(self.n_agents):
            agent_tickers = self.agent_assets[agent_id]
            asset_returns = self.returns[agent_tickers].iloc[self.current_step].values

            # Portfolio return
            portfolio_return = np.dot(self.agent_weights[agent_id], asset_returns)
            agent_returns[agent_id] = portfolio_return

            # Update capital
            self.agent_capitals[agent_id] = (
                self.agent_capitals[agent_id] * (1 + portfolio_return)
                - transaction_costs[agent_id]
            )

            # Update history
            self.agent_return_history[agent_id].append(portfolio_return)

        # Calculate rewards
        rewards = self._calculate_rewards(agent_returns)

        # Store metrics
        self.portfolio_values.append(self.agent_capitals.copy())
        self.episode_rewards.append(rewards)

        # Check if episode is done
        self.done = self.current_step >= self.episode_end - 1

        # Get next observations
        next_observations = (
            self._get_observations()
            if not self.done
            else [np.zeros_like(o) for o in self._get_observations()]
        )

        # Info dictionary
        info = {
            "agent_capitals": self.agent_capitals,
            "agent_returns": agent_returns,
            "transaction_costs": transaction_costs,
            "portfolio_values": self.agent_capitals.copy(),
            "step": self.current_step,
        }

        return next_observations, rewards, self.done, info

    def _calculate_rewards(self, agent_returns: np.ndarray) -> List[float]:
        """
        Calculate diversity-promoting rewards
        From paper Section 3.3: R_i,t = R_base,i,t - λ * D_i,t
        """
        rewards = []

        for agent_id in range(self.n_agents):
            # Base reward: Daily Sharpe ratio
            if len(self.agent_return_history[agent_id]) > 1:
                returns_array = np.array(self.agent_return_history[agent_id])
                mean_return = np.mean(returns_array)
                std_return = np.std(returns_array) + 1e-8
                sharpe = (mean_return - self.risk_free_rate) / std_return
                base_reward = sharpe
            else:
                base_reward = agent_returns[agent_id] - self.risk_free_rate

            # Diversity penalty: correlation with other agents
            diversity_penalty = 0.0

            if (
                len(self.agent_return_history[agent_id])
                >= self.config.env.correlation_window
            ):
                # Calculate average pairwise correlation
                correlations = []
                returns_i = np.array(self.agent_return_history[agent_id])

                for other_id in range(self.n_agents):
                    if (
                        other_id != agent_id
                        and len(self.agent_return_history[other_id])
                        >= self.config.env.correlation_window
                    ):
                        returns_j = np.array(self.agent_return_history[other_id])

                        # Pearson correlation
                        corr = np.corrcoef(returns_i, returns_j)[0, 1]
                        if not np.isnan(corr):
                            correlations.append(abs(corr))  # Use absolute correlation

                if correlations:
                    diversity_penalty = np.mean(correlations)

            # Total reward with diversity weight λ
            total_reward = base_reward - self.diversity_weight * diversity_penalty
            rewards.append(total_reward)

        return rewards

    def get_episode_metrics(self) -> Dict:
        """Calculate episode performance metrics"""
        portfolio_values_array = np.array(self.portfolio_values)

        metrics = {}

        for agent_id in range(self.n_agents):
            agent_values = portfolio_values_array[:, agent_id]
            agent_returns_arr = np.diff(agent_values) / agent_values[:-1]

            # Sharpe ratio
            if len(agent_returns_arr) > 1:
                sharpe = (np.mean(agent_returns_arr) - self.risk_free_rate) / (
                    np.std(agent_returns_arr) + 1e-8
                )
                sharpe_annualized = sharpe * np.sqrt(252)
            else:
                sharpe_annualized = 0.0

            # Total return
            total_return = (agent_values[-1] - agent_values[0]) / agent_values[0]

            # Maximum drawdown
            cummax = np.maximum.accumulate(agent_values)
            drawdown = (agent_values - cummax) / cummax
            max_drawdown = np.min(drawdown)

            metrics[f"agent_{agent_id}"] = {
                "sharpe_ratio": sharpe_annualized,
                "total_return": total_return,
                "max_drawdown": max_drawdown,
                "final_capital": agent_values[-1],
            }

        # Aggregate metrics
        total_capital = np.sum(portfolio_values_array, axis=1)
        aggregate_returns = np.diff(total_capital) / total_capital[:-1]

        metrics["aggregate"] = {
            "sharpe_ratio": (np.mean(aggregate_returns) - self.risk_free_rate)
            / (np.std(aggregate_returns) + 1e-8)
            * np.sqrt(252),
            "total_return": (total_capital[-1] - total_capital[0]) / total_capital[0],
            "max_drawdown": np.min(
                (total_capital - np.maximum.accumulate(total_capital))
                / np.maximum.accumulate(total_capital)
            ),
            "final_capital": total_capital[-1],
        }

        # Diversity metrics
        if len(self.agent_return_history[0]) > 0:
            all_returns = np.array(
                [list(h) for h in self.agent_return_history if len(h) > 0]
            )
            if len(all_returns) > 1:
                corr_matrix = np.corrcoef(all_returns)
                avg_correlation = (np.sum(np.abs(corr_matrix)) - self.n_agents) / (
                    self.n_agents * (self.n_agents - 1)
                )
                metrics["diversity"] = {"avg_correlation": avg_correlation}

        return metrics


if __name__ == "__main__":
    from config import Config
    from data_loader import MarketDataLoader

    config = Config()
    loader = MarketDataLoader(config)
    data = loader.prepare_environment_data()

    env = EnhancedMultiAgentPortfolioEnv(config, data)
    obs = env.reset()

    print(f"Environment initialized successfully!")
    print(f"Number of agents: {env.n_agents}")
    print(f"Assets per agent: {env.assets_per_agent}")
    print(f"Observation dimensions: {[len(o) for o in obs]}")
    print(f"Episode length: {env.episode_end - env.episode_start} days")
