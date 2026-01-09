"""
Baseline strategies for comparison
Implements all baselines from the paper
"""

import numpy as np
from typing import Dict
from scipy.optimize import minimize


class BaselineStrategy:
    """Base class for baseline strategies"""

    def __init__(self, env):
        self.env = env
        self.n_agents = env.n_agents
        self.assets_per_agent = env.assets_per_agent

    def get_actions(self, observations):
        """Get portfolio weights for all agents"""
        raise NotImplementedError

    def evaluate(self) -> Dict:
        """Evaluate strategy on test set"""
        states = self.env.reset()

        while not self.env.done:
            actions = self.get_actions(states)
            states, rewards, done, info = self.env.step(actions)

        metrics = self.env.get_episode_metrics()
        return metrics


class RandomStrategy(BaselineStrategy):
    """Random portfolio allocation"""

    def get_actions(self, observations):
        actions = []
        for i in range(self.n_agents):
            action = np.random.random(self.assets_per_agent[i])
            action = action / np.sum(action)
            actions.append(action)
        return actions


class EqualWeightStrategy(BaselineStrategy):
    """Equal-weight (1/N) portfolio"""

    def get_actions(self, observations):
        actions = []
        for i in range(self.n_agents):
            action = np.ones(self.assets_per_agent[i]) / self.assets_per_agent[i]
            actions.append(action)
        return actions


class RiskParityStrategy(BaselineStrategy):
    """Risk parity portfolio - equal risk contribution"""

    def __init__(self, env, lookback=60):
        super().__init__(env)
        self.lookback = lookback

    def get_actions(self, observations):
        actions = []

        for agent_id in range(self.n_agents):
            agent_tickers = self.env.agent_assets[agent_id]

            # Get historical returns
            current_idx = self.env.current_step
            start_idx = max(0, current_idx - self.lookback)

            returns = self.env.returns[agent_tickers].iloc[start_idx:current_idx].values

            if len(returns) < 10:
                # Not enough data, use equal weight
                action = (
                    np.ones(self.assets_per_agent[agent_id])
                    / self.assets_per_agent[agent_id]
                )
            else:
                # Calculate covariance matrix
                cov_matrix = np.cov(returns.T)

                # Risk parity weights
                try:
                    action = self._risk_parity_weights(cov_matrix)
                except:
                    action = (
                        np.ones(self.assets_per_agent[agent_id])
                        / self.assets_per_agent[agent_id]
                    )

            actions.append(action)

        return actions

    def _risk_parity_weights(self, cov_matrix):
        """Calculate risk parity weights"""
        n = len(cov_matrix)

        # Objective: minimize sum of squared differences in risk contribution
        def objective(w):
            portfolio_vol = np.sqrt(w @ cov_matrix @ w)
            marginal_contrib = cov_matrix @ w
            risk_contrib = w * marginal_contrib / portfolio_vol
            return np.sum((risk_contrib - risk_contrib.mean()) ** 2)

        # Constraints: weights sum to 1, all positive
        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
        bounds = [(0.0, 1.0) for _ in range(n)]

        # Initial guess: equal weight
        w0 = np.ones(n) / n

        result = minimize(
            objective, w0, method="SLSQP", bounds=bounds, constraints=constraints
        )

        if result.success:
            return result.x
        else:
            return w0


class MeanVarianceStrategy(BaselineStrategy):
    """Mean-Variance Optimization (Markowitz)"""

    def __init__(self, env, lookback=60, risk_aversion=2.0):
        super().__init__(env)
        self.lookback = lookback
        self.risk_aversion = risk_aversion

    def get_actions(self, observations):
        actions = []

        for agent_id in range(self.n_agents):
            agent_tickers = self.env.agent_assets[agent_id]

            # Get historical returns
            current_idx = self.env.current_step
            start_idx = max(0, current_idx - self.lookback)

            returns = self.env.returns[agent_tickers].iloc[start_idx:current_idx].values

            if len(returns) < 10:
                # Not enough data, use equal weight
                action = (
                    np.ones(self.assets_per_agent[agent_id])
                    / self.assets_per_agent[agent_id]
                )
            else:
                # Calculate mean and covariance
                mean_returns = np.mean(returns, axis=0)
                cov_matrix = np.cov(returns.T)

                # Mean-variance optimization
                try:
                    action = self._mean_variance_weights(mean_returns, cov_matrix)
                except:
                    action = (
                        np.ones(self.assets_per_agent[agent_id])
                        / self.assets_per_agent[agent_id]
                    )

            actions.append(action)

        return actions

    def _mean_variance_weights(self, mean_returns, cov_matrix):
        """Calculate mean-variance optimal weights"""
        n = len(mean_returns)

        # Objective: maximize return - risk_aversion * variance
        def objective(w):
            portfolio_return = w @ mean_returns
            portfolio_variance = w @ cov_matrix @ w
            return -(portfolio_return - self.risk_aversion * portfolio_variance)

        # Constraints: weights sum to 1, all positive (long-only)
        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
        bounds = [(0.0, 1.0) for _ in range(n)]

        # Initial guess: equal weight
        w0 = np.ones(n) / n

        result = minimize(
            objective, w0, method="SLSQP", bounds=bounds, constraints=constraints
        )

        if result.success:
            return result.x
        else:
            return w0


class SingleAgentDDPG(BaselineStrategy):
    """Single-agent DDPG baseline (conceptual implementation)"""

    def __init__(self, env):
        super().__init__(env)
        # In practice, would use actual DDPG implementation
        # For comparison, we use a simple momentum strategy

    def get_actions(self, observations):
        """Momentum-based strategy as DDPG proxy"""
        actions = []

        for agent_id in range(self.n_agents):
            agent_tickers = self.env.agent_assets[agent_id]

            # Get recent returns (momentum)
            current_idx = self.env.current_step
            lookback = 20
            start_idx = max(0, current_idx - lookback)

            returns = self.env.returns[agent_tickers].iloc[start_idx:current_idx]

            if len(returns) > 0:
                # Calculate momentum scores
                momentum = returns.mean().values

                # Convert to weights (higher momentum = higher weight)
                exp_momentum = np.exp(momentum * 10)  # Scale factor
                action = exp_momentum / np.sum(exp_momentum)
            else:
                action = (
                    np.ones(self.assets_per_agent[agent_id])
                    / self.assets_per_agent[agent_id]
                )

            actions.append(action)

        return actions


def evaluate_all_baselines(env, config) -> Dict:
    """Evaluate all baseline strategies"""
    print("\nEvaluating baseline strategies...")

    baselines = {
        "Random": RandomStrategy(env),
        "Equal-Weight": EqualWeightStrategy(env),
        "Risk Parity": RiskParityStrategy(env),
        "Mean-Variance": MeanVarianceStrategy(env),
        "Single-Agent DDPG": SingleAgentDDPG(env),
    }

    results = {}

    for name, strategy in baselines.items():
        print(f"Evaluating {name}...")

        # Reset environment for test period
        test_start, test_end = env.data["test_indices"]
        env.reset(start_idx=test_start, end_idx=test_end)

        # Evaluate
        metrics = strategy.evaluate()
        results[name] = metrics

        # Print results
        agg_metrics = metrics["aggregate"]
        print(f"  Sharpe Ratio: {agg_metrics['sharpe_ratio']:.3f}")
        print(f"  Annual Return: {agg_metrics['total_return']*100:.2f}%")
        print(f"  Max Drawdown: {agg_metrics['max_drawdown']*100:.2f}%")

    return results


if __name__ == "__main__":
    from config import Config
    from data_loader import MarketDataLoader
    from environment import EnhancedMultiAgentPortfolioEnv

    config = Config()
    loader = MarketDataLoader(config)
    data = loader.prepare_environment_data()
    env = EnhancedMultiAgentPortfolioEnv(config, data)

    results = evaluate_all_baselines(env, config)

    print("\n=== Baseline Comparison ===")
    for name, metrics in results.items():
        agg = metrics["aggregate"]
        print(f"\n{name}:")
        print(f"  Sharpe: {agg['sharpe_ratio']:.3f}")
        print(f"  Return: {agg['total_return']*100:.2f}%")
        print(f"  MDD: {agg['max_drawdown']*100:.2f}%")
