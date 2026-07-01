"""
Advanced Risk Metrics: CVaR, Sortino Ratio, Tail Risk
"""

import numpy as np


class RiskMetricsCalculator:
    def __init__(self, cvar_alpha: float = 0.95, target_return: float = 0.0):
        self.cvar_alpha = cvar_alpha
        self.target_return = target_return

    def calculate_cvar(self, returns: np.ndarray) -> float:
        """Conditional Value-at-Risk (Expected Shortfall)"""
        if len(returns) == 0:
            return 0.0
        var_threshold = np.percentile(returns, (1 - self.cvar_alpha) * 100)
        cvar = returns[returns <= var_threshold].mean()
        return float(cvar) if not np.isnan(cvar) else 0.0

    def calculate_sortino_ratio(
        self, returns: np.ndarray, periods_per_year: int = 252
    ) -> float:
        """Sortino Ratio (downside risk-adjusted return)"""
        if len(returns) == 0:
            return 0.0
        excess_returns = returns - self.target_return
        downside_returns = excess_returns[excess_returns < 0]
        if len(downside_returns) == 0:
            return 0.0
        downside_std = np.sqrt(np.mean(downside_returns**2))
        if downside_std == 0:
            return 0.0
        sortino = (np.mean(excess_returns) / downside_std) * np.sqrt(periods_per_year)
        return float(sortino)

    def calculate_max_drawdown(self, cumulative_returns: np.ndarray) -> float:
        """Maximum Drawdown"""
        if len(cumulative_returns) == 0:
            return 0.0
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - running_max) / running_max
        return float(np.min(drawdown))

    def calculate_sharpe_ratio(
        self,
        returns: np.ndarray,
        risk_free_rate: float = 0.02,
        periods_per_year: int = 252,
    ) -> float:
        """Sharpe Ratio"""
        if len(returns) == 0 or np.std(returns) == 0:
            return 0.0
        excess_returns = returns - (risk_free_rate / periods_per_year)
        sharpe = (np.mean(excess_returns) / np.std(returns)) * np.sqrt(periods_per_year)
        return float(sharpe)

    def calculate_all_metrics(
        self, returns: np.ndarray, cumulative_returns: np.ndarray = None
    ) -> dict:
        """Calculate all risk metrics"""
        if cumulative_returns is None:
            cumulative_returns = np.cumprod(1 + returns)

        return {
            "sharpe_ratio": self.calculate_sharpe_ratio(returns),
            "sortino_ratio": self.calculate_sortino_ratio(returns),
            "cvar": self.calculate_cvar(returns),
            "max_drawdown": self.calculate_max_drawdown(cumulative_returns),
            "volatility": float(np.std(returns) * np.sqrt(252)),
            "total_return": (
                float(cumulative_returns[-1] - 1)
                if len(cumulative_returns) > 0
                else 0.0
            ),
        }
