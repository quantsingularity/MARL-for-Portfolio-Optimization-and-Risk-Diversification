"""Risk Attribution Analysis"""

from typing import Dict

import numpy as np


class RiskAttributionAnalyzer:
    def __init__(self):
        self.attribution_history = []

    def compute_risk_contribution(
        self, weights: np.ndarray, cov_matrix: np.ndarray
    ) -> np.ndarray:
        """Compute marginal risk contribution of each asset"""
        portfolio_var = np.dot(weights, np.dot(cov_matrix, weights))
        if portfolio_var == 0:
            return np.zeros_like(weights)
        marginal_contrib = np.dot(cov_matrix, weights) / np.sqrt(portfolio_var)
        risk_contrib = weights * marginal_contrib
        return risk_contrib

    def compute_factor_attribution(
        self, returns: np.ndarray, factors: np.ndarray
    ) -> Dict:
        """Attribute returns to risk factors"""
        # Simple factor attribution using regression
        if len(returns) != len(factors):
            return {"systematic": 0.0, "idiosyncratic": 0.0}

        # Compute correlation
        corr = np.corrcoef(returns, factors)[0, 1] if len(returns) > 1 else 0.0
        systematic_var = (corr**2) * np.var(returns)
        idiosyncratic_var = np.var(returns) - systematic_var

        return {
            "systematic": float(systematic_var),
            "idiosyncratic": float(idiosyncratic_var),
            "correlation": float(corr),
        }
