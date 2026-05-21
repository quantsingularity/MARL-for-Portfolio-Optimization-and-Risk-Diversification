"""Tests for Risk Metrics"""

import os
import sys

import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))


def test_sharpe_ratio():
    """Test Sharpe ratio calculation"""
    from code.risk_management.risk_metrics import RiskMetricsCalculator

    calc = RiskMetricsCalculator()
    returns = np.random.randn(252) * 0.01
    sharpe = calc.calculate_sharpe_ratio(returns)
    assert isinstance(sharpe, float)


def test_cvar():
    """Test CVaR calculation"""
    from code.risk_management.risk_metrics import RiskMetricsCalculator

    calc = RiskMetricsCalculator(cvar_alpha=0.95)

    returns = -np.abs(np.random.randn(1000) * 0.01) - 0.001  # always negative
    cvar = calc.calculate_cvar(returns)
    assert isinstance(cvar, float)
    assert cvar <= 0  # CVaR of a loss-only distribution must be negative
