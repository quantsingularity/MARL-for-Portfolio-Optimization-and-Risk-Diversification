"""Tests for baseline strategies"""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))


def test_evaluate_all_baselines_runs():
    from code.baselines import evaluate_all_baselines
    from code.config import Config
    from code.data_loader import MarketDataLoader
    from code.environment import MultiAgentPortfolioEnv

    config = Config()
    config.data.data_source = "synthetic"
    loader = MarketDataLoader(config)
    data = loader.prepare_environment_data()
    env = MultiAgentPortfolioEnv(config, data)

    results = evaluate_all_baselines(env, config)
    assert isinstance(results, dict)
    assert "Equal-Weight" in results
    assert "Random" in results
    # Check aggregate metrics exist
    for name, metrics in results.items():
        assert "aggregate" in metrics
