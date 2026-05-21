"""Tests for interpretability utilities"""

import os
import sys

import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))


def test_shap_analyzer_basic():
    from code.interpretability.shap_analyzer import SHAPAnalyzer

    states = np.random.randn(5, 10)
    analyzer = SHAPAnalyzer()
    shap_vals = analyzer.compute_shap_values(states)
    assert shap_vals.shape == (5, 10)
    imp = analyzer.get_feature_importance(states)
    assert isinstance(imp, dict)
    assert len(imp) == 10


def test_model_explainer_basic():
    from code.interpretability.explainer import ModelExplainer

    state = np.random.randn(8)
    action = np.ones(4) / 4
    feature_names = [f"f{i}" for i in range(len(state))]
    explainer = ModelExplainer(None, feature_names)

    out = explainer.explain_decision(state, action)
    assert "state_contribution" in out
    assert "action_rationale" in out
    assert len(out["state_contribution"]) == len(state)
