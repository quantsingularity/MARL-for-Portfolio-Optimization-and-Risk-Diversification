"""SHAP Value Analysis for Model Interpretability"""

import warnings
from typing import Dict

import numpy as np

warnings.filterwarnings("ignore")


class SHAPAnalyzer:
    def __init__(self, model=None):
        self.model = model

    def compute_shap_values(self, states: np.ndarray) -> np.ndarray:
        """Compute SHAP-like values using a lightweight heuristic.

        If a `model` with a `predict` or callable interface is provided, we use
        the model outputs to scale contributions. Otherwise we use deviations
        from the feature means as proxy importance.
        """
        states = np.array(states, dtype=float)
        n_samples, n_features = states.shape

        # Base proxy: centered features
        centered = states - np.mean(states, axis=0, keepdims=True)

        # Scale factor from model (if available)
        scale = 0.01
        try:
            if self.model is not None:
                # Try to get deterministic scaling from model outputs
                if hasattr(self.model, "predict"):
                    preds = np.array(self.model.predict(states))
                else:
                    preds = np.array(self.model(states))
                scale = float(np.mean(np.abs(preds))) + 1e-8
        except Exception:
            scale = 0.01

        shap_vals = centered * scale / (np.std(centered, axis=0, keepdims=True) + 1e-8)
        return shap_vals.astype(float)

    def get_feature_importance(self, states: np.ndarray) -> Dict[int, float]:
        """Get feature importance scores"""
        shap_values = self.compute_shap_values(states)
        importance = np.abs(shap_values).mean(axis=0)
        return {i: float(v) for i, v in enumerate(importance)}
