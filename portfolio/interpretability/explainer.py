"""Model Explanation Tools"""

from typing import Dict, List

import numpy as np


class ModelExplainer:
    def __init__(self, model, feature_names: List[str]):
        self.model = model
        self.feature_names = feature_names

    def explain_decision(self, state: np.ndarray, action: np.ndarray) -> Dict:
        """Explain a specific decision by returning a simple feature attribution map and a short rationale.

        - `state_contribution`: normalized absolute contributions of state features
        - `action_rationale`: short textual rationale highlighting top contributing features
        """
        state = np.array(state, dtype=float)
        abs_state = np.abs(state)
        total = abs_state.sum() + 1e-12
        contributions = (abs_state / total).tolist()

        # Pick top contributing features for a short rationale
        top_idx = int(min(3, len(state)))
        top_features = sorted(
            range(len(state)), key=lambda i: abs_state[i], reverse=True
        )[:top_idx]
        top_names = [
            self.feature_names[i] if i < len(self.feature_names) else f"f{i}"
            for i in top_features
        ]
        rationale = f"Top features: {', '.join(top_names)}"

        return {
            "state_contribution": contributions,
            "action_rationale": rationale,
        }
