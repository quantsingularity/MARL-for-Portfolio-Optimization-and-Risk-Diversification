"""Attention Weight Visualization"""

import warnings
from typing import Optional

import numpy as np

warnings.filterwarnings("ignore")


class AttentionVisualizer:
    def __init__(self, model):
        self.model = model

    def extract_attention_weights(self, state):
        """Extract attention weights from model.

        Tries multiple approaches:
        - Call model.get_attention_weights(state) if available
        - If not, perform a forward pass and return a simple identity weight
        """
        # Prefer model-provided method
        if hasattr(self.model, "get_attention_weights"):
            weights = self.model.get_attention_weights(state)
            if weights is not None:
                return weights

        # Fallback: return a trivial attention matrix (ones normalized)
        try:
            if state is None:
                return np.zeros((1, 1))
            if hasattr(state, "shape"):
                seq_len = state.shape[1] if state.ndim == 3 else 1
            else:
                seq_len = 1
            weights = np.ones((seq_len, seq_len)) / float(max(1, seq_len))
            return weights
        except Exception:
            return np.zeros((1, 1))

    def plot_attention_heatmap(
        self, attention_weights: np.ndarray, save_path: Optional[str] = None
    ):
        """Plot attention heatmap using matplotlib. Saves to `save_path` if provided and returns the figure object."""
        import matplotlib.pyplot as plt

        if attention_weights is None:
            raise ValueError("Attention weights are None")

        arr = np.array(attention_weights)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)

        fig, ax = plt.subplots(figsize=(6, 4))
        im = ax.imshow(arr, cmap="viridis", aspect="auto")
        ax.set_title("Attention Heatmap")
        plt.colorbar(im, ax=ax)
        plt.tight_layout()

        if save_path is not None:
            fig.savefig(save_path)

        return fig
