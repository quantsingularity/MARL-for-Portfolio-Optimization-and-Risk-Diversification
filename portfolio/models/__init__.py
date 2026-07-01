"""
Models module for advanced MADDPG
Includes Transformer-based architectures and attention mechanisms
"""

from .attention_module import CrossAssetAttention, MultiHeadAttention
from .regime_detector import MarketRegimeDetector
from .transformer_actor import TransformerActor
from .transformer_critic import TransformerCritic

__all__ = [
    "TransformerActor",
    "TransformerCritic",
    "MultiHeadAttention",
    "CrossAssetAttention",
    "MarketRegimeDetector",
]
