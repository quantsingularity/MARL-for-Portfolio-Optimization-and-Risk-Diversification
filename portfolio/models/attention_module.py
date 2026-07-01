"""
Multi-Head Attention and Cross-Asset Attention Mechanisms
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism for financial time series"""

    def __init__(self, d_model: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # Linear projections and reshape
        q = (
            self.q_linear(query)
            .view(batch_size, -1, self.num_heads, self.d_k)
            .transpose(1, 2)
        )
        k = (
            self.k_linear(key)
            .view(batch_size, -1, self.num_heads, self.d_k)
            .transpose(1, 2)
        )
        v = (
            self.v_linear(value)
            .view(batch_size, -1, self.num_heads, self.d_k)
            .transpose(1, 2)
        )

        # Attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attention = F.softmax(scores, dim=-1)
        attention = self.dropout(attention)

        # Apply attention to values
        context = torch.matmul(attention, v)
        context = (
            context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        )

        output = self.out_linear(context)
        return output, attention


class CrossAssetAttention(nn.Module):
    """Cross-asset attention for capturing inter-asset dependencies"""

    def __init__(self, d_model: int, num_assets: int, num_heads: int = 4):
        super().__init__()
        self.num_assets = num_assets
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, asset_features):
        """
        Args:
            asset_features: [batch_size, num_assets, d_model]
        Returns:
            attended_features: [batch_size, num_assets, d_model]
        """
        attended, attention_weights = self.attention(
            asset_features, asset_features, asset_features
        )
        output = self.layer_norm(asset_features + attended)
        return output, attention_weights
