"""
Transformer-based Actor Network
Implements multi-head self-attention for temporal pattern recognition in financial data
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)


class TransformerActor(nn.Module):
    """
    Transformer-based actor network for portfolio allocation

    Features:
    - Multi-head self-attention for temporal dependencies
    - Position encoding for sequential financial data
    - 4-layer transformer encoder with 8 attention heads
    - Captures long-range dependencies in market dynamics
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        activation: str = "relu",
    ):
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.d_model = d_model

        # Input projection
        self.input_proj = nn.Linear(state_dim, d_model)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        # Output projection to action space
        self.output_proj = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, action_dim),
        )

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize network weights using Xavier initialization"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.01)

    def forward(self, state, attention_mask: Optional[torch.Tensor] = None):
        """
        Forward pass through transformer actor

        Args:
            state: Input state tensor [batch_size, state_dim] or [batch_size, seq_len, state_dim]
            attention_mask: Optional attention mask

        Returns:
            action: Portfolio weights [batch_size, action_dim] (softmax normalized)
        """
        # Handle both 2D and 3D inputs
        if state.dim() == 2:
            state = state.unsqueeze(1)  # Add sequence dimension

        batch_size, seq_len, _ = state.shape

        # Project input to d_model dimension
        x = self.input_proj(state)  # [batch_size, seq_len, d_model]

        # Add positional encoding (need to transpose for pos_encoder)
        x = x.transpose(0, 1)  # [seq_len, batch_size, d_model]
        x = self.pos_encoder(x)
        x = x.transpose(0, 1)  # [batch_size, seq_len, d_model]

        # Pass through transformer encoder
        encoded = self.transformer_encoder(x, src_key_padding_mask=attention_mask)

        # Use the last sequence output for action prediction
        last_output = encoded[:, -1, :]  # [batch_size, d_model]

        # Project to action space and apply softmax
        logits = self.output_proj(last_output)  # [batch_size, action_dim]
        action = F.softmax(logits, dim=-1)

        return action

    def get_attention_weights(self, state):
        """
        Extract attention weights for interpretability

        Args:
            state: Input state tensor

        Returns:
            attention_weights: List of attention weight matrices from each layer
        """
        if state.dim() == 2:
            state = state.unsqueeze(1)

        x = self.input_proj(state)
        x = x.transpose(0, 1)
        x = self.pos_encoder(x)
        x = x.transpose(0, 1)

        attention_weights = []
        for layer in self.transformer_encoder.layers:
            # Extract attention weights from each layer
            # Note: This requires modifying TransformerEncoderLayer to return attention weights
            # For now, we'll use a simplified version
            attention_weights.append(None)

        return attention_weights


class HybridTransformerActor(nn.Module):
    """
    Hybrid architecture combining Transformer with traditional MLP
    Useful for ablation studies
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 2,
        mlp_hidden: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Transformer branch
        self.transformer_branch = TransformerActor(
            state_dim=state_dim,
            action_dim=d_model,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dropout=dropout,
        )

        # MLP branch
        self.mlp_branch = nn.Sequential(
            nn.Linear(state_dim, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, mlp_hidden // 2),
            nn.ReLU(),
        )

        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(d_model + mlp_hidden // 2, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, action_dim),
        )

    def forward(self, state):
        """Forward pass combining both branches"""
        # Transformer features
        trans_out = self.transformer_branch(state)

        # MLP features
        if state.dim() == 3:
            state_2d = state[:, -1, :]  # Use last timestep
        else:
            state_2d = state
        mlp_out = self.mlp_branch(state_2d)

        # Concatenate and fuse
        combined = torch.cat([trans_out, mlp_out], dim=-1)
        action = F.softmax(self.fusion(combined), dim=-1)

        return action
