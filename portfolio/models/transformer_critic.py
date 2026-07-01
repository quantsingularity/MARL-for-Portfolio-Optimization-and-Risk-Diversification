"""
Transformer-based Critic Network
Centralized critic with transformer architecture for multi-agent coordination
"""

from typing import Optional

import torch
import torch.nn as nn

from .transformer_actor import PositionalEncoding


class TransformerCritic(nn.Module):
    """
    Transformer-based centralized critic network

    Takes global state and joint actions as input
    Uses transformer to model complex inter-agent dependencies
    """

    def __init__(
        self,
        global_state_dim: int,
        total_action_dim: int,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        activation: str = "relu",
    ):
        super().__init__()

        self.global_state_dim = global_state_dim
        self.total_action_dim = total_action_dim
        self.d_model = d_model

        # Input projection for state and action
        input_dim = global_state_dim + total_action_dim
        self.input_proj = nn.Linear(input_dim, d_model)

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

        # Value head (Q-value output)
        self.value_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model // 4),
            nn.ReLU(),
            nn.Linear(d_model // 4, 1),
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

    def forward(
        self, global_state, joint_actions, attention_mask: Optional[torch.Tensor] = None
    ):
        """
        Forward pass through transformer critic

        Args:
            global_state: Global state tensor [batch_size, global_state_dim]
            joint_actions: Joint actions from all agents [batch_size, total_action_dim]
            attention_mask: Optional attention mask

        Returns:
            q_value: Q-value estimate [batch_size, 1]
        """
        # Concatenate state and actions
        x = torch.cat([global_state, joint_actions], dim=-1)  # [batch_size, input_dim]

        # Add sequence dimension if needed
        if x.dim() == 2:
            x = x.unsqueeze(1)  # [batch_size, 1, input_dim]

        # Project to d_model dimension
        x = self.input_proj(x)  # [batch_size, seq_len, d_model]

        # Add positional encoding
        x = x.transpose(0, 1)  # [seq_len, batch_size, d_model]
        x = self.pos_encoder(x)
        x = x.transpose(0, 1)  # [batch_size, seq_len, d_model]

        # Pass through transformer encoder
        encoded = self.transformer_encoder(x, src_key_padding_mask=attention_mask)

        # Use the last sequence output for Q-value prediction
        last_output = encoded[:, -1, :]  # [batch_size, d_model]

        # Compute Q-value
        q_value = self.value_head(last_output)  # [batch_size, 1]

        return q_value


class DuelingTransformerCritic(nn.Module):
    """
    Dueling architecture for critic network
    Separates state value and action advantage
    """

    def __init__(
        self,
        global_state_dim: int,
        total_action_dim: int,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
    ):
        super().__init__()

        input_dim = global_state_dim + total_action_dim
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.01)

    def forward(self, global_state, joint_actions):
        """Forward pass with dueling architecture"""
        x = torch.cat([global_state, joint_actions], dim=-1)

        if x.dim() == 2:
            x = x.unsqueeze(1)

        x = self.input_proj(x)
        x = x.transpose(0, 1)
        x = self.pos_encoder(x)
        x = x.transpose(0, 1)

        encoded = self.transformer_encoder(x)
        features = encoded[:, -1, :]

        # Compute value and advantage
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)

        # Combine using dueling formula: Q = V + (A - mean(A))
        q_value = value + (advantage - advantage.mean(dim=-1, keepdim=True))

        return q_value


class EnsembleTransformerCritic(nn.Module):
    """
    Ensemble of transformer critics for uncertainty estimation
    Useful for risk-aware decision making
    """

    def __init__(
        self,
        global_state_dim: int,
        total_action_dim: int,
        num_critics: int = 3,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.num_critics = num_critics
        self.critics = nn.ModuleList(
            [
                TransformerCritic(
                    global_state_dim=global_state_dim,
                    total_action_dim=total_action_dim,
                    d_model=d_model,
                    nhead=nhead,
                    num_layers=num_layers,
                    dropout=dropout,
                )
                for _ in range(num_critics)
            ]
        )

    def forward(self, global_state, joint_actions, return_individual=False):
        """
        Forward pass through ensemble

        Args:
            global_state: Global state tensor
            joint_actions: Joint actions tensor
            return_individual: If True, return individual Q-values from each critic

        Returns:
            If return_individual is False: mean Q-value and std
            If return_individual is True: all Q-values [batch_size, num_critics]
        """
        q_values = [critic(global_state, joint_actions) for critic in self.critics]
        q_values = torch.cat(q_values, dim=-1)  # [batch_size, num_critics]

        if return_individual:
            return q_values
        else:
            mean_q = q_values.mean(dim=-1, keepdim=True)
            std_q = q_values.std(dim=-1, keepdim=True)
            return mean_q, std_q
