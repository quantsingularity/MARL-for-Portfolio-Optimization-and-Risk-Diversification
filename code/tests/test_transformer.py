"""Tests for Transformer Architecture"""

import os
import sys

import torch

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))


def test_transformer_actor():
    """Test transformer actor creation"""
    from code.models.transformer_actor import TransformerActor

    model = TransformerActor(state_dim=100, action_dim=10)
    state = torch.randn(32, 100)
    action = model(state)
    assert action.shape == (32, 10)
    assert torch.allclose(action.sum(dim=1), torch.ones(32), atol=1e-5)


def test_transformer_critic():
    """Test transformer critic creation"""
    from code.models.transformer_critic import TransformerCritic

    model = TransformerCritic(global_state_dim=400, total_action_dim=40)
    state = torch.randn(32, 400)
    actions = torch.randn(32, 40)
    q_value = model(state, actions)
    assert q_value.shape == (32, 1)
