"""Tests for Transformer Architecture"""

# === stdlib 'code' pin (Python 3.13 pdb compatibility) -- auto-added ===
import sys as _sys

if not hasattr(_sys.modules.get("code"), "InteractiveConsole"):
    import importlib.util as _ilu
    import os as _os
    import sysconfig as _sc

    _sp = _sc.get_paths()["stdlib"]
    _cspec = _ilu.spec_from_file_location("code", _os.path.join(_sp, "code.py"))
    if _cspec is not None:
        _cmod = _ilu.module_from_spec(_cspec)
        _cspec.loader.exec_module(_cmod)
        _sys.modules["code"] = _cmod
    del _ilu, _os, _sc, _sp, _cspec
# === end stdlib 'code' pin ===

import os
import sys

import torch

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))


def test_transformer_actor():
    """Test transformer actor creation"""
    from models.transformer_actor import TransformerActor

    model = TransformerActor(state_dim=100, action_dim=10)
    state = torch.randn(32, 100)
    action = model(state)
    assert action.shape == (32, 10)
    assert torch.allclose(action.sum(dim=1), torch.ones(32), atol=1e-5)


def test_transformer_critic():
    """Test transformer critic creation"""
    from models.transformer_critic import TransformerCritic

    model = TransformerCritic(global_state_dim=400, total_action_dim=40)
    state = torch.randn(32, 400)
    actions = torch.randn(32, 40)
    q_value = model(state, actions)
    assert q_value.shape == (32, 1)
