"""Experiment Logging with TensorBoard"""

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
from typing import Dict

try:
    from torch.utils.tensorboard import SummaryWriter
except:
    SummaryWriter = None


class ExperimentLogger:
    def __init__(self, log_dir: str = "./runs", experiment_name: str = "experiment"):
        self.log_dir = os.path.join(log_dir, experiment_name)
        os.makedirs(self.log_dir, exist_ok=True)
        self.writer = SummaryWriter(self.log_dir) if SummaryWriter else None

    def log_scalar(self, tag: str, value: float, step: int):
        """Log scalar value"""
        if self.writer:
            self.writer.add_scalar(tag, value, step)

    def log_scalars(self, main_tag: str, tag_value_dict: Dict, step: int):
        """Log multiple scalars"""
        if self.writer:
            self.writer.add_scalars(main_tag, tag_value_dict, step)

    def close(self):
        """Close logger"""
        if self.writer:
            self.writer.close()
