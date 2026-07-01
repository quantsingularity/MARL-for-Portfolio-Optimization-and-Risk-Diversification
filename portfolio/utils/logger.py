"""Experiment Logging with TensorBoard"""

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
