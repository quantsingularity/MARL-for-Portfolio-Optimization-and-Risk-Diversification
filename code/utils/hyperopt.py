"""Hyperparameter Optimization using Optuna"""

import warnings
from typing import Any, Callable, Dict

warnings.filterwarnings("ignore")


class HyperparameterOptimizer:
    def __init__(self, n_trials: int = 50, direction: str = "maximize"):
        self.n_trials = n_trials
        self.direction = direction
        self.study = None

    def optimize(self, objective_func: Callable, param_space: Dict[str, Any]) -> Dict:
        """Run Optuna optimization"""
        try:
            import optuna
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "optuna is required for hyperparameter optimization. "
                "Install it with `pip install optuna`."
            ) from exc
        self.study = optuna.create_study(direction=self.direction)
        self.study.optimize(objective_func, n_trials=self.n_trials)
        return self.study.best_params

    def get_best_params(self) -> Dict:
        """Get best parameters"""
        if self.study is None:
            return {}
        return self.study.best_params
