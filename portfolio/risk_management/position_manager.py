"""Position Sizing and Risk Limits"""

from typing import Dict

import numpy as np


class PositionManager:
    def __init__(
        self,
        max_position_size: float = 0.3,
        max_sector_exposure: float = 0.5,
        stop_loss: float = -0.15,
        take_profit: float = 0.25,
    ):
        self.max_position_size = max_position_size
        self.max_sector_exposure = max_sector_exposure
        self.stop_loss = stop_loss
        self.take_profit = take_profit

    def apply_position_limits(self, weights: np.ndarray) -> np.ndarray:
        """Apply maximum position size limits"""
        weights = np.clip(weights, 0, self.max_position_size)
        weights = weights / weights.sum()  # Renormalize
        return weights

    def apply_sector_limits(
        self, weights: np.ndarray, sector_map: Dict[int, str]
    ) -> np.ndarray:
        """Apply sector exposure limits"""
        sector_weights = {}
        for idx, sector in sector_map.items():
            if sector not in sector_weights:
                sector_weights[sector] = 0
            sector_weights[sector] += weights[idx]

        # Scale down if sector exceeds limit
        for sector, total_weight in sector_weights.items():
            if total_weight > self.max_sector_exposure:
                scale_factor = self.max_sector_exposure / total_weight
                for idx, s in sector_map.items():
                    if s == sector:
                        weights[idx] *= scale_factor

        weights = weights / weights.sum()  # Renormalize
        return weights

    def check_stop_loss(self, position_return: float) -> bool:
        """Check if stop loss triggered"""
        return position_return <= self.stop_loss

    def check_take_profit(self, position_return: float) -> bool:
        """Check if take profit triggered"""
        return position_return >= self.take_profit
