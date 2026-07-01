"""Dynamic Diversity Weight λ Adjustment"""

import numpy as np


class DynamicDiversityManager:
    def __init__(
        self,
        base_weight: float = 0.1,
        weight_range: tuple = (0.05, 0.2),
        vix_threshold: float = 25.0,
    ):
        self.base_weight = base_weight
        self.min_weight, self.max_weight = weight_range
        self.vix_threshold = vix_threshold
        self.current_weight = base_weight

    def adjust_diversity_weight(self, vix: float, market_volatility: float) -> float:
        """Adjust diversity weight based on market conditions"""
        # High VIX -> More diversification
        if vix > self.vix_threshold:
            vix_factor = min((vix / self.vix_threshold) - 1, 1.0)
            self.current_weight = self.base_weight + vix_factor * (
                self.max_weight - self.base_weight
            )
        else:
            # Low VIX -> Allow conviction trades
            vix_factor = vix / self.vix_threshold
            self.current_weight = self.min_weight + vix_factor * (
                self.base_weight - self.min_weight
            )

        # Clamp to range
        self.current_weight = np.clip(
            self.current_weight, self.min_weight, self.max_weight
        )
        return self.current_weight

    def get_current_weight(self) -> float:
        return self.current_weight
