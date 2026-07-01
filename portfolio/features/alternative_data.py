"""
Alternative Data Sources Provider
"""

from typing import List

import numpy as np


class AlternativeDataProvider:
    """
    Provider for alternative data sources

    In production, this would integrate:
    - Satellite imagery (parking lots, shipping)
    - Credit card transactions
    - Web traffic data
    - Social media metrics
    """

    def __init__(self):
        self.data_cache = {}

    def get_web_traffic(self, ticker: str) -> float:
        """Simulated web traffic data"""
        return np.random.uniform(0.5, 1.5)

    def get_social_mentions(self, ticker: str) -> int:
        """Simulated social media mentions"""
        return int(np.random.uniform(100, 10000))

    def get_alternative_features(self, tickers: List[str]) -> np.ndarray:
        """Get alternative data as features"""
        features = []
        for ticker in tickers:
            features.append(
                [
                    self.get_web_traffic(ticker),
                    np.log(self.get_social_mentions(ticker)),
                ]
            )
        return np.array(features)
