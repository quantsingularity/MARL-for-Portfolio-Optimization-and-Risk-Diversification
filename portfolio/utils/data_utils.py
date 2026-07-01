"""Data Processing Utilities"""

from typing import Tuple

import numpy as np
import pandas as pd


class DataUtils:
    @staticmethod
    def normalize_data(data: np.ndarray, method: str = "minmax") -> np.ndarray:
        """Normalize data"""
        if method == "minmax":
            return (data - data.min()) / (data.max() - data.min() + 1e-8)
        elif method == "zscore":
            return (data - data.mean()) / (data.std() + 1e-8)
        return data

    @staticmethod
    def train_test_split(
        data: pd.DataFrame, train_ratio: float = 0.75
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split data into train/test"""
        split_idx = int(len(data) * train_ratio)
        return data.iloc[:split_idx], data.iloc[split_idx:]

    @staticmethod
    def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
        """Compute returns from prices"""
        return prices.pct_change().fillna(0)
