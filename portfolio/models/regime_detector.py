"""
Market Regime Detection using Hidden Markov Models
Detects Bull/Bear/Sideways market regimes
"""

import numpy as np
from sklearn.mixture import GaussianMixture


class MarketRegimeDetector:
    """
    Detects market regimes using Hidden Markov Model approach

    Regimes:
    - Bull Market (0): High returns, low volatility
    - Bear Market (1): Negative returns, high volatility
    - Sideways Market (2): Low returns, moderate volatility
    """

    def __init__(self, n_regimes: int = 3, lookback_window: int = 60):
        self.n_regimes = n_regimes
        self.lookback_window = lookback_window
        self.gmm = GaussianMixture(
            n_components=n_regimes, covariance_type="full", random_state=42
        )
        self.is_fitted = False

    def extract_features(
        self, returns: np.ndarray, vix: np.ndarray = None
    ) -> np.ndarray:
        """Extract regime detection features"""
        features = []

        # Rolling return statistics
        rolling_mean = np.convolve(
            returns, np.ones(self.lookback_window) / self.lookback_window, mode="valid"
        )
        rolling_std = np.array(
            [
                np.std(returns[max(0, i - self.lookback_window) : i])
                for i in range(len(returns))
            ]
        )

        # Momentum indicator
        momentum = (
            returns[-self.lookback_window :].sum()
            if len(returns) >= self.lookback_window
            else 0
        )

        # Combine features
        min_len = min(len(rolling_mean), len(rolling_std))
        features = np.column_stack(
            [
                rolling_mean[:min_len],
                rolling_std[:min_len],
            ]
        )

        if vix is not None and len(vix) == len(features):
            features = np.column_stack([features, vix[:min_len]])

        return features

    def fit(self, returns: np.ndarray, vix: np.ndarray = None):
        """Fit HMM to historical data"""
        features = self.extract_features(returns, vix)
        self.gmm.fit(features)
        self.is_fitted = True

    def predict(self, returns: np.ndarray, vix: np.ndarray = None) -> int:
        """Predict current market regime"""
        if not self.is_fitted:
            return 0  # Default to bull market

        features = self.extract_features(returns, vix)
        if len(features) == 0:
            return 0

        regime = self.gmm.predict(features[-1:])
        return int(regime[0])

    def get_regime_probabilities(
        self, returns: np.ndarray, vix: np.ndarray = None
    ) -> np.ndarray:
        """Get probability distribution over regimes"""
        if not self.is_fitted:
            return np.array([1 / self.n_regimes] * self.n_regimes)

        features = self.extract_features(returns, vix)
        if len(features) == 0:
            return np.array([1 / self.n_regimes] * self.n_regimes)

        probs = self.gmm.predict_proba(features[-1:])
        return probs[0]
