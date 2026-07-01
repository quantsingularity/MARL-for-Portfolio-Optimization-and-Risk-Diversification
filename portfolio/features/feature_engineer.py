"""
Advanced Feature Engineering for Financial Data
"""

import numpy as np
import pandas as pd


class FeatureEngineer:
    """Advanced feature engineering for portfolio optimization"""

    @staticmethod
    def compute_technical_indicators(prices: pd.DataFrame) -> pd.DataFrame:
        """Compute technical indicators"""
        features = pd.DataFrame(index=prices.index)

        for col in prices.columns:
            price = prices[col]

            # Returns
            features[f"{col}_return"] = price.pct_change()
            features[f"{col}_log_return"] = np.log(price / price.shift(1))

            # Moving averages
            features[f"{col}_sma_20"] = price.rolling(20).mean()
            features[f"{col}_sma_50"] = price.rolling(50).mean()
            features[f"{col}_ema_12"] = price.ewm(span=12).mean()

            # Volatility
            features[f"{col}_volatility"] = price.pct_change().rolling(20).std()

            # RSI
            delta = price.diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            features[f"{col}_rsi"] = 100 - (100 / (1 + rs))

            # MACD
            ema_12 = price.ewm(span=12).mean()
            ema_26 = price.ewm(span=26).mean()
            features[f"{col}_macd"] = ema_12 - ema_26
            features[f"{col}_macd_signal"] = features[f"{col}_macd"].ewm(span=9).mean()

            # Bollinger Bands
            sma = price.rolling(20).mean()
            std = price.rolling(20).std()
            features[f"{col}_bb_upper"] = sma + (2 * std)
            features[f"{col}_bb_lower"] = sma - (2 * std)
            features[f"{col}_bb_width"] = (
                features[f"{col}_bb_upper"] - features[f"{col}_bb_lower"]
            ) / sma

        return features

    @staticmethod
    def compute_correlation_features(
        returns: pd.DataFrame, window: int = 30
    ) -> pd.DataFrame:
        """Compute rolling correlation features"""

        corr_features = pd.DataFrame(index=returns.index)

        # Rolling correlation matrix
        for i, col1 in enumerate(returns.columns):
            for j, col2 in enumerate(returns.columns):
                if i < j:  # Upper triangle only
                    corr_features[f"corr_{col1}_{col2}"] = (
                        returns[col1].rolling(window).corr(returns[col2])
                    )

        # Average correlation
        corr_features["avg_correlation"] = corr_features.mean(axis=1)

        return corr_features

    @staticmethod
    def compute_risk_metrics(returns: pd.DataFrame, window: int = 60) -> pd.DataFrame:
        """Compute rolling risk metrics"""
        risk_features = pd.DataFrame(index=returns.index)

        # Sharpe ratio (assuming risk-free rate = 0)
        risk_features["sharpe"] = (
            returns.rolling(window).mean()
            / returns.rolling(window).std()
            * np.sqrt(252)
        )

        # Sortino ratio
        downside_returns = returns.copy()
        downside_returns[downside_returns > 0] = 0
        downside_std = downside_returns.rolling(window).std()
        risk_features["sortino"] = (
            returns.rolling(window).mean() / downside_std * np.sqrt(252)
        )

        # Maximum drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.rolling(window, min_periods=1).max()
        drawdown = (cumulative - running_max) / running_max
        risk_features["max_drawdown"] = drawdown.rolling(window).min()

        return risk_features
