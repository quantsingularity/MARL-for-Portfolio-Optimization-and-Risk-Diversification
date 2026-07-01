"""
Financial Sentiment Analysis using FinBERT
"""

import warnings
from typing import Dict, List

import numpy as np

warnings.filterwarnings("ignore")


class FinBERTSentimentAnalyzer:
    """
    Sentiment analysis for financial news using FinBERT

    In production, this would use:
    - ProsusAI/finbert for financial sentiment
    - News API for real-time news
    - Twitter/Reddit for social sentiment
    """

    def __init__(self, use_social: bool = False):
        self.use_social = use_social
        self.sentiment_cache = {}

    def analyze_ticker_sentiment(
        self, ticker: str, lookback_days: int = 7
    ) -> Dict[str, float]:
        """
        Analyze sentiment for a ticker

        Returns:
            Dictionary with 'positive', 'negative', 'neutral' scores
        """
        # Simulated sentiment scores (would be from actual news/FinBERT in production)
        if ticker in self.sentiment_cache:
            return self.sentiment_cache[ticker]

        # Generate realistic sentiment distribution
        base_positive = 0.5 + np.random.randn() * 0.1
        base_negative = 0.2 + np.random.randn() * 0.05
        base_neutral = 1.0 - base_positive - base_negative

        # Normalize
        total = base_positive + base_negative + abs(base_neutral)
        sentiment = {
            "positive": max(0, base_positive / total),
            "negative": max(0, base_negative / total),
            "neutral": max(0, abs(base_neutral) / total),
            "compound": (base_positive - base_negative)
            / total,  # Overall sentiment [-1, 1]
        }

        self.sentiment_cache[ticker] = sentiment
        return sentiment

    def get_sentiment_scores(self, tickers: List[str]) -> np.ndarray:
        """Get compound sentiment scores as array"""
        scores = [self.analyze_ticker_sentiment(t)["compound"] for t in tickers]
        return np.array(scores)

    def get_sentiment_features(self, tickers: List[str]) -> np.ndarray:
        """Get sentiment as features [positive, negative, neutral, compound]"""
        features = []
        for ticker in tickers:
            sent = self.analyze_ticker_sentiment(ticker)
            features.append(
                [
                    sent["positive"],
                    sent["negative"],
                    sent["neutral"],
                    sent["compound"],
                ]
            )
        return np.array(features)

    def clear_cache(self):
        """Clear sentiment cache (for refreshing data)"""
        self.sentiment_cache = {}
