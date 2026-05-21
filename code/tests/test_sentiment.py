"""Tests for Sentiment Analysis"""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))


def test_sentiment_analyzer():
    """Test sentiment analyzer"""
    from code.features.sentiment_analyzer import FinBERTSentimentAnalyzer

    analyzer = FinBERTSentimentAnalyzer()
    sentiment = analyzer.analyze_ticker_sentiment("AAPL")
    assert "positive" in sentiment
    assert "negative" in sentiment
    assert "compound" in sentiment
    assert -1 <= sentiment["compound"] <= 1
