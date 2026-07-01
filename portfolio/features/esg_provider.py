"""
ESG Data Provider
Integrates Environmental, Social, and Governance scores
"""

import warnings
from typing import Dict, List, Optional

import numpy as np

warnings.filterwarnings("ignore")


class ESGProvider:
    """
    Provides ESG scores for portfolio assets

    In production, this would connect to:
    - MSCI ESG Ratings
    - Sustainalytics
    - Bloomberg ESG Data
    - Refinitiv ESG Scores
    """

    def __init__(self, min_score: float = 50.0):
        self.min_score = min_score
        self.esg_cache = {}

        # Simulated ESG scores for common tickers (0-100 scale)
        self.default_scores = {
            # Technology
            "AAPL": 82.5,
            "MSFT": 85.3,
            "GOOGL": 78.2,
            "META": 65.4,
            "NVDA": 72.8,
            "TSLA": 68.9,
            "AVGO": 71.2,
            "ADBE": 79.5,
            # Healthcare
            "JNJ": 88.7,
            "UNH": 76.3,
            "PFE": 81.2,
            "ABBV": 74.5,
            "TMO": 80.1,
            "MRK": 83.4,
            "LLY": 77.8,
            # Finance
            "JPM": 73.6,
            "BAC": 71.9,
            "V": 82.1,
            "MA": 81.5,
            "GS": 69.8,
            "MS": 72.3,
            "AXP": 75.7,
            # Energy
            "XOM": 58.3,
            "CVX": 61.7,
            "COP": 64.2,
            "SLB": 66.5,
            "EOG": 67.9,
            "PXD": 63.1,
            # Commodities/ETFs
            "GLD": 55.0,
            "SLV": 54.0,
            # Bonds
            "TLT": 70.0,
            "IEF": 70.0,
            "SHY": 70.0,
            "LQD": 72.0,
            # Crypto
            "BTC-USD": 35.0,
            "ETH-USD": 40.0,
            "BNB-USD": 38.0,
        }

    def get_esg_score(self, ticker: str) -> float:
        """Get ESG score for a ticker"""
        if ticker in self.esg_cache:
            return self.esg_cache[ticker]

        score = self.default_scores.get(ticker, 60.0)  # Default mid-range score
        self.esg_cache[ticker] = score
        return score

    def get_esg_scores(self, tickers: List[str]) -> Dict[str, float]:
        """Get ESG scores for multiple tickers"""
        return {ticker: self.get_esg_score(ticker) for ticker in tickers}

    def filter_by_esg(
        self, tickers: List[str], min_score: Optional[float] = None
    ) -> List[str]:
        """Filter tickers by minimum ESG score"""
        threshold = min_score if min_score is not None else self.min_score
        return [t for t in tickers if self.get_esg_score(t) >= threshold]

    def get_esg_features(self, tickers: List[str]) -> np.ndarray:
        """Get ESG scores as feature array"""
        scores = [self.get_esg_score(t) for t in tickers]
        return np.array(scores) / 100.0  # Normalize to [0, 1]

    def compute_portfolio_esg_score(
        self, tickers: List[str], weights: np.ndarray
    ) -> float:
        """Compute weighted average ESG score for portfolio"""
        scores = self.get_esg_features(tickers) * 100  # Back to [0, 100]
        portfolio_score = np.dot(scores, weights)
        return float(portfolio_score)
