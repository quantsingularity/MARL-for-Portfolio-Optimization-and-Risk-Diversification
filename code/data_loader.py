"""
Data Loading and Preprocessing Module
Handles market data acquisition, feature engineering, and preprocessing
"""

import numpy as np
import pandas as pd
import yfinance as yf
from typing import List, Tuple, Optional, Dict
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

try:
    from ta import add_all_ta_features
    from ta.volatility import BollingerBands
    from ta.momentum import RSIIndicator
    from ta.trend import MACD
except ImportError:
    print("Warning: 'ta' library not installed. Technical indicators will be limited.")
    add_all_ta_features = None

from config import Config


class MarketDataLoader:
    """
    Market data loader with feature engineering capabilities
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.tickers = config.data.tickers
        self.start_date = config.data.start_date
        self.end_date = config.data.end_date
        
        self.raw_data = None
        self.processed_data = None
        self.feature_data = None
        
    def load_data(self) -> pd.DataFrame:
        """Load market data from specified source"""
        if self.config.data.data_source == 'yfinance':
            return self._load_from_yfinance()
        elif self.config.data.data_source == 'synthetic':
            return self._generate_synthetic_data()
        elif self.config.data.data_source == 'csv':
            return self._load_from_csv()
        else:
            raise ValueError(f"Unknown data source: {self.config.data.data_source}")
    
    def _load_from_yfinance(self) -> pd.DataFrame:
        """Load data from Yahoo Finance"""
        print(f"Loading data for {len(self.tickers)} assets from {self.start_date} to {self.end_date}...")
        
        try:
            # Download data
            data = yf.download(
                self.tickers,
                start=self.start_date,
                end=self.end_date,
                progress=False,
                threads=True
            )
            
            # Extract adjusted close prices
            if len(self.tickers) == 1:
                prices = data['Adj Close'].to_frame()
                prices.columns = self.tickers
            else:
                prices = data['Adj Close']
            
            # Handle missing data
            prices = prices.fillna(method='ffill').fillna(method='bfill')
            
            print(f"✓ Successfully loaded {len(prices)} days of data")
            
            self.raw_data = prices
            return prices
            
        except Exception as e:
            print(f"Error loading data from yfinance: {e}")
            print("Falling back to synthetic data...")
            return self._generate_synthetic_data()
    
    def _generate_synthetic_data(self) -> pd.DataFrame:
        """Generate synthetic market data"""
        print("Generating synthetic market data...")
        
        n_assets = self.config.env.n_assets
        date_range = pd.date_range(
            start=self.start_date,
            end=self.end_date,
            freq='D'
        )
        n_days = len(date_range)
        
        # Generate correlated returns
        np.random.seed(42)
        correlation_matrix = self._create_correlation_matrix(n_assets)
        
        mean_returns = np.random.uniform(0.0001, 0.001, n_assets)
        volatilities = np.random.uniform(0.01, 0.03, n_assets)
        
        # Cholesky decomposition
        L = np.linalg.cholesky(correlation_matrix)
        
        returns = []
        for _ in range(n_days):
            uncorrelated = np.random.randn(n_assets)
            correlated = L @ uncorrelated
            daily_returns = mean_returns + volatilities * correlated
            returns.append(daily_returns)
        
        returns = np.array(returns)
        
        # Convert to prices
        initial_prices = np.ones(n_assets) * 100
        prices = initial_prices * np.exp(np.cumsum(returns, axis=0))
        
        # Create DataFrame
        columns = [f'asset_{i}' for i in range(n_assets)]
        df = pd.DataFrame(prices, index=date_range, columns=columns)
        
        print(f"✓ Generated {n_days} days of synthetic data")
        
        self.raw_data = df
        return df
    
    def _create_correlation_matrix(self, n_assets: int, base_corr: float = 0.3) -> np.ndarray:
        """Create realistic correlation matrix with sector clustering"""
        corr_matrix = np.eye(n_assets)
        
        # Define sector boundaries (based on agent assignment)
        sectors = self.config.env.agent_asset_assignment
        
        # Higher correlation within sectors
        for sector in sectors:
            for i in sector:
                for j in sector:
                    if i != j:
                        corr_matrix[i, j] = base_corr + np.random.uniform(0, 0.2)
        
        # Lower correlation between sectors
        for i in range(n_assets):
            for j in range(i+1, n_assets):
                if corr_matrix[i, j] == 0:  # Not in same sector
                    corr_matrix[i, j] = np.random.uniform(0.05, 0.15)
                    corr_matrix[j, i] = corr_matrix[i, j]
        
        # Ensure positive definite
        eigvals, eigvecs = np.linalg.eigh(corr_matrix)
        eigvals = np.maximum(eigvals, 0.01)
        corr_matrix = eigvecs @ np.diag(eigvals) @ eigvecs.T
        
        # Normalize diagonal
        d = np.sqrt(np.diag(corr_matrix))
        corr_matrix = corr_matrix / d[:, None] / d[None, :]
        
        return corr_matrix
    
    def _load_from_csv(self, path: str = 'market_data.csv') -> pd.DataFrame:
        """Load data from CSV file"""
        print(f"Loading data from {path}...")
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        self.raw_data = df
        return df
    
    def compute_features(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Compute comprehensive feature set including:
        - Historical returns (20-day, 60-day)
        - Technical indicators (RSI, MACD, Bollinger Bands)
        - Volatility measures
        - Macro features (if enabled)
        """
        print("Computing features...")
        
        features_list = []
        
        # 1. Price returns
        returns = prices.pct_change().fillna(0)
        
        # 2. Historical returns (20-day and 60-day)
        returns_20d = prices.pct_change(20).fillna(0)
        returns_60d = prices.pct_change(60).fillna(0)
        
        # 3. Volatility (20-day rolling)
        volatility_20d = returns.rolling(window=20).std().fillna(0)
        
        for col in prices.columns:
            asset_features = pd.DataFrame(index=prices.index)
            
            # Basic features
            asset_features[f'{col}_return'] = returns[col]
            asset_features[f'{col}_return_20d'] = returns_20d[col]
            asset_features[f'{col}_return_60d'] = returns_60d[col]
            asset_features[f'{col}_volatility_20d'] = volatility_20d[col]
            
            # Technical indicators (if enabled and library available)
            if self.config.data.use_technical_indicators:
                try:
                    # RSI
                    rsi = RSIIndicator(close=prices[col], window=14)
                    asset_features[f'{col}_rsi'] = rsi.rsi().fillna(50) / 100.0  # Normalize
                    
                    # MACD
                    macd = MACD(close=prices[col])
                    asset_features[f'{col}_macd'] = macd.macd_diff().fillna(0)
                    
                    # Bollinger Bands
                    bb = BollingerBands(close=prices[col])
                    bb_position = (prices[col] - bb.bollinger_lband()) / (bb.bollinger_hband() - bb.bollinger_lband() + 1e-8)
                    asset_features[f'{col}_bb_position'] = bb_position.fillna(0.5)
                    
                except Exception as e:
                    print(f"Warning: Could not compute technical indicators for {col}: {e}")
            
            features_list.append(asset_features)
        
        # Combine all features
        all_features = pd.concat(features_list, axis=1)
        
        # Add macro features (if enabled)
        if self.config.data.use_macro_features:
            all_features = self._add_macro_features(all_features)
        
        # Fill any remaining NaN values
        all_features = all_features.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        print(f"✓ Computed {len(all_features.columns)} features")
        
        self.feature_data = all_features
        return all_features
    
    def _add_macro_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Add macroeconomic features (VIX, Treasury yields, etc.)"""
        try:
            # Download VIX
            vix_data = yf.download('^VIX', start=features.index[0], end=features.index[-1], progress=False)
            if not vix_data.empty:
                vix = vix_data['Adj Close'].reindex(features.index, method='ffill').fillna(method='bfill')
                features['vix'] = (vix - vix.mean()) / (vix.std() + 1e-8)  # Normalize
            
            # Download 10-year Treasury yield
            treasury = yf.download('^TNX', start=features.index[0], end=features.index[-1], progress=False)
            if not treasury.empty:
                tnx = treasury['Adj Close'].reindex(features.index, method='ffill').fillna(method='bfill')
                features['treasury_10y'] = (tnx - tnx.mean()) / (tnx.std() + 1e-8)  # Normalize
            
        except Exception as e:
            print(f"Warning: Could not load macro features: {e}")
        
        return features
    
    def prepare_environment_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Prepare data for environment initialization
        Returns: (prices, returns)
        """
        if self.raw_data is None:
            self.load_data()
        
        prices = self.raw_data
        returns = prices.pct_change().fillna(0)
        
        return prices, returns
    
    def split_train_test(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split data into training and testing sets"""
        train_data = data.loc[self.config.data.train_start:self.config.data.train_end]
        test_data = data.loc[self.config.data.test_start:self.config.data.test_end]
        
        print(f"Training period: {train_data.index[0]} to {train_data.index[-1]} ({len(train_data)} days)")
        print(f"Testing period: {test_data.index[0]} to {test_data.index[-1]} ({len(test_data)} days)")
        
        return train_data, test_data
    
    def get_feature_dim(self) -> int:
        """Get total feature dimension for one asset"""
        if self.feature_data is None:
            prices, _ = self.prepare_environment_data()
            self.compute_features(prices)
        
        # Features per asset
        n_features_per_asset = len(self.feature_data.columns) // self.config.env.n_assets
        return n_features_per_asset


def download_and_preprocess_data(config: Config) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Convenience function to download and preprocess all data
    Returns: train_prices, test_prices, train_returns, test_returns
    """
    loader = MarketDataLoader(config)
    
    # Load raw data
    prices = loader.load_data()
    
    # Split into train/test
    train_prices, test_prices = loader.split_train_test(prices)
    
    # Compute returns
    train_returns = train_prices.pct_change().fillna(0)
    test_returns = test_prices.pct_change().fillna(0)
    
    return train_prices, test_prices, train_returns, test_returns


if __name__ == "__main__":
    # Test data loader
    from config import default_config
    
    loader = MarketDataLoader(default_config)
    prices = loader.load_data()
    features = loader.compute_features(prices)
    
    print("\nData shape:", prices.shape)
    print("Feature shape:", features.shape)
    print("\nFirst few rows of prices:")
    print(prices.head())
    print("\nFirst few features:")
    print(features.head())
