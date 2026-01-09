"""
Data loading and preprocessing module
Implements technical indicators and market data acquisition
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import warnings

warnings.filterwarnings("ignore")


def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Calculate Relative Strength Index"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)  # Neutral RSI for initial values


def calculate_macd(
    prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> Tuple[pd.Series, pd.Series]:
    """Calculate MACD and signal line"""
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd.fillna(0), signal_line.fillna(0)


def calculate_bollinger_bands(
    prices: pd.Series, period: int = 20, std_dev: float = 2.0
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Calculate Bollinger Bands"""
    middle = prices.rolling(window=period).mean()
    std = prices.rolling(window=period).std()
    upper = middle + (std * std_dev)
    lower = middle - (std * std_dev)
    return (
        upper.fillna(method="bfill"),
        middle.fillna(method="bfill"),
        lower.fillna(method="bfill"),
    )


def calculate_volatility(returns: pd.Series, window: int = 20) -> pd.Series:
    """Calculate rolling volatility"""
    return returns.rolling(window=window).std().fillna(0)


def calculate_historical_returns(prices: pd.Series, windows: List[int]) -> pd.DataFrame:
    """Calculate historical returns for multiple windows"""
    returns_dict = {}
    for window in windows:
        returns_dict[f"return_{window}d"] = prices.pct_change(window).fillna(0)
    return pd.DataFrame(returns_dict)


class MarketDataLoader:
    """Load and preprocess market data"""

    def __init__(self, config):
        self.config = config
        self.sector_allocations = config.env.sector_allocations
        self.all_tickers = []
        for stocks in self.sector_allocations.values():
            self.all_tickers.extend(stocks)

    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load market data based on configuration"""
        if self.config.data.data_source == "yfinance":
            return self._load_yfinance_data()
        elif self.config.data.data_source == "synthetic":
            return self._generate_synthetic_data()
        elif self.config.data.data_source == "csv":
            return self._load_csv_data()
        else:
            raise ValueError(f"Unknown data source: {self.config.data.data_source}")

    def _load_yfinance_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load real market data from Yahoo Finance"""
        try:
            import yfinance as yf

            print(f"Downloading data for {len(self.all_tickers)} stocks...")

            data = yf.download(
                self.all_tickers,
                start=self.config.data.start_date,
                end=self.config.data.end_date,
                progress=False,
            )["Adj Close"]

            if isinstance(data, pd.Series):
                data = data.to_frame()

            # Handle missing data
            data = data.fillna(method="ffill").fillna(method="bfill")

            # Calculate returns
            returns = data.pct_change().fillna(0)

            print(f"Loaded {len(data)} days of data for {len(data.columns)} stocks")
            return data, returns

        except Exception as e:
            print(f"Error loading yfinance data: {e}")
            print("Falling back to synthetic data...")
            return self._generate_synthetic_data()

    def _generate_synthetic_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Generate synthetic market data with realistic correlations"""
        print("Generating synthetic market data...")

        n_days = 2000  # ~8 years of trading days
        n_assets = len(self.all_tickers)

        # Generate correlated returns with sector clustering
        sector_returns = {}
        for sector, stocks in self.sector_allocations.items():
            n_sector_assets = len(stocks)

            # Generate sector factor
            sector_factor = np.random.randn(n_days) * 0.015

            # Generate individual stock returns correlated with sector
            stock_returns = np.zeros((n_days, n_sector_assets))
            for i in range(n_sector_assets):
                idiosyncratic = np.random.randn(n_days) * 0.01
                stock_returns[:, i] = 0.7 * sector_factor + 0.3 * idiosyncratic

            sector_returns[sector] = pd.DataFrame(stock_returns, columns=stocks)

        # Combine all returns
        returns = pd.concat(sector_returns.values(), axis=1)
        returns = returns[self.all_tickers]  # Ensure correct order

        # Generate price data from returns
        prices = (1 + returns).cumprod() * 100

        # Add dates
        dates = pd.date_range(start="2017-01-01", periods=n_days, freq="B")
        prices.index = dates
        returns.index = dates

        print(f"Generated {n_days} days of synthetic data for {n_assets} stocks")
        return prices, returns

    def _load_csv_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load data from CSV file"""
        if self.config.data.csv_path is None:
            raise ValueError("csv_path must be specified in config")

        prices = pd.read_csv(self.config.data.csv_path, index_col=0, parse_dates=True)
        returns = prices.pct_change().fillna(0)

        print(f"Loaded {len(prices)} days from CSV")
        return prices, returns

    def calculate_technical_indicators(
        self, prices: pd.DataFrame
    ) -> Dict[str, pd.DataFrame]:
        """Calculate all technical indicators for each asset"""
        indicators = {}

        print("Calculating technical indicators...")

        for ticker in prices.columns:
            price_series = prices[ticker]

            # RSI
            rsi = calculate_rsi(price_series, self.config.env.rsi_period)

            # MACD
            macd, macd_signal = calculate_macd(
                price_series,
                self.config.env.macd_fast,
                self.config.env.macd_slow,
                self.config.env.macd_signal,
            )

            # Bollinger Bands
            bb_upper, bb_middle, bb_lower = calculate_bollinger_bands(
                price_series,
                self.config.env.bollinger_period,
                self.config.env.bollinger_std,
            )

            # Normalize Bollinger Bands to [-1, 1]
            bb_position = (price_series - bb_lower) / (
                bb_upper - bb_lower + 1e-8
            ) * 2 - 1
            bb_width = (bb_upper - bb_lower) / (bb_middle + 1e-8)

            # Store indicators
            indicators[ticker] = pd.DataFrame(
                {
                    "rsi": rsi / 100.0 - 0.5,  # Normalize to [-0.5, 0.5]
                    "macd": macd
                    / price_series.rolling(20)
                    .std()
                    .fillna(1),  # Normalize by volatility
                    "macd_signal": macd_signal
                    / price_series.rolling(20).std().fillna(1),
                    "bb_position": bb_position.fillna(0),
                    "bb_width": bb_width.fillna(1),
                },
                index=prices.index,
            )

        return indicators

    def calculate_macro_features(self, dates: pd.DatetimeIndex) -> pd.DataFrame:
        """Generate macro features (VIX proxy and interest rates)"""
        # Generate synthetic VIX and Treasury yield
        n_days = len(dates)

        # VIX-like volatility index (mean-reverting process)
        vix = np.zeros(n_days)
        vix[0] = 0.15
        for t in range(1, n_days):
            vix[t] = vix[t - 1] + 0.1 * (0.15 - vix[t - 1]) + np.random.randn() * 0.02
        vix = np.clip(vix, 0.05, 0.50)

        # Treasury yield (mean-reverting around 2%)
        treasury = np.zeros(n_days)
        treasury[0] = 0.02
        for t in range(1, n_days):
            treasury[t] = (
                treasury[t - 1]
                + 0.05 * (0.02 - treasury[t - 1])
                + np.random.randn() * 0.001
            )
        treasury = np.clip(treasury, 0.0, 0.05)

        macro_df = pd.DataFrame({"vix": vix, "treasury_yield": treasury}, index=dates)

        return macro_df

    def prepare_environment_data(self) -> Dict[str, any]:
        """Prepare all data for environment"""
        # Load price data
        prices, returns = self.load_data()

        # Calculate technical indicators
        indicators = self.calculate_technical_indicators(prices)

        # Calculate macro features
        macro = self.calculate_macro_features(prices.index)

        # Calculate historical returns
        hist_returns = {}
        for ticker in prices.columns:
            returns_df = calculate_historical_returns(
                prices[ticker],
                [self.config.env.lookback_short, self.config.env.lookback_long],
            )
            hist_returns[ticker] = returns_df

        # Calculate volatility
        volatility = {}
        for ticker in returns.columns:
            vol = calculate_volatility(
                returns[ticker], self.config.env.volatility_window
            )
            volatility[ticker] = vol

        # Split train/test
        split_idx = int(len(prices) * self.config.data.train_ratio)

        data = {
            "prices": prices,
            "returns": returns,
            "indicators": indicators,
            "hist_returns": hist_returns,
            "volatility": volatility,
            "macro": macro,
            "train_indices": (0, split_idx),
            "test_indices": (split_idx, len(prices)),
            "tickers": self.all_tickers,
            "sector_allocations": self.sector_allocations,
        }

        return data


if __name__ == "__main__":
    from config import Config

    config = Config()
    loader = MarketDataLoader(config)
    data = loader.prepare_environment_data()

    print(f"\nData summary:")
    print(f"Total days: {len(data['prices'])}")
    print(f"Training days: {data['train_indices'][1] - data['train_indices'][0]}")
    print(f"Testing days: {data['test_indices'][1] - data['test_indices'][0]}")
    print(f"Assets: {len(data['tickers'])}")
    print(f"\nSample indicators for {data['tickers'][0]}:")
    print(data["indicators"][data["tickers"][0]].head())
