"""
Configuration management for MADDPG Portfolio Optimization
Complete implementation based on research paper specifications
"""

import json
from dataclasses import dataclass, asdict
from typing import List, Dict, Any


@dataclass
class EnvironmentConfig:
    """Environment configuration matching paper specifications"""

    n_agents: int = 4  # Tech, Healthcare, Finance, Energy/Commodities
    n_assets: int = 30  # 30 S&P 500 stocks
    initial_capital: float = 1_000_000  # $1M per agent
    transaction_cost: float = 0.001  # 0.1% as per paper
    risk_free_rate: float = 0.02 / 252  # Daily risk-free rate (2% annual)

    # Diversity penalty weight (optimal from ablation study)
    diversity_weight: float = 0.1  # λ = 0.1 (optimal)

    # Rolling windows for calculations
    correlation_window: int = 30  # 30-day window for diversity penalty
    volatility_window: int = 20  # 20-day rolling volatility

    # State space dimensions
    lookback_short: int = 20  # Short-term historical returns
    lookback_long: int = 60  # Long-term historical returns

    # Technical indicators
    rsi_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    bollinger_period: int = 20
    bollinger_std: float = 2.0

    # Asset allocation per agent (from paper)
    sector_allocations: Dict[str, List[str]] = None

    def __post_init__(self):
        if self.sector_allocations is None:
            # Exact allocation from paper
            self.sector_allocations = {
                "Tech": [
                    "AAPL",
                    "MSFT",
                    "NVDA",
                    "GOOGL",
                    "META",
                    "TSLA",
                    "AVGO",
                    "ADBE",
                ],
                "Healthcare": ["JNJ", "UNH", "PFE", "ABBV", "TMO", "MRK", "LLY"],
                "Finance": ["JPM", "BAC", "V", "MA", "GS", "MS", "AXP"],
                "Energy": ["XOM", "CVX", "COP", "SLB", "EOG", "PXD", "GLD", "SLV"],
            }


@dataclass
class NetworkConfig:
    """Network architecture from paper Section 4.2"""

    # Actor network: [256, 128, 64] with ReLU and BatchNorm
    actor_hidden_dims: List[int] = None
    actor_activation: str = "relu"
    actor_use_batch_norm: bool = True
    actor_output_activation: str = "softmax"  # Ensures weights sum to 1

    # Critic network: [512, 256, 128] with ReLU and BatchNorm
    critic_hidden_dims: List[int] = None
    critic_activation: str = "relu"
    critic_use_batch_norm: bool = True
    critic_output_activation: str = "linear"

    def __post_init__(self):
        if self.actor_hidden_dims is None:
            self.actor_hidden_dims = [256, 128, 64]
        if self.critic_hidden_dims is None:
            self.critic_hidden_dims = [512, 256, 128]


@dataclass
class TrainingConfig:
    """Training hyperparameters from paper Section 4.3"""

    n_episodes: int = 300  # Training episodes
    max_steps_per_episode: int = 252  # ~1 trading year

    # Learning rates from paper
    lr_actor: float = 1e-4  # Actor learning rate
    lr_critic: float = 1e-3  # Critic learning rate

    # RL hyperparameters
    gamma: float = 0.99  # Discount factor
    tau: float = 0.01  # Polyak averaging rate for target networks
    batch_size: int = 128  # Batch size from paper

    # Experience replay
    buffer_size: int = 1_000_000  # Replay buffer size
    min_buffer_size: int = 10000  # Minimum before training starts

    # Exploration
    noise_std_start: float = 0.2  # Initial exploration noise
    noise_std_end: float = 0.05  # Final exploration noise
    noise_decay: float = 0.995  # Decay rate per episode

    # Training schedule
    update_every: int = 1  # Update frequency
    updates_per_step: int = 1  # Number of updates per step

    # Checkpointing
    save_interval: int = 50  # Save model every N episodes
    eval_interval: int = 10  # Evaluate every N episodes


@dataclass
class DataConfig:
    """Data configuration"""

    data_source: str = "yfinance"  # 'yfinance', 'synthetic', or 'csv'
    start_date: str = "2017-01-01"  # Training start (paper: 2017-2024)
    end_date: str = "2024-12-31"  # Training end
    train_ratio: float = 0.75  # 75% train, 25% test (2017-2022 train, 2023-2024 test)

    # CSV data path (if using custom data)
    csv_path: str = None


@dataclass
class Config:
    """Master configuration class"""

    env: EnvironmentConfig = None
    network: NetworkConfig = None
    training: TrainingConfig = None
    data: DataConfig = None

    # Experiment settings
    seed: int = 42
    device: str = "cpu"  # 'cuda' or 'cpu'
    num_workers: int = 4  # For data loading

    def __post_init__(self):
        if self.env is None:
            self.env = EnvironmentConfig()
        if self.network is None:
            self.network = NetworkConfig()
        if self.training is None:
            self.training = TrainingConfig()
        if self.data is None:
            self.data = DataConfig()

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return {
            "env": asdict(self.env),
            "network": asdict(self.network),
            "training": asdict(self.training),
            "data": asdict(self.data),
            "seed": self.seed,
            "device": self.device,
            "num_workers": self.num_workers,
        }

    def save(self, path: str):
        """Save configuration to JSON file"""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "Config":
        """Load configuration from dictionary"""
        config = cls()
        config.env = EnvironmentConfig(**config_dict.get("env", {}))
        config.network = NetworkConfig(**config_dict.get("network", {}))
        config.training = TrainingConfig(**config_dict.get("training", {}))
        config.data = DataConfig(**config_dict.get("data", {}))
        config.seed = config_dict.get("seed", 42)
        config.device = config_dict.get("device", "cpu")
        config.num_workers = config_dict.get("num_workers", 4)
        return config

    @classmethod
    def load(cls, path: str) -> "Config":
        """Load configuration from JSON file"""
        with open(path, "r") as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)


# Default configuration instance
default_config = Config()


def get_state_dim(config: Config) -> int:
    """Calculate state dimension based on configuration"""
    n_features_per_asset = (
        2  # Historical returns (20-day, 60-day)
        + 3  # RSI, MACD, MACD Signal
        + 3  # Bollinger Bands (upper, middle, lower)
        + 1  # Volatility (20-day rolling std)
    )
    # Add macroeconomic features (VIX, Treasury yield)
    macro_features = 2

    # Per agent state = features for assigned assets + macro features
    assets_per_agent = config.env.n_assets // config.env.n_agents
    state_dim = n_features_per_asset * assets_per_agent + macro_features

    return state_dim


def get_action_dim(config: Config) -> int:
    """Calculate action dimension (portfolio weights per agent)"""
    return config.env.n_assets // config.env.n_agents


if __name__ == "__main__":
    # Test configuration
    config = Config()
    print("Configuration initialized successfully!")
    print(f"State dimension: {get_state_dim(config)}")
    print(f"Action dimension: {get_action_dim(config)}")
    print(f"\nSector allocations:")
    for sector, stocks in config.env.sector_allocations.items():
        print(f"  {sector}: {len(stocks)} stocks - {', '.join(stocks[:3])}...")
