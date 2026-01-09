"""
Configuration Management
Adds support for Transformers, ESG, Dynamic Diversity, Multi-Asset Classes
"""

import json
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Any, Optional

@dataclass
class EnvironmentConfig:
    """Environment configuration"""
    n_agents: int = 4
    n_assets: int = 30
    initial_capital: float = 1_000_000
    transaction_cost: float = 0.001
    risk_free_rate: float = 0.02 / 252
    
    # Diversity settings
    diversity_weight: float = 0.1
    dynamic_diversity: bool = True  # NEW: Enable dynamic λ adjustment
    diversity_weight_range: tuple = (0.05, 0.2)  # NEW: Range for dynamic λ
    
    # Rolling windows
    correlation_window: int = 30
    volatility_window: int = 20
    
    # State space
    lookback_short: int = 20
    lookback_long: int = 60
    
    # Technical indicators
    rsi_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    bollinger_period: int = 20
    bollinger_std: float = 2.0
    
    # NEW: ESG Integration
    use_esg: bool = True
    esg_weight: float = 0.05
    min_esg_score: float = 50.0
    
    # NEW: Sentiment Analysis
    use_sentiment: bool = True
    sentiment_weight: float = 0.03
    
    # NEW: Multi-Asset Class Support
    asset_classes: List[str] = field(default_factory=lambda: ["equities"])
    include_crypto: bool = False
    include_bonds: bool = False
    include_commodities: bool = True
    
    # Asset allocation
    sector_allocations: Dict[str, List[str]] = None
    
    def __post_init__(self):
        if self.sector_allocations is None:
            self.sector_allocations = {
                "Tech": ["AAPL", "MSFT", "NVDA", "GOOGL", "META", "TSLA", "AVGO", "ADBE"],
                "Healthcare": ["JNJ", "UNH", "PFE", "ABBV", "TMO", "MRK", "LLY"],
                "Finance": ["JPM", "BAC", "V", "MA", "GS", "MS", "AXP"],
                "Energy": ["XOM", "CVX", "COP", "SLB", "EOG", "PXD", "GLD", "SLV"],
            }
            
            # Add crypto if enabled
            if self.include_crypto:
                self.sector_allocations["Crypto"] = ["BTC-USD", "ETH-USD", "BNB-USD"]
            
            # Add bonds if enabled
            if self.include_bonds:
                self.sector_allocations["Bonds"] = ["TLT", "IEF", "SHY", "LQD"]

@dataclass
class NetworkConfig:
    """Enhanced network architecture configuration"""
    # Actor network
    actor_hidden_dims: List[int] = field(default_factory=lambda: [256, 128, 64])
    actor_activation: str = "relu"
    actor_use_batch_norm: bool = True
    actor_output_activation: str = "softmax"
    
    # Critic network
    critic_hidden_dims: List[int] = field(default_factory=lambda: [512, 256, 128])
    critic_activation: str = "relu"
    critic_use_batch_norm: bool = True
    critic_output_activation: str = "linear"
    
    # NEW: Transformer Architecture
    use_transformer: bool = True
    transformer_heads: int = 8
    transformer_layers: int = 4
    transformer_dim: int = 256
    transformer_dropout: float = 0.1
    
    # NEW: Attention Mechanism
    use_attention: bool = True
    attention_heads: int = 4

@dataclass
class TrainingConfig:
    """Enhanced training configuration"""
    n_episodes: int = 300
    max_steps_per_episode: int = 252
    
    # Learning rates
    lr_actor: float = 1e-4
    lr_critic: float = 1e-3
    
    # RL hyperparameters
    gamma: float = 0.99
    tau: float = 0.01
    batch_size: int = 128
    
    # Experience replay
    buffer_size: int = 1_000_000
    min_buffer_size: int = 10000
    
    # Exploration
    noise_std_start: float = 0.2
    noise_std_end: float = 0.05
    noise_decay: float = 0.995
    
    # Training schedule
    update_every: int = 1
    updates_per_step: int = 1
    
    # Checkpointing
    save_interval: int = 50
    eval_interval: int = 10
    
    # NEW: Hyperparameter Optimization
    use_optuna: bool = False
    optuna_trials: int = 50
    
    # NEW: Advanced Training Features
    use_curriculum_learning: bool = False
    use_prioritized_replay: bool = False
    use_hindsight_replay: bool = False

@dataclass
class RiskManagementConfig:
    """NEW: Risk management configuration"""
    # Risk metrics
    use_cvar: bool = True
    cvar_alpha: float = 0.95
    use_sortino: bool = True
    target_return: float = 0.0
    
    # Position limits
    max_position_size: float = 0.3
    max_sector_exposure: float = 0.5
    
    # Stop loss/take profit
    use_stop_loss: bool = True
    stop_loss_threshold: float = -0.15
    use_take_profit: bool = True
    take_profit_threshold: float = 0.25
    
    # Market regime detection
    use_regime_detection: bool = True
    regime_indicators: List[str] = field(default_factory=lambda: ["vix", "yield_spread", "momentum"])

@dataclass
class DataConfig:
    """Data configuration"""
    data_source: str = "yfinance"
    start_date: str = "2017-01-01"
    end_date: str = "2024-12-31"
    train_ratio: float = 0.75
    csv_path: Optional[str] = None
    
    # NEW: Alternative Data Sources
    use_alternative_data: bool = True
    news_sources: List[str] = field(default_factory=lambda: ["reuters", "bloomberg"])
    social_sentiment: bool = False
    
    # NEW: Real-time Data
    use_realtime: bool = False
    realtime_interval: int = 60  # seconds

@dataclass
class Config:
    """Master configuration class"""
    env: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    risk: RiskManagementConfig = field(default_factory=RiskManagementConfig)
    data: DataConfig = field(default_factory=DataConfig)
    
    seed: int = 42
    device: str = "cpu"
    num_workers: int = 4
    
    # NEW: Experiment Tracking
    use_tensorboard: bool = True
    use_wandb: bool = False
    experiment_name: str = "enhanced_maddpg"
    
    # NEW: Interpretability
    use_shap: bool = True
    use_attention_visualization: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return {
            "env": asdict(self.env),
            "network": asdict(self.network),
            "training": asdict(self.training),
            "risk": asdict(self.risk),
            "data": asdict(self.data),
            "seed": self.seed,
            "device": self.device,
            "num_workers": self.num_workers,
            "use_tensorboard": self.use_tensorboard,
            "use_wandb": self.use_wandb,
            "experiment_name": self.experiment_name,
            "use_shap": self.use_shap,
            "use_attention_visualization": self.use_attention_visualization,
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
        config.risk = RiskManagementConfig(**config_dict.get("risk", {}))
        config.data = DataConfig(**config_dict.get("data", {}))
        config.seed = config_dict.get("seed", 42)
        config.device = config_dict.get("device", "cpu")
        config.num_workers = config_dict.get("num_workers", 4)
        config.use_tensorboard = config_dict.get("use_tensorboard", True)
        config.use_wandb = config_dict.get("use_wandb", False)
        config.experiment_name = config_dict.get("experiment_name", "enhanced_maddpg")
        config.use_shap = config_dict.get("use_shap", True)
        config.use_attention_visualization = config_dict.get("use_attention_visualization", True)
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
    """Calculate enhanced state dimension"""
    n_features_per_asset = (
        2  # Historical returns
        + 3  # RSI, MACD, MACD Signal
        + 3  # Bollinger Bands
        + 1  # Volatility
    )
    
    # Add ESG if enabled
    if config.env.use_esg:
        n_features_per_asset += 1
    
    # Add sentiment if enabled
    if config.env.use_sentiment:
        n_features_per_asset += 1
    
    # Macro features
    macro_features = 2  # VIX, Treasury yield
    
    # Per agent state
    assets_per_agent = config.env.n_assets // config.env.n_agents
    state_dim = n_features_per_asset * assets_per_agent + macro_features
    
    return state_dim

def get_action_dim(config: Config) -> int:
    """Calculate action dimension"""
    return config.env.n_assets // config.env.n_agents

if __name__ == "__main__":
    config = Config()
    print("Configuration initialized!")
    print(f"State dimension: {get_state_dim(config)}")
    print(f"Action dimension: {get_action_dim(config)}")
    print(f"Using Transformer: {config.network.use_transformer}")
    print(f"Using ESG: {config.env.use_esg}")
    print(f"Using Sentiment: {config.env.use_sentiment}")
    print(f"Dynamic Diversity: {config.env.dynamic_diversity}")
