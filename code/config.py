"""
Configuration file for MADDPG Portfolio Optimization
Contains all hyperparameters, network architectures, and system settings
"""

import torch
from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class EnvironmentConfig:
    """Environment configuration"""
    # Asset and Agent Configuration
    n_agents: int = 4  # As per paper: Tech, Healthcare, Finance, Energy/Commodities
    n_assets: int = 30  # 30 large-cap US stocks from S&P 500
    
    # Agent-Asset Assignment (sector-based)
    agent_asset_assignment: List[List[int]] = field(default_factory=lambda: [
        list(range(0, 8)),    # Agent 1: Tech (8 stocks)
        list(range(8, 15)),   # Agent 2: Healthcare (7 stocks)
        list(range(15, 22)),  # Agent 3: Finance (7 stocks)
        list(range(22, 30))   # Agent 4: Energy/Commodities (8 stocks)
    ])
    
    # Capital Configuration
    initial_capital: float = 1_000_000.0  # Initial capital per agent
    
    # Transaction Costs
    transaction_cost: float = 0.001  # 0.1% (10 bps)
    
    # State Space Configuration
    lookback_window_20: int = 20  # 20-day historical returns
    lookback_window_60: int = 60  # 60-day historical returns
    
    # Reward Configuration
    diversity_weight: float = 0.1  # λ = 0.1 (optimal from ablation study)
    risk_free_rate: float = 0.02 / 252  # Daily risk-free rate (2% annual)
    correlation_window: int = 30  # 30-day rolling correlation for diversity penalty
    
    # Episode Configuration
    max_episode_steps: Optional[int] = None  # Will be set based on data length


@dataclass
class NetworkConfig:
    """Neural network architecture configuration"""
    # Actor Network (Policy)
    actor_hidden_dims: List[int] = field(default_factory=lambda: [256, 256, 128])
    actor_activation: str = 'relu'
    actor_use_layer_norm: bool = True
    
    # Critic Network (Value)
    critic_hidden_dims: List[int] = field(default_factory=lambda: [512, 256, 128, 64])
    critic_activation: str = 'relu'
    critic_use_layer_norm: bool = True
    
    # Initialization
    weight_init: str = 'xavier_uniform'  # xavier_uniform, kaiming_normal, orthogonal


@dataclass
class TrainingConfig:
    """Training configuration"""
    # Episode Configuration
    n_episodes: int = 300  # As per paper
    
    # Replay Buffer
    buffer_capacity: int = 100_000
    batch_size: int = 64
    min_buffer_size: int = 1000  # Minimum samples before training starts
    
    # Learning Rates
    lr_actor: float = 1e-4  # Actor learning rate
    lr_critic: float = 1e-3  # Critic learning rate
    
    # RL Hyperparameters
    gamma: float = 0.99  # Discount factor
    tau: float = 0.01  # Soft update rate (Polyak averaging)
    
    # Exploration
    initial_noise_scale: float = 0.2
    noise_decay: float = 0.9995
    min_noise_scale: float = 0.01
    
    # Gradient Clipping
    max_grad_norm: float = 1.0
    
    # Entropy Regularization
    entropy_coef: float = 0.01
    
    # Checkpoint Configuration
    save_interval: int = 20  # Save model every N episodes
    eval_interval: int = 10  # Evaluate every N episodes
    
    # Device
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'


@dataclass
class DataConfig:
    """Data configuration"""
    # Data Source
    data_source: str = 'yfinance'  # 'yfinance', 'synthetic', 'csv'
    
    # Date Range
    start_date: str = '2017-01-01'
    end_date: str = '2024-12-31'
    
    # Train/Test Split
    train_start: str = '2017-01-01'
    train_end: str = '2022-12-31'
    test_start: str = '2023-01-01'
    test_end: str = '2024-12-31'
    
    # Asset Tickers (30 S&P 500 stocks by sector)
    tickers: List[str] = field(default_factory=lambda: [
        # Tech (8 stocks) - Agent 1
        'AAPL', 'MSFT', 'NVDA', 'GOOGL', 'META', 'TSLA', 'ADBE', 'CRM',
        # Healthcare (7 stocks) - Agent 2
        'JNJ', 'PFE', 'ABBV', 'UNH', 'LLY', 'MRK', 'TMO',
        # Finance (7 stocks) - Agent 3
        'JPM', 'BAC', 'V', 'MA', 'WFC', 'GS', 'MS',
        # Energy/Commodities (8 stocks) - Agent 4
        'XOM', 'CVX', 'COP', 'SLB', 'MPC', 'VLO', 'PSX', 'HES'
    ])
    
    # Technical Indicators
    use_technical_indicators: bool = True
    use_sentiment: bool = False  # Set to True if sentiment data available
    use_macro_features: bool = True  # VIX, Treasury yields, etc.


@dataclass
class EvaluationConfig:
    """Evaluation configuration"""
    # Number of evaluation episodes
    n_eval_episodes: int = 10
    
    # Baseline strategies to compare
    baselines: List[str] = field(default_factory=lambda: [
        'equal_weight',
        'random',
        'risk_parity',
        'mean_variance',
        'single_agent_ddpg',
        'maddpg_no_diversity'
    ])


# Global configuration instance
class Config:
    """Main configuration class"""
    def __init__(self):
        self.env = EnvironmentConfig()
        self.network = NetworkConfig()
        self.training = TrainingConfig()
        self.data = DataConfig()
        self.evaluation = EvaluationConfig()
    
    def to_dict(self):
        """Convert config to dictionary"""
        return {
            'environment': self.env.__dict__,
            'network': self.network.__dict__,
            'training': self.training.__dict__,
            'data': self.data.__dict__,
            'evaluation': self.evaluation.__dict__
        }
    
    def save(self, path: str):
        """Save configuration to JSON file"""
        import json
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: str):
        """Load configuration from JSON file"""
        import json
        with open(path, 'r') as f:
            config_dict = json.load(f)
        
        config = cls()
        for key, value in config_dict.get('environment', {}).items():
            setattr(config.env, key, value)
        for key, value in config_dict.get('network', {}).items():
            setattr(config.network, key, value)
        for key, value in config_dict.get('training', {}).items():
            setattr(config.training, key, value)
        for key, value in config_dict.get('data', {}).items():
            setattr(config.data, key, value)
        for key, value in config_dict.get('evaluation', {}).items():
            setattr(config.evaluation, key, value)
        
        return config


# Default configuration
default_config = Config()
