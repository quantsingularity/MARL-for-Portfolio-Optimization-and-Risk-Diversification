# Multi-Agent Reinforcement Learning for Portfolio Optimization and Risk Diversification

## Complete Implementation of Research Paper

This repository contains the full implementation of the MADDPG (Multi-Agent Deep Deterministic Policy Gradient) framework for portfolio optimization, as described in the research paper "Multi-Agent Reinforcement Learning for Portfolio Optimization and Risk Diversification".

## 📚 Overview

Traditional portfolio optimization strategies often rely on single-agent models that struggle to adapt to dynamic market conditions and fail to achieve optimal diversification. This implementation introduces a novel MADDPG framework that leverages **cooperative reinforcement learning** to build diversified portfolios with superior risk-adjusted returns.

### Key Features

✅ **Multi-Agent Architecture**: 4 independent agents managing sector-specific sub-portfolios (Tech, Healthcare, Finance, Energy/Commodities)

✅ **Diversity-Promoting Rewards**: Explicit diversity penalty based on rolling correlation of agent returns

✅ **Centralized Training, Decentralized Execution (CTDE)**: Centralized critic sees global state during training

✅ **Comprehensive State Space**: Historical returns (20-day, 60-day), technical indicators (RSI, MACD, Bollinger Bands), volatility measures

✅ **Realistic Market Simulation**: Transaction costs, risk-adjusted metrics, sector-based asset allocation

✅ **Complete Evaluation Suite**: Comparison with Equal-Weight, Random, Risk Parity, Mean-Variance, Single-Agent DDPG baselines

✅ **Publication-Quality Figures**: Training curves, portfolio evolution, diversification analysis, drawdown analysis

## 📊 Research Paper Results

The framework achieves:
- **Sharpe Ratio**: 1.52 (57% improvement over equal-weight baseline)
- **Annualized Return**: 24.5%
- **Maximum Drawdown**: -8.7% (39% reduction vs. equal-weight)
- **Average Correlation**: 0.23 (highly decorrelated sub-portfolios)
- **Portfolio Turnover**: 0.18 (low transaction costs)

## 🏗️ Project Structure

```
MARL_Portfolio_Full_Implementation/
│
├── config.py                      # Configuration management
├── data_loader.py                 # Data acquisition and preprocessing
├── enhanced_environment.py        # Complete multi-agent environment
├── maddpg_agent.py               # MADDPG algorithm implementation
├── main.py                        # Main training/evaluation script
├── generate_figures.py            # Figure generation utilities
├── requirements.txt               # Python dependencies
│
├── results/                       # Training results (created at runtime)
│   ├── best_model/               # Best model checkpoints
│   ├── final_model/              # Final model checkpoints
│   ├── checkpoints/              # Periodic checkpoints
│   ├── training_history.json     # Training metrics
│   ├── test_results.json         # Test set evaluation
│   └── figures/                  # Generated figures
│
└── README.md                      # This file
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository_url>
cd MARL_Portfolio_Full_Implementation

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Demo (5 minutes)

Quick demonstration with synthetic data:

```bash
python main.py --mode demo --data-source synthetic
```

### 3. Train from Scratch

Full training with 300 episodes (~2-4 hours depending on hardware):

```bash
# Using synthetic data (faster, no API required)
python main.py --mode train --data-source synthetic --episodes 300

# Using real market data (requires yfinance)
python main.py --mode train --data-source yfinance --episodes 300
```

### 4. Evaluate Trained Model

```bash
python main.py --mode eval --load-model ./results/best_model
```

## 🎯 Training Options

### Command-Line Arguments

```bash
python main.py [OPTIONS]

Options:
  --mode {train,eval,demo}     Mode: train new model, evaluate existing, or run demo
  --episodes INT               Number of training episodes (default: 300)
  --data-source {yfinance,synthetic,csv}  Data source selection
  --save-dir PATH              Directory to save results (default: ./results)
  --load-model PATH            Path to load pre-trained model
  --config PATH                Path to custom configuration JSON
```

### Examples

```bash
# Quick demo with 5 episodes
python main.py --mode demo

# Full training with custom save directory
python main.py --mode train --episodes 300 --save-dir ./my_experiment

# Evaluation with pre-trained model
python main.py --mode eval --load-model ./results/best_model

# Training with real market data
python main.py --mode train --data-source yfinance --episodes 200
```

## ⚙️ Configuration

The system is highly configurable through `config.py`. Key parameters:

### Environment Configuration
```python
n_agents = 4                    # Number of agents (Tech, Healthcare, Finance, Energy)
n_assets = 30                   # Total assets (30 S&P 500 stocks)
initial_capital = 1,000,000     # Initial capital per agent
transaction_cost = 0.001        # 0.1% transaction cost
diversity_weight = 0.1          # λ = 0.1 (optimal from ablation study)
```

### Network Architecture
```python
actor_hidden_dims = [256, 256, 128]      # Actor network layers
critic_hidden_dims = [512, 256, 128, 64] # Critic network layers
```

### Training Hyperparameters
```python
n_episodes = 300                # Training episodes
batch_size = 64                 # Batch size for updates
lr_actor = 1e-4                 # Actor learning rate
lr_critic = 1e-3                # Critic learning rate
gamma = 0.99                    # Discount factor
tau = 0.01                      # Soft update rate
```

## 📈 Results and Metrics

### Performance Metrics

The framework tracks comprehensive metrics:

1. **Risk-Adjusted Returns**
   - Sharpe Ratio
   - Sortino Ratio
   - Annualized Return

2. **Risk Measures**
   - Volatility
   - Maximum Drawdown
   - Downside Deviation

3. **Diversification**
   - Average Pairwise Correlation
   - Portfolio Turnover
   - Herfindahl Index

4. **Training Progress**
   - Episode rewards
   - Q-values
   - Actor/Critic losses

### Generated Figures

The system automatically generates publication-quality figures:

1. **Training Curves**: Rewards, Sharpe ratio, returns, volatility, drawdown over episodes
2. **Portfolio Evolution**: Individual and aggregate capital over time
3. **Position Heatmap**: Agent allocations across assets and time
4. **Drawdown Analysis**: Maximum drawdown visualization
5. **Diversification Analysis**: Herfindahl index, effective assets, agent similarity
6. **Risk-Return Scatter**: Comparison with baselines
7. **Architecture Diagram**: MADDPG system visualization

## 🔬 Methodology

### Multi-Agent Deep Deterministic Policy Gradient (MADDPG)

The framework implements:

1. **Centralized Training, Decentralized Execution (CTDE)**
   - Centralized critic observes global state and all agent actions during training
   - Decentralized actors operate independently during execution

2. **Diversity-Promoting Reward Structure**
   ```
   r_i(t) = r_base,i(t) - λ × Diversity_Penalty
   ```
   where:
   - `r_base,i(t)`: Agent i's risk-adjusted return (Sharpe-style)
   - `λ = 0.1`: Diversity weight (optimal from ablation study)
   - `Diversity_Penalty`: Average pairwise correlation of agent returns over 30-day window

3. **Sector-Based Agent Assignment**
   - Agent 1: Technology (8 stocks)
   - Agent 2: Healthcare (7 stocks)
   - Agent 3: Finance (7 stocks)
   - Agent 4: Energy/Commodities (8 stocks)

## 📊 Data Sources

### 1. Yahoo Finance (yfinance)
- Real market data for 30 S&P 500 stocks
- Historical period: 2017-2024
- Train: 2017-2022, Test: 2023-2024

### 2. Synthetic Data
- Realistic correlated returns with sector clustering
- Configurable volatility and correlation structure
- Useful for rapid prototyping and testing

### 3. Custom CSV
- Load your own market data
- Format: Date column + asset price columns

## 🎓 Research Paper Implementation

This implementation includes **all** components described in the research paper:

- ✅ Complete MDP formulation with comprehensive state space
- ✅ Actor-Critic network architectures with layer normalization
- ✅ Diversity-promoting reward structure with correlation penalty
- ✅ Sector-based agent assignment (Tech, Healthcare, Finance, Energy)
- ✅ Transaction costs and realistic market simulation
- ✅ Comprehensive evaluation metrics (Sharpe, Sortino, MDD, turnover)
- ✅ Baseline comparisons (Equal-Weight, Random, Risk Parity)
- ✅ Ablation studies framework (diversity weight, number of agents)
- ✅ Publication-quality figure generation
- ✅ Training/testing split and out-of-sample evaluation

## 🛠️ Advanced Usage

### Custom Configuration

Create a custom configuration file:

```python
from config import Config

# Create custom config
config = Config()
config.env.n_agents = 5  # Use 5 agents
config.env.diversity_weight = 0.15  # Higher diversity emphasis
config.training.n_episodes = 500  # More training episodes

# Save configuration
config.save('my_config.json')
```

Then use it:

```bash
python main.py --config my_config.json
```

### Using Custom Data

```python
# Load your data
import pandas as pd
data = pd.read_csv('my_market_data.csv', index_col=0, parse_dates=True)

# Ensure format: Date index + asset price columns
# Train environment
from enhanced_environment import EnhancedMultiAgentPortfolioEnv
from config import default_config

returns = data.pct_change().fillna(0)
env = EnhancedMultiAgentPortfolioEnv(default_config, returns)
```

### Hyperparameter Tuning

The framework supports easy hyperparameter experimentation:

```python
# Example: Test different diversity weights
diversity_weights = [0.05, 0.1, 0.15, 0.2]

for weight in diversity_weights:
    config = Config()
    config.env.diversity_weight = weight
    # ... run training ...
```

## 📄 License

This project is licensed under the MIT License - see LICENSE file for details.
