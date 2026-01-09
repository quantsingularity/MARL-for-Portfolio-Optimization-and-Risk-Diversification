# Multi-Agent Reinforcement Learning for Portfolio Optimization and Risk Diversification

## Complete Implementation of Research Paper

This repository contains the **complete, fully-implemented** version of the MADDPG (Multi-Agent Deep Deterministic Policy Gradient) framework for portfolio optimization, as described in the research paper "Multi-Agent Reinforcement Learning for Portfolio Optimization and Risk Diversification" by Abrar Ahmed (January 2025).

---

## 🎯 Overview

Traditional portfolio optimization strategies often rely on single-agent models that struggle to adapt to dynamic market conditions and fail to achieve optimal diversification. This implementation introduces a novel MADDPG framework that leverages **cooperative reinforcement learning** to build diversified portfolios with superior risk-adjusted returns.

### Key Features

✅ **Multi-Agent Architecture**: 4 independent agents managing sector-specific sub-portfolios (Tech, Healthcare, Finance, Energy/Commodities)

✅ **Diversity-Promoting Rewards**: Explicit diversity penalty (λ = 0.1) based on rolling correlation of agent returns

✅ **Centralized Training, Decentralized Execution (CTDE)**: Centralized critic sees global state during training

✅ **Comprehensive State Space**: Historical returns (20-day, 60-day), technical indicators (RSI, MACD, Bollinger Bands), volatility measures, macroeconomic variables

✅ **Realistic Market Simulation**: Transaction costs (0.1%), risk-adjusted metrics, sector-based asset allocation

✅ **Complete Evaluation Suite**: Comparison with Equal-Weight, Random, Risk Parity, Mean-Variance, Single-Agent DDPG baselines

✅ **Publication-Quality Figures**: Training curves, ablation studies, comparative analysis

---

## 📊 Research Paper Results

The framework achieves (as reported in the paper):

- **Sharpe Ratio**: 1.42 (82% improvement over equal-weight baseline)
- **Annualized Return**: 18.4%
- **Maximum Drawdown**: 12.3% (50% reduction vs. equal-weight)
- **Average Agent Correlation**: 0.14 (highly decorrelated sub-portfolios)
- **Portfolio Turnover**: 0.12 (low transaction costs)

---

## 📁 Project Structure

```
marl_portfolio_complete/
│
├── config.py                  # Complete configuration management
├── data_loader.py             # Data acquisition and technical indicators
├── environment.py             # Multi-agent portfolio environment
├── maddpg_agent.py           # MADDPG algorithm implementation
├── baselines.py              # Baseline strategies for comparison
├── main.py                   # Main training/evaluation script
├── visualize.py              # Figure generation utilities
├── requirements.txt          # Python dependencies
│
├── results/                  # Training results (created at runtime)
│   ├── best_model/          # Best model checkpoints
│   ├── final_model/         # Final model checkpoints
│   ├── checkpoints/         # Periodic checkpoints
│   ├── training_history.json
│   ├── test_results.json
│   └── figures/             # Generated figures
│
└── README.md                # This file
```

---

## 🚀 Quick Start

### 1. Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

### 2. Run Demo (5 minutes)

Quick demonstration with synthetic data:

```bash
python main.py --mode demo --data-source synthetic
```

### 3. Train from Scratch

Full training with 300 episodes:

```bash
# Using synthetic data (faster, no API required)
python main.py --mode train --data-source synthetic --episodes 300

# Using real market data (requires yfinance)
python main.py --mode train --data-source yfinance --episodes 300
```

### 4. Evaluate Trained Model

```bash
python main.py --mode eval --load-model ./results/<timestamp>/best_model
```

### 5. Generate Figures

```bash
python visualize.py ./results/<timestamp>
```

---

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

### Network Architecture (from Paper Section 4.2)

```python
# Actor Network: [256, 128, 64] with ReLU, BatchNorm, Softmax
actor_hidden_dims = [256, 128, 64]

# Critic Network: [512, 256, 128] with ReLU, BatchNorm
critic_hidden_dims = [512, 256, 128]
```

### Training Hyperparameters (from Paper Section 4.3)

```python
n_episodes = 300                # Training episodes
batch_size = 128                # Batch size for updates
lr_actor = 1e-4                 # Actor learning rate
lr_critic = 1e-3                # Critic learning rate
gamma = 0.99                    # Discount factor
tau = 0.01                      # Polyak averaging rate (soft update)
```

---

## 📈 Implementation Details

### Complete MDP Formulation (Section 3 of Paper)

#### State Space

For each agent, the observation includes:
- **Historical Returns**: 20-day and 60-day returns
- **Technical Indicators**: RSI, MACD, MACD Signal, Bollinger Bands (position & width)
- **Volatility**: 20-day rolling standard deviation
- **Macroeconomic Variables**: VIX index, 10-year Treasury yield
- **Current Portfolio Weights**

#### Action Space

Continuous portfolio weight vectors for each agent's sub-portfolio:
- Constrained to [0, 1] (long-only)
- Sum to 1 (fully invested)
- Enforced by Softmax activation

#### Reward Function (Section 3.3)

The diversity-promoting reward structure:

```
R_i,t = R_base,i,t - λ × D_i,t
```

Where:
- **R_base,i,t**: Daily Sharpe ratio = (r_i,t - r_f) / σ_i,t
- **D_i,t**: Average pairwise correlation over 30-day window
- **λ = 0.1**: Diversity weight (optimal from ablation study)

### Network Architectures

#### Actor Network (Decentralized)

```
Input → [256, ReLU, BatchNorm] → [128, ReLU, BatchNorm] → [64, ReLU] → [Output, Softmax]
```

- **Input**: Local observation (agent-specific features)
- **Output**: Portfolio weights (valid probability distribution)

#### Critic Network (Centralized)

```
Input → [512, ReLU, BatchNorm] → [256, ReLU, BatchNorm] → [128, ReLU] → [1, Linear]
```

- **Input**: Global state + Joint actions
- **Output**: Q-value estimate

### Loss Functions

**Critic Loss (TD Error)**:
```
L(θ_i) = E[(Q_i(s, a_1, ..., a_N) - y)²]
where y = r_i + γ × Q_i'(s', a'_1, ..., a'_N)
```

**Actor Loss (Policy Gradient)**:
```
∇_θ_i J = E[∇_θ_i μ_i(a_i|o_i) × ∇_a_i Q_i(s, a_1, ..., a_N)]
```

**Target Network Update (Polyak Averaging)**:
```
θ' ← τ × θ + (1 - τ) × θ'  (τ = 0.01)
```

---

## 🎯 Asset Allocation

The 30 S&P 500 stocks are divided into 4 sectors:

| Agent | Sector               | Stocks | Tickers                                    |
|-------|----------------------|--------|--------------------------------------------|
| 1     | Technology           | 8      | AAPL, MSFT, NVDA, GOOGL, META, TSLA, AVGO, ADBE |
| 2     | Healthcare           | 7      | JNJ, UNH, PFE, ABBV, TMO, MRK, LLY        |
| 3     | Finance              | 7      | JPM, BAC, V, MA, GS, MS, AXP              |
| 4     | Energy/Commodities   | 8      | XOM, CVX, COP, SLB, EOG, PXD, GLD, SLV    |

---

## 📊 Evaluation Metrics

### Risk-Adjusted Returns
- **Sharpe Ratio**: (Mean Return - Risk-Free Rate) / Volatility
- **Sortino Ratio**: Downside risk-adjusted returns
- **Annualized Return**: Cumulative return scaled to annual

### Risk Measures
- **Volatility**: Standard deviation of returns
- **Maximum Drawdown (MDD)**: Peak-to-trough decline
- **Downside Deviation**: Volatility of negative returns

### Diversification
- **Average Pairwise Correlation**: Between agent portfolios
- **Portfolio Turnover**: Trading activity measure
- **Herfindahl Index**: Portfolio concentration

---

## 🏆 Baseline Comparisons

The implementation includes 5 baseline strategies from the paper:

1. **Random Allocation**: Random portfolio weights
2. **Equal-Weight (1/N)**: Uniform distribution across assets
3. **Risk Parity**: Equal risk contribution from each asset
4. **Mean-Variance Optimization (MVO)**: Markowitz optimization
5. **Single-Agent DDPG**: Single centralized agent (no diversification)

### Results Comparison (from Paper)

| Strategy          | Sharpe | Return | Max DD | Avg Corr |
|-------------------|--------|--------|--------|----------|
| **MADDPG (λ=0.1)**| **1.42** | **18.4%** | **12.3%** | **0.14** |
| MADDPG (λ=0)     | 1.13   | 16.2%  | 18.9%  | 0.42     |
| Single-Agent DDPG | 1.05   | 14.8%  | 20.2%  | 0.55     |
| Mean-Variance    | 0.88   | 12.5%  | 22.4%  | 0.48     |
| Risk Parity      | 0.82   | 10.2%  | 15.8%  | 0.35     |
| Equal-Weight     | 0.78   | 9.8%   | 24.5%  | 0.62     |
| Random           | 0.48   | 5.4%   | 35.6%  | 0.68     |

---

## 🔬 Ablation Studies (from Paper)

### Diversity Weight (λ) Impact

| λ    | Sharpe | Return | Max DD | Avg Corr |
|------|--------|--------|--------|----------|
| 0.0  | 1.13   | 16.2%  | 18.9%  | 0.42     |
| 0.05 | 1.31   | 17.8%  | 14.5%  | 0.25     |
| **0.1** | **1.42** | **18.4%** | **12.3%** | **0.14** |
| 0.2  | 1.28   | 15.5%  | 11.8%  | 0.08     |

**Optimal λ = 0.1** balances return and diversification.

### Number of Agents Impact

| # Agents | Sharpe | Max DD | Avg Corr | Training Time |
|----------|--------|--------|----------|---------------|
| 2        | 1.18   | 16.5%  | 0.28     | 1.0×          |
| **4**    | **1.42** | **12.3%** | **0.14** | **2.4×**  |
| 6        | 1.45   | 11.9%  | 0.12     | 5.8×          |
| 8        | 1.46   | 11.7%  | 0.11     | 12.1×         |

**4 agents selected** for optimal performance-efficiency trade-off.

---

## 📝 Command-Line Options

```bash
python main.py [OPTIONS]

Options:
  --mode {train,eval,demo}     Mode: train new model, evaluate existing, or run demo
  --episodes INT               Number of training episodes (default: 300)
  --data-source {yfinance,synthetic,csv}  Data source selection
  --save-dir PATH              Directory to save results (default: ./results)
  --load-model PATH            Path to load pre-trained model
  --config PATH                Path to custom configuration JSON
  --seed INT                   Random seed (default: 42)
```

### Examples

```bash
# Quick demo (5 episodes)
python main.py --mode demo

# Full training with synthetic data
python main.py --mode train --episodes 300 --data-source synthetic

# Training with real market data
python main.py --mode train --episodes 300 --data-source yfinance

# Evaluation
python main.py --mode eval --load-model ./results/20250109_120000/best_model

# Custom configuration
python main.py --mode train --config custom_config.json
```

---

## 📜 License

MIT License - See LICENSE file for details

---
