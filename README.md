# Code Documentation

## Overview

This directory contains the complete implementation of the MADDPG framework for portfolio optimization.

## File Structure

```
code/
├── marl_portfolio_env.py          # Multi-agent portfolio environment
├── maddpg_agent.py                 # MADDPG algorithm implementation
├── train_maddpg.py                 # Training pipeline
├── generate_figures.py             # Figure generation
├── generate_synthetic_results.py   # Quick demo
├── requirements.txt                # Dependencies
└── models/                         # Saved model directory
```

## Quick Start

### 1. Installation

```bash
pip install -r requirements.txt
```

### 2. Generate Demo Results (Fast)

```bash
python generate_synthetic_results.py
```

This creates synthetic results and all figures in ~2 minutes without training.

### 3. Train from Scratch (Slow)

```bash
python train_maddpg.py
```

This trains for 300 episodes (~2 hours) and saves models to `models/`.

### 4. Generate Figures

```bash
python generate_figures.py
```

## Detailed Documentation

### marl_portfolio_env.py

**Purpose:** Implements the multi-agent portfolio optimization environment.

**Key Classes:**
- `MultiAgentPortfolioEnv`: Main environment following Gymnasium API

**Key Methods:**
- `reset()`: Initialize environment for new episode
- `step(actions)`: Execute one timestep with agent actions
- `get_portfolio_metrics()`: Calculate performance metrics

**Parameters:**
- `n_agents`: Number of agents (default: 3)
- `n_assets`: Number of assets (default: 10)
- `initial_capital`: Starting capital per agent (default: 1,000,000)
- `transaction_cost`: Trading cost rate (default: 0.001)
- `diversity_bonus`: Diversity reward weight (default: 0.1)
- `lookback_window`: Historical window (default: 20)

### maddpg_agent.py

**Purpose:** Implements the MADDPG algorithm with actor-critic networks.

**Key Classes:**
- `Actor`: Policy network (state → portfolio weights)
- `Critic`: Value network (state-action → Q-value)
- `MADDPGAgent`: Single agent with actor and critic
- `MADDPG`: Multi-agent coordinator
- `ReplayBuffer`: Experience replay memory

**Network Architecture:**
- Actor: [256, 256, 128] → Softmax
- Critic: [512, 256, 128, 64] → Linear
- Activation: ReLU
- Normalization: Layer Normalization

### train_maddpg.py

**Purpose:** Training pipeline with baselines and evaluation.

**Key Functions:**
- `generate_synthetic_market_data()`: Create realistic market data
- `train_maddpg()`: Main training loop
- `evaluate_maddpg()`: Test trained agents
- `compare_with_baselines()`: Compare with Equal-Weight and Random

**Command Line Arguments:**
```bash
python train_maddpg.py --episodes 300 --agents 3 --assets 10 --lr 1e-4
```

## Hyperparameters

### Environment
- Assets: 10
- Agents: 3
- Initial capital: $1,000,000
- Transaction cost: 0.1%
- Diversity bonus: 0.1
- Lookback window: 20 days

### Training
- Episodes: 300
- Episode length: Variable (500-700 steps)
- Replay buffer: 100,000
- Batch size: 64
- Learning rate (actor): 1e-4
- Learning rate (critic): 1e-3
- Discount factor: 0.99
- Soft update: τ = 0.01

## Customization

### Change Number of Agents

```python
env = MultiAgentPortfolioEnv(n_agents=5)  # Use 5 agents
maddpg = MADDPG(n_agents=5, ...)
```

### Adjust Diversity Bonus

```python
env = MultiAgentPortfolioEnv(diversity_bonus=0.2)  # Stronger diversity
```

### Use Real Market Data

```python
import pandas as pd

# Load your data (columns: Date, Asset1, Asset2, ...)
data = pd.read_csv('market_data.csv')
returns = data.pct_change().dropna()

env = MultiAgentPortfolioEnv(data=returns.values)
```

## Performance Tips

1. **Use GPU:** Set `device='cuda'` in MADDPG initialization
2. **Parallel Training:** Use multiple environments for data collection
3. **Hyperparameter Search:** Use tools like Optuna for optimization
4. **Early Stopping:** Monitor validation performance to prevent overfitting

## Troubleshooting

### Training Unstable
- Reduce learning rates
- Increase replay buffer size
- Add gradient clipping
- Check data normalization

### Poor Performance
- Increase network capacity
- Adjust diversity bonus
- Extend training episodes
- Check transaction costs

### Out of Memory
- Reduce batch size
- Decrease replay buffer
- Use smaller networks
- Clear gradients frequently
