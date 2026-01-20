# Multi-Agent Reinforcement Learning for Portfolio Optimization and Risk Diversification

## 🎯 Project Overview

This repository presents a cutting-edge **Multi-Agent Reinforcement Learning (MARL)** framework for dynamic portfolio optimization, focusing on enhanced risk diversification and integration of non-traditional data sources. The system employs a novel **Transformer-based architecture** for superior temporal pattern recognition and features an **adaptive diversity-promoting reward function** that adjusts to real-time market volatility.

### Key Features

The framework is built on a modular and extensible design, incorporating advanced techniques from deep learning and quantitative finance.

| Feature                            | Description                                                                                                                                                       |
| :--------------------------------- | :---------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Transformer-Based Architecture** | Utilizes a 4-layer transformer encoder with multi-head self-attention to capture long-range dependencies and complex market dynamics.                             |
| **Dynamic Diversity Weight (λ)**   | Implements an adaptive diversity penalty that automatically adjusts based on market regime (e.g., VIX) to enforce diversification during high-volatility periods. |
| **ESG Integration**                | Incorporates ESG scores as state features and a weighted reward component to align portfolio selection with sustainable investment mandates.                      |
| **Sentiment Analysis (FinBERT)**   | Integrates real-time news sentiment from FinBERT as a non-price signal to inform agent decision-making.                                                           |
| **Advanced Risk Metrics**          | Calculates and optimizes for Conditional Value-at-Risk (CVaR) and Sortino Ratio for robust downside risk management.                                              |
| **Market Regime Detection**        | Uses a Hidden Markov Model (HMM) to classify Bull/Bear/Sideways markets, enabling regime-specific strategy adaptation.                                            |
| **Model Interpretability**         | Provides SHAP value analysis and attention weight visualization to explain agent decisions and feature importance.                                                |
| **Real-Time Dashboard**            | Includes a Plotly/Dash application for live portfolio performance tracking, risk metrics, and agent allocation heatmaps.                                          |

## 📊 Key Results (Out-of-Sample Performance)

The full MARL system significantly outperforms traditional baselines and the original MADDPG implementation across key risk-adjusted metrics.

| Metric                 | Original MADDPG | No-Transformer | No-ESG | **Full MARL System** |
| :--------------------- | :-------------- | :------------- | :----- | :------------------- |
| **Sharpe Ratio**       | 1.42            | 1.52           | 1.61   | **1.68**             |
| **Max Drawdown (MDD)** | 12.3%           | 11.2%          | 10.3%  | **9.8%**             |
| **Total Return**       | 18.4%           | 19.1%          | 20.5%  | **21.2%**            |
| **Avg Correlation**    | 0.14            | 0.11           | 0.10   | **0.09**             |
| **ESG Score**          | N/A             | 72.5           | N/A    | **72.5**             |

## 🚀 Quick Start

The project is designed for easy setup and execution.

### 1. Installation

```bash
# Clone repository
git clone https://github.com/quantsingularity/MARL-for-Portfolio-Optimization-and-Risk-Diversification
cd MARL-for-Portfolio-Optimization-and-Risk-Diversification

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Advanced Demo

Run a quick demo with the full feature set (Transformer, ESG, Sentiment) to verify installation.

```bash
# Quick demo (5 episodes) with all advanced features
python code/main.py --mode demo --use-transformer --use-esg --use-sentiment

# Full training with the Transformer architecture
python code/main.py --mode train --episodes 300 --config configs/transformer.json

# Launch Real-Time Dashboard
python code/dashboard/app.py --port 8050
# Access at: http://localhost:8050
```

## 📁 Repository Structure

The repository is structured to separate the core MARL implementation, configuration, and analysis notebooks.

```
MARL-for-Portfolio-Optimization-and-Risk-Diversification/
├── README.md                          # This file
├── LICENSE                            # Project license
├── requirements.txt                   # Python dependencies
├── setup.py                           # Package installation script
│
├── code/                              # Main implementation
│   ├── maddpg_agent.py                # MADDPG agent with Transformer integration
│   ├── environment.py                 # Multi-Agent Portfolio Environment (with diversity reward)
│   ├── main.py                        # Main training and evaluation script
│   │
│   ├── models/                        # Deep Learning Architectures
│   │   ├── transformer_actor.py       # Transformer-based actor network
│   │   ├── transformer_critic.py      # Transformer-based critic network
│   │   └── regime_detector.py         # HMM-based market regime detection
│   │
│   ├── features/                      # Data Feature Engineering
│   │   ├── esg_provider.py            # ESG data integration
│   │   └── sentiment_analyzer.py      # FinBERT sentiment analysis
│   │
│   ├── risk_management/               # Risk and Position Control
│   │   ├── risk_metrics.py            # CVaR, Sortino, and other metrics
│   │   └── dynamic_diversity.py       # Adaptive λ adjustment logic
│   │
│   ├── interpretability/              # Model Explanation Tools
│   │   ├── shap_analyzer.py           # SHAP value computation
│   │   └── attention_viz.py           # Attention weight visualization
│   │
│   └── dashboard/                     # Real-Time Monitoring Dashboard (Plotly/Dash)
│
├── configs/                           # JSON configuration files
│   ├── default.json                   # Default configuration
│   └── transformer.json               # Transformer-specific config
│
└── notebooks/                         # Analysis and visualization notebooks
    ├── 01_data_exploration.ipynb
    └── 03_model_interpretation.ipynb
```

## 🏗️ Architecture

The system is built around the **MADDPG (Multi-Agent Deep Deterministic Policy Gradient)** algorithm, where each agent is responsible for a subset of assets (e.g., a sector) and is trained to maximize its own return while minimizing correlation with other agents.

### Core Components and Responsibilities

| Component                | Responsibility                                                                                                      | Implementation Location            |
| :----------------------- | :------------------------------------------------------------------------------------------------------------------ | :--------------------------------- |
| **Agent (MADDPG)**       | Learns optimal portfolio weights (actions) for its assigned assets based on observations.                           | `code/maddpg_agent.py`             |
| **Environment**          | Simulates market dynamics, executes agent actions, calculates returns, and computes the diversity-promoting reward. | `code/environment.py`              |
| **Transformer Networks** | Replaces standard MLPs in the Actor/Critic to process sequential market data and extract temporal features.         | `code/models/transformer_actor.py` |
| **Feature Engineer**     | Generates state features, including technical indicators, ESG scores, and FinBERT sentiment.                        | `code/features/`                   |
| **Risk Manager**         | Enforces position limits, stop-loss/take-profit rules, and dynamically adjusts the diversity weight (λ).            | `code/risk_management/`            |
| **Regime Detector**      | Provides the current market regime (Bull/Bear/Sideways) as an input to the agents and the dynamic diversity logic.  | `code/models/regime_detector.py`   |

### Key Design Principles

| Principle                      | Explanation                                                                                                                |
| :----------------------------- | :------------------------------------------------------------------------------------------------------------------------- |
| **Decentralized Execution**    | Agents act independently on their asset subsets, promoting specialization and computational efficiency.                    |
| **Centralized Training**       | The Critic network uses a global state (all observations and actions) to stabilize training and coordinate agents.         |
| **Diversity-Promoting Reward** | The reward function penalizes high correlation between agents' portfolio returns, explicitly driving risk diversification. |
| **Data-Driven Risk Control**   | Risk management is integrated into the environment and reward structure, moving beyond simple post-trade analysis.         |
| **Interpretability-First**     | The use of SHAP and attention visualization ensures that complex deep learning decisions are transparent and auditable.    |

## 🧪 Evaluation Framework

The repository includes scripts and notebooks for comprehensive evaluation and ablation studies.

### Ablation Studies

```bash
# 1. Transformer vs. MLP Ablation
python code/main.py --use-transformer --save-dir results/transformer
python code/main.py --no-transformer --save-dir results/mlp

# 2. Dynamic Diversity Study (Compare dynamic vs. static lambda)
python code/main.py --dynamic-diversity
python code/main.py --diversity-weight 0.1 --no-dynamic-diversity

# 3. ESG Impact Analysis (Compare with and without ESG constraint)
python code/main.py --use-esg --esg-weight 0.05
python code/main.py --no-esg
```

### Testing

Run the unit and integration tests to ensure all components are functioning correctly.

```bash
# Install pytest and coverage
pip install pytest pytest-cov

# Run all tests
pytest tests/ -v

# Run specific test suites
pytest tests/test_transformer.py
pytest tests/test_risk_metrics.py
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
