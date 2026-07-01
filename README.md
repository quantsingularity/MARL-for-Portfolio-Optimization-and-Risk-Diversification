# MARL Portfolio Optimization

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Docker](https://img.shields.io/badge/docker-ready-blue.svg)](https://www.docker.com/)
[![Tests](https://img.shields.io/badge/tests-82%25%20coverage-green.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A production-ready Multi-Agent Reinforcement Learning system for portfolio optimization and risk diversification. Built on MADDPG with Transformer and MLP variants, dynamic volatility-based diversification, ESG and sentiment integration, and a full FastAPI deployment stack.

---

## Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Model Configurations](#model-configurations)
- [Repository Structure](#repository-structure)
- [Analysis Tools](#analysis-tools)
- [Deployment](#deployment)
- [Documentation](#documentation)
- [License](#license)

---

## Overview

| Feature               | Description                                                                           |
| :-------------------- | :------------------------------------------------------------------------------------ |
| **MADDPG Agents**     | Multi-agent collaborative portfolio allocation with MLP and Transformer architectures |
| **MARL-Lite**         | Simplified MLP using top 5 features — 65% faster training, 65% less memory            |
| **Dynamic Diversity** | Adjusts diversification penalty based on real-time VIX for regime-aware risk control  |
| **ESG and Sentiment** | FinBERT-based sentiment and simulated ESG scores as additional state signals          |
| **FastAPI**           | 15+ endpoints for model serving, rebalancing, and health checks                       |
| **Risk Monitor**      | Background service for real-time risk metrics and alert generation                    |
| **Ablation Study**    | Feature importance scripts to identify the most valuable model inputs                 |

---

## Quick Start

### Docker (Recommended)

```bash
git clone https://github.com/quantsingularity/MARL-for-Portfolio-Optimization-and-Risk-Diversification.git
cd MARL-for-Portfolio-Optimization-and-Risk-Diversification

docker-compose --profile production up -d
```

| Service              | URL                          |
| :------------------- | :--------------------------- |
| API Documentation    | `http://localhost:8000/docs` |
| Monitoring Dashboard | `http://localhost:8050`      |

### Local Setup

```bash
chmod +x setup.sh && ./setup.sh
source venv/bin/activate

python portfolio/main.py --mode demo
python portfolio/main.py --mode train --config configs/marl_lite.json
```

---

<!-- output-note -->

### Output

Console output stays readable: benign third-party warnings and library progress bars are suppressed, so only meaningful log lines remain. Each run finishes with a clean, aligned summary block reporting episodes trained, number of agents and the best Sharpe ratio achieved.

## Model Configurations

### Comparison

| Configuration | Architecture | Features                                    | Resource Usage     | Use Case                                          |
| :------------ | :----------- | :------------------------------------------ | :----------------- | :------------------------------------------------ |
| **MARL-Full** | Transformer  | All (ESG, Sentiment, full TA)               | 2.8 GB RAM, 45s/ep | Maximum performance and research                  |
| **MARL-Lite** | MLP          | Top 5 (Returns, Volatility, RSI, MACD, VIX) | 980 MB RAM, 16s/ep | Fast iteration, resource-constrained environments |

### Performance vs. Equal-Weight Baseline

| Metric        | MARL-Full | MARL-Lite    | Equal-Weight |
| :------------ | :-------- | :----------- | :----------- |
| Sharpe Ratio  | **1.68**  | 1.34         | 0.93         |
| Total Return  | 21.2%     | 18.5%        | 12.1%        |
| Max Drawdown  | -8.5%     | -10.1%       | -15.5%       |
| Training Time | 45.2s/ep  | **15.8s/ep** | N/A          |
| Memory Usage  | 2.8 GB    | **980 MB**   | N/A          |

MARL-Lite achieves 80% of the full model Sharpe Ratio at 35% of the training cost, making it the recommended starting point.

---

## Repository Structure

| Path                          | Description                                                 |
| :---------------------------- | :---------------------------------------------------------- |
| `portfolio/`                  | Training loop, MADDPG implementation, portfolio environment |
| `portfolio/models/`           | Transformer actor, MLP critic, regime detector              |
| `portfolio/risk_management/`  | Dynamic diversity, VaR, CVaR, risk metrics                  |
| `portfolio/features/`         | ESG provider, FinBERT sentiment analyzer                    |
| `portfolio/api/`              | FastAPI endpoints for model serving and rebalancing         |
| `portfolio/production/`       | Scheduler and real-time risk monitor services               |
| `portfolio/analysis/`         | Feature importance, rebalancing optimization scripts        |
| `portfolio/interpretability/` | Feature attribution and decision rationale                  |
| `configs/`                    | JSON configs for default, MARL-Lite, and Transformer        |
| `tests/`                      | Pytest suite with 82%+ coverage                             |
| `notebooks/`                  | Feature analysis and model interpretation notebooks         |

---

## Analysis Tools

| Script                                           | Purpose                                                     | Output                          |
| :----------------------------------------------- | :---------------------------------------------------------- | :------------------------------ |
| `portfolio/analysis/feature_importance.py`       | Ablation study ranking features by Sharpe Ratio impact      | `results/feature_analysis/`     |
| `portfolio/analysis/rebalancing_optimization.py` | Optimal rebalancing frequency vs. transaction cost analysis | `results/rebalancing_analysis/` |
| `portfolio/benchmarks/run_benchmarks.py`         | Full, Lite, and baseline model comparison                   | `results/benchmarks/`           |
| `portfolio/interpretability/explainer.py`        | Feature attribution map for individual decisions            | Used by API                     |

---

## Deployment

### Docker Compose Profiles

| Profile      | Services                                | Access          |
| :----------- | :-------------------------------------- | :-------------- |
| `production` | API, dashboard, scheduler, risk monitor | Port 8000, 8050 |
| `train-cpu`  | Trainer (CPU)                           | Logs            |
| `train-gpu`  | Trainer (GPU, requires NVIDIA runtime)  | Logs            |
| `train-lite` | MARL-Lite trainer                       | Logs            |
| `test`       | Full Pytest suite                       | Logs            |

### Makefile Commands

| Command           | Action                       |
| :---------------- | :--------------------------- |
| `make install`    | Install local dependencies   |
| `make test`       | Run test suite with coverage |
| `make train`      | Train full MARL model (CPU)  |
| `make train-lite` | Train MARL-Lite model        |
| `make benchmark`  | Run performance benchmarks   |
| `make docker-up`  | Start production stack       |
| `make api`        | Start FastAPI server locally |

---

## Documentation

| Resource          | Location                     |
| :---------------- | :--------------------------- |
| Quick Start Guide | `docs/QUICKSTART.md`         |
| API Documentation | `http://localhost:8000/docs` |
| Jupyter Notebooks | `notebooks/`                 |
| Test Suite        | `tests/`                     |

---

## License

Licensed under the **MIT License**. See [LICENSE](LICENSE) for details.
