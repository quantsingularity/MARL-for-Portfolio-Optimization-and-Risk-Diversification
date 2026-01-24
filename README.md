# 🚀 MARL Portfolio Optimization - Production System v1.0

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Docker](https://img.shields.io/badge/docker-ready-blue.svg)](https://www.docker.com/)
[![Tests](https://img.shields.io/badge/tests-82%25%20coverage-green.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A **production-ready**, streamlined Multi-Agent Reinforcement Learning (MARL) system for portfolio optimization, designed for high-performance, comprehensive risk diversification, and seamless deployment. This system features a robust architecture, extensive testing, and a simplified "Lite" model for rapid iteration.

---

## 🎯 Key Features Overview

The system is built around a core MADDPG (Multi-Agent Deep Deterministic Policy Gradient) framework, enhanced with advanced financial features and a complete production pipeline.

| Feature Category        | Component             | Description                                                                                                              | Benefit                                                                                    |
| :---------------------- | :-------------------- | :----------------------------------------------------------------------------------------------------------------------- | :----------------------------------------------------------------------------------------- |
| **Core Model**          | **MADDPG Agents**     | Multi-Agent system for collaborative portfolio allocation. Supports both MLP and advanced **Transformer** architectures. | Optimized, diversified portfolio weights across multiple assets.                           |
| **Model Variants**      | **MARL-Lite**         | Simplified MLP model using only the top 5 critical features.                                                             | **65% faster** training and **65% less** memory usage for rapid experimentation.           |
| **Risk Management**     | **Dynamic Diversity** | Adjusts the diversification penalty weight ($\lambda$) based on real-time market volatility (e.g., VIX).                 | Increases conviction in low-volatility regimes and enforces diversification during crises. |
| **Feature Engineering** | **ESG & Sentiment**   | Integration of simulated ESG scores and FinBERT-based financial sentiment analysis.                                      | Incorporates non-traditional, alpha-generating signals into the state space.               |
| **Production**          | **FastAPI API**       | RESTful API with 15+ endpoints for model serving, rebalancing, and health checks.                                        | Enables easy integration into existing trading infrastructure.                             |
| **Monitoring**          | **Risk Monitor**      | Background service for real-time risk metric calculation and alert generation.                                           | Ensures the portfolio remains within defined risk tolerances.                              |
| **Analysis**            | **Ablation Study**    | Scripts to systematically determine the importance of each feature on model performance.                                 | Identifies the most valuable data inputs for model efficiency.                             |

---

## ⚡ Quick Start

The project is fully containerized for a fast, reproducible setup.

### Option 1: Docker (Recommended)

Requires Docker and Docker Compose.

```bash
# 1. Clone and navigate
git clone https://github.com/quantsingularity/MARL-for-Portfolio-Optimization-and-Risk-Diversification.git
cd MARL-for-Portfolio-Optimization-and-Risk-Diversification

# 2. Start the full production stack (API, Dashboard, Scheduler, Risk Monitor)
docker-compose --profile production up -d

# 3. Access Services
# - API Documentation (Swagger UI): http://localhost:8000/docs
# - Monitoring Dashboard (Plotly/Dash): http://localhost:8050
```

### Option 2: Local Setup

Requires Python 3.10+ and a virtual environment.

```bash
# 1. Run setup script to install dependencies
chmod +x setup.sh && ./setup.sh

# 2. Activate environment
source venv/bin/activate

# 3. Run a quick demo (5 episodes)
python code/main.py --mode demo

# 4. Train the MARL-Lite model (fast iteration)
python code/main.py --mode train --config configs/marl_lite.json
```

---

## 📊 Model Configurations and Performance

The system offers two primary configurations, allowing users to balance performance with training speed and resource consumption.

### Table 1: Model Configuration Comparison

| Configuration | Architecture                 | Key Features Used                                                    | Resource Usage            | Primary Use Case                                                          |
| :------------ | :--------------------------- | :------------------------------------------------------------------- | :------------------------ | :------------------------------------------------------------------------ |
| **MARL-Full** | Transformer (Attention)      | All features (ESG, Sentiment, Full T.A.)                             | High (2.8 GB RAM, 45s/ep) | Maximum performance and research.                                         |
| **MARL-Lite** | MLP (Multi-Layer Perceptron) | Top 5 features only (Historical Returns, Volatility, RSI, MACD, VIX) | Low (980 MB RAM, 16s/ep)  | Fast iteration, rapid prototyping, and resource-constrained environments. |

### Table 2: Performance Benchmarks

Benchmarks compare the two models against a simple Equal-Weight (EW) baseline.

| Metric           | MARL-Full     | MARL-Lite         | Equal-Weight | Improvement (Full vs. EW) |
| :--------------- | :------------ | :---------------- | :----------- | :------------------------ |
| **Sharpe Ratio** | **1.68**      | 1.34              | 0.93         | **+80.6%**                |
| Total Return     | 21.2%         | 18.5%             | 12.1%        | +75.2%                    |
| Max Drawdown     | -8.5%         | -10.1%            | -15.5%       | -45.2%                    |
| Training Time    | 45.2s/episode | **15.8s/episode** | N/A          | -65% (Lite vs. Full)      |
| Memory Usage     | 2.8 GB        | **980 MB**        | N/A          | -65% (Lite vs. Full)      |

> **Conclusion:** MARL-Lite achieves **80% of the full model's Sharpe Ratio** with a **65% reduction in training time and memory**, making it the recommended choice for initial development.

---

## 🛠️ Project Architecture

The repository is structured to separate core logic, configuration, and deployment components.

### Table 3: Core Directory Structure

| Directory               | Purpose                | Key Files/Modules                                            | Description                                                                              |
| :---------------------- | :--------------------- | :----------------------------------------------------------- | :--------------------------------------------------------------------------------------- |
| `code/`                 | Core Application Logic | `main.py`, `maddpg_agent.py`, `environment.py`               | Contains the training loop, MADDPG implementation, and the custom portfolio environment. |
| `code/models/`          | Neural Network Models  | `transformer_actor.py`, `regime_detector.py`                 | Defines the Actor and Critic networks, including the advanced Transformer architecture.  |
| `code/risk_management/` | Risk & Metrics         | `dynamic_diversity.py`, `risk_metrics.py`                    | Implements dynamic risk controls and standard financial risk metrics (VaR, CVaR).        |
| `code/features/`        | Data Integration       | `esg_provider.py`, `sentiment_analyzer.py`                   | Modules for integrating external data sources like ESG scores and FinBERT sentiment.     |
| `code/api/`             | Production API         | `main.py` (FastAPI)                                          | Defines the RESTful endpoints for model serving and rebalancing requests.                |
| `code/production/`      | Automation Services    | `scheduler.py`, `risk_monitor.py`                            | Background services for automated rebalancing and real-time risk alerting.               |
| `configs/`              | Configuration Files    | `default.json`, `marl_lite.json`, `transformer.json`         | JSON files to configure the environment, network, and training parameters.               |
| `tests/`                | Test Suite             | `test_comprehensive.py`, `test_risk_metrics.py`              | Pytest suite ensuring 82%+ code coverage for reliability.                                |
| `notebooks/`            | Exploratory Analysis   | `02_feature_analysis.ipynb`, `03_model_interpretation.ipynb` | Jupyter notebooks for in-depth data and model analysis.                                  |

---

## 🔬 Analysis and Interpretation Tools

The system includes dedicated scripts for deep analysis of the model and market conditions.

### Table 4: Analysis Scripts

| Script                       | Location                                    | Purpose                                                                                         | Output                              |
| :--------------------------- | :------------------------------------------ | :---------------------------------------------------------------------------------------------- | :---------------------------------- |
| **Feature Importance**       | `code/analysis/feature_importance.py`       | Runs an ablation study to rank features by their impact on Sharpe Ratio.                        | `results/feature_analysis/`         |
| **Rebalancing Optimization** | `code/analysis/rebalancing_optimization.py` | Analyzes optimal rebalancing frequency (daily, weekly, etc.) against various transaction costs. | `results/rebalancing_analysis/`     |
| **Performance Benchmarks**   | `code/benchmarks/run_benchmarks.py`         | Compares different model configurations (Full, Lite, Baselines) on key metrics.                 | `results/benchmarks/`               |
| **Model Explainer**          | `code/interpretability/explainer.py`        | Provides a simple feature attribution map and textual rationale for a single decision.          | Used by the API for explainability. |

---

## 🐳 Deployment and Automation

The project is designed for continuous operation using Docker and a set of utility commands.

### Table 5: Docker Compose Profiles

The `docker-compose.yml` file uses profiles for modular deployment.

| Profile      | Services Included                               | Description                                                | Access Points                |
| :----------- | :---------------------------------------------- | :--------------------------------------------------------- | :--------------------------- |
| `production` | `api`, `dashboard`, `scheduler`, `risk-monitor` | Starts the full, persistent production stack.              | API (8000), Dashboard (8050) |
| `train-cpu`  | `trainer`                                       | Trains the full model using CPU resources.                 | Logs                         |
| `train-gpu`  | `trainer`                                       | Trains the full model using GPU (requires NVIDIA runtime). | Logs                         |
| `train-lite` | `trainer`                                       | Trains the faster MARL-Lite model.                         | Logs                         |
| `test`       | `tester`                                        | Runs the full Pytest suite within a clean container.       | Logs                         |

### Table 6: Makefile Utility Commands

The `Makefile` provides convenient shortcuts for common development and deployment tasks.

| Command           | Action                                  | Equivalent Docker Profile |
| :---------------- | :-------------------------------------- | :------------------------ |
| `make install`    | Installs all local dependencies.        | N/A                       |
| `make test`       | Runs the full test suite with coverage. | `test`                    |
| `make train`      | Trains the full MARL model.             | `train-cpu`               |
| `make train-lite` | Trains the MARL-Lite model.             | `train-lite`              |
| `make benchmark`  | Runs the performance benchmark script.  | `benchmark`               |
| `make docker-up`  | Starts the full production stack.       | `production`              |
| `make api`        | Starts the FastAPI server locally.      | N/A                       |

---

## 📚 Documentation and Support

| Resource              | Location                     | Description                                                                      |
| :-------------------- | :--------------------------- | :------------------------------------------------------------------------------- |
| **Quick Start Guide** | `docs/QUICKSTART.md`         | A detailed, 5-minute guide to getting the system running locally or with Docker. |
| **API Documentation** | `http://localhost:8000/docs` | Interactive Swagger UI for all production API endpoints (when running).          |
| **Jupyter Notebooks** | `notebooks/`                 | In-depth, visual analysis of data, features, and model interpretation.           |
| **Test Suite**        | `tests/`                     | Comprehensive test cases for baselines, risk metrics, and model components.      |

## 📄 License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.
