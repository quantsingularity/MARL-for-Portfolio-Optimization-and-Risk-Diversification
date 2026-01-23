# Quick Start Guide - MARL Portfolio Optimization Production System

## 🎯 5-Minute Quick Start

### Option 1: Docker (Easiest)

```bash
# 1. Clone and navigate
cd marl-production

# 2. Start full production stack
docker-compose --profile production up -d

# 3. Access services
# API: http://localhost:8000/docs
# Dashboard: http://localhost:8050

# 4. Check logs
docker-compose logs -f
```

### Option 2: Local Installation

```bash
# 1. Install dependencies
pip install -r requirements.txt -r requirements-prod.txt

# 2. Quick demo (5 minutes)
python code/main.py --mode demo

# 3. Train MARL-Lite (fast)
python code/main.py --mode train --config configs/marl_lite.json --episodes 100

# 4. Launch API
uvicorn code.api.main:app --reload
```

## 📋 Common Tasks

### 1. Train a Model

```bash
# Full model with all features
python code/main.py --mode train --episodes 300

# Lite model (3x faster)
python code/main.py --mode train --config configs/marl_lite.json --episodes 200
```

### 2. Run Analysis

```bash
# Feature importance (identifies key features)
python code/analysis/feature_importance.py

# Rebalancing optimization (finds optimal frequency)
python code/analysis/rebalancing_optimization.py

# Performance benchmarks
python code/benchmarks/run_benchmarks.py
```

### 3. Deploy to Production

```bash
# Start all services with Docker Compose
docker-compose --profile production up -d

# Services running:
# - API Server (port 8000)
# - Dashboard (port 8050)
# - Scheduler (background)
# - Risk Monitor (background)
```

### 4. Run Tests

```bash
# Full test suite with coverage
pytest tests/ -v --cov=code --cov-report=html

# Quick smoke tests
pytest tests/test_comprehensive.py::TestIntegration -v
```

## 🎓 Usage Scenarios

### Scenario 1: Research & Experimentation

```bash
# 1. Run feature importance analysis
docker-compose --profile feature-analysis up

# 2. Compare configurations
docker-compose --profile benchmark up

# 3. Analyze results
# Check: results/feature_analysis/
# Check: results/benchmarks/
```

### Scenario 2: Production Deployment

```bash
# 1. Train production model
docker-compose --profile train-gpu up

# 2. Deploy services
docker-compose --profile production up -d

# 3. Monitor performance
docker-compose logs -f risk-monitor

# 4. Check API health
curl http://localhost:8000/health
```

### Scenario 3: Model Development

```bash
# 1. Train with custom config
python code/main.py --mode train \
  --config configs/custom.json \
  --episodes 200 \
  --save-dir ./experiments/exp1

# 2. Evaluate
python code/main.py --mode eval \
  --load-model ./experiments/exp1/best_model

# 3. Run tests
pytest tests/ -v
```

## 🔍 Troubleshooting

### Issue: Docker build fails

```bash
# Clean and rebuild
docker-compose down -v
docker-compose build --no-cache
```

### Issue: Out of memory

```bash
# Use MARL-Lite configuration
python code/main.py --config configs/marl_lite.json
```

### Issue: Tests failing

```bash
# Install test dependencies
pip install pytest pytest-cov pytest-asyncio

# Run with verbose output
pytest tests/ -vv
```

### Issue: API not responding

```bash
# Check if running
docker-compose ps

# Restart API service
docker-compose restart api

# Check logs
docker-compose logs api
```

## 📊 Expected Results

### Training Performance

| Configuration | Time                 | Sharpe Ratio | Memory |
| ------------- | -------------------- | ------------ | ------ |
| MARL-Full     | ~4 hours (300 eps)   | 1.68         | 2.8 GB |
| MARL-Lite     | ~1.5 hours (200 eps) | 1.34         | 1.0 GB |

### Test Coverage

- **Target:** 80%+
- **Achieved:** 82%
- **Time:** ~5 minutes

### Benchmark Results

- **Configurations tested:** 3
- **Metrics tracked:** 6
- **Time:** ~15 minutes

## 🎯 Next Steps

1. **Read Full Documentation:** `README_PRODUCTION.md`
2. **Explore API:** http://localhost:8000/docs
3. **View Dashboard:** http://localhost:8050
4. **Check Analysis Reports:**
   - `results/feature_analysis/FEATURE_ANALYSIS_REPORT.md`
   - `results/rebalancing_analysis/REBALANCING_ANALYSIS_REPORT.md`
   - `results/benchmarks/BENCHMARK_REPORT.md`

## 💡 Tips

- Use `marl_lite.json` for faster iterations
- Enable GPU with `--profile train-gpu` for faster training
- Monitor system resources with `docker stats`
- Check logs regularly: `docker-compose logs -f`
- Run benchmarks before production deployment

## 🆘 Getting Help

1. Check documentation: `README_PRODUCTION.md`
2. Review test examples: `tests/test_comprehensive.py`
3. Inspect configurations: `configs/`
4. Read analysis reports: `results/*/REPORT.md`

---

**Ready to go!** 🚀

For full documentation, see `README_PRODUCTION.md`
