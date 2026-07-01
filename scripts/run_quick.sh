#!/bin/bash
# Quick run: trains MADDPG on synthetic market data (no network required).
set -e
cd "$(dirname "$0")/.."
echo "=========================================="
echo "MARL Portfolio Optimization - Quick Run"
echo "=========================================="
export PYTHONPATH="$(pwd)/portfolio:${PYTHONPATH}"
python portfolio/main.py --mode train --data-source synthetic --episodes 20 --seed 42
echo ""
echo "Quick run complete. Results in portfolio/results/"
