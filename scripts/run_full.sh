#!/bin/bash
# Full run: trains MADDPG for more episodes on synthetic market data.
set -e
cd "$(dirname "$0")/.."
echo "=========================================="
echo "MARL Portfolio Optimization - Full Run"
echo "=========================================="
export PYTHONPATH="$(pwd)/portfolio:${PYTHONPATH}"
python portfolio/main.py --mode train --data-source synthetic --episodes 300 --seed 42
echo ""
echo "Full run complete. Results in portfolio/results/"
