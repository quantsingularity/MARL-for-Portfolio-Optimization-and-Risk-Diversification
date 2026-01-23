#!/bin/bash
# Production Setup Script for MARL Portfolio Optimization

set -e

echo "=========================================="
echo "MARL Portfolio Optimization - Production Setup"
echo "=========================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check Python version
print_info "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
print_info "Python version: $python_version"

# Create virtual environment
print_info "Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
print_info "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
print_info "Installing core dependencies..."
pip install -r requirements.txt

print_info "Installing production dependencies..."
pip install -r requirements-prod.txt

# Install package in editable mode
print_info "Installing package..."
pip install -e .

# Create necessary directories
print_info "Creating project directories..."
mkdir -p data logs models results
mkdir -p results/{training,benchmarks,feature_analysis,rebalancing_analysis}

# Check if Docker is installed
if command -v docker &> /dev/null; then
    print_info "Docker found: $(docker --version)"
    
    # Check if Docker Compose is installed
    if command -v docker-compose &> /dev/null; then
        print_info "Docker Compose found: $(docker-compose --version)"
    else
        print_warn "Docker Compose not found. Install it for container orchestration."
    fi
else
    print_warn "Docker not found. Install Docker for containerized deployment."
fi

# Run tests
print_info "Running tests..."
pytest tests/test_comprehensive.py::TestConfig -v || print_warn "Some tests failed"

# Generate sample configs if they don't exist
if [ ! -f "configs/default.json" ]; then
    print_info "Generating default configuration..."
    python -c "from code.config import Config; Config().save('configs/default.json')"
fi

echo ""
echo "=========================================="
echo -e "${GREEN}Setup completed successfully!${NC}"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Activate virtual environment: source venv/bin/activate"
echo "  2. Run quick demo: python code/main.py --mode demo"
echo "  3. Or train model: python code/main.py --mode train --config configs/marl_lite.json"
echo "  4. Or use Docker: docker-compose --profile train-cpu up"
echo ""
echo "For more information, see:"
echo "  - QUICKSTART.md - Quick start guide"
echo "  - README_PRODUCTION.md - Full documentation"
echo "  - DEPLOYMENT_CHECKLIST.md - Deployment guide"
echo ""
echo "Run 'make help' to see available commands"
echo ""
