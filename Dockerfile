# Multi-stage Dockerfile for MARL Portfolio Optimization
# Supports both CPU and GPU execution

ARG CUDA_VERSION=11.8.0
ARG PYTHON_VERSION=3.10

# Base stage
FROM nvidia/cuda:${CUDA_VERSION}-cudnn8-runtime-ubuntu22.04 as base

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install Python and system dependencies
RUN apt-get update && apt-get install -y \
    python${PYTHON_VERSION} \
    python3-pip \
    git \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic link for python
RUN ln -s /usr/bin/python${PYTHON_VERSION} /usr/bin/python

# Set working directory
WORKDIR /app

# CPU-only stage
FROM base as cpu

# Copy requirements
COPY requirements.txt requirements-prod.txt ./

# Install Python dependencies (CPU version)
RUN pip install --no-cache-dir -r requirements.txt -r requirements-prod.txt
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# GPU stage
FROM base as gpu

# Copy requirements
COPY requirements.txt requirements-prod.txt ./

# Install Python dependencies (GPU version)
RUN pip install --no-cache-dir -r requirements.txt -r requirements-prod.txt
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Final stage
FROM ${COMPUTE_TYPE:-cpu} as final

# Copy application code
COPY . /app

# Install the package
RUN pip install -e .

# Create necessary directories
RUN mkdir -p /app/data /app/logs /app/models /app/results

# Set environment variables
ENV PYTHONPATH=/app:$PYTHONPATH
ENV DATA_DIR=/app/data
ENV MODELS_DIR=/app/models
ENV LOGS_DIR=/app/logs
ENV RESULTS_DIR=/app/results

# Expose ports for API and dashboard
EXPOSE 8000 8050

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import sys; sys.exit(0)"

# Default command
CMD ["python", "portfolio/main.py", "--help"]
