"""Setup script for MARL Portfolio Optimization"""

from setuptools import setup, find_packages

setup(
    name="marl-portfolio",
    version="1.0.0",
    description="Multi-Agent Reinforcement Learning for Portfolio Optimization",
    author="Abrar Ahmed",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "torch>=1.10.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "scipy>=1.7.0",
        "yfinance>=0.1.70",
        "tqdm>=4.62.0",
        "scikit-learn>=1.0.0",
        "transformers>=4.30.0",
        "optuna>=3.0.0",
        "tensorboard>=2.13.0",
        "plotly>=5.14.0",
        "dash>=2.11.0",
    ],
    python_requires=">=3.8",
)
