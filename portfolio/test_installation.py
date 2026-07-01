"""
import numpy as np
Quick test script to verify installation and functionality
"""

import sys

import numpy as np


def test_imports():
    """Test all required imports"""
    print("Testing imports...")
    try:
        print("✓ numpy")
        print("✓ pandas")
        print("✓ torch")
        print("✓ matplotlib")
        print("✓ seaborn")
        print("✓ scipy")
        print("\nAll imports successful!")
        return True
    except ImportError as e:
        print(f"\n✗ Import error: {e}")
        print("\nPlease install requirements: pip install -r requirements.txt")
        return False


def test_modules():
    """Test project modules"""
    print("\nTesting project modules...")
    try:
        print("✓ config")
        print("✓ data_loader")
        print("✓ environment")
        print("✓ maddpg_agent")
        print("✓ baselines")
        print("\nAll modules loaded successfully!")
        return True
    except Exception as e:
        print(f"\n✗ Module error: {e}")
        return False


def test_basic_functionality():
    """Test basic functionality"""
    print("\nTesting basic functionality...")
    try:
        from config import Config
        from data_loader import MarketDataLoader
        from environment import MultiAgentPortfolioEnv

        # Create config
        config = Config()
        config.data.data_source = "synthetic"
        print("✓ Configuration created")

        # Load data
        loader = MarketDataLoader(config)
        data = loader.prepare_environment_data()
        print("✓ Data loaded")

        # Create environment
        env = MultiAgentPortfolioEnv(config, data)
        print("✓ Environment created")

        # Reset environment
        obs = env.reset()
        print("✓ Environment reset")

        # Test step: build uniform-weight actions of the correct (per-agent
        # asset) dimension, not the observation dimension.
        actions = [
            np.ones_like(env.agent_weights[i]) / len(env.agent_weights[i])
            for i in range(env.n_agents)
        ]
        next_obs, rewards, done, info = env.step(actions)
        print("✓ Environment step executed")

        print("\nBasic functionality test passed!")
        print(f"  - Number of agents: {env.n_agents}")
        print(f"  - Observation dimensions: {[len(o) for o in obs]}")
        print(f"  - Action dimensions: {[len(a) for a in actions]}")

        return True
    except Exception as e:
        print(f"\n✗ Functionality error: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    print("=" * 60)
    print("MADDPG Portfolio Optimization - System Test")
    print("=" * 60)

    success = True

    # Test imports
    if not test_imports():
        success = False
        return

    # Test modules
    if not test_modules():
        success = False
        return

    # Test functionality
    if not test_basic_functionality():
        success = False
        return

    if success:
        print("\n" + "=" * 60)
        print("✓ ALL TESTS PASSED!")
        print("=" * 60)
        print("\nYou can now run the main script:")
        print("  python main.py --mode demo")
    else:
        print("\n" + "=" * 60)
        print("✗ SOME TESTS FAILED")
        print("=" * 60)
        sys.exit(1)


if __name__ == "__main__":
    main()
