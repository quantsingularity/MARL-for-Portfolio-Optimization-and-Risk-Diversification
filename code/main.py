"""
Main training and evaluation script
"""

import os
import json
import argparse
import numpy as np
import torch
from datetime import datetime

from config import Config
from data_loader import MarketDataLoader
from environment import EnhancedMultiAgentPortfolioEnv
from maddpg_agent import MADDPGTrainer
from baselines import evaluate_all_baselines


def set_seed(seed: int):
    """Set random seeds for reproducibility"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def train(config: Config, save_dir: str):
    """Train MADDPG agents"""
    print("=" * 80)
    print("MADDPG Portfolio Optimization - Training")
    print("=" * 80)

    # Set seed
    set_seed(config.seed)

    # Load data
    print("\nLoading market data...")
    loader = MarketDataLoader(config)
    data = loader.prepare_environment_data()

    # Create environment
    print("Creating environment...")
    env = EnhancedMultiAgentPortfolioEnv(config, data)

    # Create trainer
    print("Initializing MADDPG trainer...")
    trainer = MADDPGTrainer(env, config)

    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(f"{save_dir}/checkpoints", exist_ok=True)
    os.makedirs(f"{save_dir}/best_model", exist_ok=True)
    os.makedirs(f"{save_dir}/final_model", exist_ok=True)

    # Save configuration
    config.save(f"{save_dir}/config.json")

    # Training loop
    print(f"\nTraining for {config.training.n_episodes} episodes...")
    print("-" * 80)

    best_sharpe = -np.inf
    training_history = []

    for episode in range(config.training.n_episodes):
        # Train episode
        episode_result = trainer.train_episode()

        # Extract metrics
        episode_reward = episode_result["episode_reward"]
        metrics = episode_result["metrics"]
        agg_metrics = metrics["aggregate"]

        # Store history
        training_history.append(
            {
                "episode": episode,
                "reward": float(np.mean(episode_reward)),
                "sharpe_ratio": float(agg_metrics["sharpe_ratio"]),
                "total_return": float(agg_metrics["total_return"]),
                "max_drawdown": float(agg_metrics["max_drawdown"]),
                "final_capital": float(agg_metrics["final_capital"]),
            }
        )

        # Print progress
        if (episode + 1) % 10 == 0:
            print(f"Episode {episode+1}/{config.training.n_episodes}")
            print(f"  Reward: {np.mean(episode_reward):.4f}")
            print(f"  Sharpe: {agg_metrics['sharpe_ratio']:.4f}")
            print(f"  Return: {agg_metrics['total_return']*100:.2f}%")
            print(f"  Max DD: {agg_metrics['max_drawdown']*100:.2f}%")
            print(f"  Noise: {trainer.agents[0].noise_std:.4f}")

            if "diversity" in metrics:
                print(
                    f"  Avg Correlation: {metrics['diversity']['avg_correlation']:.4f}"
                )
            print()

        # Save best model
        if agg_metrics["sharpe_ratio"] > best_sharpe:
            best_sharpe = agg_metrics["sharpe_ratio"]
            for agent in trainer.agents:
                agent.save(f"{save_dir}/best_model")
            print(f"  *** New best Sharpe ratio: {best_sharpe:.4f} ***")

        # Save checkpoint
        if (episode + 1) % config.training.save_interval == 0:
            checkpoint_dir = f"{save_dir}/checkpoints/episode_{episode+1}"
            os.makedirs(checkpoint_dir, exist_ok=True)
            for agent in trainer.agents:
                agent.save(checkpoint_dir)

    # Save final model
    for agent in trainer.agents:
        agent.save(f"{save_dir}/final_model")

    # Save training history
    with open(f"{save_dir}/training_history.json", "w") as f:
        json.dump(training_history, f, indent=2)

    print("\n" + "=" * 80)
    print("Training completed!")
    print(f"Best Sharpe ratio: {best_sharpe:.4f}")
    print(f"Results saved to: {save_dir}")
    print("=" * 80)

    return trainer, training_history


def evaluate(config: Config, model_path: str, save_dir: str):
    """Evaluate trained MADDPG agents"""
    print("=" * 80)
    print("MADDPG Portfolio Optimization - Evaluation")
    print("=" * 80)

    # Set seed
    set_seed(config.seed)

    # Load data
    print("\nLoading market data...")
    loader = MarketDataLoader(config)
    data = loader.prepare_environment_data()

    # Create environment
    print("Creating environment...")
    env = EnhancedMultiAgentPortfolioEnv(config, data)

    # Create trainer and load model
    print(f"Loading model from {model_path}...")
    trainer = MADDPGTrainer(env, config)
    for agent in trainer.agents:
        agent.load(model_path)

    # Evaluate on test set
    print("\nEvaluating MADDPG on test set...")
    test_start, test_end = data["test_indices"]
    env.reset(start_idx=test_start, end_idx=test_end)

    states = env.reset(start_idx=test_start, end_idx=test_end)

    while not env.done:
        actions = [
            agent.select_action(states[i], add_noise=False)
            for i, agent in enumerate(trainer.agents)
        ]
        states, rewards, done, info = env.step(actions)

    maddpg_metrics = env.get_episode_metrics()

    # Evaluate baselines
    baseline_results = evaluate_all_baselines(env, config)

    # Combine results
    all_results = {"MADDPG": maddpg_metrics, **baseline_results}

    # Save results
    os.makedirs(save_dir, exist_ok=True)
    with open(f"{save_dir}/test_results.json", "w") as f:
        json.dump(
            {
                k: {
                    "aggregate": {
                        key: float(val) for key, val in v["aggregate"].items()
                    }
                }
                for k, v in all_results.items()
            },
            f,
            indent=2,
        )

    # Print comparison
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)
    print(f"\n{'Strategy':<20} {'Sharpe':<10} {'Return':<12} {'Max DD':<12}")
    print("-" * 80)

    for name, metrics in all_results.items():
        agg = metrics["aggregate"]
        print(
            f"{name:<20} {agg['sharpe_ratio']:>9.3f} "
            f"{agg['total_return']*100:>10.2f}% "
            f"{agg['max_drawdown']*100:>10.2f}%"
        )

    print("=" * 80)

    # Calculate improvements
    maddpg_sharpe = maddpg_metrics["aggregate"]["sharpe_ratio"]
    ew_sharpe = baseline_results["Equal-Weight"]["aggregate"]["sharpe_ratio"]
    improvement = (maddpg_sharpe - ew_sharpe) / ew_sharpe * 100

    print(f"\nMADDPG vs Equal-Weight: {improvement:+.1f}% Sharpe improvement")

    return all_results


def demo(config: Config, save_dir: str):
    """Quick demo with 5 episodes"""
    print("=" * 80)
    print("MADDPG Portfolio Optimization - Quick Demo")
    print("=" * 80)

    config.training.n_episodes = 5
    config.data.data_source = "synthetic"

    trainer, history = train(config, save_dir)

    print("\nDemo completed! For full training, use --mode train")

    return trainer, history


def main():
    parser = argparse.ArgumentParser(description="MADDPG Portfolio Optimization")
    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        choices=["train", "eval", "demo"],
        help="Mode: train new model, evaluate existing, or run demo",
    )
    parser.add_argument(
        "--episodes", type=int, default=300, help="Number of training episodes"
    )
    parser.add_argument(
        "--data-source",
        type=str,
        default="synthetic",
        choices=["yfinance", "synthetic", "csv"],
        help="Data source selection",
    )
    parser.add_argument(
        "--save-dir", type=str, default="./results", help="Directory to save results"
    )
    parser.add_argument(
        "--load-model", type=str, default=None, help="Path to load pre-trained model"
    )
    parser.add_argument(
        "--config", type=str, default=None, help="Path to custom configuration JSON"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    # Load configuration
    if args.config:
        config = Config.load(args.config)
    else:
        config = Config()

    # Override with command line arguments
    config.training.n_episodes = args.episodes
    config.data.data_source = args.data_source
    config.seed = args.seed

    # Create save directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"{args.save_dir}/{timestamp}"

    # Execute mode
    if args.mode == "train":
        train(config, save_dir)
    elif args.mode == "eval":
        if args.load_model is None:
            print("Error: --load-model must be specified for evaluation mode")
            return
        evaluate(config, args.load_model, save_dir)
    elif args.mode == "demo":
        demo(config, save_dir)


if __name__ == "__main__":
    main()
