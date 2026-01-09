"""
Main Training Script for MADDPG Portfolio Optimization
Complete implementation as per research paper
"""

import os
import sys
import argparse
import json
import numpy as np
import pandas as pd
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Import modules
from config import Config, default_config
from data_loader import MarketDataLoader
from enhanced_environment import EnhancedMultiAgentPortfolioEnv
from maddpg_agent import MADDPG
from generate_figures import *


def train_maddpg(env, maddpg, config, save_dir='./results'):
    """
    Train MADDPG agents with full tracking and checkpointing
    """
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(f"{save_dir}/checkpoints", exist_ok=True)
    
    episode_rewards = []
    episode_metrics = []
    best_sharpe = -np.inf
    
    print(f"\n{'='*60}")
    print(f"Starting Training: {config.training.n_episodes} episodes")
    print(f"{'='*60}\n")
    
    for episode in tqdm(range(config.training.n_episodes), desc="Training"):
        states, _ = env.reset()
        episode_reward = np.zeros(env.n_agents)
        done = False
        step_count = 0
        
        while not done:
            # Select actions
            actions = maddpg.select_actions(states, explore=True)
            
            # Take step
            next_states, rewards, terminated, truncated, info = env.step(actions)
            done = terminated or truncated
            
            # Store experience
            maddpg.replay_buffer.push(
                states, actions, rewards, next_states, [done] * env.n_agents
            )
            
            # Update agents
            if len(maddpg.replay_buffer) >= config.training.min_buffer_size:
                losses = maddpg.update(config.training.batch_size)
            
            episode_reward += np.array(rewards)
            states = next_states
            step_count += 1
        
        # Get metrics
        metrics = env.get_portfolio_metrics()
        episode_rewards.append(episode_reward.tolist())
        episode_metrics.append(metrics)
        
        # Save best model
        if metrics['sharpe_ratio'] > best_sharpe:
            best_sharpe = metrics['sharpe_ratio']
            maddpg.save(f"{save_dir}/best_model")
            print(f"\n✓ New best model saved! Sharpe: {best_sharpe:.4f}")
        
        # Periodic logging
        if (episode + 1) % config.training.eval_interval == 0:
            recent_metrics = episode_metrics[-config.training.eval_interval:]
            avg_sharpe = np.mean([m['sharpe_ratio'] for m in recent_metrics])
            avg_return = np.mean([m['annualized_return'] for m in recent_metrics])
            avg_mdd = np.mean([m['max_drawdown'] for m in recent_metrics])
            avg_corr = np.mean([m['avg_correlation'] for m in recent_metrics])
            
            print(f"\n{'='*60}")
            print(f"Episode {episode + 1}/{config.training.n_episodes}")
            print(f"{'-'*60}")
            print(f"  Sharpe Ratio:      {avg_sharpe:.4f}")
            print(f"  Ann. Return:       {avg_return:.4f}")
            print(f"  Max Drawdown:      {avg_mdd:.4f}")
            print(f"  Avg Correlation:   {avg_corr:.4f}")
            print(f"  Best Sharpe:       {best_sharpe:.4f}")
            print(f"{'='*60}\n")
        
        # Save checkpoint
        if (episode + 1) % config.training.save_interval == 0:
            maddpg.save(f"{save_dir}/checkpoints/episode_{episode+1}")
    
    # Save final model
    maddpg.save(f"{save_dir}/final_model")
    
    # Save training history
    history = {
        'episode_rewards': episode_rewards,
        'episode_metrics': episode_metrics,
        'config': config.to_dict()
    }
    
    with open(f"{save_dir}/training_history.json", 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\n✓ Training completed! Results saved to {save_dir}")
    
    return episode_rewards, episode_metrics


def evaluate_baselines(env, maddpg, config, n_episodes=10):
    """
    Evaluate multiple baseline strategies
    """
    results = {}
    
    # 1. MADDPG (trained)
    print("\nEvaluating MADDPG...")
    maddpg_metrics = []
    for _ in range(n_episodes):
        states, _ = env.reset()
        done = False
        while not done:
            actions = maddpg.select_actions(states, explore=False)
            next_states, rewards, terminated, truncated, info = env.step(actions)
            done = terminated or truncated
            states = next_states
        metrics = env.get_portfolio_metrics()
        maddpg_metrics.append(metrics)
    
    results['MADDPG'] = aggregate_metrics(maddpg_metrics)
    
    # 2. Equal Weight
    print("Evaluating Equal Weight...")
    ew_metrics = []
    for _ in range(n_episodes):
        states, _ = env.reset()
        done = False
        while not done:
            # Equal weight for each agent's assets
            actions = []
            for agent_id in range(env.n_agents):
                n_assets = len(env.agent_asset_assignment[agent_id])
                action = np.ones(n_assets) / n_assets
                actions.append(action)
            next_states, rewards, terminated, truncated, info = env.step(actions)
            done = terminated or truncated
            states = next_states
        metrics = env.get_portfolio_metrics()
        ew_metrics.append(metrics)
    
    results['Equal_Weight'] = aggregate_metrics(ew_metrics)
    
    # 3. Random
    print("Evaluating Random...")
    random_metrics = []
    for _ in range(n_episodes):
        states, _ = env.reset()
        done = False
        while not done:
            actions = []
            for agent_id in range(env.n_agents):
                n_assets = len(env.agent_asset_assignment[agent_id])
                action = np.random.dirichlet(np.ones(n_assets))
                actions.append(action)
            next_states, rewards, terminated, truncated, info = env.step(actions)
            done = terminated or truncated
            states = next_states
        metrics = env.get_portfolio_metrics()
        random_metrics.append(metrics)
    
    results['Random'] = aggregate_metrics(random_metrics)
    
    return results


def aggregate_metrics(metrics_list):
    """Aggregate metrics across episodes"""
    agg = {}
    keys = metrics_list[0].keys()
    for key in keys:
        values = [m[key] for m in metrics_list]
        agg[f'avg_{key}'] = np.mean(values)
        agg[f'std_{key}'] = np.std(values)
    return agg


def generate_all_figures(env, maddpg, config, save_dir='./results/figures'):
    """Generate all publication figures"""
    os.makedirs(save_dir, exist_ok=True)
    
    print("\nGenerating figures...")
    
    # Load history
    with open(f"{save_dir}/../training_history.json", 'r') as f:
        history = json.load(f)
    
    # 1. Training curves
    plot_training_curves(f"{save_dir}/../training_history.json", 
                        f"{save_dir}/figure1_training_curves.png")
    
    # 2. Run final evaluation to get data
    states, _ = env.reset()
    done = False
    while not done:
        actions = maddpg.select_actions(states, explore=False)
        next_states, rewards, terminated, truncated, info = env.step(actions)
        done = terminated or truncated
        states = next_states
    
    # 3. Portfolio evolution
    plot_portfolio_evolution(env.capital_history, env.n_agents,
                            f"{save_dir}/figure2_portfolio_evolution.png")
    
    # 4. Position heatmap
    position_array = np.array([[pos for pos in positions] for positions in env.position_history])
    n_assets_per_agent = [len(env.agent_asset_assignment[i]) for i in range(env.n_agents)]
    plot_position_heatmap(position_array.tolist(), env.n_agents, max(n_assets_per_agent),
                         f"{save_dir}/figure3_position_heatmap.png")
    
    # 5. Drawdown analysis
    plot_drawdown_analysis(env.capital_history,
                          f"{save_dir}/figure4_drawdown_analysis.png")
    
    # 6. Diversification analysis
    plot_diversification_analysis(position_array.tolist(), env.n_agents, max(n_assets_per_agent),
                                 f"{save_dir}/figure5_diversification_analysis.png")
    
    # 7. Architecture diagram
    plot_architecture_diagram(f"{save_dir}/figure6_architecture_diagram.png")
    
    print(f"✓ All figures saved to {save_dir}")


def main():
    parser = argparse.ArgumentParser(description='MADDPG Portfolio Optimization')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'eval', 'demo'],
                       help='Mode: train, eval, or demo')
    parser.add_argument('--episodes', type=int, default=None, help='Number of training episodes')
    parser.add_argument('--data-source', type=str, default='synthetic', 
                       choices=['yfinance', 'synthetic', 'csv'], help='Data source')
    parser.add_argument('--save-dir', type=str, default='./results', help='Save directory')
    parser.add_argument('--load-model', type=str, default=None, help='Path to load model from')
    parser.add_argument('--config', type=str, default=None, help='Path to config JSON')
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config:
        config = Config.load(args.config)
    else:
        config = default_config
    
    # Override config with args
    if args.episodes:
        config.training.n_episodes = args.episodes
    if args.data_source:
        config.data.data_source = args.data_source
    
    print(f"\n{'='*60}")
    print(f"MADDPG Portfolio Optimization")
    print(f"{'='*60}")
    print(f"Mode: {args.mode}")
    print(f"Data Source: {config.data.data_source}")
    print(f"Agents: {config.env.n_agents}")
    print(f"Assets: {config.env.n_assets}")
    print(f"Diversity Weight: {config.env.diversity_weight}")
    print(f"{'='*60}\n")
    
    # Load data
    print("Loading data...")
    loader = MarketDataLoader(config)
    prices = loader.load_data()
    train_prices, test_prices = loader.split_train_test(prices)
    train_returns = train_prices.pct_change().fillna(0)
    test_returns = test_prices.pct_change().fillna(0)
    
    # Create environment
    print("Creating environment...")
    train_env = EnhancedMultiAgentPortfolioEnv(config, train_returns, mode='train')
    test_env = EnhancedMultiAgentPortfolioEnv(config, test_returns, mode='test')
    
    # Create MADDPG
    print("Creating MADDPG agents...")
    maddpg = MADDPG(
        n_agents=config.env.n_agents,
        state_dim=train_env.observation_space.shape[0],
        action_dim=train_env.action_space.shape[0],
        lr_actor=config.training.lr_actor,
        lr_critic=config.training.lr_critic,
        gamma=config.training.gamma,
        tau=config.training.tau,
        hidden_dim=config.network.actor_hidden_dims[0],
        buffer_capacity=config.training.buffer_capacity
    )
    
    # Load model if specified
    if args.load_model:
        print(f"Loading model from {args.load_model}...")
        maddpg.load(args.load_model)
    
    # Execute mode
    if args.mode == 'train':
        # Train
        episode_rewards, episode_metrics = train_maddpg(train_env, maddpg, config, args.save_dir)
        
        # Load best model
        maddpg.load(f"{args.save_dir}/best_model")
        
        # Evaluate on test set
        print("\nEvaluating on test set...")
        test_results = evaluate_baselines(test_env, maddpg, config, n_episodes=10)
        
        # Print results
        print(f"\n{'='*60}")
        print(f"TEST SET RESULTS")
        print(f"{'='*60}")
        for strategy, metrics in test_results.items():
            print(f"\n{strategy}:")
            print(f"  Ann. Return:  {metrics['avg_annualized_return']:.4f} ± {metrics['std_annualized_return']:.4f}")
            print(f"  Sharpe Ratio: {metrics['avg_sharpe_ratio']:.4f} ± {metrics['std_sharpe_ratio']:.4f}")
            print(f"  Max Drawdown: {metrics['avg_max_drawdown']:.4f} ± {metrics['std_max_drawdown']:.4f}")
            print(f"  Volatility:   {metrics['avg_volatility']:.4f} ± {metrics['std_volatility']:.4f}")
        
        # Save results
        with open(f"{args.save_dir}/test_results.json", 'w') as f:
            json.dump(test_results, f, indent=2)
        
        # Generate figures
        generate_all_figures(test_env, maddpg, config, f"{args.save_dir}/figures")
        
    elif args.mode == 'eval':
        # Evaluation only
        print("\nEvaluating...")
        results = evaluate_baselines(test_env, maddpg, config, n_episodes=config.evaluation.n_eval_episodes)
        
        print(f"\n{'='*60}")
        print(f"EVALUATION RESULTS")
        print(f"{'='*60}")
        for strategy, metrics in results.items():
            print(f"\n{strategy}:")
            for key, value in metrics.items():
                print(f"  {key}: {value:.4f}")
        
    elif args.mode == 'demo':
        # Quick demo with synthetic data
        print("\nRunning demo (5 episodes, synthetic data)...")
        config.training.n_episodes = 5
        config.data.data_source = 'synthetic'
        
        episode_rewards, episode_metrics = train_maddpg(train_env, maddpg, config, args.save_dir)
        print("\n✓ Demo completed!")
    
    print(f"\n{'='*60}")
    print(f"Completed successfully!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
