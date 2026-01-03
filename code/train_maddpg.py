"""
Training script for Multi-Agent Portfolio Optimization
"""

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import json
import os

from marl_portfolio_env import MultiAgentPortfolioEnv
from maddpg_agent import MADDPG


def generate_synthetic_market_data(
    n_assets: int = 10,
    n_days: int = 1000,
    seed: int = 42
) -> pd.DataFrame:
    """
    Generate synthetic market data with realistic properties.
    
    Args:
        n_assets: Number of assets
        n_days: Number of trading days
        seed: Random seed
        
    Returns:
        DataFrame with date and asset price columns
    """
    np.random.seed(seed)
    
    # Generate correlated asset returns
    # Create correlation matrix
    base_corr = 0.3
    correlation_matrix = np.ones((n_assets, n_assets)) * base_corr
    np.fill_diagonal(correlation_matrix, 1.0)
    
    # Generate returns with correlation
    mean_returns = np.random.uniform(0.0001, 0.001, n_assets)
    volatilities = np.random.uniform(0.01, 0.03, n_assets)
    
    # Cholesky decomposition for correlated returns
    L = np.linalg.cholesky(correlation_matrix)
    
    returns = []
    for _ in range(n_days):
        uncorrelated = np.random.randn(n_assets)
        correlated = L @ uncorrelated
        daily_returns = mean_returns + volatilities * correlated
        returns.append(daily_returns)
    
    returns = np.array(returns)
    
    # Convert returns to prices
    initial_prices = np.ones(n_assets) * 100
    prices = initial_prices * np.exp(np.cumsum(returns, axis=0))
    
    # Create DataFrame
    dates = pd.date_range(start='2020-01-01', periods=n_days, freq='D')
    df = pd.DataFrame(prices, columns=[f'asset_{i}' for i in range(n_assets)])
    df.insert(0, 'date', dates)
    
    # Normalize prices to returns for the environment
    for col in df.columns[1:]:
        df[col] = df[col].pct_change().fillna(0)
    
    return df


def train_maddpg(
    env: MultiAgentPortfolioEnv,
    maddpg: MADDPG,
    n_episodes: int = 500,
    batch_size: int = 64,
    save_dir: str = './models'
):
    """
    Train MADDPG agents.
    
    Args:
        env: Multi-agent portfolio environment
        maddpg: MADDPG instance
        n_episodes: Number of training episodes
        batch_size: Batch size for updates
        save_dir: Directory to save models
    """
    os.makedirs(save_dir, exist_ok=True)
    
    episode_rewards = []
    episode_metrics = []
    
    best_sharpe = -np.inf
    
    print(f"Starting training for {n_episodes} episodes...")
    
    for episode in tqdm(range(n_episodes)):
        states, _ = env.reset()
        episode_reward = np.zeros(env.n_agents)
        done = False
        
        while not done:
            # Select actions
            actions = maddpg.select_actions(states, explore=True)
            
            # Take step in environment
            next_states, rewards, terminated, truncated, info = env.step(actions)
            done = terminated or truncated
            
            # Store in replay buffer
            maddpg.replay_buffer.push(
                states,
                actions,
                rewards,
                next_states,
                [done] * env.n_agents
            )
            
            # Update agents
            if len(maddpg.replay_buffer) >= batch_size:
                losses = maddpg.update(batch_size)
            
            episode_reward += np.array(rewards)
            states = next_states
        
        # Get portfolio metrics
        metrics = env.get_portfolio_metrics()
        episode_rewards.append(episode_reward)
        episode_metrics.append(metrics)
        
        # Save best model based on Sharpe ratio
        if metrics['sharpe_ratio'] > best_sharpe:
            best_sharpe = metrics['sharpe_ratio']
            maddpg.save(f"{save_dir}/best_model")
        
        # Logging
        if (episode + 1) % 20 == 0:
            avg_reward = np.mean([np.sum(r) for r in episode_rewards[-20:]])
            avg_sharpe = np.mean([m['sharpe_ratio'] for m in episode_metrics[-20:]])
            avg_return = np.mean([m['annualized_return'] for m in episode_metrics[-20:]])
            
            print(f"\nEpisode {episode + 1}/{n_episodes}")
            print(f"  Avg Reward: {avg_reward:.4f}")
            print(f"  Avg Sharpe: {avg_sharpe:.4f}")
            print(f"  Avg Return: {avg_return:.4f}")
            print(f"  Best Sharpe: {best_sharpe:.4f}")
    
    # Save final model
    maddpg.save(f"{save_dir}/final_model")
    
    # Save training history
    history = {
        'episode_rewards': [r.tolist() for r in episode_rewards],
        'episode_metrics': episode_metrics
    }
    
    with open(f"{save_dir}/training_history.json", 'w') as f:
        json.dump(history, f, indent=2)
    
    return episode_rewards, episode_metrics


def evaluate_maddpg(
    env: MultiAgentPortfolioEnv,
    maddpg: MADDPG,
    n_episodes: int = 10
):
    """
    Evaluate trained MADDPG agents.
    
    Args:
        env: Multi-agent portfolio environment
        maddpg: Trained MADDPG instance
        n_episodes: Number of evaluation episodes
        
    Returns:
        Dictionary of evaluation metrics
    """
    all_metrics = []
    all_capital_histories = []
    all_position_histories = []
    
    for episode in range(n_episodes):
        states, _ = env.reset()
        done = False
        
        while not done:
            # Select actions without exploration
            actions = maddpg.select_actions(states, explore=False)
            
            # Take step
            next_states, rewards, terminated, truncated, info = env.step(actions)
            done = terminated or truncated
            
            states = next_states
        
        # Get metrics
        metrics = env.get_portfolio_metrics()
        all_metrics.append(metrics)
        all_capital_histories.append(env.capital_history)
        all_position_histories.append(env.position_history)
    
    # Aggregate metrics
    avg_metrics = {}
    for key in all_metrics[0].keys():
        values = [m[key] for m in all_metrics]
        avg_metrics[f'avg_{key}'] = np.mean(values)
        avg_metrics[f'std_{key}'] = np.std(values)
    
    return {
        'metrics': avg_metrics,
        'all_metrics': all_metrics,
        'capital_histories': all_capital_histories,
        'position_histories': all_position_histories
    }


def compare_with_baselines(
    env: MultiAgentPortfolioEnv,
    maddpg: MADDPG,
    n_episodes: int = 10
):
    """
    Compare MADDPG with baseline strategies.
    
    Args:
        env: Multi-agent portfolio environment
        maddpg: Trained MADDPG instance
        n_episodes: Number of evaluation episodes
        
    Returns:
        Dictionary comparing different strategies
    """
    results = {}
    
    # 1. MADDPG Strategy
    print("Evaluating MADDPG...")
    maddpg_results = evaluate_maddpg(env, maddpg, n_episodes)
    results['MADDPG'] = maddpg_results['metrics']
    
    # 2. Equal Weight Baseline
    print("Evaluating Equal Weight baseline...")
    equal_weight_metrics = []
    
    for _ in range(n_episodes):
        states, _ = env.reset()
        done = False
        
        # Equal weight actions for all agents
        equal_weight_action = np.ones(env.n_assets) / env.n_assets
        
        while not done:
            actions = [equal_weight_action.copy() for _ in range(env.n_agents)]
            next_states, rewards, terminated, truncated, info = env.step(actions)
            done = terminated or truncated
            states = next_states
        
        metrics = env.get_portfolio_metrics()
        equal_weight_metrics.append(metrics)
    
    avg_ew_metrics = {}
    for key in equal_weight_metrics[0].keys():
        values = [m[key] for m in equal_weight_metrics]
        avg_ew_metrics[f'avg_{key}'] = np.mean(values)
        avg_ew_metrics[f'std_{key}'] = np.std(values)
    
    results['Equal_Weight'] = avg_ew_metrics
    
    # 3. Random Strategy
    print("Evaluating Random baseline...")
    random_metrics = []
    
    for _ in range(n_episodes):
        states, _ = env.reset()
        done = False
        
        while not done:
            # Random portfolio weights
            actions = []
            for _ in range(env.n_agents):
                random_weights = np.random.dirichlet(np.ones(env.n_assets))
                actions.append(random_weights)
            
            next_states, rewards, terminated, truncated, info = env.step(actions)
            done = terminated or truncated
            states = next_states
        
        metrics = env.get_portfolio_metrics()
        random_metrics.append(metrics)
    
    avg_random_metrics = {}
    for key in random_metrics[0].keys():
        values = [m[key] for m in random_metrics]
        avg_random_metrics[f'avg_{key}'] = np.mean(values)
        avg_random_metrics[f'std_{key}'] = np.std(values)
    
    results['Random'] = avg_random_metrics
    
    return results


if __name__ == "__main__":
    # Configuration
    N_ASSETS = 10
    N_AGENTS = 3
    N_TRAIN_DAYS = 300
    N_EPISODES = 100
    BATCH_SIZE = 64
    
    # Generate data
    print("Generating synthetic market data...")
    data = generate_synthetic_market_data(n_assets=N_ASSETS, n_days=N_TRAIN_DAYS)
    
    # Create environment
    print("Creating environment...")
    env = MultiAgentPortfolioEnv(
        data=data,
        n_agents=N_AGENTS,
        initial_capital=1000000.0,
        transaction_cost=0.001,
        lookback_window=20,
        diversity_bonus=0.1
    )
    
    # Create MADDPG
    print("Creating MADDPG agents...")
    maddpg = MADDPG(
        n_agents=N_AGENTS,
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0],
        lr_actor=1e-4,
        lr_critic=1e-3,
        gamma=0.99,
        tau=0.01,
        hidden_dim=256,
        buffer_capacity=100000
    )
    
    # Train
    episode_rewards, episode_metrics = train_maddpg(
        env=env,
        maddpg=maddpg,
        n_episodes=N_EPISODES,
        batch_size=BATCH_SIZE,
        save_dir='./models'
    )
    
    # Load best model
    maddpg.load('./models/best_model')
    
    # Evaluate and compare
    print("\nComparing with baselines...")
    comparison_results = compare_with_baselines(env, maddpg, n_episodes=10)
    
    # Print results
    print("\n" + "="*60)
    print("PERFORMANCE COMPARISON")
    print("="*60)
    
    for strategy, metrics in comparison_results.items():
        print(f"\n{strategy}:")
        print(f"  Annualized Return: {metrics['avg_annualized_return']:.4f} ± {metrics['std_annualized_return']:.4f}")
        print(f"  Sharpe Ratio: {metrics['avg_sharpe_ratio']:.4f} ± {metrics['std_sharpe_ratio']:.4f}")
        print(f"  Max Drawdown: {metrics['avg_max_drawdown']:.4f} ± {metrics['std_max_drawdown']:.4f}")
        print(f"  Volatility: {metrics['avg_volatility']:.4f} ± {metrics['std_volatility']:.4f}")
    
    print("\nTraining completed!")
