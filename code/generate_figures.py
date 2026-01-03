"""
Generate high-quality figures for the research paper.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from matplotlib.patches import Rectangle
from scipy import stats

# Set publication-quality style
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'serif'
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.titlesize'] = 13


def plot_training_curves(history_path: str, save_path: str):
    """Plot training performance curves."""
    with open(history_path, 'r') as f:
        history = json.load(f)
    
    episode_rewards = history['episode_rewards']
    episode_metrics = history['episode_metrics']
    
    # Create subplots
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle('Multi-Agent RL Training Progress', fontsize=14, fontweight='bold')
    
    # Total rewards over episodes
    total_rewards = [np.sum(r) for r in episode_rewards]
    smoothed_rewards = pd.Series(total_rewards).rolling(window=20, min_periods=1).mean()
    
    axes[0, 0].plot(total_rewards, alpha=0.3, color='steelblue', label='Raw')
    axes[0, 0].plot(smoothed_rewards, color='darkblue', linewidth=2, label='Smoothed')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Total Reward')
    axes[0, 0].set_title('Cumulative Agent Rewards')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Sharpe ratio over episodes
    sharpe_ratios = [m['sharpe_ratio'] for m in episode_metrics]
    smoothed_sharpe = pd.Series(sharpe_ratios).rolling(window=20, min_periods=1).mean()
    
    axes[0, 1].plot(sharpe_ratios, alpha=0.3, color='forestgreen', label='Raw')
    axes[0, 1].plot(smoothed_sharpe, color='darkgreen', linewidth=2, label='Smoothed')
    axes[0, 1].axhline(y=0, color='red', linestyle='--', alpha=0.5)
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Sharpe Ratio')
    axes[0, 1].set_title('Risk-Adjusted Performance')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Annualized return over episodes
    annual_returns = [m['annualized_return'] for m in episode_metrics]
    smoothed_returns = pd.Series(annual_returns).rolling(window=20, min_periods=1).mean()
    
    axes[0, 2].plot(annual_returns, alpha=0.3, color='coral', label='Raw')
    axes[0, 2].plot(smoothed_returns, color='darkred', linewidth=2, label='Smoothed')
    axes[0, 2].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    axes[0, 2].set_xlabel('Episode')
    axes[0, 2].set_ylabel('Annualized Return')
    axes[0, 2].set_title('Expected Returns')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # Volatility over episodes
    volatilities = [m['volatility'] for m in episode_metrics]
    smoothed_vol = pd.Series(volatilities).rolling(window=20, min_periods=1).mean()
    
    axes[1, 0].plot(volatilities, alpha=0.3, color='purple', label='Raw')
    axes[1, 0].plot(smoothed_vol, color='indigo', linewidth=2, label='Smoothed')
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Volatility')
    axes[1, 0].set_title('Portfolio Volatility')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Maximum drawdown over episodes
    max_drawdowns = [m['max_drawdown'] for m in episode_metrics]
    smoothed_dd = pd.Series(max_drawdowns).rolling(window=20, min_periods=1).mean()
    
    axes[1, 1].plot(max_drawdowns, alpha=0.3, color='orangered', label='Raw')
    axes[1, 1].plot(smoothed_dd, color='darkred', linewidth=2, label='Smoothed')
    axes[1, 1].set_xlabel('Episode')
    axes[1, 1].set_ylabel('Max Drawdown')
    axes[1, 1].set_title('Downside Risk')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # Turnover over episodes
    turnovers = [m['avg_turnover'] for m in episode_metrics]
    smoothed_turn = pd.Series(turnovers).rolling(window=20, min_periods=1).mean()
    
    axes[1, 2].plot(turnovers, alpha=0.3, color='teal', label='Raw')
    axes[1, 2].plot(smoothed_turn, color='darkcyan', linewidth=2, label='Smoothed')
    axes[1, 2].set_xlabel('Episode')
    axes[1, 2].set_ylabel('Avg Turnover')
    axes[1, 2].set_title('Portfolio Turnover')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"Saved training curves to {save_path}")


def plot_comparison_bars(comparison_results: dict, save_path: str):
    """Create bar chart comparing different strategies."""
    strategies = list(comparison_results.keys())
    
    metrics = ['avg_annualized_return', 'avg_sharpe_ratio', 'avg_sortino_ratio', 'avg_max_drawdown']
    metric_labels = ['Annualized Return', 'Sharpe Ratio', 'Sortino Ratio', 'Max Drawdown']
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('Performance Comparison: MADDPG vs Baselines', fontsize=14, fontweight='bold')
    
    axes = axes.flatten()
    colors = ['#2E86AB', '#A23B72', '#F18F01']
    
    for idx, (metric, label) in enumerate(zip(metrics, metric_labels)):
        means = [comparison_results[s][metric] for s in strategies]
        stds = [comparison_results[s][f'std_{metric.split("avg_")[1]}'] for s in strategies]
        
        bars = axes[idx].bar(strategies, means, yerr=stds, capsize=5, 
                            color=colors, alpha=0.8, edgecolor='black', linewidth=1.2)
        
        axes[idx].set_ylabel(label)
        axes[idx].set_title(f'{label} Comparison')
        axes[idx].grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, mean, std in zip(bars, means, stds):
            height = bar.get_height()
            axes[idx].text(bar.get_x() + bar.get_width()/2., height,
                          f'{mean:.3f}',
                          ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"Saved comparison bars to {save_path}")


def plot_portfolio_evolution(capital_history: list, n_agents: int, save_path: str):
    """Plot portfolio value evolution for all agents."""
    capital_history = np.array(capital_history)
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    fig.suptitle('Portfolio Evolution Over Time', fontsize=14, fontweight='bold')
    
    # Individual agent portfolios
    for agent_id in range(n_agents):
        agent_capital = capital_history[:, agent_id]
        axes[0].plot(agent_capital, label=f'Agent {agent_id+1}', linewidth=2, alpha=0.8)
    
    axes[0].set_xlabel('Time Step')
    axes[0].set_ylabel('Portfolio Value ($)')
    axes[0].set_title('Individual Agent Performance')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Aggregate portfolio
    aggregate_capital = np.sum(capital_history, axis=1)
    cumulative_return = (aggregate_capital - aggregate_capital[0]) / aggregate_capital[0] * 100
    
    axes[1].plot(aggregate_capital, color='darkblue', linewidth=2.5)
    axes[1].fill_between(range(len(aggregate_capital)), aggregate_capital, 
                         aggregate_capital[0], alpha=0.3, color='lightblue')
    axes[1].set_xlabel('Time Step')
    axes[1].set_ylabel('Aggregate Portfolio Value ($)')
    axes[1].set_title(f'Combined Portfolio (Final Return: {cumulative_return[-1]:.2f}%)')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"Saved portfolio evolution to {save_path}")


def plot_position_heatmap(position_history: list, n_agents: int, n_assets: int, save_path: str):
    """Plot heatmap of agent positions over time."""
    position_history = np.array(position_history)
    
    fig, axes = plt.subplots(1, n_agents, figsize=(15, 5))
    fig.suptitle('Agent Portfolio Allocations Over Time', fontsize=14, fontweight='bold')
    
    if n_agents == 1:
        axes = [axes]
    
    for agent_id in range(n_agents):
        agent_positions = position_history[:, agent_id, :]
        
        im = axes[agent_id].imshow(agent_positions.T, aspect='auto', cmap='YlOrRd', 
                                   interpolation='nearest', vmin=0, vmax=1)
        
        axes[agent_id].set_xlabel('Time Step')
        axes[agent_id].set_ylabel('Asset ID')
        axes[agent_id].set_title(f'Agent {agent_id+1}')
        axes[agent_id].set_yticks(range(n_assets))
        axes[agent_id].set_yticklabels([f'A{i+1}' for i in range(n_assets)])
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=axes[agent_id], fraction=0.046, pad=0.04)
        cbar.set_label('Weight', rotation=270, labelpad=15)
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"Saved position heatmap to {save_path}")


def plot_risk_return_scatter(comparison_results: dict, save_path: str):
    """Create risk-return scatter plot."""
    fig, ax = plt.subplots(figsize=(10, 7))
    
    strategies = list(comparison_results.keys())
    colors = ['#2E86AB', '#A23B72', '#F18F01']
    markers = ['o', 's', '^']
    
    for idx, strategy in enumerate(strategies):
        metrics = comparison_results[strategy]
        ret = metrics['avg_annualized_return']
        vol = metrics['avg_volatility']
        ret_std = metrics['std_annualized_return']
        vol_std = metrics['std_volatility']
        
        ax.scatter(vol, ret, s=200, c=[colors[idx]], marker=markers[idx], 
                  alpha=0.7, edgecolors='black', linewidth=1.5, label=strategy)
        
        # Error bars
        ax.errorbar(vol, ret, xerr=vol_std, yerr=ret_std, 
                   fmt='none', ecolor=colors[idx], alpha=0.5, capsize=5)
    
    # Add efficient frontier reference line
    vols = np.linspace(0, max([comparison_results[s]['avg_volatility'] for s in strategies]) * 1.2, 100)
    efficient_returns = vols * 2  # Theoretical efficient frontier
    ax.plot(vols, efficient_returns, '--', color='gray', alpha=0.5, 
            linewidth=1, label='Theoretical Efficient Frontier')
    
    ax.set_xlabel('Volatility (Risk)', fontsize=12)
    ax.set_ylabel('Annualized Return', fontsize=12)
    ax.set_title('Risk-Return Trade-off', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', frameon=True, shadow=True)
    ax.grid(True, alpha=0.3)
    
    # Add quadrant labels
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"Saved risk-return scatter to {save_path}")


def plot_diversification_analysis(position_history: list, n_agents: int, n_assets: int, save_path: str):
    """Analyze and plot diversification metrics."""
    position_history = np.array(position_history)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Portfolio Diversification Analysis', fontsize=14, fontweight='bold')
    
    # 1. Herfindahl Index over time (concentration measure)
    herfindahl_indices = []
    for t in range(len(position_history)):
        hi_values = []
        for agent_id in range(n_agents):
            positions = position_history[t, agent_id, :]
            hi = np.sum(positions ** 2)
            hi_values.append(hi)
        herfindahl_indices.append(np.mean(hi_values))
    
    axes[0, 0].plot(herfindahl_indices, color='darkblue', linewidth=2)
    axes[0, 0].axhline(y=1/n_assets, color='red', linestyle='--', 
                      label=f'Equal Weight ({1/n_assets:.3f})', alpha=0.7)
    axes[0, 0].set_xlabel('Time Step')
    axes[0, 0].set_ylabel('Herfindahl Index')
    axes[0, 0].set_title('Portfolio Concentration (Lower = More Diversified)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Effective number of assets
    effective_n = [1/hi if hi > 0 else n_assets for hi in herfindahl_indices]
    axes[0, 1].plot(effective_n, color='forestgreen', linewidth=2)
    axes[0, 1].axhline(y=n_assets, color='red', linestyle='--', 
                      label=f'Max Diversity ({n_assets})', alpha=0.7)
    axes[0, 1].set_xlabel('Time Step')
    axes[0, 1].set_ylabel('Effective # of Assets')
    axes[0, 1].set_title('Effective Number of Assets (Higher = More Diversified)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Agent similarity over time (cosine similarity)
    similarities = []
    for t in range(len(position_history)):
        sim_values = []
        for i in range(n_agents):
            for j in range(i+1, n_agents):
                pos_i = position_history[t, i, :]
                pos_j = position_history[t, j, :]
                # Cosine similarity
                sim = np.dot(pos_i, pos_j) / (np.linalg.norm(pos_i) * np.linalg.norm(pos_j) + 1e-8)
                sim_values.append(sim)
        similarities.append(np.mean(sim_values) if sim_values else 0)
    
    axes[1, 0].plot(similarities, color='purple', linewidth=2)
    axes[1, 0].axhline(y=0, color='green', linestyle='--', 
                      label='Orthogonal (Perfect Diversity)', alpha=0.7)
    axes[1, 0].set_xlabel('Time Step')
    axes[1, 0].set_ylabel('Avg Agent Similarity')
    axes[1, 0].set_title('Inter-Agent Portfolio Similarity (Lower = More Diverse)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Distribution of final positions
    final_positions = position_history[-1, :, :]
    
    positions_flat = final_positions.flatten()
    axes[1, 1].hist(positions_flat, bins=30, color='coral', alpha=0.7, edgecolor='black')
    axes[1, 1].axvline(x=1/n_assets, color='red', linestyle='--', 
                      label=f'Equal Weight ({1/n_assets:.3f})', linewidth=2)
    axes[1, 1].set_xlabel('Position Weight')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Distribution of Final Portfolio Weights')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"Saved diversification analysis to {save_path}")


def plot_drawdown_analysis(capital_history: list, save_path: str):
    """Plot drawdown analysis."""
    capital_history = np.array(capital_history)
    aggregate_capital = np.sum(capital_history, axis=1)
    
    # Calculate drawdown
    cumulative_returns = aggregate_capital / aggregate_capital[0]
    running_max = np.maximum.accumulate(cumulative_returns)
    drawdown = (cumulative_returns - running_max) / running_max * 100
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    fig.suptitle('Drawdown Analysis', fontsize=14, fontweight='bold')
    
    # Portfolio value and running maximum
    axes[0].plot(cumulative_returns, label='Portfolio Value', color='darkblue', linewidth=2)
    axes[0].plot(running_max, label='Running Maximum', color='red', 
                linestyle='--', linewidth=2, alpha=0.7)
    axes[0].fill_between(range(len(cumulative_returns)), cumulative_returns, 
                        running_max, alpha=0.3, color='red', label='Drawdown')
    axes[0].set_xlabel('Time Step')
    axes[0].set_ylabel('Cumulative Return')
    axes[0].set_title('Portfolio Value vs Running Maximum')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Drawdown over time
    axes[1].fill_between(range(len(drawdown)), drawdown, 0, 
                        color='crimson', alpha=0.6)
    axes[1].plot(drawdown, color='darkred', linewidth=1.5)
    axes[1].axhline(y=np.min(drawdown), color='black', linestyle='--', 
                   label=f'Max Drawdown: {np.min(drawdown):.2f}%', linewidth=2)
    axes[1].set_xlabel('Time Step')
    axes[1].set_ylabel('Drawdown (%)')
    axes[1].set_title('Drawdown Over Time')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"Saved drawdown analysis to {save_path}")


def plot_architecture_diagram(save_path: str):
    """Create MADDPG architecture diagram."""
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Title
    ax.text(5, 9.5, 'MADDPG Architecture for Portfolio Optimization', 
            ha='center', fontsize=14, fontweight='bold')
    
    # Environment
    env_rect = Rectangle((0.5, 6.5), 2, 1.5, facecolor='lightblue', 
                         edgecolor='black', linewidth=2)
    ax.add_patch(env_rect)
    ax.text(1.5, 7.25, 'Market\nEnvironment', ha='center', va='center', 
           fontsize=10, fontweight='bold')
    
    # Agents (4 agents as per paper: Tech, Healthcare, Finance, Energy/Commodities)
    agent_colors = ['lightcoral', 'lightgreen', 'lightyellow', 'lightskyblue']
    agent_positions = [(3.5, 7.5), (4.7, 7.5), (5.9, 7.5), (7.1, 7.5)]
    agent_labels = ['Agent 1\n(Tech)', 'Agent 2\n(Health)', 'Agent 3\n(Finance)', 'Agent 4\n(Energy)']
    
    for idx, (x, y) in enumerate(agent_positions):
        # Actor
        actor_rect = Rectangle((x, y-0.3), 1, 0.6, facecolor=agent_colors[idx], 
                              edgecolor='black', linewidth=1.5)
        ax.add_patch(actor_rect)
        ax.text(x+0.5, y, f'{agent_labels[idx]}\nActor', ha='center', va='center', 
               fontsize=8, fontweight='bold')
        
        # Arrows from environment to actors
        ax.annotate('', xy=(x, y), xytext=(2.5, 7.25),
                   arrowprops=dict(arrowstyle='->', lw=1.5, color='black'))
        
        # Arrows from actors to environment
        ax.annotate('', xy=(2.5, 7.25), xytext=(x+1, y),
                   arrowprops=dict(arrowstyle='->', lw=1.5, color='blue'))
    
    # Centralized Critic
    critic_rect = Rectangle((3.5, 5), 3, 1, facecolor='plum', 
                           edgecolor='black', linewidth=2)
    ax.add_patch(critic_rect)
    ax.text(5, 5.5, 'Centralized Critic\n(Sees all states & actions)', 
           ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Arrows from agents to critic
    for x, y in agent_positions:
        ax.annotate('', xy=(5, 6), xytext=(x+0.5, y-0.3),
                   arrowprops=dict(arrowstyle='->', lw=1.5, color='green'))
    
    # Replay Buffer
    buffer_rect = Rectangle((3.5, 3.5), 3, 0.8, facecolor='wheat', 
                           edgecolor='black', linewidth=2)
    ax.add_patch(buffer_rect)
    ax.text(5, 3.9, 'Shared Replay Buffer', ha='center', va='center', 
           fontsize=10, fontweight='bold')
    
    # Arrow from critic to buffer
    ax.annotate('', xy=(5, 4.3), xytext=(5, 5),
               arrowprops=dict(arrowstyle='->', lw=1.5, color='purple'))
    
    # Output
    output_rect = Rectangle((3.5, 1.5), 3, 0.8, facecolor='lightgray', 
                           edgecolor='black', linewidth=2)
    ax.add_patch(output_rect)
    ax.text(5, 1.9, 'Diversified Portfolio', ha='center', va='center', 
           fontsize=10, fontweight='bold')
    
    # Arrow from buffer to output
    ax.annotate('', xy=(5, 2.3), xytext=(5, 3.5),
               arrowprops=dict(arrowstyle='->', lw=1.5, color='black'))
    
    # Add legend
    legend_elements = [
        Rectangle((0, 0), 1, 1, facecolor='lightblue', label='Environment'),
        Rectangle((0, 0), 1, 1, facecolor='lightcoral', label='Agents'),
        Rectangle((0, 0), 1, 1, facecolor='plum', label='Critic'),
        Rectangle((0, 0), 1, 1, facecolor='wheat', label='Buffer')
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"Saved architecture diagram to {save_path}")


if __name__ == "__main__":
    print("Figure generation script loaded successfully!")
    print("Use individual functions to generate specific figures.")
