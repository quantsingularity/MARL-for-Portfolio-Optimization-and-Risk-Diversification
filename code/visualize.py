"""
Visualization and figure generation
Creates publication-quality figures from training results
"""

import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List
import json
import os

# Set style
plt.style.use("seaborn-v0_8-darkgrid")
sns.set_palette("husl")


def plot_training_curves(history: List[Dict], save_path: str):
    """Plot training progress curves"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    episodes = [h["episode"] for h in history]
    rewards = [h["reward"] for h in history]
    sharpes = [h["sharpe_ratio"] for h in history]
    returns = [h["total_return"] * 100 for h in history]
    drawdowns = [h["max_drawdown"] * 100 for h in history]

    # Reward
    axes[0, 0].plot(episodes, rewards, linewidth=2)
    axes[0, 0].set_title("Episode Reward", fontsize=14, fontweight="bold")
    axes[0, 0].set_xlabel("Episode")
    axes[0, 0].set_ylabel("Average Reward")
    axes[0, 0].grid(True, alpha=0.3)

    # Sharpe Ratio
    axes[0, 1].plot(episodes, sharpes, linewidth=2, color="green")
    axes[0, 1].set_title("Sharpe Ratio", fontsize=14, fontweight="bold")
    axes[0, 1].set_xlabel("Episode")
    axes[0, 1].set_ylabel("Sharpe Ratio")
    axes[0, 1].grid(True, alpha=0.3)

    # Returns
    axes[1, 0].plot(episodes, returns, linewidth=2, color="orange")
    axes[1, 0].set_title("Total Return", fontsize=14, fontweight="bold")
    axes[1, 0].set_xlabel("Episode")
    axes[1, 0].set_ylabel("Return (%)")
    axes[1, 0].grid(True, alpha=0.3)

    # Max Drawdown
    axes[1, 1].plot(episodes, drawdowns, linewidth=2, color="red")
    axes[1, 1].set_title("Maximum Drawdown", fontsize=14, fontweight="bold")
    axes[1, 1].set_xlabel("Episode")
    axes[1, 1].set_ylabel("Max Drawdown (%)")
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{save_path}/training_curves.png", dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved training curves to {save_path}/training_curves.png")


def plot_comparison_table(results: Dict, save_path: str):
    """Create comparison table of all methods"""
    methods = list(results.keys())
    metric_names = ["Sharpe Ratio", "Annual Return (%)", "Max Drawdown (%)"]

    # Extract data
    data = []
    for method in methods:
        row = [method]
        agg = results[method]["aggregate"]
        row.append(f"{agg['sharpe_ratio']:.3f}")
        row.append(f"{agg['total_return']*100:.2f}")
        row.append(f"{agg['max_drawdown']*100:.2f}")
        data.append(row)

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis("tight")
    ax.axis("off")

    table = ax.table(
        cellText=data,
        colLabels=["Strategy"] + metric_names,
        cellLoc="center",
        loc="center",
        colWidths=[0.3, 0.2, 0.25, 0.25],
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    # Style header
    for i in range(4):
        table[(0, i)].set_facecolor("#40466e")
        table[(0, i)].set_text_props(weight="bold", color="white")

    # Highlight MADDPG row
    for i in range(4):
        table[(1, i)].set_facecolor("#90EE90")
        table[(1, i)].set_text_props(weight="bold")

    plt.title("Performance Comparison", fontsize=16, fontweight="bold", pad=20)
    plt.savefig(f"{save_path}/comparison_table.png", dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved comparison table to {save_path}/comparison_table.png")


def plot_risk_return_scatter(results: Dict, save_path: str):
    """Create risk-return scatter plot"""
    fig, ax = plt.subplots(figsize=(10, 8))

    methods = []
    returns = []
    risks = []

    for method, metrics in results.items():
        agg = metrics["aggregate"]
        methods.append(method)
        returns.append(agg["total_return"] * 100)

        # Estimate risk from max drawdown (proxy)
        risks.append(abs(agg["max_drawdown"]) * 100)

    # Plot points
    colors = ["red" if m == "MADDPG" else "blue" for m in methods]
    sizes = [200 if m == "MADDPG" else 100 for m in methods]

    for i, method in enumerate(methods):
        ax.scatter(
            risks[i], returns[i], s=sizes[i], c=colors[i], alpha=0.6, edgecolors="black"
        )
        ax.annotate(
            method,
            (risks[i], returns[i]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=10,
            fontweight="bold" if method == "MADDPG" else "normal",
        )

    ax.set_xlabel("Risk (Max Drawdown %)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Return (%)", fontsize=12, fontweight="bold")
    ax.set_title("Risk-Return Profile", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{save_path}/risk_return_scatter.png", dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved risk-return scatter to {save_path}/risk_return_scatter.png")


def plot_ablation_study(save_path: str):
    """Plot ablation study results from paper"""
    # Data from paper Table 3 (Diversity Weight Ablation)
    lambdas = [0.0, 0.05, 0.1, 0.2]
    sharpes = [1.13, 1.31, 1.42, 1.28]
    returns = [16.2, 17.8, 18.4, 15.5]
    mdds = [18.9, 14.5, 12.3, 11.8]
    correlations = [0.42, 0.25, 0.14, 0.08]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Sharpe Ratio
    axes[0, 0].plot(lambdas, sharpes, "o-", linewidth=2, markersize=8, color="green")
    axes[0, 0].axvline(
        x=0.1, color="red", linestyle="--", alpha=0.5, label="Optimal λ=0.1"
    )
    axes[0, 0].set_title(
        "Sharpe Ratio vs Diversity Weight", fontsize=12, fontweight="bold"
    )
    axes[0, 0].set_xlabel("Diversity Weight (λ)")
    axes[0, 0].set_ylabel("Sharpe Ratio")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Annual Return
    axes[0, 1].plot(lambdas, returns, "o-", linewidth=2, markersize=8, color="blue")
    axes[0, 1].axvline(
        x=0.1, color="red", linestyle="--", alpha=0.5, label="Optimal λ=0.1"
    )
    axes[0, 1].set_title(
        "Annual Return vs Diversity Weight", fontsize=12, fontweight="bold"
    )
    axes[0, 1].set_xlabel("Diversity Weight (λ)")
    axes[0, 1].set_ylabel("Annual Return (%)")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Max Drawdown
    axes[1, 0].plot(lambdas, mdds, "o-", linewidth=2, markersize=8, color="red")
    axes[1, 0].axvline(
        x=0.1, color="red", linestyle="--", alpha=0.5, label="Optimal λ=0.1"
    )
    axes[1, 0].set_title(
        "Max Drawdown vs Diversity Weight", fontsize=12, fontweight="bold"
    )
    axes[1, 0].set_xlabel("Diversity Weight (λ)")
    axes[1, 0].set_ylabel("Max Drawdown (%)")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Average Correlation
    axes[1, 1].plot(
        lambdas, correlations, "o-", linewidth=2, markersize=8, color="purple"
    )
    axes[1, 1].axvline(
        x=0.1, color="red", linestyle="--", alpha=0.5, label="Optimal λ=0.1"
    )
    axes[1, 1].set_title(
        "Agent Correlation vs Diversity Weight", fontsize=12, fontweight="bold"
    )
    axes[1, 1].set_xlabel("Diversity Weight (λ)")
    axes[1, 1].set_ylabel("Average Pairwise Correlation")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle(
        "Ablation Study: Effect of Diversity Weight",
        fontsize=16,
        fontweight="bold",
        y=1.00,
    )
    plt.tight_layout()
    plt.savefig(f"{save_path}/ablation_study.png", dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved ablation study to {save_path}/ablation_study.png")


def plot_agent_ablation(save_path: str):
    """Plot agent number ablation study from paper"""
    # Data from paper Table 4
    n_agents = [2, 4, 6, 8]
    sharpes = [1.18, 1.42, 1.45, 1.46]
    mdds = [16.5, 12.3, 11.9, 11.7]
    correlations = [0.28, 0.14, 0.12, 0.11]
    times = [1.0, 2.4, 5.8, 12.1]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Sharpe Ratio
    axes[0, 0].plot(n_agents, sharpes, "o-", linewidth=2, markersize=8, color="green")
    axes[0, 0].axvline(
        x=4, color="red", linestyle="--", alpha=0.5, label="Selected: 4 agents"
    )
    axes[0, 0].set_title(
        "Sharpe Ratio vs Number of Agents", fontsize=12, fontweight="bold"
    )
    axes[0, 0].set_xlabel("Number of Agents")
    axes[0, 0].set_ylabel("Sharpe Ratio")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Max Drawdown
    axes[0, 1].plot(n_agents, mdds, "o-", linewidth=2, markersize=8, color="red")
    axes[0, 1].axvline(
        x=4, color="red", linestyle="--", alpha=0.5, label="Selected: 4 agents"
    )
    axes[0, 1].set_title(
        "Max Drawdown vs Number of Agents", fontsize=12, fontweight="bold"
    )
    axes[0, 1].set_xlabel("Number of Agents")
    axes[0, 1].set_ylabel("Max Drawdown (%)")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Correlation
    axes[1, 0].plot(
        n_agents, correlations, "o-", linewidth=2, markersize=8, color="purple"
    )
    axes[1, 0].axvline(
        x=4, color="red", linestyle="--", alpha=0.5, label="Selected: 4 agents"
    )
    axes[1, 0].set_title(
        "Agent Correlation vs Number of Agents", fontsize=12, fontweight="bold"
    )
    axes[1, 0].set_xlabel("Number of Agents")
    axes[1, 0].set_ylabel("Average Pairwise Correlation")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Training Time
    axes[1, 1].plot(n_agents, times, "o-", linewidth=2, markersize=8, color="orange")
    axes[1, 1].axvline(
        x=4, color="red", linestyle="--", alpha=0.5, label="Selected: 4 agents"
    )
    axes[1, 1].set_title(
        "Training Time vs Number of Agents", fontsize=12, fontweight="bold"
    )
    axes[1, 1].set_xlabel("Number of Agents")
    axes[1, 1].set_ylabel("Relative Training Time")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle(
        "Ablation Study: Effect of Number of Agents",
        fontsize=16,
        fontweight="bold",
        y=1.00,
    )
    plt.tight_layout()
    plt.savefig(f"{save_path}/agent_ablation.png", dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved agent ablation study to {save_path}/agent_ablation.png")


def generate_all_figures(results_dir: str):
    """Generate all figures for a completed experiment"""
    print("\nGenerating figures...")

    figures_dir = f"{results_dir}/figures"
    os.makedirs(figures_dir, exist_ok=True)

    # Load training history
    history_path = f"{results_dir}/training_history.json"
    if os.path.exists(history_path):
        with open(history_path, "r") as f:
            history = json.load(f)
        plot_training_curves(history, figures_dir)

    # Load test results
    results_path = f"{results_dir}/test_results.json"
    if os.path.exists(results_path):
        with open(results_path, "r") as f:
            results = json.load(f)
        plot_comparison_table(results, figures_dir)
        plot_risk_return_scatter(results, figures_dir)

    # Generate ablation studies (from paper)
    plot_ablation_study(figures_dir)
    plot_agent_ablation(figures_dir)

    print(f"\nAll figures saved to {figures_dir}/")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        results_dir = sys.argv[1]
        generate_all_figures(results_dir)
    else:
        print("Usage: python visualize.py <results_directory>")
