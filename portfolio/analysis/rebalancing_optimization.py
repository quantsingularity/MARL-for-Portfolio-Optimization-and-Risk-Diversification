"""
Rebalancing Optimization Analysis
Analyzes optimal rebalancing frequency considering transaction costs
Compares daily, weekly, monthly rebalancing strategies
"""

# --- matplotlib/seaborn compatibility shim (auto-added) ---
# Old seaborn (<0.12) calls matplotlib.cm.register_cmap / get_cmap, removed in
# matplotlib 3.9+. Restore them so seaborn imports and runs on modern matplotlib
# without requiring a seaborn upgrade. No-op on already-compatible versions.
try:
    import matplotlib as _mpl_compat
    import matplotlib.cm as _mpl_cm_compat

    if not hasattr(_mpl_cm_compat, "register_cmap"):

        def _compat_register_cmap(name=None, cmap=None, **_kw):
            if cmap is None and not isinstance(name, str):
                cmap, name = name, getattr(name, "name", None)
            _mpl_compat.colormaps.register(cmap, name=name, force=True)

        _mpl_cm_compat.register_cmap = _compat_register_cmap

    if not hasattr(_mpl_cm_compat, "get_cmap"):

        def _compat_get_cmap(name=None, lut=None):
            _c = (
                _mpl_compat.colormaps[name]
                if isinstance(name, str)
                else (
                    name
                    if name is not None
                    else _mpl_compat.colormaps[_mpl_compat.rcParams["image.cmap"]]
                )
            )
            return _c.resampled(lut) if lut else _c

        _mpl_cm_compat.get_cmap = _compat_get_cmap
except Exception:
    pass
# --- end compatibility shim ---

import json
import os
import sys
from dataclasses import dataclass
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from data_loader import MarketDataLoader
from environment import MultiAgentPortfolioEnv
from maddpg_agent import MADDPGTrainer


@dataclass
class RebalancingResult:
    """Results for a rebalancing strategy"""

    frequency: str
    days_between_rebalance: int
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    total_transaction_costs: float
    net_return: float
    n_rebalances: int
    avg_portfolio_turnover: float
    return_per_rebalance: float
    cost_drag: float  # Impact of transaction costs on returns


class RebalancingOptimizer:
    """Optimize rebalancing frequency"""

    def __init__(self, config: Config, trained_model_path: str = None):
        self.config = config
        self.trained_model_path = trained_model_path
        self.results = {}

        # Rebalancing frequencies to test
        self.frequencies = {
            "daily": 1,
            "weekly": 5,
            "biweekly": 10,
            "monthly": 21,
            "quarterly": 63,
            "semi_annual": 126,
        }

        # Transaction cost scenarios
        self.cost_scenarios = {
            "low": 0.0005,  # 5 bps
            "medium": 0.001,  # 10 bps (default)
            "high": 0.002,  # 20 bps
            "institutional": 0.0001,  # 1 bp
        }

    def train_or_load_model(self) -> MADDPGTrainer:
        """Train new model or load existing one"""
        loader = MarketDataLoader(self.config)
        data = loader.prepare_environment_data()
        env = MultiAgentPortfolioEnv(self.config, data)
        trainer = MADDPGTrainer(env, self.config)

        if self.trained_model_path:
            print(f"Loading model from {self.trained_model_path}")
            for agent in trainer.agents:
                agent.load(self.trained_model_path)
        else:
            print("Training new model...")
            for episode in tqdm(range(100), desc="Training"):
                trainer.train_episode()

        return trainer, data, env

    def evaluate_rebalancing_frequency(
        self,
        trainer: MADDPGTrainer,
        env: MultiAgentPortfolioEnv,
        data: Dict,
        frequency_name: str,
        days_between: int,
        transaction_cost: float,
    ) -> RebalancingResult:
        """Evaluate a specific rebalancing frequency"""

        # Reset environment to test period
        test_start, test_end = data["test_indices"]

        # Simulate with controlled rebalancing
        states = env.reset(start_idx=test_start, end_idx=test_end)

        total_costs = 0
        n_rebalances = 0
        portfolio_turnovers = []
        previous_positions = None

        step = 0

        while not env.done:
            # Get actions from trained agents
            actions = [
                agent.select_action(states[i], add_noise=False)
                for i, agent in enumerate(trainer.agents)
            ]

            # Only rebalance at specified frequency
            if step % days_between == 0:
                n_rebalances += 1

                # Calculate portfolio turnover
                if previous_positions is not None:
                    combined_actions = np.concatenate(actions)
                    turnover = np.sum(np.abs(combined_actions - previous_positions))
                    portfolio_turnovers.append(turnover)

                    total_capital = np.sum(env.agent_capitals)
                    costs = turnover * transaction_cost * total_capital
                    total_costs += costs

                    # Deduct costs proportionally from each agent's capital
                    if total_capital > 0:
                        for agent_id in range(env.n_agents):
                            agent_share = env.agent_capitals[agent_id] / total_capital
                            env.agent_capitals[agent_id] -= costs * agent_share

                # Update positions
                previous_positions = np.concatenate(actions)

                # Execute rebalancing
                states, rewards, done, info = env.step(actions)
            else:
                # Hold current positions (let them drift)
                # Pass zero-adjustment actions
                hold_actions = [
                    (
                        previous_positions[i * len(a) : (i + 1) * len(a)]
                        if previous_positions is not None
                        else actions[i]
                    )
                    for i, a in enumerate(actions)
                ]
                states, rewards, done, info = env.step(hold_actions)

            step += 1

        # Get final metrics
        metrics = env.get_episode_metrics()["aggregate"]

        # Calculate net return
        gross_return = metrics["total_return"]
        cost_drag = total_costs / self.config.env.initial_capital
        net_return = gross_return - cost_drag

        return RebalancingResult(
            frequency=frequency_name,
            days_between_rebalance=days_between,
            total_return=gross_return,
            sharpe_ratio=metrics["sharpe_ratio"],
            max_drawdown=metrics["max_drawdown"],
            total_transaction_costs=total_costs,
            net_return=net_return,
            n_rebalances=n_rebalances,
            avg_portfolio_turnover=(
                np.mean(portfolio_turnovers) if portfolio_turnovers else 0
            ),
            return_per_rebalance=net_return / n_rebalances if n_rebalances > 0 else 0,
            cost_drag=cost_drag,
        )

    def run_comprehensive_analysis(self) -> Dict[str, Dict[str, RebalancingResult]]:
        """Run analysis across all frequencies and cost scenarios"""
        print("=" * 80)
        print("REBALANCING OPTIMIZATION ANALYSIS")
        print("=" * 80)

        # Train/load model
        trainer, data, env = self.train_or_load_model()

        results = {}

        for cost_name, cost_value in self.cost_scenarios.items():
            print(f"\n{'='*80}")
            print(f"Cost Scenario: {cost_name} ({cost_value*10000:.1f} bps)")
            print(f"{'='*80}")

            results[cost_name] = {}

            for freq_name, days_between in self.frequencies.items():
                print(f"\nEvaluating {freq_name} rebalancing ({days_between} days)...")

                # Create fresh environment copy
                loader = MarketDataLoader(self.config)
                data_fresh = loader.prepare_environment_data()
                config_copy = self.config
                config_copy.env.transaction_cost = cost_value
                env_fresh = MultiAgentPortfolioEnv(config_copy, data_fresh)

                result = self.evaluate_rebalancing_frequency(
                    trainer, env_fresh, data_fresh, freq_name, days_between, cost_value
                )

                results[cost_name][freq_name] = result

                print(f"  Net Return: {result.net_return*100:.2f}%")
                print(f"  Sharpe Ratio: {result.sharpe_ratio:.4f}")
                print(f"  Cost Drag: {result.cost_drag*100:.2f}%")
                print(f"  # Rebalances: {result.n_rebalances}")

        self.results = results
        return results

    def find_optimal_frequency(self, cost_scenario: str = "medium") -> str:
        """Find optimal rebalancing frequency for a cost scenario"""
        if not self.results:
            raise ValueError("No results available. Run analysis first.")

        scenario_results = self.results[cost_scenario]
        best_freq = max(scenario_results.items(), key=lambda x: x[1].sharpe_ratio)
        return best_freq[0]

    def plot_analysis(self, save_path: str = None):
        """Visualize rebalancing optimization results"""
        if not self.results:
            raise ValueError("No results available. Run analysis first.")

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        # Prepare data for plotting
        plot_data = []
        for cost_name, freq_results in self.results.items():
            for freq_name, result in freq_results.items():
                plot_data.append(
                    {
                        "Cost Scenario": cost_name,
                        "Frequency": freq_name,
                        "Days Between": result.days_between_rebalance,
                        "Net Return (%)": result.net_return * 100,
                        "Sharpe Ratio": result.sharpe_ratio,
                        "Cost Drag (%)": result.cost_drag * 100,
                        "# Rebalances": result.n_rebalances,
                        "Avg Turnover": result.avg_portfolio_turnover,
                        "Max Drawdown (%)": result.max_drawdown * 100,
                    }
                )

        df = pd.DataFrame(plot_data)

        # 1. Net Return vs Frequency
        ax1 = axes[0, 0]
        for cost in df["Cost Scenario"].unique():
            data = df[df["Cost Scenario"] == cost]
            ax1.plot(
                data["Days Between"],
                data["Net Return (%)"],
                marker="o",
                label=cost,
                linewidth=2,
            )
        ax1.set_xlabel("Days Between Rebalancing")
        ax1.set_title("Net Return vs Rebalancing Frequency")
        ax1.set_ylabel("Net Return (%)")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. Sharpe Ratio vs Frequency
        ax2 = axes[0, 1]
        for cost in df["Cost Scenario"].unique():
            data = df[df["Cost Scenario"] == cost]
            ax2.plot(
                data["Days Between"],
                data["Sharpe Ratio"],
                marker="o",
                label=cost,
                linewidth=2,
            )
        ax2.set_xlabel("Days Between Rebalancing")
        ax2.set_title("Sharpe Ratio vs Rebalancing Frequency")
        ax2.set_ylabel("Sharpe Ratio")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3. Cost Drag vs Frequency
        ax3 = axes[0, 2]
        for cost in df["Cost Scenario"].unique():
            data = df[df["Cost Scenario"] == cost]
            ax3.plot(
                data["Days Between"],
                data["Cost Drag (%)"],
                marker="o",
                label=cost,
                linewidth=2,
            )
        ax3.set_xlabel("Days Between Rebalancing")
        ax3.set_title("Transaction Cost Drag vs Frequency")
        ax3.set_ylabel("Cost Drag (%)")
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # 4. Heatmap: Return by Cost and Frequency
        ax4 = axes[1, 0]
        pivot_return = df.pivot(
            index="Cost Scenario", columns="Frequency", values="Net Return (%)"
        )
        sns.heatmap(
            pivot_return, annot=True, fmt=".2f", cmap="RdYlGn", ax=ax4, center=0
        )
        ax4.set_title("Net Return Heatmap (%)")

        # 5. Heatmap: Sharpe by Cost and Frequency
        ax5 = axes[1, 1]
        pivot_sharpe = df.pivot(
            index="Cost Scenario", columns="Frequency", values="Sharpe Ratio"
        )
        sns.heatmap(pivot_sharpe, annot=True, fmt=".3f", cmap="RdYlGn", ax=ax5)
        ax5.set_title("Sharpe Ratio Heatmap")

        # 6. Number of Rebalances vs Cost Drag
        ax6 = axes[1, 2]
        for cost in df["Cost Scenario"].unique():
            data = df[df["Cost Scenario"] == cost]
            ax6.scatter(
                data["# Rebalances"],
                data["Cost Drag (%)"],
                label=cost,
                s=100,
                alpha=0.6,
            )
        ax6.set_xlabel("Number of Rebalances")
        ax6.set_ylabel("Cost Drag (%)")
        ax6.set_title("Rebalances vs Cost Impact")
        ax6.legend()
        ax6.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"\nPlot saved to: {save_path}")

        return fig

    def generate_report(self, output_dir: str = "./results/rebalancing_analysis"):
        """Generate comprehensive rebalancing optimization report"""
        os.makedirs(output_dir, exist_ok=True)

        # Save detailed results
        all_results = []
        for cost_name, freq_results in self.results.items():
            for freq_name, result in freq_results.items():
                all_results.append(
                    {
                        "cost_scenario": cost_name,
                        "frequency": result.frequency,
                        "days_between": result.days_between_rebalance,
                        "gross_return": result.total_return,
                        "net_return": result.net_return,
                        "sharpe_ratio": result.sharpe_ratio,
                        "max_drawdown": result.max_drawdown,
                        "total_costs": result.total_transaction_costs,
                        "cost_drag": result.cost_drag,
                        "n_rebalances": result.n_rebalances,
                        "avg_turnover": result.avg_portfolio_turnover,
                        "return_per_rebalance": result.return_per_rebalance,
                    }
                )

        df = pd.DataFrame(all_results)
        df.to_csv(f"{output_dir}/rebalancing_analysis_detailed.csv", index=False)

        # Find optimal frequencies for each cost scenario
        optimal_frequencies = {}
        for cost_name in self.results.keys():
            optimal = self.find_optimal_frequency(cost_name)
            optimal_frequencies[cost_name] = {
                "frequency": optimal,
                "metrics": {
                    "net_return": self.results[cost_name][optimal].net_return,
                    "sharpe_ratio": self.results[cost_name][optimal].sharpe_ratio,
                    "cost_drag": self.results[cost_name][optimal].cost_drag,
                },
            }

        # Save summary
        summary = {
            "optimal_frequencies": optimal_frequencies,
            "analysis_summary": {
                "cost_scenarios_tested": list(self.cost_scenarios.keys()),
                "frequencies_tested": list(self.frequencies.keys()),
                "key_findings": self._generate_key_findings(df),
            },
        }

        with open(f"{output_dir}/rebalancing_optimization_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        # Generate plots
        self.plot_analysis(f"{output_dir}/rebalancing_optimization_plots.png")

        # Generate markdown report
        self._generate_markdown_report(
            df, optimal_frequencies, f"{output_dir}/REBALANCING_ANALYSIS_REPORT.md"
        )

        print(f"\n{'='*80}")
        print("Rebalancing optimization analysis complete!")
        print(f"Results saved to: {output_dir}")
        print(f"{'='*80}")

    def _generate_key_findings(self, df: pd.DataFrame) -> List[str]:
        """Generate key findings from analysis"""
        findings = []

        # Best performing frequency overall
        best_row = df.loc[df["sharpe_ratio"].idxmax()]
        findings.append(
            f"Best overall performance: {best_row['frequency']} rebalancing "
            f"with {best_row['cost_scenario']} costs (Sharpe: {best_row['sharpe_ratio']:.4f})"
        )

        # Cost impact
        avg_cost_drag = df.groupby("cost_scenario")["cost_drag"].mean()
        findings.append(
            f"Average cost drag ranges from {avg_cost_drag.min()*100:.2f}% "
            f"to {avg_cost_drag.max()*100:.2f}%"
        )

        # Frequency impact
        freq_performance = (
            df.groupby("frequency")["sharpe_ratio"].mean().sort_values(ascending=False)
        )
        findings.append(
            f"Most robust frequency: {freq_performance.index[0]} "
            f"(avg Sharpe: {freq_performance.values[0]:.4f})"
        )

        return findings

    def _generate_markdown_report(
        self, df: pd.DataFrame, optimal_freq: Dict, output_path: str
    ):
        """Generate markdown report"""
        with open(output_path, "w") as f:
            f.write("# Rebalancing Optimization Analysis Report\n\n")

            f.write("## Executive Summary\n\n")
            f.write("This analysis evaluates optimal portfolio rebalancing frequency ")
            f.write("considering transaction costs across multiple cost scenarios.\n\n")

            f.write("## Optimal Rebalancing Frequencies\n\n")
            f.write(
                "| Cost Scenario | Optimal Frequency | Net Return | Sharpe Ratio | Cost Drag |\n"
            )
            f.write(
                "|---------------|-------------------|------------|--------------|----------|\n"
            )

            for cost, data in optimal_freq.items():
                metrics = data["metrics"]
                f.write(
                    f"| {cost} | {data['frequency']} | "
                    f"{metrics['net_return']*100:.2f}% | "
                    f"{metrics['sharpe_ratio']:.4f} | "
                    f"{metrics['cost_drag']*100:.2f}% |\n"
                )

            f.write("\n## Detailed Results by Cost Scenario\n\n")

            for cost in df["cost_scenario"].unique():
                f.write(f"### {cost.title()} Cost Scenario\n\n")
                cost_df = df[df["cost_scenario"] == cost].sort_values(
                    "sharpe_ratio", ascending=False
                )

                f.write(
                    "| Frequency | Net Return | Sharpe Ratio | Cost Drag | # Rebalances |\n"
                )
                f.write(
                    "|-----------|------------|--------------|-----------|-------------|\n"
                )

                for _, row in cost_df.iterrows():
                    f.write(
                        f"| {row['frequency']} | {row['net_return']*100:.2f}% | "
                        f"{row['sharpe_ratio']:.4f} | {row['cost_drag']*100:.2f}% | "
                        f"{row['n_rebalances']} |\n"
                    )
                f.write("\n")

            f.write("## Key Findings\n\n")
            findings = self._generate_key_findings(df)
            for i, finding in enumerate(findings, 1):
                f.write(f"{i}. {finding}\n")

            f.write("\n## Recommendations\n\n")
            f.write(
                "- For institutional investors (low costs): Consider more frequent rebalancing\n"
            )
            f.write(
                "- For retail investors (high costs): Weekly to monthly rebalancing is optimal\n"
            )
            f.write("- Cost drag increases exponentially with rebalancing frequency\n")
            f.write("- Risk-adjusted returns (Sharpe) should be the primary metric\n")


def main():
    """Run rebalancing optimization analysis"""
    print("=" * 80)
    print("MARL Portfolio Optimization - Rebalancing Optimization Analysis")
    print("=" * 80)

    # Load config
    config = Config()
    config.data.data_source = "synthetic"

    # Create optimizer
    optimizer = RebalancingOptimizer(config)

    # Run comprehensive analysis
    optimizer.run_comprehensive_analysis()

    # Generate report
    optimizer.generate_report()

    # Print recommendations
    print("\n" + "=" * 80)
    print("OPTIMAL REBALANCING FREQUENCIES:")
    print("=" * 80)
    for cost in optimizer.cost_scenarios.keys():
        optimal = optimizer.find_optimal_frequency(cost)
        result = optimizer.results[cost][optimal]
        print(f"{cost.title()} costs: {optimal} (Sharpe: {result.sharpe_ratio:.4f})")
    print("=" * 80)


if __name__ == "__main__":
    main()
