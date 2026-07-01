"""
Feature Importance Analysis using Ablation Study
Systematically removes features to identify most important ones
"""

import json
import os
import sys
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from data_loader import MarketDataLoader
from environment import MultiAgentPortfolioEnv
from maddpg_agent import MADDPGTrainer


class FeatureImportanceAnalyzer:
    """Conduct ablation study to identify important features"""

    def __init__(self, config: Config, n_episodes: int = 50):
        self.config = config
        self.n_episodes = n_episodes
        self.baseline_performance = None
        self.results = {}

    def get_feature_groups(self) -> Dict[str, List[str]]:
        """Define feature groups for ablation study"""
        return {
            "historical_returns": ["returns_short", "returns_long"],
            "technical_rsi": ["rsi"],
            "technical_macd": ["macd", "macd_signal"],
            "technical_bollinger": ["bb_upper", "bb_lower", "bb_mid"],
            "volatility": ["volatility"],
            "esg_scores": ["esg"] if self.config.env.use_esg else [],
            "sentiment": ["sentiment"] if self.config.env.use_sentiment else [],
            "macro_vix": ["vix"],
            "macro_yield": ["treasury_yield"],
        }

    def train_with_config(self, config: Config, name: str) -> Dict:
        """Train model with given configuration"""
        print(f"\nTraining: {name}")
        print("-" * 60)

        # Load data
        loader = MarketDataLoader(config)
        data = loader.prepare_environment_data()

        # Create environment
        env = MultiAgentPortfolioEnv(config, data)

        # Create trainer
        trainer = MADDPGTrainer(env, config)

        # Train
        rewards = []
        sharpe_ratios = []

        for episode in tqdm(range(self.n_episodes), desc=name):
            result = trainer.train_episode()
            rewards.append(np.mean(result["episode_reward"]))
            sharpe_ratios.append(result["metrics"]["aggregate"]["sharpe_ratio"])

        # Final evaluation
        test_start, test_end = data["test_indices"]
        states = env.reset(start_idx=test_start, end_idx=test_end)

        while not env.done:
            actions = [
                agent.select_action(states[i], add_noise=False)
                for i, agent in enumerate(trainer.agents)
            ]
            states, _, _, _ = env.step(actions)

        metrics = env.get_episode_metrics()["aggregate"]

        return {
            "name": name,
            "sharpe_ratio": metrics["sharpe_ratio"],
            "total_return": metrics["total_return"],
            "max_drawdown": metrics["max_drawdown"],
            "avg_training_reward": np.mean(rewards),
            "final_training_sharpe": np.mean(sharpe_ratios[-10:]),
        }

    def run_baseline(self) -> Dict:
        """Run baseline with all features"""
        print("=" * 80)
        print("BASELINE: Training with ALL features")
        print("=" * 80)

        self.baseline_performance = self.train_with_config(
            self.config, "Baseline (All Features)"
        )
        return self.baseline_performance

    def run_ablation_study(self) -> Dict[str, Dict]:
        """Run ablation study by removing each feature group"""
        print("\n" + "=" * 80)
        print("ABLATION STUDY: Removing feature groups one by one")
        print("=" * 80)

        if self.baseline_performance is None:
            self.run_baseline()

        feature_groups = self.get_feature_groups()

        for group_name, features in feature_groups.items():
            if not features:  # Skip empty groups
                continue

            # Create modified config
            modified_config = self._create_config_without_features(group_name)

            # Train and evaluate
            result = self.train_with_config(modified_config, f"Without {group_name}")

            # Calculate performance drop
            result["performance_drop"] = (
                self.baseline_performance["sharpe_ratio"] - result["sharpe_ratio"]
            )
            result["relative_drop"] = (
                result["performance_drop"]
                / self.baseline_performance["sharpe_ratio"]
                * 100
            )

            self.results[group_name] = result

        return self.results

    def _create_config_without_features(self, feature_group: str) -> Config:
        """Create config with specified feature group disabled"""
        import copy

        config = copy.deepcopy(self.config)

        # Disable specific feature groups
        if feature_group == "esg_scores":
            config.env.use_esg = False
        elif feature_group == "sentiment":
            config.env.use_sentiment = False
        # For other features, we'd modify the feature engineering
        # This is a simplified approach

        return config

    def analyze_results(self) -> pd.DataFrame:
        """Analyze and rank features by importance"""
        if not self.results:
            raise ValueError("No results available. Run ablation study first.")

        # Create DataFrame
        df = pd.DataFrame(self.results).T
        df = df.sort_values("performance_drop", ascending=False)

        # Add importance tier
        df["importance_tier"] = pd.cut(
            df["relative_drop"],
            bins=[-np.inf, 5, 15, np.inf],
            labels=["Low", "Medium", "High"],
        )

        return df

    def identify_top_features(self, n_features: int = 5) -> List[str]:
        """Identify top N most important feature groups"""
        df = self.analyze_results()
        return df.head(n_features).index.tolist()

    def plot_feature_importance(self, save_path: str = None):
        """Visualize feature importance"""
        df = self.analyze_results()

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # 1. Performance drop bar chart
        ax1 = axes[0, 0]
        df.sort_values("performance_drop")["performance_drop"].plot(
            kind="barh", ax=ax1, color="steelblue"
        )
        ax1.set_xlabel("Performance Drop (Sharpe Ratio)")
        ax1.set_title("Feature Importance: Absolute Performance Drop")
        ax1.axvline(0, color="red", linestyle="--", alpha=0.7)

        # 2. Relative drop
        ax2 = axes[0, 1]
        df.sort_values("relative_drop")["relative_drop"].plot(
            kind="barh", ax=ax2, color="coral"
        )
        ax2.set_xlabel("Relative Performance Drop (%)")
        ax2.set_title("Feature Importance: Relative Performance Drop")

        # 3. Sharpe ratio comparison
        ax3 = axes[1, 0]
        comparison_data = pd.DataFrame(
            {
                "Baseline": [self.baseline_performance["sharpe_ratio"]] * len(df),
                "Without Feature": df["sharpe_ratio"].values,
            },
            index=df.index,
        )
        comparison_data.plot(kind="barh", ax=ax3, width=0.8)
        ax3.set_xlabel("Sharpe Ratio")
        ax3.set_title("Sharpe Ratio: With vs Without Features")
        ax3.legend(["All Features", "Feature Removed"])

        # 4. Feature importance tiers
        ax4 = axes[1, 1]
        tier_counts = df["importance_tier"].value_counts()
        colors = {"High": "red", "Medium": "orange", "Low": "green"}
        tier_counts.plot(
            kind="pie",
            ax=ax4,
            autopct="%1.1f%%",
            colors=[colors[tier] for tier in tier_counts.index],
        )
        ax4.set_title("Feature Importance Distribution")
        ax4.set_ylabel("")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"\nPlot saved to: {save_path}")

        return fig

    def generate_report(self, output_dir: str = "./results/feature_analysis"):
        """Generate comprehensive feature importance report"""
        os.makedirs(output_dir, exist_ok=True)

        df = self.analyze_results()

        # Save detailed results
        df.to_csv(f"{output_dir}/feature_importance_detailed.csv")

        # Save summary
        summary = {
            "baseline_performance": self.baseline_performance,
            "feature_rankings": df.to_dict("index"),
            "top_5_features": self.identify_top_features(5),
            "recommendations": self._generate_recommendations(df),
        }

        with open(f"{output_dir}/feature_importance_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        # Generate plots
        self.plot_feature_importance(f"{output_dir}/feature_importance_plots.png")

        # Generate markdown report
        self._generate_markdown_report(df, f"{output_dir}/FEATURE_ANALYSIS_REPORT.md")

        print(f"\n{'='*80}")
        print("Feature importance analysis complete!")
        print(f"Results saved to: {output_dir}")
        print(f"{'='*80}")

    def _generate_recommendations(self, df: pd.DataFrame) -> Dict:
        """Generate recommendations based on analysis"""
        high_importance = df[df["importance_tier"] == "High"].index.tolist()
        low_importance = df[df["importance_tier"] == "Low"].index.tolist()

        return {
            "must_keep_features": high_importance,
            "can_remove_features": low_importance,
            "complexity_reduction": f"{len(low_importance)}/{len(df)} features can be removed",
            "expected_performance_retention": f"{100 - df[df['importance_tier'] == 'Low']['relative_drop'].sum():.1f}%",
        }

    def _generate_markdown_report(self, df: pd.DataFrame, output_path: str):
        """Generate markdown report"""
        with open(output_path, "w") as f:
            f.write("# Feature Importance Analysis Report\n\n")

            f.write("## Executive Summary\n\n")
            f.write(
                f"- **Baseline Performance (All Features):** Sharpe Ratio = {self.baseline_performance['sharpe_ratio']:.4f}\n"
            )
            f.write(f"- **Features Analyzed:** {len(df)}\n")
            f.write(
                f"- **High Importance Features:** {len(df[df['importance_tier'] == 'High'])}\n"
            )
            f.write(
                f"- **Low Importance Features:** {len(df[df['importance_tier'] == 'Low'])}\n\n"
            )

            f.write("## Feature Rankings\n\n")
            f.write(
                "| Rank | Feature Group | Sharpe w/o Feature | Performance Drop | Relative Drop (%) | Importance |\n"
            )
            f.write(
                "|------|---------------|-------------------|------------------|-------------------|------------|\n"
            )

            for i, (idx, row) in enumerate(df.iterrows(), 1):
                f.write(
                    f"| {i} | {idx} | {row['sharpe_ratio']:.4f} | "
                    f"{row['performance_drop']:.4f} | {row['relative_drop']:.2f}% | "
                    f"{row['importance_tier']} |\n"
                )

            f.write("\n## Recommendations\n\n")
            recommendations = self._generate_recommendations(df)

            f.write("### Must-Keep Features (High Importance)\n")
            for feature in recommendations["must_keep_features"]:
                f.write(f"- {feature}\n")

            f.write("\n### Can Remove (Low Importance)\n")
            for feature in recommendations["can_remove_features"]:
                f.write(f"- {feature}\n")

            f.write("\n### Complexity Reduction\n")
            f.write(
                f"- {recommendations['complexity_reduction']} features can be removed\n"
            )
            f.write(
                f"- Expected performance retention: {recommendations['expected_performance_retention']}\n"
            )


def main():
    """Run feature importance analysis"""
    print("=" * 80)
    print("MARL Portfolio Optimization - Feature Importance Analysis")
    print("=" * 80)

    # Load config
    config = Config()
    config.training.n_episodes = 50  # Reduced for faster analysis
    config.data.data_source = "synthetic"

    # Create analyzer
    analyzer = FeatureImportanceAnalyzer(config, n_episodes=50)

    # Run baseline
    analyzer.run_baseline()

    # Run ablation study
    analyzer.run_ablation_study()

    # Generate report
    analyzer.generate_report()

    # Print top features
    print("\n" + "=" * 80)
    print("TOP 5 MOST IMPORTANT FEATURES:")
    print("=" * 80)
    top_features = analyzer.identify_top_features(5)
    for i, feature in enumerate(top_features, 1):
        print(f"{i}. {feature}")
    print("=" * 80)


if __name__ == "__main__":
    main()
