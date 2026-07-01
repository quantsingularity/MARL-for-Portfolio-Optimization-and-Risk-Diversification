"""
Comprehensive Benchmarking Suite
Measures runtime and memory performance across different configurations
"""

import os
import sys
import time
from dataclasses import dataclass
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from data_loader import MarketDataLoader
from environment import MultiAgentPortfolioEnv
from maddpg_agent import MADDPGTrainer


@dataclass
class BenchmarkResult:
    config_name: str
    episodes: int
    avg_episode_time: float
    total_time: float
    peak_memory_mb: float
    avg_memory_mb: float
    gpu_used: bool
    final_sharpe: float

    def to_dict(self):
        return {
            "config_name": self.config_name,
            "episodes": self.episodes,
            "avg_episode_time_sec": self.avg_episode_time,
            "total_time_sec": self.total_time,
            "peak_memory_mb": self.peak_memory_mb,
            "avg_memory_mb": self.avg_memory_mb,
            "gpu_used": self.gpu_used,
            "final_sharpe_ratio": self.final_sharpe,
        }


class PerformanceBenchmark:
    """Benchmark system performance"""

    def __init__(self):
        self.results = []
        self.process = psutil.Process()

    def benchmark_config(
        self, config: Config, name: str, n_episodes: int = 10
    ) -> BenchmarkResult:
        """Benchmark a specific configuration"""
        print(f"\nBenchmarking: {name}")
        print("-" * 60)

        # Create environment
        loader = MarketDataLoader(config)
        data = loader.prepare_environment_data()
        env = MultiAgentPortfolioEnv(config, data)
        trainer = MADDPGTrainer(env, config)

        # Check GPU usage
        gpu_used = torch.cuda.is_available() and config.device == "cuda"

        # Benchmark training
        episode_times = []
        memory_samples = []
        sharpe_ratios = []

        start_time = time.time()

        for episode in range(n_episodes):
            episode_start = time.time()

            # Train episode
            result = trainer.train_episode()

            episode_time = time.time() - episode_start
            episode_times.append(episode_time)

            # Sample memory
            current_memory = self.process.memory_info().rss / 1024 / 1024
            memory_samples.append(current_memory)

            # Record performance
            sharpe = result["metrics"]["aggregate"]["sharpe_ratio"]
            sharpe_ratios.append(sharpe)

            print(
                f"  Episode {episode+1}/{n_episodes}: {episode_time:.2f}s, "
                f"Memory: {current_memory:.1f}MB, Sharpe: {sharpe:.4f}"
            )

        total_time = time.time() - start_time

        # Create result
        result = BenchmarkResult(
            config_name=name,
            episodes=n_episodes,
            avg_episode_time=np.mean(episode_times),
            total_time=total_time,
            peak_memory_mb=max(memory_samples),
            avg_memory_mb=np.mean(memory_samples),
            gpu_used=gpu_used,
            final_sharpe=np.mean(sharpe_ratios[-3:]),
        )

        print("\nResults:")
        print(f"  Avg Episode Time: {result.avg_episode_time:.2f}s")
        print(f"  Total Time: {result.total_time:.2f}s")
        print(f"  Peak Memory: {result.peak_memory_mb:.1f}MB")
        print(f"  Final Sharpe: {result.final_sharpe:.4f}")

        return result

    def run_all_benchmarks(self) -> List[BenchmarkResult]:
        """Run benchmarks for all configurations"""
        print("=" * 80)
        print("COMPREHENSIVE PERFORMANCE BENCHMARKS")
        print("=" * 80)

        benchmarks = [
            ("MARL-Lite (No Transformer)", Config.load("configs/marl_lite.json")),
            ("MARL-Full (With Transformer)", Config()),
        ]

        # Add transformer-only config if it exists
        try:
            transformer_config = Config.load("configs/transformer.json")
            benchmarks.append(("Transformer-Optimized", transformer_config))
        except Exception:
            pass

        results = []
        for name, config in benchmarks:
            config.data.data_source = "synthetic"  # Use synthetic for speed
            result = self.benchmark_config(config, name, n_episodes=10)
            results.append(result)
            self.results.append(result)

        return results

    def compare_results(self, save_path: str = None):
        """Compare benchmark results"""
        if not self.results:
            print("No results to compare")
            return

        df = pd.DataFrame([r.to_dict() for r in self.results])

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # 1. Episode time comparison
        ax1 = axes[0, 0]
        df.plot(
            x="config_name", y="avg_episode_time_sec", kind="bar", ax=ax1, legend=False
        )
        ax1.set_title("Average Episode Training Time")
        ax1.set_ylabel("Time (seconds)")
        ax1.set_xlabel("")
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha="right")

        # 2. Memory usage
        ax2 = axes[0, 1]
        df[["config_name", "peak_memory_mb", "avg_memory_mb"]].plot(
            x="config_name", kind="bar", ax=ax2
        )
        ax2.set_title("Memory Usage")
        ax2.set_ylabel("Memory (MB)")
        ax2.set_xlabel("")
        ax2.legend(["Peak", "Average"])
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha="right")

        # 3. Performance vs Time
        ax3 = axes[1, 0]
        ax3.scatter(df["avg_episode_time_sec"], df["final_sharpe_ratio"], s=100)
        for idx, row in df.iterrows():
            ax3.annotate(
                row["config_name"],
                (row["avg_episode_time_sec"], row["final_sharpe_ratio"]),
                fontsize=8,
                ha="right",
            )
        ax3.set_xlabel("Avg Episode Time (s)")
        ax3.set_ylabel("Final Sharpe Ratio")
        ax3.set_title("Performance vs Training Speed")
        ax3.grid(True, alpha=0.3)

        # 4. Efficiency score (Sharpe / Time)
        ax4 = axes[1, 1]
        df["efficiency"] = df["final_sharpe_ratio"] / df["avg_episode_time_sec"]
        df.plot(
            x="config_name",
            y="efficiency",
            kind="bar",
            ax=ax4,
            legend=False,
            color="green",
        )
        ax4.set_title("Training Efficiency (Sharpe / Time)")
        ax4.set_ylabel("Efficiency Score")
        ax4.set_xlabel("")
        plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45, ha="right")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"\nComparison plot saved: {save_path}")

        return fig

    def generate_report(self, output_dir: str = "./results/benchmarks"):
        """Generate benchmark report"""
        os.makedirs(output_dir, exist_ok=True)

        # Save detailed results
        df = pd.DataFrame([r.to_dict() for r in self.results])
        df.to_csv(f"{output_dir}/benchmark_results.csv", index=False)

        # Generate comparison plots
        self.compare_results(f"{output_dir}/benchmark_comparison.png")

        # Generate markdown report
        with open(f"{output_dir}/BENCHMARK_REPORT.md", "w") as f:
            f.write("# Performance Benchmark Report\n\n")
            f.write(f"Generated: {pd.Timestamp.now()}\n\n")

            f.write("## Summary\n\n")
            f.write(
                "| Configuration | Avg Time/Episode | Peak Memory | Sharpe Ratio | Efficiency |\n"
            )
            f.write(
                "|---------------|------------------|-------------|--------------|------------|\n"
            )

            for result in self.results:
                efficiency = result.final_sharpe / result.avg_episode_time
                f.write(
                    f"| {result.config_name} | {result.avg_episode_time:.2f}s | "
                    f"{result.peak_memory_mb:.1f}MB | {result.final_sharpe:.4f} | "
                    f"{efficiency:.4f} |\n"
                )

            f.write("\n## Recommendations\n\n")

            # Find fastest
            fastest = min(self.results, key=lambda x: x.avg_episode_time)
            f.write(
                f"- **Fastest Training:** {fastest.config_name} ({fastest.avg_episode_time:.2f}s/episode)\n"
            )

            # Find best performance
            best_perf = max(self.results, key=lambda x: x.final_sharpe)
            f.write(
                f"- **Best Performance:** {best_perf.config_name} (Sharpe: {best_perf.final_sharpe:.4f})\n"
            )

            # Find most efficient
            efficiencies = [
                (r, r.final_sharpe / r.avg_episode_time) for r in self.results
            ]
            most_efficient = max(efficiencies, key=lambda x: x[1])
            f.write(
                f"- **Most Efficient:** {most_efficient[0].config_name} (Score: {most_efficient[1]:.4f})\n"
            )

        print(f"\n{'='*80}")
        print(f"Benchmark report generated: {output_dir}")
        print(f"{'='*80}")


def main():
    """Run comprehensive benchmarks"""
    benchmark = PerformanceBenchmark()
    benchmark.run_all_benchmarks()
    benchmark.generate_report()


if __name__ == "__main__":
    main()
