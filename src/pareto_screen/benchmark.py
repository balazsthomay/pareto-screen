"""Runnable benchmark: BO vs baselines on EGFR data with plots.

Usage: uv run python -m pareto_screen.benchmark [--n-molecules 500] [--n-iterations 20]
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from pareto_screen.baselines.greedy import GreedySelector
from pareto_screen.baselines.pareto_ranking import ParetoRankingSelector
from pareto_screen.baselines.random import RandomSelector
from pareto_screen.bo.pareto import pareto_frontier
from pareto_screen.bo.selector import BOSelector
from pareto_screen.data.chembl import load_chembl_activities
from pareto_screen.data.dataset import MoleculeDataset
from pareto_screen.evaluation.benchmark import BenchmarkRunner
from pareto_screen.evaluation.oracle import Oracle
from pareto_screen.types import ObjectiveConfig


OBJECTIVE_CONFIGS = [
    ObjectiveConfig(name="qed", direction="maximize"),
    ObjectiveConfig(name="sa_score", direction="minimize"),
    ObjectiveConfig(name="logp", direction="maximize"),
    ObjectiveConfig(name="pic50", direction="maximize"),
]


def load_data(n_molecules: int, seed: int = 42) -> MoleculeDataset:
    """Load EGFR data and build dataset."""
    print("Loading EGFR bioactivity data from ChEMBL REST API...")
    df = load_chembl_activities("CHEMBL203")
    print(f"  Loaded {len(df)} EGFR compounds")

    df_subset = df.sample(n_molecules, random_state=seed).reset_index(drop=True)
    print(f"  Sampled {len(df_subset)} for benchmark")

    print("Computing molecular properties and featurizing...")
    dataset = MoleculeDataset.from_dataframe(
        df_subset, OBJECTIVE_CONFIGS, n_pca_components=20
    )
    print(f"  Dataset ready: {len(dataset)} molecules, {len(OBJECTIVE_CONFIGS)} objectives")
    return dataset


def run_benchmark(
    dataset: MoleculeDataset,
    n_iterations: int = 20,
    batch_size: int = 5,
    n_initial: int = 20,
    n_repeats: int = 3,
    seed: int = 42,
) -> dict:
    """Run all strategies and return results."""
    X, Y = dataset.get_tensors()
    oracle = Oracle(Y=Y, objective_configs=OBJECTIVE_CONFIGS)

    strategies = {
        "Random": RandomSelector(seed=seed),
        "Greedy (QED)": GreedySelector(objective_index=0, pool_Y=Y),
        "Pareto Ranking": ParetoRankingSelector(pool_Y=Y),
        "BO (qLogNEHVI)": BOSelector(
            objective_configs=OBJECTIVE_CONFIGS,
            ref_point=oracle.ref_point,
        ),
    }

    print(f"\nRunning benchmark: {n_repeats} repeats x {n_iterations} iterations x batch {batch_size}")
    print(f"  Initial sample: {n_initial}, Total evaluated per run: {n_initial + n_iterations * batch_size}")
    print(f"  True Pareto HV: {oracle.true_pareto_hypervolume:.4f}\n")

    start = time.time()
    runner = BenchmarkRunner(
        pool_X=X,
        oracle=oracle,
        strategies=strategies,
        n_iterations=n_iterations,
        batch_size=batch_size,
        n_initial=n_initial,
        n_repeats=n_repeats,
        seed=seed,
    )
    result = runner.run()
    elapsed = time.time() - start
    print(f"  Completed in {elapsed:.1f}s\n")

    return {
        "result": result,
        "X": X,
        "Y": Y,
        "oracle": oracle,
        "n_initial": n_initial,
        "batch_size": batch_size,
    }


def print_summary(data: dict) -> None:
    """Print results table."""
    result = data["result"]
    true_hv = result.true_hypervolume

    print("=" * 70)
    print(f"{'Strategy':<20s}  {'HV@mid':>10s}  {'HV@final':>10s}  {'% of true':>10s}")
    print("-" * 70)
    for name in result.strategy_names:
        curves = result.hypervolume_curves[name]
        mid_idx = len(curves[0]) // 2
        mid_hvs = [c[mid_idx] for c in curves]
        final_hvs = [c[-1] for c in curves]
        avg_mid = np.mean(mid_hvs)
        avg_final = np.mean(final_hvs)
        print(f"{name:<20s}  {avg_mid:>10.2f}  {avg_final:>10.2f}  {avg_final / true_hv * 100:>9.1f}%")
    print("=" * 70)
    print(f"True Pareto HV: {true_hv:.4f}")
    print(f"Pool size: {data['Y'].shape[0]}, Objectives: {data['Y'].shape[1]}")


def plot_hypervolume_curves(data: dict, output_dir: Path) -> None:
    """Plot hypervolume progression over iterations."""
    result = data["result"]
    n_initial = data["n_initial"]
    batch_size = data["batch_size"]

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = {"Random": "#999999", "Greedy (QED)": "#e69f00", "Pareto Ranking": "#56b4e9", "BO (qLogNEHVI)": "#d55e00"}

    for name in result.strategy_names:
        curves = np.array(result.hypervolume_curves[name])
        n_iter = curves.shape[1]
        x = np.arange(n_iter)
        evaluations = n_initial + (x + 1) * batch_size

        mean = curves.mean(axis=0)
        std = curves.std(axis=0)
        color = colors.get(name, "#333333")

        ax.plot(evaluations, mean, label=name, color=color, linewidth=2)
        ax.fill_between(evaluations, mean - std, mean + std, alpha=0.15, color=color)

    # True HV reference line
    ax.axhline(y=result.true_hypervolume, color="black", linestyle="--", linewidth=1, label="True Pareto HV")

    ax.set_xlabel("Molecules Evaluated", fontsize=12)
    ax.set_ylabel("Hypervolume", fontsize=12)
    ax.set_title("Multi-Objective BO for Compound Selection (EGFR, 4 objectives)", fontsize=13)
    ax.legend(fontsize=10, loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    path = output_dir / "hypervolume_curves.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_pareto_frontier_2d(data: dict, output_dir: Path) -> None:
    """Plot 2D projections of the Pareto frontier (QED vs pIC50, QED vs SA)."""
    Y = data["Y"]
    obj_names = ["QED", "-SA Score", "LogP", "pIC50"]

    # Two 2D projections: QED vs pIC50, QED vs SA
    pairs = [(0, 3, "QED", "pIC50"), (0, 1, "QED", "-SA Score (higher=better)")]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, (i, j, xlabel, ylabel) in zip(axes, pairs):
        y_2d = Y[:, [i, j]]
        front = pareto_frontier(y_2d)

        ax.scatter(Y[:, i].numpy(), Y[:, j].numpy(), alpha=0.3, s=15, color="#999999", label="All molecules")
        ax.scatter(front[:, 0].numpy(), front[:, 1].numpy(), s=30, color="#d55e00", zorder=5, label="Pareto front")

        # Sort front for line
        sort_idx = front[:, 0].argsort()
        ax.plot(front[sort_idx, 0].numpy(), front[sort_idx, 1].numpy(), color="#d55e00", alpha=0.5, linewidth=1)

        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Pareto Frontiers (2D Projections, EGFR Dataset)", fontsize=13)
    fig.tight_layout()

    path = output_dir / "pareto_frontiers.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run pareto-screen benchmark on EGFR data")
    parser.add_argument("--n-molecules", type=int, default=500, help="Number of molecules to sample")
    parser.add_argument("--n-iterations", type=int, default=20, help="BO iterations")
    parser.add_argument("--n-repeats", type=int, default=3, help="Number of random seed repeats")
    parser.add_argument("--batch-size", type=int, default=5, help="Molecules selected per iteration")
    parser.add_argument("--n-initial", type=int, default=20, help="Initial random sample size")
    parser.add_argument("--output-dir", type=str, default="results", help="Directory for output plots")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    dataset = load_data(args.n_molecules, args.seed)
    data = run_benchmark(
        dataset,
        n_iterations=args.n_iterations,
        batch_size=args.batch_size,
        n_initial=args.n_initial,
        n_repeats=args.n_repeats,
        seed=args.seed,
    )

    print_summary(data)

    print("\nGenerating plots...")
    plot_hypervolume_curves(data, output_dir)
    plot_pareto_frontier_2d(data, output_dir)
    print("\nDone.")


if __name__ == "__main__":
    main()
