"""Benchmark runner: compares strategies on the same dataset."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from pareto_screen.bo.pareto import compute_hypervolume, pareto_frontier
from pareto_screen.evaluation.oracle import Oracle
from pareto_screen.types import SelectionStrategy


@dataclass
class BenchmarkResult:
    """Results from a full benchmark run across multiple strategies and repeats."""

    strategy_names: list[str]
    hypervolume_curves: dict[str, list[list[float]]] = field(default_factory=dict)
    pareto_coverages: dict[str, list[float]] = field(default_factory=dict)
    true_hypervolume: float = 0.0


class BenchmarkRunner:
    """Runs all strategies on the same dataset and collects comparable results."""

    def __init__(
        self,
        pool_X: torch.Tensor,
        oracle: Oracle,
        strategies: dict[str, SelectionStrategy],
        n_iterations: int = 50,
        batch_size: int = 5,
        n_initial: int = 20,
        n_repeats: int = 5,
        seed: int = 42,
    ):
        self.pool_X = pool_X
        self.oracle = oracle
        self.strategies = strategies
        self.n_iterations = n_iterations
        self.batch_size = batch_size
        self.n_initial = n_initial
        self.n_repeats = n_repeats
        self.seed = seed

    def run(self) -> BenchmarkResult:
        """Run all strategies for n_repeats with different initial samples."""
        result = BenchmarkResult(
            strategy_names=list(self.strategies.keys()),
            true_hypervolume=self.oracle.true_pareto_hypervolume,
        )

        for name in self.strategies:
            result.hypervolume_curves[name] = []
            result.pareto_coverages[name] = []

        for repeat in range(self.n_repeats):
            # Generate shared initial sample for this repeat
            gen = torch.Generator()
            gen.manual_seed(self.seed + repeat)
            initial_perm = torch.randperm(self.oracle.pool_size, generator=gen)
            initial_indices = initial_perm[: self.n_initial]

            for name, strategy in self.strategies.items():
                hv_curve = self._run_strategy(strategy, initial_indices)
                result.hypervolume_curves[name].append(hv_curve)

        return result

    def _run_strategy(
        self,
        strategy: SelectionStrategy,
        initial_indices: torch.Tensor,
    ) -> list[float]:
        """Run a single strategy for one repeat."""
        ref_point = self.oracle.ref_point
        observed_indices = initial_indices.clone()
        observed_Y = self.oracle._Y[observed_indices]
        hv_curve: list[float] = []

        for _ in range(self.n_iterations):
            # Record hypervolume
            front = pareto_frontier(observed_Y)
            hv = compute_hypervolume(front, ref_point)
            hv_curve.append(hv)

            # Select next batch
            batch = strategy.select_batch(
                self.pool_X, observed_indices, observed_Y, self.batch_size
            )

            # Reveal and update
            new_Y = self.oracle._Y[batch]
            observed_indices = torch.cat([observed_indices, batch])
            observed_Y = torch.cat([observed_Y, new_Y])

        return hv_curve
