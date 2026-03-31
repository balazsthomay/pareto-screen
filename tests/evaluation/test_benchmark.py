"""Tests for the benchmark runner."""

import torch
import pytest

from pareto_screen.baselines.random import RandomSelector
from pareto_screen.baselines.greedy import GreedySelector
from pareto_screen.evaluation.benchmark import BenchmarkRunner, BenchmarkResult
from pareto_screen.evaluation.oracle import Oracle
from pareto_screen.types import ObjectiveConfig


@pytest.fixture
def benchmark_setup():
    torch.manual_seed(42)
    n, d, m = 30, 5, 2
    X = torch.randn(n, d, dtype=torch.double)
    Y = torch.zeros(n, m, dtype=torch.double)
    Y[:, 0] = X[:, 0] + 0.1 * torch.randn(n, dtype=torch.double)
    Y[:, 1] = X[:, 1] + 0.1 * torch.randn(n, dtype=torch.double)

    configs = [
        ObjectiveConfig(name="obj1", direction="maximize"),
        ObjectiveConfig(name="obj2", direction="maximize"),
    ]
    oracle = Oracle(Y=Y, objective_configs=configs)

    strategies = {
        "random": RandomSelector(seed=42),
        "greedy_obj0": GreedySelector(objective_index=0, pool_Y=Y),
    }
    return X, oracle, strategies, configs


class TestBenchmarkRunner:
    def test_run_completes(self, benchmark_setup):
        X, oracle, strategies, configs = benchmark_setup
        runner = BenchmarkRunner(
            pool_X=X,
            oracle=oracle,
            strategies=strategies,
            n_iterations=3,
            batch_size=2,
            n_initial=5,
            n_repeats=2,
            seed=42,
        )
        result = runner.run()
        assert isinstance(result, BenchmarkResult)

    def test_result_has_all_strategies(self, benchmark_setup):
        X, oracle, strategies, configs = benchmark_setup
        runner = BenchmarkRunner(
            pool_X=X,
            oracle=oracle,
            strategies=strategies,
            n_iterations=3,
            batch_size=2,
            n_initial=5,
            n_repeats=2,
            seed=42,
        )
        result = runner.run()
        assert set(result.strategy_names) == {"random", "greedy_obj0"}

    def test_hv_curves_shape(self, benchmark_setup):
        X, oracle, strategies, configs = benchmark_setup
        runner = BenchmarkRunner(
            pool_X=X,
            oracle=oracle,
            strategies=strategies,
            n_iterations=3,
            batch_size=2,
            n_initial=5,
            n_repeats=2,
            seed=42,
        )
        result = runner.run()
        for name in result.strategy_names:
            curves = result.hypervolume_curves[name]
            assert len(curves) == 2  # n_repeats
            for curve in curves:
                assert len(curve) == 3  # n_iterations
