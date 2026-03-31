"""Tests for the Bayesian optimization loop."""

import torch
import pytest

from pareto_screen.bo.optimizer import BayesianOptimizer
from pareto_screen.types import ObjectiveConfig


@pytest.fixture
def synthetic_pool():
    """Synthetic pool with known Pareto structure.

    Creates a 2-objective problem where the Pareto front is along the diagonal:
    points where obj1 + obj2 is maximized, with a tradeoff between the two.
    """
    torch.manual_seed(42)
    n, d = 50, 5
    X = torch.randn(n, d, dtype=torch.double)
    # Objectives correlated with first two feature dimensions
    Y = torch.zeros(n, 2, dtype=torch.double)
    Y[:, 0] = X[:, 0] + 0.1 * torch.randn(n, dtype=torch.double)
    Y[:, 1] = X[:, 1] + 0.1 * torch.randn(n, dtype=torch.double)
    return X, Y


@pytest.fixture
def objective_configs_2d():
    return [
        ObjectiveConfig(name="obj1", direction="maximize"),
        ObjectiveConfig(name="obj2", direction="maximize"),
    ]


class TestBayesianOptimizer:
    def test_run_completes(self, synthetic_pool, objective_configs_2d):
        X, Y = synthetic_pool
        optimizer = BayesianOptimizer(
            objective_configs=objective_configs_2d,
            batch_size=2,
            n_initial=10,
            n_iterations=3,
            seed=42,
        )
        result = optimizer.run(X, Y)
        assert result is not None
        assert len(result.selected_indices) == 3
        assert len(result.hypervolumes) == 3

    def test_hypervolume_nondecreasing(self, synthetic_pool, objective_configs_2d):
        X, Y = synthetic_pool
        optimizer = BayesianOptimizer(
            objective_configs=objective_configs_2d,
            batch_size=2,
            n_initial=10,
            n_iterations=5,
            seed=42,
        )
        result = optimizer.run(X, Y)
        # Hypervolume should be non-decreasing (adding points can't shrink it)
        for i in range(1, len(result.hypervolumes)):
            assert result.hypervolumes[i] >= result.hypervolumes[i - 1] - 1e-6

    def test_no_duplicate_selections(self, synthetic_pool, objective_configs_2d):
        X, Y = synthetic_pool
        optimizer = BayesianOptimizer(
            objective_configs=objective_configs_2d,
            batch_size=2,
            n_initial=10,
            n_iterations=5,
            seed=42,
        )
        result = optimizer.run(X, Y)
        all_indices = []
        for batch in result.selected_indices:
            all_indices.extend(batch)
        assert len(all_indices) == len(set(all_indices))

    def test_observed_y_shape(self, synthetic_pool, objective_configs_2d):
        X, Y = synthetic_pool
        optimizer = BayesianOptimizer(
            objective_configs=objective_configs_2d,
            batch_size=2,
            n_initial=10,
            n_iterations=3,
            seed=42,
        )
        result = optimizer.run(X, Y)
        # 10 initial + 3*2 = 16 total observed
        assert result.observed_Y.shape == (16, 2)

    def test_seed_determinism(self, synthetic_pool, objective_configs_2d):
        X, Y = synthetic_pool
        results = []
        for _ in range(2):
            optimizer = BayesianOptimizer(
                objective_configs=objective_configs_2d,
                batch_size=2,
                n_initial=10,
                n_iterations=3,
                seed=42,
            )
            results.append(optimizer.run(X, Y))
        assert results[0].selected_indices == results[1].selected_indices
