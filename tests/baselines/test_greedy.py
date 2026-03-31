"""Tests for greedy selection baseline."""

import torch
import pytest

from pareto_screen.baselines.greedy import GreedySelector
from pareto_screen.types import SelectionStrategy


class TestGreedySelector:
    def test_implements_protocol(self):
        assert isinstance(GreedySelector(objective_index=0), SelectionStrategy)

    def test_selects_best_by_objective(self):
        pool_Y = torch.tensor([
            [0.1, 0.5],
            [0.2, 0.3],
            [0.9, 0.1],  # best obj0 among unobserved
            [0.3, 0.8],
        ], dtype=torch.double)
        selector = GreedySelector(objective_index=0, pool_Y=pool_Y)
        pool_X = torch.randn(4, 3, dtype=torch.double)
        observed = torch.tensor([0, 1])
        batch = selector.select_batch(pool_X, observed, None, batch_size=1)
        assert batch.shape == (1,)
        assert batch[0].item() == 2  # index 2 has obj0=0.9

    def test_selects_top_k(self):
        selector = GreedySelector(objective_index=0, pool_Y=torch.tensor([
            [5.0, 1.0],
            [3.0, 2.0],
            [4.0, 3.0],
            [1.0, 4.0],
            [2.0, 5.0],
        ], dtype=torch.double))
        pool_X = torch.randn(5, 3, dtype=torch.double)
        observed = torch.tensor([0])  # index 0 has obj0=5.0, already observed
        batch = selector.select_batch(pool_X, observed, None, batch_size=2)
        # Should pick index 2 (obj0=4.0) and index 1 (obj0=3.0)
        assert set(batch.tolist()) == {2, 1}
