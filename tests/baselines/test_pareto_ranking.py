"""Tests for Pareto ranking selection baseline."""

import torch
import pytest

from pareto_screen.baselines.pareto_ranking import ParetoRankingSelector
from pareto_screen.types import SelectionStrategy


class TestParetoRankingSelector:
    def test_implements_protocol(self):
        pool_Y = torch.randn(10, 2, dtype=torch.double)
        assert isinstance(ParetoRankingSelector(pool_Y=pool_Y), SelectionStrategy)

    def test_selects_from_pareto_front(self):
        # Clear Pareto structure: (3,1), (2,2), (1,3) are Pareto, (0,0) is dominated
        pool_Y = torch.tensor([
            [3.0, 1.0],
            [2.0, 2.0],
            [1.0, 3.0],
            [0.0, 0.0],
        ], dtype=torch.double)
        selector = ParetoRankingSelector(pool_Y=pool_Y)
        pool_X = torch.randn(4, 3, dtype=torch.double)
        observed = torch.tensor([0])
        batch = selector.select_batch(pool_X, observed, None, batch_size=2)
        assert batch.shape == (2,)
        # Should select from Pareto front (indices 1, 2), not dominated (3)
        for idx in batch:
            assert idx.item() in {1, 2}

    def test_no_duplicates(self):
        pool_Y = torch.randn(20, 2, dtype=torch.double)
        selector = ParetoRankingSelector(pool_Y=pool_Y)
        pool_X = torch.randn(20, 3, dtype=torch.double)
        observed = torch.arange(5)
        batch = selector.select_batch(pool_X, observed, None, batch_size=5)
        assert len(set(batch.tolist())) == 5
        for idx in batch:
            assert idx.item() not in set(observed.tolist())
