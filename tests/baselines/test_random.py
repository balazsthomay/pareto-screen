"""Tests for random selection baseline."""

import torch
import pytest

from pareto_screen.baselines.random import RandomSelector
from pareto_screen.types import SelectionStrategy


class TestRandomSelector:
    def test_implements_protocol(self):
        assert isinstance(RandomSelector(seed=42), SelectionStrategy)

    def test_select_batch_size(self):
        selector = RandomSelector(seed=42)
        pool_X = torch.randn(20, 5, dtype=torch.double)
        observed = torch.tensor([0, 1, 2])
        batch = selector.select_batch(pool_X, observed, None, batch_size=3)
        assert batch.shape == (3,)

    def test_no_duplicates_with_observed(self):
        selector = RandomSelector(seed=42)
        pool_X = torch.randn(20, 5, dtype=torch.double)
        observed = torch.arange(10)
        batch = selector.select_batch(pool_X, observed, None, batch_size=5)
        # Selected indices should not overlap with observed
        for idx in batch:
            assert idx.item() not in set(observed.tolist())

    def test_deterministic_with_seed(self):
        pool_X = torch.randn(20, 5, dtype=torch.double)
        observed = torch.tensor([0, 1])
        b1 = RandomSelector(seed=42).select_batch(pool_X, observed, None, 3)
        b2 = RandomSelector(seed=42).select_batch(pool_X, observed, None, 3)
        assert torch.equal(b1, b2)
