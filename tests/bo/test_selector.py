"""Tests for BOSelector strategy."""

import torch
import pytest

from pareto_screen.bo.selector import BOSelector
from pareto_screen.types import ObjectiveConfig, SelectionStrategy


@pytest.fixture
def synthetic_setup():
    torch.manual_seed(42)
    n, d, m = 30, 5, 2
    X = torch.randn(n, d, dtype=torch.double)
    Y = torch.zeros(n, m, dtype=torch.double)
    Y[:, 0] = X[:, 0] + 0.1 * torch.randn(n, dtype=torch.double)
    Y[:, 1] = X[:, 1] + 0.1 * torch.randn(n, dtype=torch.double)
    ref_point = Y.min(dim=0).values - 0.1 * (Y.max(dim=0).values - Y.min(dim=0).values)
    configs = [
        ObjectiveConfig(name="obj1", direction="maximize"),
        ObjectiveConfig(name="obj2", direction="maximize"),
    ]
    return X, Y, ref_point, configs


class TestBOSelector:
    def test_implements_protocol(self):
        configs = [ObjectiveConfig(name="obj1", direction="maximize")]
        selector = BOSelector(configs, ref_point=torch.tensor([0.0]))
        assert isinstance(selector, SelectionStrategy)

    def test_select_batch_shape(self, synthetic_setup):
        X, Y, ref_point, configs = synthetic_setup
        selector = BOSelector(configs, ref_point=ref_point)
        observed = torch.arange(10)
        observed_Y = Y[:10]
        batch = selector.select_batch(X, observed, observed_Y, batch_size=3)
        assert batch.shape == (3,)

    def test_no_overlap_with_observed(self, synthetic_setup):
        X, Y, ref_point, configs = synthetic_setup
        selector = BOSelector(configs, ref_point=ref_point)
        observed = torch.arange(15)
        observed_Y = Y[:15]
        batch = selector.select_batch(X, observed, observed_Y, batch_size=5)
        for idx in batch:
            assert idx.item() not in set(range(15))

    def test_fallback_with_no_observed_y(self, synthetic_setup):
        X, Y, ref_point, configs = synthetic_setup
        selector = BOSelector(configs, ref_point=ref_point)
        observed = torch.arange(1)
        batch = selector.select_batch(X, observed, None, batch_size=3)
        assert batch.shape == (3,)
