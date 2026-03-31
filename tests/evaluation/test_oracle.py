"""Tests for the Oracle pattern."""

import torch
import pytest

from pareto_screen.evaluation.oracle import Oracle
from pareto_screen.types import ObjectiveConfig


@pytest.fixture
def oracle():
    Y = torch.tensor([
        [1.0, 3.0],
        [2.0, 2.0],
        [3.0, 1.0],
        [0.0, 0.0],
    ], dtype=torch.double)
    configs = [
        ObjectiveConfig(name="obj1", direction="maximize"),
        ObjectiveConfig(name="obj2", direction="maximize"),
    ]
    return Oracle(Y=Y, objective_configs=configs)


class TestOracle:
    def test_initial_state(self, oracle):
        assert oracle.n_revealed == 0

    def test_reveal(self, oracle):
        indices = torch.tensor([0, 2])
        values = oracle.reveal(indices)
        assert values.shape == (2, 2)
        assert oracle.n_revealed == 2

    def test_reveal_values_correct(self, oracle):
        indices = torch.tensor([1])
        values = oracle.reveal(indices)
        assert torch.allclose(values, torch.tensor([[2.0, 2.0]], dtype=torch.double))

    def test_reveal_tracks_state(self, oracle):
        oracle.reveal(torch.tensor([0]))
        oracle.reveal(torch.tensor([1]))
        assert oracle.n_revealed == 2

    def test_true_pareto_hypervolume(self, oracle):
        # Pareto front: (1,3), (2,2), (3,1). Ref point from data minus margin.
        hv = oracle.true_pareto_hypervolume
        assert hv > 0

    def test_ref_point(self, oracle):
        ref = oracle.ref_point
        assert ref.shape == (2,)
        # Should be below all points
        assert (ref < oracle._Y.min(dim=0).values).all()
