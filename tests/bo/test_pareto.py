"""Tests for Pareto frontier utilities."""

import pytest
import torch

from pareto_screen.bo.pareto import compute_hypervolume, is_pareto_optimal, pareto_frontier


class TestIsParetoOptimal:
    def test_2d_simple(self):
        # Points: (1,3), (2,2), (3,1), (0,0)
        # Pareto front: (1,3), (2,2), (3,1) — each dominates (0,0)
        Y = torch.tensor([[1.0, 3.0], [2.0, 2.0], [3.0, 1.0], [0.0, 0.0]])
        mask = is_pareto_optimal(Y)
        assert mask.tolist() == [True, True, True, False]

    def test_all_pareto(self):
        Y = torch.tensor([[1.0, 2.0], [2.0, 1.0]])
        mask = is_pareto_optimal(Y)
        assert mask.tolist() == [True, True]

    def test_single_point(self):
        Y = torch.tensor([[1.0, 2.0]])
        mask = is_pareto_optimal(Y)
        assert mask.tolist() == [True]

    def test_dominated_point(self):
        # (1,1) is dominated by (2,2)
        Y = torch.tensor([[2.0, 2.0], [1.0, 1.0]])
        mask = is_pareto_optimal(Y)
        assert mask.tolist() == [True, False]

    def test_3d(self):
        Y = torch.tensor([
            [3.0, 1.0, 1.0],
            [1.0, 3.0, 1.0],
            [1.0, 1.0, 3.0],
            [0.0, 0.0, 0.0],
        ])
        mask = is_pareto_optimal(Y)
        assert mask.tolist() == [True, True, True, False]


class TestComputeHypervolume:
    def test_2d_known(self):
        # Single point (2, 3), ref_point (0, 0) -> HV = 2*3 = 6
        Y = torch.tensor([[2.0, 3.0]])
        ref_point = torch.tensor([0.0, 0.0])
        hv = compute_hypervolume(Y, ref_point)
        assert hv == pytest.approx(6.0)

    def test_2d_two_points(self):
        # Points (1,3), (3,1), ref (0,0)
        # HV = 1*3 + (3-1)*1 = 3 + 2 = 5... actually:
        # The dominated area = 1*3 + 2*1 = 5
        Y = torch.tensor([[1.0, 3.0], [3.0, 1.0]])
        ref_point = torch.tensor([0.0, 0.0])
        hv = compute_hypervolume(Y, ref_point)
        assert hv == pytest.approx(5.0)

    def test_empty_returns_zero(self):
        Y = torch.empty(0, 2)
        ref_point = torch.tensor([0.0, 0.0])
        hv = compute_hypervolume(Y, ref_point)
        assert hv == pytest.approx(0.0)


class TestParetoFrontier:
    def test_returns_pareto_points(self):
        Y = torch.tensor([[1.0, 3.0], [2.0, 2.0], [3.0, 1.0], [0.0, 0.0]])
        front = pareto_frontier(Y)
        assert front.shape == (3, 2)

    def test_single_point(self):
        Y = torch.tensor([[1.0, 2.0]])
        front = pareto_frontier(Y)
        assert front.shape == (1, 2)
