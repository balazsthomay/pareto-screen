"""Tests for evaluation metrics."""

import torch
import pytest

from pareto_screen.evaluation.metrics import (
    hypervolume_indicator,
    pareto_coverage,
    selection_efficiency,
)


class TestHypervolumeIndicator:
    def test_single_point(self):
        Y = torch.tensor([[2.0, 3.0]])
        ref = torch.tensor([0.0, 0.0])
        assert hypervolume_indicator(Y, ref) == pytest.approx(6.0)

    def test_empty(self):
        Y = torch.empty(0, 2)
        ref = torch.tensor([0.0, 0.0])
        assert hypervolume_indicator(Y, ref) == pytest.approx(0.0)


class TestParetoCoverage:
    def test_full_coverage(self):
        true_pareto = torch.tensor([[1.0, 3.0], [2.0, 2.0], [3.0, 1.0]])
        discovered = true_pareto.clone()
        assert pareto_coverage(discovered, true_pareto) == pytest.approx(1.0)

    def test_no_coverage(self):
        true_pareto = torch.tensor([[10.0, 10.0]])
        discovered = torch.tensor([[0.0, 0.0]])
        assert pareto_coverage(discovered, true_pareto) == pytest.approx(0.0)

    def test_partial_coverage(self):
        true_pareto = torch.tensor([[1.0, 3.0], [2.0, 2.0], [3.0, 1.0]])
        # Discover one of three
        discovered = torch.tensor([[1.0, 3.0]])
        cov = pareto_coverage(discovered, true_pareto)
        assert 0.0 < cov < 1.0


class TestSelectionEfficiency:
    def test_reaches_target(self):
        hv_curve = [0.0, 5.0, 8.0, 9.5, 10.0]
        true_hv = 10.0
        eff = selection_efficiency(hv_curve, true_hv, target_fraction=0.9)
        assert eff == 3  # First index where HV >= 9.0 is index 3

    def test_never_reaches(self):
        hv_curve = [0.0, 1.0, 2.0]
        true_hv = 100.0
        assert selection_efficiency(hv_curve, true_hv, target_fraction=0.9) is None

    def test_immediately_reached(self):
        hv_curve = [10.0, 10.0]
        true_hv = 10.0
        assert selection_efficiency(hv_curve, true_hv, target_fraction=0.9) == 0
