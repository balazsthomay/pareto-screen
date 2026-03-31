"""Tests for acquisition function construction and candidate evaluation."""

import torch
import pytest

from pareto_screen.bo.acquisition import build_acquisition, evaluate_candidates
from pareto_screen.bo.surrogate import build_model, fit_model


@pytest.fixture
def fitted_model():
    torch.manual_seed(42)
    n, d, m = 15, 5, 2
    X = torch.randn(n, d, dtype=torch.double)
    Y = torch.randn(n, m, dtype=torch.double)
    model = build_model(X, Y)
    fit_model(model)
    return model, X, Y


class TestBuildAcquisition:
    def test_builds_acquisition(self, fitted_model):
        model, X, Y = fitted_model
        ref_point = Y.min(dim=0).values - 0.1
        acq = build_acquisition(model, X, Y, ref_point)
        assert acq is not None

    def test_acquisition_evaluates(self, fitted_model):
        model, X, Y = fitted_model
        ref_point = Y.min(dim=0).values - 0.1
        acq = build_acquisition(model, X, Y, ref_point)
        # Evaluate at a single candidate
        test_X = torch.randn(1, 1, 5, dtype=torch.double)
        with torch.no_grad():
            val = acq(test_X)
        assert torch.isfinite(val).all()


class TestEvaluateCandidates:
    def test_returns_values_for_all_candidates(self, fitted_model):
        model, X, Y = fitted_model
        ref_point = Y.min(dim=0).values - 0.1
        acq = build_acquisition(model, X, Y, ref_point)
        candidates = torch.randn(10, 5, dtype=torch.double)
        values = evaluate_candidates(acq, candidates)
        assert values.shape == (10,)
        assert torch.isfinite(values).all()

    def test_values_vary(self, fitted_model):
        """Different candidates should generally get different acquisition values."""
        model, X, Y = fitted_model
        ref_point = Y.min(dim=0).values - 0.1
        acq = build_acquisition(model, X, Y, ref_point)
        candidates = torch.randn(10, 5, dtype=torch.double)
        values = evaluate_candidates(acq, candidates)
        # Not all values should be identical
        assert values.std() > 0
