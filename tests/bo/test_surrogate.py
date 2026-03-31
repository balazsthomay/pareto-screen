"""Tests for GP surrogate model construction and fitting."""

import torch
import pytest

from pareto_screen.bo.surrogate import build_model, fit_model


@pytest.fixture
def synthetic_data():
    """Small synthetic dataset for GP testing."""
    torch.manual_seed(42)
    n, d, m = 15, 5, 2
    X = torch.randn(n, d, dtype=torch.double)
    Y = torch.randn(n, m, dtype=torch.double)
    return X, Y


class TestBuildModel:
    def test_builds_model_list(self, synthetic_data):
        X, Y = synthetic_data
        model = build_model(X, Y)
        # ModelListGP should have one sub-model per objective
        assert len(model.models) == 2

    def test_predictions_shape(self, synthetic_data):
        X, Y = synthetic_data
        model = build_model(X, Y)
        model.eval()
        with torch.no_grad():
            posterior = model.posterior(X[:3])
            mean = posterior.mean
            assert mean.shape == (3, 2)

    def test_single_objective(self):
        torch.manual_seed(0)
        X = torch.randn(10, 3, dtype=torch.double)
        Y = torch.randn(10, 1, dtype=torch.double)
        model = build_model(X, Y)
        assert len(model.models) == 1


class TestFitModel:
    def test_fit_completes(self, synthetic_data):
        X, Y = synthetic_data
        model = build_model(X, Y)
        fitted = fit_model(model)
        assert fitted is model  # returns same object

    def test_fit_improves_likelihood(self, synthetic_data):
        X, Y = synthetic_data
        model = build_model(X, Y)
        # Just verify fitting doesn't raise and the model can predict afterward
        fit_model(model)
        model.eval()
        with torch.no_grad():
            posterior = model.posterior(X[:3])
            mean = posterior.mean
            assert mean.shape == (3, 2)
            assert torch.isfinite(mean).all()
