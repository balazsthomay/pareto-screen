"""GP surrogate model construction and fitting."""

from __future__ import annotations

import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models.gp_regression import SingleTaskGP
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
from gpytorch.mlls import SumMarginalLogLikelihood


def build_model(train_X: torch.Tensor, train_Y: torch.Tensor) -> ModelListGP:
    """Build a ModelListGP with one SingleTaskGP per objective.

    Args:
        train_X: Training inputs, shape [n, d].
        train_Y: Training objectives, shape [n, m].

    Returns:
        ModelListGP wrapping independent per-objective GPs.
    """
    models = []
    for i in range(train_Y.shape[1]):
        model = SingleTaskGP(
            train_X,
            train_Y[:, i : i + 1],
            input_transform=Normalize(d=train_X.shape[-1]),
            outcome_transform=Standardize(m=1),
        )
        models.append(model)
    return ModelListGP(*models)


def fit_model(model: ModelListGP) -> ModelListGP:
    """Fit GP hyperparameters via marginal likelihood optimization.

    Returns the same model (mutated in place).
    """
    mll = SumMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)
    return model
