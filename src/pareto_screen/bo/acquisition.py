"""Multi-objective acquisition function construction and evaluation."""

from __future__ import annotations

import torch
from botorch.acquisition.multi_objective.logei import (
    qLogNoisyExpectedHypervolumeImprovement,
)
from botorch.models.model_list_gp_regression import ModelListGP


def build_acquisition(
    model: ModelListGP,
    train_X: torch.Tensor,
    train_Y: torch.Tensor,
    ref_point: torch.Tensor,
) -> qLogNoisyExpectedHypervolumeImprovement:
    """Construct qLogNEHVI acquisition function.

    Args:
        model: Fitted ModelListGP.
        train_X: Training inputs, shape [n, d].
        train_Y: Training objectives, shape [n, m].
        ref_point: Reference point for hypervolume, shape [m].

    Returns:
        qLogNoisyExpectedHypervolumeImprovement instance.
    """
    return qLogNoisyExpectedHypervolumeImprovement(
        model=model,
        ref_point=ref_point,
        X_baseline=train_X,
        cache_root=False,
    )


def evaluate_candidates(
    acq_func: qLogNoisyExpectedHypervolumeImprovement,
    candidate_X: torch.Tensor,
) -> torch.Tensor:
    """Evaluate acquisition function at all candidate points.

    Args:
        acq_func: The acquisition function.
        candidate_X: Candidate features, shape [n_candidates, d].

    Returns:
        Acquisition values, shape [n_candidates].
    """
    # acq_func expects shape [batch, q, d] — we evaluate each candidate independently
    with torch.no_grad():
        values = acq_func(candidate_X.unsqueeze(1))
    return values
