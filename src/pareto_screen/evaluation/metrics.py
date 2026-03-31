"""Evaluation metrics for multi-objective optimization."""

from __future__ import annotations

import torch

from pareto_screen.bo.pareto import compute_hypervolume


def hypervolume_indicator(Y: torch.Tensor, ref_point: torch.Tensor) -> float:
    """Compute hypervolume of the dominated region."""
    return compute_hypervolume(Y, ref_point)


def pareto_coverage(
    discovered_Y: torch.Tensor,
    true_pareto_Y: torch.Tensor,
    epsilon: float = 0.05,
) -> float:
    """Fraction of true Pareto points epsilon-dominated by discovered points.

    A true Pareto point p is "covered" if there exists a discovered point d
    such that d >= p - epsilon * range for all objectives.
    """
    if true_pareto_Y.shape[0] == 0:
        return 1.0
    if discovered_Y.shape[0] == 0:
        return 0.0

    y_range = true_pareto_Y.max(dim=0).values - true_pareto_Y.min(dim=0).values
    y_range = torch.clamp(y_range, min=1e-8)
    tolerance = epsilon * y_range

    covered = 0
    for p in true_pareto_Y:
        # Check if any discovered point epsilon-dominates p
        diffs = discovered_Y - p.unsqueeze(0) + tolerance.unsqueeze(0)
        if (diffs >= 0).all(dim=1).any():
            covered += 1

    return covered / true_pareto_Y.shape[0]


def selection_efficiency(
    hypervolumes: list[float],
    true_hv: float,
    target_fraction: float = 0.9,
) -> int | None:
    """Number of evaluations to reach target_fraction of true hypervolume.

    Returns None if the target was never reached.
    """
    target = target_fraction * true_hv
    for i, hv in enumerate(hypervolumes):
        if hv >= target:
            return i
    return None
