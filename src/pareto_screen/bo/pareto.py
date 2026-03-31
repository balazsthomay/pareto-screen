"""Pareto frontier utilities and hypervolume computation."""

from __future__ import annotations

import torch
from botorch.utils.multi_objective.box_decompositions.dominated import (
    DominatedPartitioning,
)


def is_pareto_optimal(Y: torch.Tensor) -> torch.Tensor:
    """Return boolean mask of Pareto-optimal points (assuming maximization).

    A point is Pareto-optimal if no other point is strictly better in all objectives.

    Args:
        Y: Objective values, shape [n, m].

    Returns:
        Boolean tensor of shape [n].
    """
    n = Y.shape[0]
    is_optimal = torch.ones(n, dtype=torch.bool)
    for i in range(n):
        if not is_optimal[i]:
            continue
        for j in range(n):
            if i == j:
                continue
            # j dominates i if j >= i in all objectives and j > i in at least one
            if (Y[j] >= Y[i]).all() and (Y[j] > Y[i]).any():
                is_optimal[i] = False
                break
    return is_optimal


def compute_hypervolume(Y: torch.Tensor, ref_point: torch.Tensor) -> float:
    """Compute hypervolume indicator using BoTorch's DominatedPartitioning.

    Args:
        Y: Objective values of Pareto-optimal points, shape [n, m].
        ref_point: Reference point, shape [m].

    Returns:
        Hypervolume of the dominated region.
    """
    if Y.shape[0] == 0:
        return 0.0

    Y = Y.to(torch.double)
    ref_point = ref_point.to(torch.double)

    partitioning = DominatedPartitioning(ref_point=ref_point, Y=Y)
    return partitioning.compute_hypervolume().item()


def pareto_frontier(Y: torch.Tensor) -> torch.Tensor:
    """Return the Pareto-optimal subset of Y."""
    mask = is_pareto_optimal(Y)
    return Y[mask]
