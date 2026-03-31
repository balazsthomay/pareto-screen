"""Single-objective greedy selection baseline."""

from __future__ import annotations

import torch


class GreedySelector:
    """Selects molecules with the best value for a single objective.

    Requires pool_Y (all objective values for the full pool) to rank candidates.
    This baseline has access to the true objective values — it represents
    the strategy of greedily picking the "best" molecules by one criterion
    without considering trade-offs.
    """

    def __init__(self, objective_index: int = 0, pool_Y: torch.Tensor | None = None):
        self.objective_index = objective_index
        self._pool_Y = pool_Y

    def select_batch(
        self,
        pool_X: torch.Tensor,
        observed_indices: torch.Tensor,
        observed_Y: torch.Tensor | None,
        batch_size: int,
    ) -> torch.Tensor:
        if self._pool_Y is None:
            raise ValueError("GreedySelector requires pool_Y to be set")

        n_pool = pool_X.shape[0]
        mask = torch.ones(n_pool, dtype=torch.bool)
        mask[observed_indices] = False
        candidates = mask.nonzero(as_tuple=True)[0]

        # Rank by the target objective (descending — assumes maximization)
        obj_values = self._pool_Y[candidates, self.objective_index]
        top = torch.argsort(obj_values, descending=True)[:batch_size]
        return candidates[top]
