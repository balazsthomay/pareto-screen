"""Non-dominated sorting (Pareto ranking) selection baseline."""

from __future__ import annotations

import torch

from pareto_screen.bo.pareto import is_pareto_optimal


class ParetoRankingSelector:
    """Selects molecules using non-dominated sorting on known objectives.

    Ranks all unobserved molecules by Pareto front membership. Selects from
    the first (best) front first, then the second front, etc.

    This baseline handles multiple objectives but has NO uncertainty awareness —
    it is the key comparison for showing that BO's uncertainty modeling adds value.
    """

    def __init__(self, pool_Y: torch.Tensor):
        self._pool_Y = pool_Y

    def select_batch(
        self,
        pool_X: torch.Tensor,
        observed_indices: torch.Tensor,
        observed_Y: torch.Tensor | None,
        batch_size: int,
    ) -> torch.Tensor:
        n_pool = pool_X.shape[0]
        mask = torch.ones(n_pool, dtype=torch.bool)
        mask[observed_indices] = False
        candidates = mask.nonzero(as_tuple=True)[0]

        if len(candidates) == 0:
            return torch.tensor([], dtype=torch.long)

        # Non-dominated sorting: iteratively find Pareto fronts
        selected: list[int] = []
        remaining_Y = self._pool_Y[candidates].clone()
        remaining_local_idx = torch.arange(len(candidates))

        while len(selected) < batch_size and len(remaining_local_idx) > 0:
            pareto_mask = is_pareto_optimal(remaining_Y)
            front_local = remaining_local_idx[pareto_mask]

            needed = batch_size - len(selected)
            if len(front_local) <= needed:
                selected.extend(candidates[front_local].tolist())
            else:
                # Take a subset from this front (first `needed` by index)
                selected.extend(candidates[front_local[:needed]].tolist())

            # Remove this front from remaining
            keep = ~pareto_mask
            remaining_Y = remaining_Y[keep]
            remaining_local_idx = remaining_local_idx[keep]

        return torch.tensor(selected[:batch_size], dtype=torch.long)
