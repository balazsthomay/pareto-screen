"""Random selection baseline."""

from __future__ import annotations

import torch


class RandomSelector:
    """Selects molecules uniformly at random from the unobserved pool."""

    def __init__(self, seed: int | None = None):
        self._generator = torch.Generator()
        if seed is not None:
            self._generator.manual_seed(seed)

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
        perm = torch.randperm(len(candidates), generator=self._generator)
        return candidates[perm[:batch_size]]
