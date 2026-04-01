"""BO-based selection strategy implementing the SelectionStrategy protocol."""

from __future__ import annotations

import torch

from pareto_screen.bo.acquisition import build_acquisition, evaluate_candidates
from pareto_screen.bo.pareto import compute_hypervolume
from pareto_screen.bo.surrogate import build_model, fit_model
from pareto_screen.types import ObjectiveConfig


class BOSelector:
    """Bayesian optimization selection strategy.

    Fits GP surrogates on observed data and uses qLogNEHVI acquisition
    to select the most promising unobserved candidates.
    """

    def __init__(
        self,
        objective_configs: list[ObjectiveConfig],
        ref_point: torch.Tensor,
    ):
        self.objective_configs = objective_configs
        self.ref_point = ref_point

    def select_batch(
        self,
        pool_X: torch.Tensor,
        observed_indices: torch.Tensor,
        observed_Y: torch.Tensor | None,
        batch_size: int,
    ) -> torch.Tensor:
        if observed_Y is None or observed_Y.shape[0] < 2:
            # Not enough data to fit GPs — fall back to random
            n_pool = pool_X.shape[0]
            mask = torch.ones(n_pool, dtype=torch.bool)
            mask[observed_indices] = False
            candidates = mask.nonzero(as_tuple=True)[0]
            perm = torch.randperm(len(candidates))
            return candidates[perm[:batch_size]]

        # Get observed data
        train_X = pool_X[observed_indices]
        train_Y = observed_Y

        # Get unobserved candidates
        n_pool = pool_X.shape[0]
        mask = torch.ones(n_pool, dtype=torch.bool)
        mask[observed_indices] = False
        candidate_indices = mask.nonzero(as_tuple=True)[0]

        if len(candidate_indices) == 0:
            return torch.tensor([], dtype=torch.long)

        candidate_X = pool_X[candidate_indices]

        # Build, fit, and evaluate
        model = build_model(train_X, train_Y)
        fit_model(model)
        model.eval()

        acq = build_acquisition(model, train_X, train_Y, self.ref_point)
        acq_values = evaluate_candidates(acq, candidate_X)

        n_select = min(batch_size, len(candidate_indices))
        top_local = torch.argsort(acq_values, descending=True)[:n_select]
        return candidate_indices[top_local]
