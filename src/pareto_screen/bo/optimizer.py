"""Multi-objective Bayesian optimization loop for molecule selection."""

from __future__ import annotations

import torch

from pareto_screen.bo.acquisition import build_acquisition, evaluate_candidates
from pareto_screen.bo.pareto import compute_hypervolume, is_pareto_optimal
from pareto_screen.bo.surrogate import build_model, fit_model
from pareto_screen.types import ObjectiveConfig, OptimizationResult


class BayesianOptimizer:
    """Multi-objective BO loop for selecting molecules from a finite pool.

    Operates in discrete pool mode: evaluates acquisition function at all
    unobserved candidates and selects the top-batch_size by acquisition value.
    """

    def __init__(
        self,
        objective_configs: list[ObjectiveConfig],
        batch_size: int = 1,
        n_initial: int = 20,
        n_iterations: int = 50,
        seed: int | None = None,
    ):
        self.objective_configs = objective_configs
        self.batch_size = batch_size
        self.n_initial = n_initial
        self.n_iterations = n_iterations
        self.seed = seed

    def run(
        self,
        pool_X: torch.Tensor,
        oracle_Y: torch.Tensor,
    ) -> OptimizationResult:
        """Run the full BO loop.

        Args:
            pool_X: All candidate features, shape [N, d].
            oracle_Y: True objectives for all candidates (hidden), shape [N, m].
                      Already direction-normalized (all maximize).

        Returns:
            OptimizationResult with selection history and metrics.
        """
        generator = torch.Generator()
        if self.seed is not None:
            generator.manual_seed(self.seed)

        n_pool = pool_X.shape[0]

        # Compute reference point from full data (fixed for entire run)
        ref_point = oracle_Y.min(dim=0).values - 0.1 * (
            oracle_Y.max(dim=0).values - oracle_Y.min(dim=0).values
        )

        # Select initial random sample
        initial_perm = torch.randperm(n_pool, generator=generator)
        initial_indices = initial_perm[: self.n_initial].tolist()

        observed_mask = torch.zeros(n_pool, dtype=torch.bool)
        observed_mask[initial_indices] = True

        selected_indices_history: list[list[int]] = []
        hypervolumes: list[float] = []

        for iteration in range(self.n_iterations):
            # Get observed data
            obs_idx = observed_mask.nonzero(as_tuple=True)[0]
            train_X = pool_X[obs_idx]
            train_Y = oracle_Y[obs_idx]

            # Compute current hypervolume
            hv = compute_hypervolume(train_Y, ref_point)
            hypervolumes.append(hv)

            # Get unobserved candidates
            unobs_mask = ~observed_mask
            if not unobs_mask.any():
                break
            candidate_indices = unobs_mask.nonzero(as_tuple=True)[0]
            candidate_X = pool_X[candidate_indices]

            # Build and fit GP model
            model = build_model(train_X, train_Y)
            fit_model(model)
            model.eval()

            # Build acquisition and evaluate candidates
            acq = build_acquisition(model, train_X, train_Y, ref_point)
            acq_values = evaluate_candidates(acq, candidate_X)

            # Select top-batch_size candidates
            n_select = min(self.batch_size, len(candidate_indices))
            top_local = torch.argsort(acq_values, descending=True)[:n_select]
            selected = candidate_indices[top_local].tolist()

            # Update observed set
            for idx in selected:
                observed_mask[idx] = True
            selected_indices_history.append(selected)

        # Final state
        obs_idx = observed_mask.nonzero(as_tuple=True)[0]
        final_Y = oracle_Y[obs_idx]
        pareto_mask = is_pareto_optimal(final_Y)
        pareto_indices = obs_idx[pareto_mask].tolist()

        return OptimizationResult(
            selected_indices=selected_indices_history,
            observed_Y=final_Y,
            hypervolumes=hypervolumes,
            pareto_indices=pareto_indices,
        )
