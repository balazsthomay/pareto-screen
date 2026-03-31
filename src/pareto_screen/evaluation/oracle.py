"""Oracle: holds ground truth objectives, reveals on request."""

from __future__ import annotations

import torch

from pareto_screen.bo.pareto import compute_hypervolume, pareto_frontier
from pareto_screen.types import ObjectiveConfig


class Oracle:
    """Holds ground truth objective values and reveals them on request.

    Simulates the real-world scenario where evaluating a molecule
    (sending it to a wet lab) is expensive.
    """

    def __init__(self, Y: torch.Tensor, objective_configs: list[ObjectiveConfig]):
        self._Y = Y
        self.objective_configs = objective_configs
        self._revealed = torch.zeros(Y.shape[0], dtype=torch.bool)

        # Fixed reference point for the entire run
        y_range = Y.max(dim=0).values - Y.min(dim=0).values
        self._ref_point = Y.min(dim=0).values - 0.1 * y_range

    def reveal(self, indices: torch.Tensor) -> torch.Tensor:
        """Reveal objective values for selected indices."""
        self._revealed[indices] = True
        return self._Y[indices]

    @property
    def n_revealed(self) -> int:
        return self._revealed.sum().item()

    @property
    def ref_point(self) -> torch.Tensor:
        return self._ref_point

    @property
    def true_pareto_hypervolume(self) -> float:
        """Hypervolume of the true Pareto frontier (theoretical maximum)."""
        front = pareto_frontier(self._Y)
        return compute_hypervolume(front, self._ref_point)

    @property
    def pool_size(self) -> int:
        return self._Y.shape[0]
