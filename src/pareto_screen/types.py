"""Shared data types for the pareto-screen framework."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Protocol, runtime_checkable

import torch


@dataclass(frozen=True)
class MoleculeRecord:
    """A single molecule with its computed properties and features."""

    smiles: str
    mol_id: str
    objectives: dict[str, float]
    features: torch.Tensor | None = field(default=None, compare=False)


@dataclass(frozen=True)
class ObjectiveConfig:
    """Configuration for a single optimization objective."""

    name: str
    direction: Literal["maximize", "minimize"]
    bounds: tuple[float, float] | None = None


@dataclass
class ScreeningPool:
    """Pool of candidate molecules with their objective values."""

    records: list[MoleculeRecord]
    objective_configs: list[ObjectiveConfig]

    def to_tensors(self, dtype: torch.dtype = torch.double) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (X features, Y objectives) as tensors.

        Y columns follow objective_configs order. Minimize-direction objectives
        are negated so the BO framework always maximizes.
        """
        features = []
        objectives = []
        for record in self.records:
            if record.features is None:
                raise ValueError(f"Record {record.mol_id} has no features")
            features.append(record.features)
            row = []
            for config in self.objective_configs:
                val = record.objectives[config.name]
                row.append(-val if config.direction == "minimize" else val)
            objectives.append(row)

        X = torch.stack(features).to(dtype)
        Y = torch.tensor(objectives, dtype=dtype)
        return X, Y

    def __len__(self) -> int:
        return len(self.records)


@runtime_checkable
class SelectionStrategy(Protocol):
    """Protocol for molecule selection strategies (BO and baselines)."""

    def select_batch(
        self,
        pool_X: torch.Tensor,
        observed_indices: torch.Tensor,
        observed_Y: torch.Tensor | None,
        batch_size: int,
    ) -> torch.Tensor:
        """Return indices of the next batch of molecules to evaluate."""
        ...


@dataclass
class OptimizationResult:
    """Complete record of an optimization run."""

    selected_indices: list[list[int]]
    observed_Y: torch.Tensor
    hypervolumes: list[float]
    pareto_indices: list[int]
