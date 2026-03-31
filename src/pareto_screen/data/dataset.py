"""MoleculeDataset: assembles molecules, properties, and features into BO-ready tensors."""

from __future__ import annotations

import torch
import pandas as pd

from pareto_screen.data.featurizer import MoleculeFeaturizer
from pareto_screen.data.properties import compute_properties
from pareto_screen.types import MoleculeRecord, ObjectiveConfig, ScreeningPool


class MoleculeDataset:
    """A fully prepared dataset of molecules ready for Bayesian optimization."""

    def __init__(
        self,
        pool: ScreeningPool,
        featurizer: MoleculeFeaturizer,
    ):
        self._pool = pool
        self._featurizer = featurizer

    @classmethod
    def from_smiles(
        cls,
        smiles_list: list[str],
        objective_configs: list[ObjectiveConfig],
        n_pca_components: int = 50,
    ) -> MoleculeDataset:
        """Build dataset from SMILES strings. Computes RDKit properties and featurizes."""
        records = []
        valid_smiles = []

        for i, smiles in enumerate(smiles_list):
            props = compute_properties(smiles)
            if props is None:
                continue
            records.append(
                MoleculeRecord(
                    smiles=smiles,
                    mol_id=f"MOL_{i:06d}",
                    objectives=props,
                )
            )
            valid_smiles.append(smiles)

        featurizer = MoleculeFeaturizer(n_components=n_pca_components)
        featurizer.fit(valid_smiles)

        featurized_records = []
        for record, smiles in zip(records, valid_smiles):
            features = featurizer.transform(smiles)
            featurized_records.append(
                MoleculeRecord(
                    smiles=record.smiles,
                    mol_id=record.mol_id,
                    objectives=record.objectives,
                    features=features,
                )
            )

        pool = ScreeningPool(records=featurized_records, objective_configs=objective_configs)
        return cls(pool=pool, featurizer=featurizer)

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        objective_configs: list[ObjectiveConfig],
        n_pca_components: int = 50,
    ) -> MoleculeDataset:
        """Build dataset from a DataFrame with canonical_smiles and optional pic50 columns.

        Computes RDKit properties (qed, sa_score, logp) and merges with any
        additional columns present in the DataFrame (e.g. pic50).
        """
        records = []
        valid_smiles = []

        for idx, row in df.iterrows():
            smiles = row["canonical_smiles"]
            props = compute_properties(smiles)
            if props is None:
                continue

            # Merge any extra objective columns from the dataframe
            for config in objective_configs:
                if config.name not in props and config.name in df.columns:
                    props[config.name] = float(row[config.name])

            records.append(
                MoleculeRecord(
                    smiles=smiles,
                    mol_id=f"MOL_{idx:06d}",
                    objectives=props,
                )
            )
            valid_smiles.append(smiles)

        featurizer = MoleculeFeaturizer(n_components=n_pca_components)
        featurizer.fit(valid_smiles)

        featurized_records = []
        for record, smiles in zip(records, valid_smiles):
            features = featurizer.transform(smiles)
            featurized_records.append(
                MoleculeRecord(
                    smiles=record.smiles,
                    mol_id=record.mol_id,
                    objectives=record.objectives,
                    features=features,
                )
            )

        pool = ScreeningPool(records=featurized_records, objective_configs=objective_configs)
        return cls(pool=pool, featurizer=featurizer)

    def get_tensors(self, dtype: torch.dtype = torch.double) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (X features, Y objectives) tensors. Minimize objectives are negated."""
        return self._pool.to_tensors(dtype=dtype)

    def get_record(self, index: int) -> MoleculeRecord:
        return self._pool.records[index]

    @property
    def objective_configs(self) -> list[ObjectiveConfig]:
        return self._pool.objective_configs

    def __len__(self) -> int:
        return len(self._pool)
