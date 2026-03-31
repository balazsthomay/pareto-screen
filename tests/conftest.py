"""Shared test fixtures."""

import pytest
import torch

from pareto_screen.types import MoleculeRecord, ObjectiveConfig, ScreeningPool

# Well-known molecules for testing
ASPIRIN_SMILES = "CC(=O)Oc1ccccc1C(=O)O"
CAFFEINE_SMILES = "Cn1c(=O)c2c(ncn2C)n(C)c1=O"
IBUPROFEN_SMILES = "CC(C)Cc1ccc(cc1)C(C)C(=O)O"
BENZENE_SMILES = "c1ccccc1"
INVALID_SMILES = "NOT_A_MOLECULE"

TEST_SMILES = [ASPIRIN_SMILES, CAFFEINE_SMILES, IBUPROFEN_SMILES, BENZENE_SMILES]


@pytest.fixture
def objective_configs() -> list[ObjectiveConfig]:
    return [
        ObjectiveConfig(name="qed", direction="maximize"),
        ObjectiveConfig(name="sa_score", direction="minimize"),
        ObjectiveConfig(name="logp", direction="maximize"),
    ]


@pytest.fixture
def sample_records() -> list[MoleculeRecord]:
    """Small set of molecule records with synthetic objectives and features."""
    records = []
    for i, smiles in enumerate(TEST_SMILES):
        records.append(
            MoleculeRecord(
                smiles=smiles,
                mol_id=f"MOL_{i:04d}",
                objectives={"qed": 0.5 + i * 0.1, "sa_score": 2.0 + i * 0.5, "logp": 1.0 + i * 0.3},
                features=torch.randn(10, dtype=torch.double),
            )
        )
    return records


@pytest.fixture
def screening_pool(sample_records, objective_configs) -> ScreeningPool:
    return ScreeningPool(records=sample_records, objective_configs=objective_configs)
