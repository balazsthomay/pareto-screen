"""Tests for MoleculeDataset."""

import pandas as pd
import pytest
import torch

from pareto_screen.data.dataset import MoleculeDataset
from pareto_screen.types import ObjectiveConfig


SMILES_LIST = [
    "CC(=O)Oc1ccccc1C(=O)O",  # aspirin
    "Cn1c(=O)c2c(ncn2C)n(C)c1=O",  # caffeine
    "CC(C)Cc1ccc(cc1)C(C)C(=O)O",  # ibuprofen
    "c1ccccc1",  # benzene
    "CCO",  # ethanol
    "CC(=O)O",  # acetic acid
]


@pytest.fixture
def objective_configs_with_pic50():
    return [
        ObjectiveConfig(name="qed", direction="maximize"),
        ObjectiveConfig(name="sa_score", direction="minimize"),
        ObjectiveConfig(name="logp", direction="maximize"),
    ]


class TestMoleculeDatasetFromSmiles:
    def test_basic_construction(self, objective_configs_with_pic50):
        dataset = MoleculeDataset.from_smiles(
            SMILES_LIST, objective_configs_with_pic50, n_pca_components=3
        )
        assert len(dataset) == 6

    def test_get_tensors_shape(self, objective_configs_with_pic50):
        dataset = MoleculeDataset.from_smiles(
            SMILES_LIST, objective_configs_with_pic50, n_pca_components=3
        )
        X, Y = dataset.get_tensors()
        assert X.shape == (6, 3)  # 6 molecules, 3 PCA components
        assert Y.shape == (6, 3)  # 6 molecules, 3 objectives

    def test_get_tensors_dtype(self, objective_configs_with_pic50):
        dataset = MoleculeDataset.from_smiles(
            SMILES_LIST, objective_configs_with_pic50, n_pca_components=3
        )
        X, Y = dataset.get_tensors()
        assert X.dtype == torch.double
        assert Y.dtype == torch.double

    def test_sa_score_negated(self, objective_configs_with_pic50):
        dataset = MoleculeDataset.from_smiles(
            SMILES_LIST, objective_configs_with_pic50, n_pca_components=3
        )
        X, Y = dataset.get_tensors()
        # SA score column (index 1) should be negated (all negative since SA > 0)
        assert (Y[:, 1] <= 0).all()

    def test_qed_positive(self, objective_configs_with_pic50):
        dataset = MoleculeDataset.from_smiles(
            SMILES_LIST, objective_configs_with_pic50, n_pca_components=3
        )
        X, Y = dataset.get_tensors()
        # QED column (index 0) should be positive
        assert (Y[:, 0] > 0).all()

    def test_skips_invalid_smiles(self, objective_configs_with_pic50):
        smiles_with_bad = SMILES_LIST + ["NOT_A_MOLECULE", "ALSO_BAD"]
        dataset = MoleculeDataset.from_smiles(
            smiles_with_bad, objective_configs_with_pic50, n_pca_components=3
        )
        assert len(dataset) == 6  # only valid molecules

    def test_get_record(self, objective_configs_with_pic50):
        dataset = MoleculeDataset.from_smiles(
            SMILES_LIST, objective_configs_with_pic50, n_pca_components=3
        )
        record = dataset.get_record(0)
        assert record.smiles == SMILES_LIST[0]
        assert "qed" in record.objectives
        assert record.features is not None


class TestMoleculeDatasetFromDataFrame:
    def test_with_pic50(self):
        df = pd.DataFrame({
            "canonical_smiles": SMILES_LIST[:4],
            "pic50": [7.5, 6.0, 8.0, 5.0],
        })
        configs = [
            ObjectiveConfig(name="qed", direction="maximize"),
            ObjectiveConfig(name="sa_score", direction="minimize"),
            ObjectiveConfig(name="logp", direction="maximize"),
            ObjectiveConfig(name="pic50", direction="maximize"),
        ]
        dataset = MoleculeDataset.from_dataframe(df, configs, n_pca_components=3)
        assert len(dataset) == 4
        X, Y = dataset.get_tensors()
        assert Y.shape == (4, 4)  # 4 objectives
        # pic50 should be in the last column, positive (maximize)
        assert (Y[:, 3] > 0).all()
