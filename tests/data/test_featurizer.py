"""Tests for molecular featurization."""

import numpy as np
import pytest
import torch

from pareto_screen.data.featurizer import MoleculeFeaturizer

SMILES = [
    "CC(=O)Oc1ccccc1C(=O)O",  # aspirin
    "Cn1c(=O)c2c(ncn2C)n(C)c1=O",  # caffeine
    "CC(C)Cc1ccc(cc1)C(C)C(=O)O",  # ibuprofen
    "c1ccccc1",  # benzene
    "CCO",  # ethanol
    "CC(=O)O",  # acetic acid
    "c1ccc2ccccc2c1",  # naphthalene
    "CC(C)O",  # isopropanol
]


class TestMoleculeFeaturizer:
    def test_fit_returns_self(self):
        featurizer = MoleculeFeaturizer(n_components=3)
        result = featurizer.fit(SMILES)
        assert result is featurizer

    def test_transform_shape(self):
        featurizer = MoleculeFeaturizer(n_components=5)
        featurizer.fit(SMILES)
        vec = featurizer.transform(SMILES[0])
        assert vec is not None
        assert vec.shape == (5,)
        assert vec.dtype == torch.double

    def test_transform_batch_shape(self):
        featurizer = MoleculeFeaturizer(n_components=5)
        featurizer.fit(SMILES)
        batch = featurizer.transform_batch(SMILES[:4])
        assert batch.shape == (4, 5)
        assert batch.dtype == torch.double

    def test_transform_invalid_returns_none(self):
        featurizer = MoleculeFeaturizer(n_components=3)
        featurizer.fit(SMILES)
        assert featurizer.transform("NOT_A_MOLECULE") is None

    def test_transform_before_fit_raises(self):
        featurizer = MoleculeFeaturizer(n_components=3)
        with pytest.raises(RuntimeError):
            featurizer.transform(SMILES[0])

    def test_deterministic(self):
        featurizer = MoleculeFeaturizer(n_components=5)
        featurizer.fit(SMILES)
        v1 = featurizer.transform(SMILES[0])
        v2 = featurizer.transform(SMILES[0])
        assert torch.allclose(v1, v2)

    def test_different_molecules_different_features(self):
        featurizer = MoleculeFeaturizer(n_components=5)
        featurizer.fit(SMILES)
        v1 = featurizer.transform(SMILES[0])  # aspirin
        v2 = featurizer.transform(SMILES[3])  # benzene
        assert not torch.allclose(v1, v2)

    def test_n_components_capped_at_samples(self):
        """If n_components > n_samples, PCA should adjust gracefully."""
        featurizer = MoleculeFeaturizer(n_components=100)
        featurizer.fit(SMILES[:3])  # only 3 samples
        vec = featurizer.transform(SMILES[0])
        assert vec is not None
        assert vec.shape[0] <= 3
