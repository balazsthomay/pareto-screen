"""Tests for molecular property computation."""

import pytest

from pareto_screen.data.properties import compute_properties, compute_properties_batch


class TestComputeProperties:
    def test_aspirin(self):
        props = compute_properties("CC(=O)Oc1ccccc1C(=O)O")
        assert props is not None
        assert 0.0 < props["qed"] <= 1.0
        assert 1.0 <= props["sa_score"] <= 10.0
        assert isinstance(props["logp"], float)

    def test_caffeine(self):
        props = compute_properties("Cn1c(=O)c2c(ncn2C)n(C)c1=O")
        assert props is not None
        assert 0.0 < props["qed"] <= 1.0

    def test_benzene(self):
        props = compute_properties("c1ccccc1")
        assert props is not None
        # Benzene LogP is around 1.6
        assert 1.0 < props["logp"] < 2.5

    def test_invalid_smiles_returns_none(self):
        assert compute_properties("NOT_A_MOLECULE") is None

    def test_empty_smiles_returns_none(self):
        assert compute_properties("") is None

    def test_all_keys_present(self):
        props = compute_properties("c1ccccc1")
        assert props is not None
        assert set(props.keys()) == {"qed", "sa_score", "logp"}


class TestComputePropertiesBatch:
    def test_batch(self):
        smiles = ["CC(=O)Oc1ccccc1C(=O)O", "c1ccccc1", "NOT_A_MOLECULE"]
        results = compute_properties_batch(smiles)
        assert len(results) == 3
        assert results[0] is not None
        assert results[1] is not None
        assert results[2] is None

    def test_empty_batch(self):
        assert compute_properties_batch([]) == []
