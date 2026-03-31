"""Tests for shared types."""

import torch

from pareto_screen.types import (
    MoleculeRecord,
    ObjectiveConfig,
    OptimizationResult,
    ScreeningPool,
)


class TestMoleculeRecord:
    def test_frozen(self):
        record = MoleculeRecord(smiles="C", mol_id="test", objectives={"qed": 0.5})
        try:
            record.smiles = "CC"
            assert False, "Should be frozen"
        except AttributeError:
            pass

    def test_optional_features(self):
        record = MoleculeRecord(smiles="C", mol_id="test", objectives={"qed": 0.5})
        assert record.features is None


class TestObjectiveConfig:
    def test_maximize(self):
        config = ObjectiveConfig(name="qed", direction="maximize")
        assert config.direction == "maximize"
        assert config.bounds is None

    def test_with_bounds(self):
        config = ObjectiveConfig(name="qed", direction="maximize", bounds=(0.0, 1.0))
        assert config.bounds == (0.0, 1.0)


class TestScreeningPool:
    def test_len(self, screening_pool):
        assert len(screening_pool) == 4

    def test_to_tensors_shape(self, screening_pool):
        X, Y = screening_pool.to_tensors()
        assert X.shape == (4, 10)  # 4 molecules, 10-dim features
        assert Y.shape == (4, 3)  # 4 molecules, 3 objectives

    def test_to_tensors_dtype(self, screening_pool):
        X, Y = screening_pool.to_tensors()
        assert X.dtype == torch.double
        assert Y.dtype == torch.double

    def test_to_tensors_negates_minimize(self, screening_pool):
        X, Y = screening_pool.to_tensors()
        # sa_score is "minimize" — should be negated
        # First record: sa_score=2.0, so Y[0, 1] should be -2.0
        assert Y[0, 1].item() == -2.0
        # qed is "maximize" — should be positive
        assert Y[0, 0].item() == 0.5

    def test_to_tensors_missing_features_raises(self, objective_configs):
        records = [
            MoleculeRecord(smiles="C", mol_id="test", objectives={"qed": 0.5, "sa_score": 2.0, "logp": 1.0})
        ]
        pool = ScreeningPool(records=records, objective_configs=objective_configs)
        try:
            pool.to_tensors()
            assert False, "Should raise ValueError"
        except ValueError:
            pass


class TestOptimizationResult:
    def test_creation(self):
        result = OptimizationResult(
            selected_indices=[[0, 1], [2, 3]],
            observed_Y=torch.randn(4, 2),
            hypervolumes=[0.5, 0.8],
            pareto_indices=[0, 2],
        )
        assert len(result.selected_indices) == 2
        assert len(result.hypervolumes) == 2
