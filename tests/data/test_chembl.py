"""Tests for ChEMBL data loading."""

import numpy as np
import pandas as pd
import pytest

from pareto_screen.data.chembl import convert_to_pic50, deduplicate_activities, load_chembl_activities


class TestConvertToPic50:
    def test_basic_conversion(self):
        # IC50 = 1 nM -> pIC50 = 9 - log10(1) = 9.0
        assert convert_to_pic50(1.0) == pytest.approx(9.0)

    def test_10nm(self):
        # IC50 = 10 nM -> pIC50 = 9 - log10(10) = 8.0
        assert convert_to_pic50(10.0) == pytest.approx(8.0)

    def test_1000nm(self):
        # IC50 = 1000 nM -> pIC50 = 9 - log10(1000) = 6.0
        assert convert_to_pic50(1000.0) == pytest.approx(6.0)

    def test_zero_returns_nan(self):
        assert np.isnan(convert_to_pic50(0.0))

    def test_negative_returns_nan(self):
        assert np.isnan(convert_to_pic50(-1.0))


class TestDeduplicateActivities:
    def test_median_aggregation(self):
        df = pd.DataFrame({
            "canonical_smiles": ["CCO", "CCO", "CCO", "c1ccccc1"],
            "standard_value": [10.0, 20.0, 30.0, 100.0],
        })
        result = deduplicate_activities(df)
        assert len(result) == 2
        # Median of 10, 20, 30 = 20
        ethanol_row = result[result["canonical_smiles"] == "CCO"]
        assert ethanol_row["standard_value"].iloc[0] == pytest.approx(20.0)

    def test_single_entry_preserved(self):
        df = pd.DataFrame({
            "canonical_smiles": ["CCO"],
            "standard_value": [42.0],
        })
        result = deduplicate_activities(df)
        assert len(result) == 1
        assert result["standard_value"].iloc[0] == pytest.approx(42.0)


class TestLoadChemblActivities:
    @pytest.mark.slow
    def test_egfr_returns_data(self):
        """Integration test: requires ChEMBL download (~2GB first time)."""
        df = load_chembl_activities(target_chembl_id="CHEMBL203")
        assert len(df) > 100
        assert "canonical_smiles" in df.columns
        assert "pic50" in df.columns
        assert df["pic50"].notna().all()
