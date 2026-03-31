"""ChEMBL bioactivity data loading."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


def convert_to_pic50(ic50_nm: float) -> float:
    """Convert IC50 in nM to pIC50 = -log10(IC50 in M) = 9 - log10(IC50_nM)."""
    if ic50_nm <= 0:
        return float("nan")
    return 9.0 - math.log10(ic50_nm)


def deduplicate_activities(df: pd.DataFrame) -> pd.DataFrame:
    """Deduplicate by taking the median standard_value per SMILES."""
    return df.groupby("canonical_smiles", as_index=False).agg(
        standard_value=("standard_value", "median")
    )


def load_chembl_activities(
    target_chembl_id: str = "CHEMBL203",
    activity_types: tuple[str, ...] = ("IC50",),
    units: str = "nM",
) -> pd.DataFrame:
    """Load bioactivity data from ChEMBL for a given target.

    Returns DataFrame with columns: canonical_smiles, standard_value, pic50.
    Deduplicates by median per compound and filters invalid values.
    """
    import chembl_downloader

    activity_type_list = ", ".join(f"'{t}'" for t in activity_types)
    query = f"""
        SELECT
            cs.canonical_smiles,
            act.standard_value
        FROM activities act
        JOIN assays a ON act.assay_id = a.assay_id
        JOIN target_dictionary td ON a.tid = td.tid
        JOIN compound_structures cs ON act.molregno = cs.molregno
        WHERE td.chembl_id = '{target_chembl_id}'
          AND act.standard_type IN ({activity_type_list})
          AND act.standard_units = '{units}'
          AND act.standard_relation = '='
          AND act.standard_value IS NOT NULL
          AND act.standard_value > 0
    """

    df = chembl_downloader.query(query)
    df["standard_value"] = pd.to_numeric(df["standard_value"], errors="coerce")
    df = df.dropna(subset=["standard_value", "canonical_smiles"])
    df = deduplicate_activities(df)
    df["pic50"] = df["standard_value"].apply(convert_to_pic50)
    df = df.dropna(subset=["pic50"])
    return df.reset_index(drop=True)
