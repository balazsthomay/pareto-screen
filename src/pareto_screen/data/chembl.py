"""ChEMBL bioactivity data loading via REST API."""

from __future__ import annotations

import math

import pandas as pd
import requests


CHEMBL_API_BASE = "https://www.ebi.ac.uk/chembl/api/data"


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


def _fetch_activities_page(
    target_chembl_id: str,
    activity_type: str,
    units: str,
    limit: int,
    offset: int,
) -> dict:
    """Fetch a single page of activity data from ChEMBL REST API."""
    url = f"{CHEMBL_API_BASE}/activity.json"
    params = {
        "target_chembl_id": target_chembl_id,
        "standard_type": activity_type,
        "standard_units": units,
        "standard_relation": "=",
        "limit": limit,
        "offset": offset,
    }
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()


def load_chembl_activities(
    target_chembl_id: str = "CHEMBL203",
    activity_types: tuple[str, ...] = ("IC50",),
    units: str = "nM",
    page_size: int = 1000,
) -> pd.DataFrame:
    """Load bioactivity data from ChEMBL REST API for a given target.

    Returns DataFrame with columns: canonical_smiles, standard_value, pic50.
    Deduplicates by median per compound and filters invalid values.
    """
    rows: list[dict[str, str | float]] = []

    for activity_type in activity_types:
        offset = 0
        while True:
            data = _fetch_activities_page(
                target_chembl_id, activity_type, units, page_size, offset
            )
            activities = data.get("activities", [])
            if not activities:
                break

            for act in activities:
                smiles = act.get("canonical_smiles")
                value = act.get("standard_value")
                if smiles and value is not None:
                    try:
                        rows.append({
                            "canonical_smiles": smiles,
                            "standard_value": float(value),
                        })
                    except (ValueError, TypeError):
                        continue

            # Check if there are more pages
            if data.get("page_meta", {}).get("next") is None:
                break
            offset += page_size

    if not rows:
        return pd.DataFrame(columns=["canonical_smiles", "standard_value", "pic50"])

    df = pd.DataFrame(rows)
    df = df[df["standard_value"] > 0]
    df = deduplicate_activities(df)
    df["pic50"] = df["standard_value"].apply(convert_to_pic50)
    df = df.dropna(subset=["pic50"])
    return df.reset_index(drop=True)
