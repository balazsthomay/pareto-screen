"""Molecular property computation using RDKit."""

from __future__ import annotations

from rdkit import Chem
from rdkit.Chem import Descriptors, QED
from rdkit.Contrib.SA_Score import sascorer


def compute_properties(smiles: str) -> dict[str, float] | None:
    """Compute QED, SA Score, and LogP for a molecule.

    Returns None if SMILES cannot be parsed by RDKit.
    """
    if not smiles:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() == 0:
        return None

    return {
        "qed": QED.qed(mol),
        "sa_score": sascorer.calculateScore(mol),
        "logp": Descriptors.MolLogP(mol),
    }


def compute_properties_batch(smiles_list: list[str]) -> list[dict[str, float] | None]:
    """Compute properties for a batch of molecules."""
    return [compute_properties(s) for s in smiles_list]
