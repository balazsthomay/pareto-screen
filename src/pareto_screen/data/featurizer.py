"""Molecular featurization: SMILES → fixed-size feature vectors via Morgan FP + PCA."""

from __future__ import annotations

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.decomposition import PCA


class MoleculeFeaturizer:
    """Converts SMILES to fixed-size feature vectors for GP input.

    Strategy: Morgan fingerprints (2048-bit) → PCA reduction to n_components.
    Must be fit on the full pool before transforming individual molecules.
    """

    def __init__(self, n_components: int = 50, radius: int = 2, fp_size: int = 2048):
        self.n_components = n_components
        self.radius = radius
        self.fp_size = fp_size
        self._fpgen = AllChem.GetMorganGenerator(radius=radius, fpSize=fp_size)
        self._pca: PCA | None = None
        self._is_fitted = False

    def _smiles_to_fp(self, smiles: str) -> np.ndarray | None:
        """Convert SMILES to Morgan fingerprint bit vector."""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        fp = self._fpgen.GetFingerprint(mol)
        return np.array(fp, dtype=np.float64)

    def fit(self, smiles_list: list[str]) -> MoleculeFeaturizer:
        """Fit PCA on Morgan fingerprints of the molecule pool."""
        fps = []
        for s in smiles_list:
            fp = self._smiles_to_fp(s)
            if fp is not None:
                fps.append(fp)

        if not fps:
            raise ValueError("No valid SMILES to fit on")

        X = np.stack(fps)
        n_components = min(self.n_components, X.shape[0], X.shape[1])
        self._pca = PCA(n_components=n_components)
        self._pca.fit(X)
        self._is_fitted = True
        return self

    def transform(self, smiles: str) -> torch.Tensor | None:
        """Transform a single SMILES to a reduced feature vector."""
        if not self._is_fitted:
            raise RuntimeError("Featurizer must be fit before transform")

        fp = self._smiles_to_fp(smiles)
        if fp is None:
            return None

        reduced = self._pca.transform(fp.reshape(1, -1))
        return torch.tensor(reduced[0], dtype=torch.double)

    def transform_batch(self, smiles_list: list[str]) -> torch.Tensor:
        """Transform a batch of SMILES. Skips invalid SMILES.

        Returns tensor of shape [n_valid, n_components].
        Caller must track which indices were valid.
        """
        if not self._is_fitted:
            raise RuntimeError("Featurizer must be fit before transform")

        fps = []
        for s in smiles_list:
            fp = self._smiles_to_fp(s)
            if fp is not None:
                fps.append(fp)

        if not fps:
            return torch.empty(0, self._pca.n_components_, dtype=torch.double)

        X = np.stack(fps)
        reduced = self._pca.transform(X)
        return torch.tensor(reduced, dtype=torch.double)
