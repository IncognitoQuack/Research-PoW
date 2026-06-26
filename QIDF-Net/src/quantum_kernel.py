"""
Quantum Frequency Kernel (QFK) — analytical implementation.

Implements the closed-form kernel corresponding to the ZZ-FeatureMap circuit
used in quantum machine learning [Havlíček et al., Nature 2019].

For normalised inputs x_a, x_b ∈ [0, 1]^d:

  K_Q(x_a, x_b) =  ∏_j  cos²(π(x_a^j − x_b^j))          [single-qubit term]
                 × ∏_{j<k} cos²(π(x_a^j x_a^k − x_b^j x_b^k))  [interaction term]

This is mathematically equivalent to the inner-product kernel induced by the
ZZ-FeatureMap quantum circuit, computed analytically on classical hardware — the
definition of a quantum-inspired kernel.

Scalability is achieved via the Nyström approximation, which avoids constructing
the full N×N kernel matrix by projecting through a set of m landmark points.
"""

from pathlib import Path
from typing import Optional

import numpy as np
from scipy.linalg import svd
from sklearn.decomposition import PCA

from .utils import get_logger, cache_path, save_cache, load_cache

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Low-level kernel computation
# ---------------------------------------------------------------------------

def _to_unit(X: np.ndarray) -> np.ndarray:
    """Min-max normalise each feature to [0, 1]."""
    lo = X.min(axis=0)
    hi = X.max(axis=0)
    rng = hi - lo
    rng[rng < 1e-12] = 1.0
    return (X - lo) / rng


def zz_kernel_matrix(X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
    """
    Compute the ZZ-FeatureMap quantum kernel matrix K ∈ ℝ^{n1 × n2}.

    Both arrays are normalised internally to [0, 1]^d.

    Parameters
    ----------
    X1 : ndarray (n1, d)
    X2 : ndarray (n2, d)

    Returns
    -------
    K  : ndarray (n1, n2)  with values in [0, 1]
    """
    X1 = _to_unit(X1.astype(np.float64))
    X2 = _to_unit(X2.astype(np.float64))
    n1, d = X1.shape
    n2 = X2.shape[0]

    # Single-qubit term: vectorised over all pairs
    diff = X1[:, np.newaxis, :] - X2[np.newaxis, :, :]   # (n1, n2, d)
    K = np.prod(np.cos(np.pi * diff) ** 2, axis=2)        # (n1, n2)

    # Two-qubit interaction terms
    for j in range(d - 1):
        for k in range(j + 1, d):
            p1 = X1[:, j] * X1[:, k]                       # (n1,)
            p2 = X2[:, j] * X2[:, k]                       # (n2,)
            d2 = p1[:, np.newaxis] - p2[np.newaxis, :]     # (n1, n2)
            K *= np.cos(np.pi * d2) ** 2

    return K.astype(np.float32)


# ---------------------------------------------------------------------------
# Nyström Projector
# ---------------------------------------------------------------------------

class QuantumKernelProjector:
    """
    Projects feature vectors into a quantum-kernel-induced space via
    the Nyström approximation.

    Workflow:
      1. fit(X_train):  select m landmark points; compute K_mm; factoise.
      2. transform(X): compute K_nm; project to approximate RKHS embedding.
      3. PCA reduces the m-dim embedding to n_components.

    Parameters
    ----------
    n_landmarks   : number of Nyström anchor points (default 300)
    n_components  : output dimensionality (default 64)
    random_state  : RNG seed for reproducible landmark selection
    """

    def __init__(
        self,
        n_landmarks: int = 300,
        n_components: int = 64,
        random_state: int = 42,
    ) -> None:
        self.n_landmarks = n_landmarks
        self.n_components = n_components
        self.random_state = random_state
        self._landmarks: Optional[np.ndarray] = None
        self._K_inv_sqrt: Optional[np.ndarray] = None
        self._pca: Optional[PCA] = None

    def fit(self, X: np.ndarray) -> "QuantumKernelProjector":
        rng = np.random.default_rng(self.random_state)
        m = min(self.n_landmarks, len(X))
        idx = rng.choice(len(X), size=m, replace=False)
        self._landmarks = X[idx].copy()

        log.info("Computing Nyström kernel matrix (%d × %d) …", m, m)
        K_mm = zz_kernel_matrix(self._landmarks, self._landmarks)
        K_mm += 1e-8 * np.eye(m)                         # numerical regularisation

        U, s, _ = svd(K_mm, full_matrices=False)
        self._K_inv_sqrt = U @ np.diag(1.0 / np.sqrt(s + 1e-12)) @ U.T

        # Project training data and fit PCA on the embedding
        phi_train = self._project(X)
        self._pca = PCA(
            n_components=min(self.n_components, phi_train.shape[1]),
            random_state=self.random_state,
        )
        self._pca.fit(phi_train)
        log.info("QFK projector fitted — explained variance: %.2f%%",
                 100 * self._pca.explained_variance_ratio_.sum())
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self._landmarks is None:
            raise RuntimeError("Call fit() before transform().")
        phi = self._project(X)
        return self._pca.transform(phi).astype(np.float32)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    def _project(self, X: np.ndarray) -> np.ndarray:
        """K_nm × K_mm^{-1/2}"""
        K_nm = zz_kernel_matrix(X, self._landmarks)       # (n, m)
        return K_nm @ self._K_inv_sqrt                     # (n, m)


# ---------------------------------------------------------------------------
# Convenience wrapper with caching
# ---------------------------------------------------------------------------

def extract_quantum_features(
    dct_features: np.ndarray,
    projector: Optional[QuantumKernelProjector],
    split_tag: str,
    cache_dir: str = "outputs/cache",
    n_landmarks: int = 300,
    n_components: int = 64,
    seed: int = 42,
) -> tuple:
    """
    Fit (on train) or transform (on val/test) the quantum kernel projector.

    Returns (features_ndarray, projector).  Pass projector=None for the
    training split; pass the returned projector for val/test splits.
    """
    key = f"qfk_{split_tag}_{dct_features.shape}_{n_landmarks}_{n_components}"
    cp = cache_path(cache_dir, key)

    if cp.exists() and projector is not None:
        log.info("Loading cached quantum features for split=%s", split_tag)
        return load_cache(cp), projector

    if projector is None:
        log.info("Fitting QFK projector on %d training samples …", len(dct_features))
        projector = QuantumKernelProjector(n_landmarks, n_components, seed)
        feats = projector.fit_transform(dct_features)
    else:
        log.info("Projecting %d samples with fitted QFK …", len(dct_features))
        feats = projector.transform(dct_features)

    save_cache(feats, cp)
    return feats, projector
