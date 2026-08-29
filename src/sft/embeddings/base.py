"""Feature-embedding interface and shared utilities.

Every condition in Phase 1 (see docs/phase1_spec.md) is the SAME architecture with a different
`e_j`. An embedder maps the FeatureMeta table -> {feature_id: vector}. To keep the comparison
construct-matched, all conditions are padded to a common dimension D (so no condition gets more
parameters than another) via `pad_to`, and the shuffled-KG control is produced by `shuffle_assign`
(permute which feature gets which embedding, leaving the embedding *set* and D unchanged).
"""
from __future__ import annotations

from typing import Protocol

import numpy as np
import pandas as pd


class FeatureEmbedder(Protocol):
    name: str
    def embed(self, features: pd.DataFrame) -> dict[str, np.ndarray]: ...


def pad_to(vecs: dict[str, np.ndarray], dim: int) -> dict[str, np.ndarray]:
    """Fit every vector to exactly `dim` so all conditions share D with NO information advantage:
    zero-pad when smaller; PCA-reduce (distance-preserving) when larger, never arbitrary truncation.
    Truncation would keep the first `dim` coordinates and silently cripple high-dim embeddings (text,
    KG) while leaving low-dim ones (metadata) intact, confounding the comparison."""
    keys = list(vecs.keys())
    M = np.stack([np.asarray(vecs[k], dtype=np.float32).ravel() for k in keys])   # (N, D)
    if M.shape[1] > dim:
        Mc = M - M.mean(axis=0, keepdims=True)
        # top principal directions (preserves the most pairwise-distance variance); rank is bounded
        # by the number of features, so this may yield < dim components and is zero-padded below.
        _, _, Vt = np.linalg.svd(Mc, full_matrices=False)
        M = (Mc @ Vt[:dim].T).astype(np.float32)
    if M.shape[1] < dim:                                        # pad up to exactly `dim`
        M = np.concatenate([M, np.zeros((len(keys), dim - M.shape[1]), dtype=np.float32)], axis=1)
    return {k: M[i].astype(np.float32) for i, k in enumerate(keys)}


def shuffle_assign(vecs: dict[str, np.ndarray], seed: int) -> dict[str, np.ndarray]:
    """Shuffled-KG control (C6): keep the exact embedding set and D, permute feature->vector.

    If Correct-KG ~= Shuffled-KG, the benefit is architectural, not semantic (invariant 3)."""
    rng = np.random.default_rng(seed)
    keys = list(vecs.keys())
    perm = rng.permutation(len(keys))
    values = [vecs[k] for k in keys]
    return {keys[i]: values[perm[i]] for i in range(len(keys))}


def l2_normalize(vecs: dict[str, np.ndarray], eps: float = 1e-8) -> dict[str, np.ndarray]:
    return {k: v / (np.linalg.norm(v) + eps) for k, v in vecs.items()}
