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
    """Zero-pad (or truncate) every vector to `dim` so all conditions share D."""
    out = {}
    for k, v in vecs.items():
        v = np.asarray(v, dtype=np.float32).ravel()
        if v.shape[0] < dim:
            v = np.concatenate([v, np.zeros(dim - v.shape[0], dtype=np.float32)])
        else:
            v = v[:dim]
        out[k] = v
    return out


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
