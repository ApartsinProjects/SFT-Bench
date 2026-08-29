"""The 7 Phase-1 conditions (docs/phase1_spec.md section 2). All share one architecture; only e_j
differs. A condition resolves to either a precomputed embedding table (fixed e_j) or a model flag
(C0 zero, C2 learned-ID).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..embeddings.base import pad_to, shuffle_assign
from ..embeddings.embedders import RandomEmbedder, MetadataEmbedder, KGEmbedder, TextEmbedder


@dataclass
class Condition:
    cid: str
    name: str
    kind: str            # "zero" | "learned_id" | "fixed"
    tests: str           # what this condition establishes


CONDITIONS = [
    Condition("C0", "value_only", "zero",       "floor: no semantics"),
    Condition("C1", "random",     "fixed",      "controls for extra dims / bare identifier"),
    Condition("C2", "learned_id", "learned_id", "in-distribution ceiling; cannot zero-shot"),
    Condition("C3", "text",       "fixed",      "THE competitor: text embedding of same metadata"),
    Condition("C4", "metadata",   "fixed",      "isolates topology's value (type+comp+unit only)"),
    Condition("C5", "kg",         "fixed",      "proposed: inductive KG embedding"),
    Condition("C6", "kg_shuffled","fixed",      "semantic vs architectural (permuted KG)"),
]


def build_embedding_table(cond: Condition, features: pd.DataFrame, dim: int, seed: int
                          ) -> dict[str, np.ndarray] | None:
    """Return {feature_id: e_j (dim,)} for a fixed-e condition, or None for zero/learned_id."""
    if cond.kind in ("zero", "learned_id"):
        return None
    if cond.name == "random":
        vecs = RandomEmbedder(dim=dim, seed=seed).embed(features)
    elif cond.name == "metadata":
        vecs = MetadataEmbedder().embed(features)
    elif cond.name == "text":
        vecs = TextEmbedder().embed(features)
    elif cond.name in ("kg", "kg_shuffled"):
        vecs = KGEmbedder().embed(features)
        if cond.name == "kg_shuffled":
            vecs = shuffle_assign(vecs, seed=seed)
    else:
        raise ValueError(cond.name)
    return pad_to(vecs, dim)
