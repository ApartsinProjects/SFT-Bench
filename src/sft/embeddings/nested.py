"""Nested semantic representations that isolate the marginal value of graph topology (TMLR review W3).

The prior metadata-vs-KG contrast was confounded: the KG vector dropped unit/component fields that
metadata kept, so a difference could come from lost metadata rather than topology. Here every richer
representation is a strict superset of a poorer one, so paired differences isolate one factor:

  A0 value       no semantics (handled by the model, not here)
  A1 random      fixed random vector (capacity control)
  A3 type        one-hot measurement type only
  A4 metadata    type + unit + component-type one-hot (core metadata)          <- topology baseline
  A5 text        sentence embedding of the feature description
  A6 metaTopo    A4 concatenated with the graph-topology token multi-hot        <- A6-A4 = topology gain
  A8 topoShuf    A4 concatenated with topology tokens SHUFFLED WITHIN type      <- A6-A8 = topology correctly attached
  A9 allShuf     A6 with all feature->vector assignments permuted              (semantic-assignment control)

Topology tokens are the structural relations only (isPartOf, component.feeds, component.cools, ...),
disjoint from the metadata fields, so A6 = A4 + topology is a genuine nesting.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .base import pad_to, l2_normalize
from .embedders import TextEmbedder, _parse_relations


def _onehot(series: pd.Series, prefix: str) -> np.ndarray:
    return pd.get_dummies(series.fillna(""), prefix=prefix).to_numpy(dtype=np.float32)


def _metadata_core(features: pd.DataFrame) -> np.ndarray:
    return np.concatenate([
        _onehot(features["measurement_type"], "type"),
        _onehot(features["unit"], "unit"),
        _onehot(features["component_type"], "comp"),
    ], axis=1)


def _topology_multihot(features: pd.DataFrame):
    """Multi-hot over structural relation tokens (NOT measurement type / metadata)."""
    vocab: dict[str, int] = {}
    per_feat = []
    for _, row in features.iterrows():
        toks = [f"{r}->{t}" for r, t in _parse_relations(row.get("relations", ""))]
        per_feat.append(toks)
        for tk in toks:
            vocab.setdefault(tk, len(vocab))
    M = np.zeros((len(features), len(vocab)), dtype=np.float32)
    for i, toks in enumerate(per_feat):
        for tk in toks:
            M[i, vocab[tk]] = 1.0
    return M


def build_nested_tables(features: pd.DataFrame, dim: int, seed: int) -> dict[str, dict]:
    """Return {name: {feature_id: vec(dim)}} for A1,A3,A4,A5,A6,A8,A9 (A0 handled in-model)."""
    rng = np.random.default_rng(seed)
    fids = list(features["feature_id"])
    meta = _metadata_core(features)                                  # (N, Dm)
    topo = _topology_multihot(features)                              # (N, Dt)
    types = features["measurement_type"].to_numpy()

    a1 = rng.standard_normal((len(fids), 32)).astype(np.float32)
    a3 = _onehot(features["measurement_type"], "type")
    a4 = meta
    a6 = np.concatenate([meta, topo], axis=1)

    # A8: shuffle the topology block among features of the SAME measurement type; keep metadata intact
    topo_shuf = topo.copy()
    for t in np.unique(types):
        idx = np.where(types == t)[0]
        topo_shuf[idx] = topo[rng.permutation(idx)]
    a8 = np.concatenate([meta, topo_shuf], axis=1)

    # A9: permute the full A6 vector across all features (semantic-assignment control)
    a9 = a6[rng.permutation(len(fids))]

    # A5 text (real sentence embedding). Prefer a precomputed cache to avoid loading MiniLM at run
    # time (its transformer load is the dominant memory spike on commit-limited hosts).
    from pathlib import Path
    cache = Path(__file__).resolve().parents[3] / "results/tep_text_emb.npy"
    if cache.exists():
        a5 = np.load(cache).astype(np.float32)
    else:
        text_vecs = TextEmbedder().embed(features)
        a5 = np.stack([text_vecs[f] for f in fids])

    raw = {"A1_random": a1, "A3_type": a3, "A4_metadata": a4, "A5_text": a5,
           "A6_metaTopo": a6, "A8_topoShuf": a8, "A9_allShuf": a9}
    out = {}
    for name, M in raw.items():
        vecs = {fids[i]: M[i] for i in range(len(fids))}
        out[name] = pad_to(vecs, dim)
    return out
