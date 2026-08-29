"""The concrete feature embedders for conditions C1, C3, C4, C5.

C0 (value-only) and C2 (learned-ID) are handled in the model, not here: C0 uses a zero embedding,
C2 uses a trainable nn.Embedding keyed on feature index (and by construction cannot embed an unseen
feature, invariant 5). C6 (shuffled-KG) is C5 passed through base.shuffle_assign.

    C1 RandomEmbedder     fixed random vector per feature           (controls for extra dims)
    C3 TextEmbedder       sentence embedding of FeatureMeta.text_blob (the real competitor)
    C4 MetadataEmbedder   one-hot of (measurement_type, component_type, unit), no topology
    C5 KGEmbedder         inductive multi-hot over (relation, neighbour-class) pairs

The KG embedder is inductive: an unseen feature gets a vector purely from its own relations, so
zero-shot feature cold-start (Experiment 3) is well defined. It is deliberately simple (a
relation-typed bag-of-neighbours); the documented upgrade path is an R-GCN / GraphSAGE encoder
trained on the graph. Keeping C5 simple first makes the KG-vs-text and KG-vs-metadata reads honest
before adding model capacity that could confound them.
"""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from .base import pad_to, l2_normalize


def _parse_relations(cell: str) -> list[tuple[str, str]]:
    if not isinstance(cell, str) or not cell:
        return []
    out = []
    for part in cell.split(";"):
        if "->" in part:
            r, t = part.split("->", 1)
            out.append((r, t))
    return out


class RandomEmbedder:
    name = "random"
    def __init__(self, dim: int = 32, seed: int = 0):
        self.dim, self.seed = dim, seed
    def embed(self, features: pd.DataFrame) -> dict[str, np.ndarray]:
        rng = np.random.default_rng(self.seed)
        return {fid: rng.standard_normal(self.dim).astype(np.float32)
                for fid in features["feature_id"]}


class MetadataEmbedder:
    """One-hot over categorical metadata columns; no graph topology (condition C4)."""
    name = "metadata"
    def __init__(self, cols=("measurement_type", "component_type", "unit")):
        self.cols = cols
    def embed(self, features: pd.DataFrame) -> dict[str, np.ndarray]:
        onehots = [pd.get_dummies(features[c].fillna(""), prefix=c) for c in self.cols]
        mat = pd.concat(onehots, axis=1).to_numpy(dtype=np.float32)
        return {fid: mat[i] for i, fid in enumerate(features["feature_id"])}


class KGEmbedder:
    """Inductive relation-typed bag-of-neighbour-classes (condition C5).

    For each feature, build a multi-hot over (relation, neighbour-local-name) pairs observed in its
    graph neighbourhood. Vocabulary is fit on the training features; an unseen feature is embedded by
    the same vocabulary from its own relations (inductive). Neighbour-class generalisation (mapping a
    neighbour instance to its Brick class) is the natural next refinement.
    """
    name = "kg"
    def __init__(self):
        self.vocab: dict[str, int] = {}
    def _tokens(self, meta_row) -> list[str]:
        toks = [f"self.type={meta_row['measurement_type']}"]
        for r, t in _parse_relations(meta_row.get("relations", "")):
            toks.append(f"{r}={t}")
        return toks
    def fit(self, features: pd.DataFrame) -> "KGEmbedder":
        vocab: dict[str, int] = {}
        for _, row in features.iterrows():
            for tok in self._tokens(row):
                vocab.setdefault(tok, len(vocab))
        self.vocab = vocab
        return self
    def embed(self, features: pd.DataFrame) -> dict[str, np.ndarray]:
        if not self.vocab:
            self.fit(features)
        D = len(self.vocab)
        out = {}
        for _, row in features.iterrows():
            v = np.zeros(D, dtype=np.float32)
            for tok in self._tokens(row):
                j = self.vocab.get(tok)
                if j is not None:
                    v[j] = 1.0
            out[row["feature_id"]] = v
        return out


class TextEmbedder:
    """Sentence embedding of FeatureMeta.text_blob (condition C3, the decisive competitor).

    Prefers sentence-transformers ('all-MiniLM-L6-v2'); if unavailable, falls back to a deterministic
    hashed-bag-of-words placeholder and PRINTS A WARNING. The placeholder is only to keep the pipeline
    runnable in smoke mode; a real text encoder is required before any KG-vs-text claim is reported,
    because a weak text baseline would make KG look better than it is (invariant-style discipline).
    """
    name = "text"
    _CACHE: dict = {}          # class-level cache so the transformer loads at most once per process
    def __init__(self, model: str = "all-MiniLM-L6-v2", dim: int = 256):
        self.model_name, self.dim, self._model = model, dim, None
    def _load(self):
        if self._model is not None:
            return
        if self.model_name in TextEmbedder._CACHE:
            self._model = TextEmbedder._CACHE[self.model_name]
            return
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
        except Exception as e:  # noqa: BLE001
            print(f"[TextEmbedder] WARNING: sentence-transformers unavailable ({e}); "
                  f"using hashed-BoW PLACEHOLDER. Do NOT report KG-vs-text from this.")
            self._model = "hash"
        TextEmbedder._CACHE[self.model_name] = self._model
    def _hash_embed(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        for w in text.lower().split():
            h = int(hashlib.md5(w.encode()).hexdigest(), 16)  # noqa: S324 - not security
            v[h % self.dim] += 1.0
        return v
    def embed(self, features: pd.DataFrame) -> dict[str, np.ndarray]:
        from ..schema import FeatureMeta  # local import to avoid cycle
        self._load()
        blobs, fids = [], []
        for _, row in features.iterrows():
            fm = FeatureMeta(feature_id=row["feature_id"], name=row["name"], dataset=row["dataset"],
                             unit=row.get("unit", ""), measurement_type=row.get("measurement_type", ""),
                             component=row.get("component", ""), description=row.get("description", ""))
            blobs.append(fm.text_blob()); fids.append(row["feature_id"])
        if self._model == "hash":
            vecs = [self._hash_embed(b) for b in blobs]
        else:
            vecs = list(self._model.encode(blobs, normalize_embeddings=True))
        return {fid: np.asarray(v, dtype=np.float32) for fid, v in zip(fids, vecs)}
