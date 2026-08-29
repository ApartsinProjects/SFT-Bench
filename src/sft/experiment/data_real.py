"""Assemble real SDAHU windows into train/test tensors for each Phase-1 split.

What this measures (stated honestly after inspecting the data): SDAHU bias faults are logged as the
TRUE sensor value and manifest as SYSTEM-WIDE control-loop shifts, not a clean offset on one logged
column (verified: oa_bias shows 0.00 shift on the logged OA_TEMP, which is the shared TMY weather
input). So this is fault-FAMILY transfer, not clean feature-localised transfer. The transfer question
is: does training on some fault families, plus semantic feature embeddings e_j, help detect a
HELD-OUT fault family vs normal? The hypothesised mechanism is that the shared encoder F_theta(h_j,
e_j), keyed on semantics, relates the held-out family's affected component/sensors to trained ones;
whether that actually helps here is exactly what Phase 1 measures (uncertain, not assumed).

Transfer pairs are chosen so the held-out family has a semantic SIBLING in training:
  standard             all families pooled, window-level random split.
  feature_cold_start   hold out `oa_bias`; sibling `coi_bias` (both temperature-sensor biases) stays
                       in train; test oa_bias-vs-normal; few-shot adds K held-out windows back.
  component_cold_start hold out `damper_stuck`; sibling `coi_stuck` (both actuator-stuck) in train;
                       test damper_stuck-vs-normal.

Normalization is robust (median/IQR) per feature, estimated from ALL windows unlabelled
(label-independent), matching the deployment assumption in the research plan.
"""
from __future__ import annotations

import numpy as np

import pandas as pd

from ..datasets.lbnl_sdahu import INVENTORY, EXPECTED_POINTS, CSV_DIR, FaultCase


def _windows_for(case: FaultCase, window: int, cap: int) -> np.ndarray:
    """Up to `cap` non-overlapping windows spread across the file. Memory-lean: reads the CSV once
    (only the 30 point columns, float32) and slices ONLY the windows we keep, instead of stacking
    every window then subsampling (which peaks at ~600 MB/file)."""
    path = CSV_DIR / case.filename
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run `python -m sft.datasets.lbnl_sdahu --download`")
    cols = {c.strip(): c for c in pd.read_csv(path, nrows=0).columns}
    usecols = [cols[p] for p in EXPECTED_POINTS]
    arr = pd.read_csv(path, usecols=usecols, dtype="float32")[usecols].to_numpy()   # (T, F)
    T = arr.shape[0]
    n_full = (T - window) // window + 1
    if n_full <= 0:
        return np.empty((0, len(EXPECTED_POINTS), window), dtype=np.float32)
    starts = (np.linspace(0, n_full - 1, min(cap, n_full)).round().astype(int)) * window
    out = np.empty((len(starts), len(EXPECTED_POINTS), window), dtype=np.float32)
    for i, s in enumerate(starts):
        out[i] = arr[s:s + window].T
    return out


def assemble(window: int = 60, cap_per_file: int = 200, cache: dict | None = None):
    """Return dict: family -> (X, meta) with X (n, F, W) and per-window family/component/label.
    Caches raw per-family window arrays so repeated split builds don't re-read the CSVs."""
    if cache is not None and "byfam" in cache:
        return cache["byfam"]
    byfam: dict[str, dict] = {}
    for case in INVENTORY:
        X = _windows_for(case, window, cap_per_file)
        fam = case.family
        d = byfam.setdefault(fam, {"X": [], "component": case.component, "label": case.label})
        d["X"].append(X)
    for fam, d in byfam.items():
        d["X"] = np.concatenate(d["X"], axis=0) if d["X"] else np.empty((0,))
    if cache is not None:
        cache["byfam"] = byfam
    return byfam


def robust_stats(X_all: np.ndarray):
    """Per-feature median and IQR over pooled windows (label-independent)."""
    flat = X_all.transpose(1, 0, 2).reshape(X_all.shape[1], -1)   # (F, n*W)
    med = np.median(flat, axis=1)
    q1, q3 = np.percentile(flat, [25, 75], axis=1)
    iqr = np.where((q3 - q1) > 1e-6, q3 - q1, 1.0)
    return med.astype(np.float32), iqr.astype(np.float32)


def _norm(X, med, iqr):
    return ((X - med[None, :, None]) / iqr[None, :, None]).astype(np.float32)


def build_split(split_name: str, byfam: dict, seed: int, K: int | str = "full",
                test_frac: float = 0.3):
    """Return (Xtr, ytr, Xte, yte). Normal windows come from the `fault_free` family.

    For cold-start, `holdout` fault family is absent from training except K windows (few-shot);
    the test set is held-out-fault vs held-out-normal.
    """
    rng = np.random.default_rng(seed)
    normal = byfam["fault_free"]["X"]
    fault_families = [f for f in byfam if f != "fault_free"]

    # pooled robust stats over everything (unlabelled)
    pool = np.concatenate([byfam[f]["X"] for f in byfam if len(byfam[f]["X"])], axis=0)
    med, iqr = robust_stats(pool)

    def split_idx(n):
        perm = rng.permutation(n)
        cut = int(n * (1 - test_frac))
        return perm[:cut], perm[cut:]

    n_tr, n_te = split_idx(len(normal))
    neg_tr, neg_te = normal[n_tr], normal[n_te]

    if split_name == "standard":
        pos = np.concatenate([byfam[f]["X"] for f in fault_families], axis=0)
        p_tr, p_te = split_idx(len(pos))
        Xtr = np.concatenate([neg_tr, pos[p_tr]]); ytr = np.r_[np.zeros(len(neg_tr)), np.ones(len(p_tr))]
        Xte = np.concatenate([neg_te, pos[p_te]]); yte = np.r_[np.zeros(len(neg_te)), np.ones(len(p_te))]
    else:
        holdout = {"feature_cold_start": "oa_bias",
                   "component_cold_start": "damper_stuck"}[split_name]
        train_fams = [f for f in fault_families if f != holdout]
        pos_tr = np.concatenate([byfam[f]["X"] for f in train_fams], axis=0)
        hold = byfam[holdout]["X"]
        h_tr, h_te = split_idx(len(hold))
        # few-shot: add K held-out-fault windows to training (K=0 => zero-shot)
        if K == "full":
            add = hold[h_tr]
        else:
            add = hold[h_tr[:int(K)]]
        pos_tr_all = np.concatenate([pos_tr, add], axis=0) if len(add) else pos_tr
        Xtr = np.concatenate([neg_tr, pos_tr_all])
        ytr = np.r_[np.zeros(len(neg_tr)), np.ones(len(pos_tr_all))]
        Xte = np.concatenate([neg_te, hold[h_te]])
        yte = np.r_[np.zeros(len(neg_te)), np.ones(len(h_te))]

    p = rng.permutation(len(ytr))
    return (_norm(Xtr[p], med, iqr), ytr[p].astype(np.float32),
            _norm(Xte, med, iqr), yte.astype(np.float32))
