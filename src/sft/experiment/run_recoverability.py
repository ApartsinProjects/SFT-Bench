"""E1: the recoverability law. Does a SOURCE-ONLY recoverability statistic predict the realized
zero-shot semantic transfer gain, while semantic similarity does not?

For each held sensor j:
  R_j        source-only recoverability: cross-validated R^2 of predicting j from its same-type
             SIBLINGS with ridge, computed on the TRAINING trajectories only (no test-regime labels).
  sim_j      semantic similarity: cosine of j's metadata embedding to its nearest same-type sibling.
  G_j        realized transfer gain: skill(metadata) - skill(random) at K=0 for j, from the strict
             imputation run (results/tep_strict_seed*.parquet).

Report corr(R, G) and corr(sim, G). The thesis predicts the first is strong and positive, the second
near zero: semantics gate eligibility (same-type siblings), recoverability decides the outcome.

Run AFTER the strict imputation with the E1 held set:
  python -m sft.experiment.run_imputation_strict --seeds 10 --epochs 8 --cpu   (writes per-seed parquets)
  python -m sft.experiment.run_recoverability --cpu
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..datasets.tep import build_tep_features, COLUMNS
from .run_imputation_strict import HELD, HELD_IDX, TRAIN_RUNS, load_run
from ..embeddings.nested import _metadata_core

ROOT = Path(__file__).resolve().parents[3]


def ridge_cv_r2(y, X, folds=5, lam=1.0, seed=0):
    """K-fold CV R^2 of ridge y ~ X (standardized). Pure numpy."""
    n = len(y)
    if X.shape[1] == 0 or n < folds * 2:
        return float("nan")
    rng = np.random.default_rng(seed); idx = rng.permutation(n)
    fold = np.array_split(idx, folds)
    preds = np.zeros(n)
    Xs = (X - X.mean(0)) / (X.std(0) + 1e-9)
    for f in range(folds):
        te = fold[f]; tr = np.concatenate([fold[g] for g in range(folds) if g != f])
        Xtr, ytr = Xs[tr], y[tr]
        A = Xtr.T @ Xtr + lam * np.eye(Xtr.shape[1])
        w = np.linalg.solve(A, Xtr.T @ (ytr - ytr.mean()))
        preds[te] = Xs[te] @ w + ytr.mean()
    ss_res = np.sum((y - preds) ** 2); ss_tot = np.sum((y - y.mean()) ** 2) + 1e-12
    return float(1 - ss_res / ss_tot)


def main():
    import argparse
    argparse.ArgumentParser().parse_args()
    feats = build_tep_features()
    types = dict(zip(feats["feature_id"].str.split(":").str[1], feats["measurement_type"]))
    meta = _metadata_core(feats)                                    # per-feature metadata one-hot
    name_to_row = {c: i for i, c in enumerate(COLUMNS)}

    # source (training) rows, robust-normalized per feature
    tr = np.concatenate([load_run(k, i) for k, i in TRAIN_RUNS]).astype(np.float32)
    med = np.median(tr, 0); iqr = np.subtract(*np.percentile(tr, [75, 25], 0)); iqr = np.where(iqr > 1e-6, iqr, 1.0)
    Xn = (tr - med) / iqr

    # G_j from the strict run
    parts = glob.glob(str(ROOT / "results/tep_strict_seed*.parquet"))
    df = pd.concat([pd.read_parquet(p) for p in parts]); d0 = df[df.K == 0]
    gain = {c: float(d0[d0.condition == "A4_metadata"][c].mean() - d0[d0.condition == "A1_random"][c].mean())
            for c in HELD}

    rows = []
    for j in HELD:
        jt = types[j]
        sibs = [c for c in COLUMNS if types.get(c) == jt and c != j and c not in HELD]  # same-type, in training
        Xsib = Xn[:, [name_to_row[s] for s in sibs]] if sibs else np.zeros((len(Xn), 0))
        R = ridge_cv_r2(Xn[:, name_to_row[j]], Xsib)
        # similarity: cosine of j's metadata to nearest same-type sibling
        mj = meta[name_to_row[j]]
        sims = [float(mj @ meta[name_to_row[s]] / (np.linalg.norm(mj) * np.linalg.norm(meta[name_to_row[s]]) + 1e-9))
                for s in sibs] or [0.0]
        rows.append(dict(sensor=j, mtype=jt, n_sibs=len(sibs), recoverability=R,
                         similarity=max(sims), gain=gain[j]))
    R = pd.DataFrame(rows)
    R.to_csv(ROOT / "results/tep_recoverability.csv", index=False)
    print(R.round(3).to_string(index=False))

    def corr(a, b, method):
        s = R[[a, b]].dropna()
        return float(s[a].corr(s[b], method=method))
    out = {
        "pearson_recoverability_gain": corr("recoverability", "gain", "pearson"),
        "spearman_recoverability_gain": corr("recoverability", "gain", "spearman"),
        "pearson_similarity_gain": corr("similarity", "gain", "pearson"),
        "spearman_similarity_gain": corr("similarity", "gain", "spearman"),
        "n_sensors": int(len(R)),
    }
    (ROOT / "results/tep_recoverability_corr.json").write_text(json.dumps(out, indent=2))
    print("\n=== THE RECOVERABILITY LAW ===")
    print(f"  corr(recoverability, gain): Pearson {out['pearson_recoverability_gain']:+.3f}  "
          f"Spearman {out['spearman_recoverability_gain']:+.3f}")
    print(f"  corr(similarity,     gain): Pearson {out['pearson_similarity_gain']:+.3f}  "
          f"Spearman {out['spearman_similarity_gain']:+.3f}   (n={out['n_sensors']})")
    print("  thesis: recoverability predicts gain; semantic similarity does not.")
    print("wrote tep_recoverability.csv, tep_recoverability_corr.json")


if __name__ == "__main__":
    main()
