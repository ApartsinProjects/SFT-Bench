"""Pre-registered invariants (docs/phase1_spec.md section 4). Each check states its expected outcome
BEFORE the run; a violation is a bug to fix, not a finding to report. Input is the tidy results frame
with columns: condition, split, K, seed, auroc (and optionally a plain-MLP baseline auroc).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def _mean(df, **sel):
    q = df
    for k, v in sel.items():
        q = q[q[k] == v]
    return float(q["auroc"].mean()) if len(q) else float("nan")


def check_invariants(results: pd.DataFrame, mlp_baseline_auroc: float | None = None) -> dict:
    checks: list[dict] = []

    def record(name, expected, passed, detail):
        checks.append({"invariant": name, "expected": expected,
                       "passed": bool(passed), "detail": detail})

    # 3. ordering Correct-KG >= Shuffled-KG >= Random at low K
    lowK = results[results["K"] <= 5]
    kg = _mean(lowK, condition="C5"); sh = _mean(lowK, condition="C6"); rnd = _mean(lowK, condition="C1")
    record("ordering_kg>=shuffled>=random", "C5>=C6>=C1 at K<=5",
           not (np.isnan(kg) or np.isnan(sh) or np.isnan(rnd)) and kg + 1e-6 >= sh >= rnd - 1e-6,
           {"C5": kg, "C6": sh, "C1": rnd})

    # 2. monotonicity in K (mean auroc non-decreasing per condition, allowing small noise)
    mono = {}
    for c in results["condition"].unique():
        by_k = results[results["condition"] == c].groupby("K")["auroc"].mean().sort_index()
        diffs = np.diff(by_k.to_numpy())
        mono[c] = bool((diffs >= -0.02).all())        # tolerate 0.02 AUROC dips as noise
    record("monotonic_in_K", "auroc non-decreasing in K (tol 0.02)", all(mono.values()), mono)

    # 5. learned-ID cannot zero-shot: C2 at K==0 on feature_cold_start ~ chance
    c2_zs = _mean(results, condition="C2", split="feature_cold_start", K=0)
    record("learned_id_cannot_zeroshot", "C2 K=0 on feature_cold_start ~ 0.5",
           np.isnan(c2_zs) or c2_zs <= 0.6, {"C2_zeroshot_auroc": c2_zs})

    # 1. degenerate: value-only full-data ~ plain MLP (only if a baseline was supplied)
    if mlp_baseline_auroc is not None:
        c0_full = _mean(results, condition="C0", split="standard")
        record("degenerate_matches_mlp", "C0 standard ~ plain MLP (within 0.03)",
               np.isnan(c0_full) or abs(c0_full - mlp_baseline_auroc) <= 0.03,
               {"C0": c0_full, "mlp": mlp_baseline_auroc})

    passed = sum(c["passed"] for c in checks)
    return {"n_checks": len(checks), "n_passed": passed,
            "all_passed": passed == len(checks), "checks": checks}


def write_sanity(results: pd.DataFrame, out_path: str | Path, **kw) -> dict:
    rep = check_invariants(results, **kw)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(rep, indent=2))
    return rep
