"""Cross-domain replication of the signed-recoverability law on PhysioNet-2012 ICU data.

Cross-sectional: each patient is a sample, each clinical variable a feature. Impute a held-out variable
from the others (query-conditioned) using its text embedding, strict (held variable absent from every
training input, never a training target; patient-level train/test split). Per variable compute the
source-only SIGNED recoverability (best same-GROUP sibling correlation on training patients) and the
realized zero-shot imputation skill, then correlate. Pre-registered prediction (from clinical coupling):
blood-pressure triad and acid-base trio and BUN/Creatinine are recoverable and transfer; Glucose,
Lactate, Temperature are near-independent and do not. Same measurement-type-vs-statistics dissociation
as TEP, in a second, real, clinical domain.
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from ..model.sft_model import SFTImputer
from ..datasets.physionet2012 import build_physionet_features, load_matrix, NAMES, VARIABLES
from ..embeddings.nested import build_nested_tables
from .run_imputation_strict import pearson

ROOT = Path(__file__).resolve().parents[3]
CONDS = ["A1_random", "A4_metadata", "A5_text"]
# spread of targets across the recoverability range (each keeps same-group siblings)
TARGETS = ["MAP", "DiasABP", "NIMAP", "NIDiasABP",   # BP triad (recoverable)
           "pH", "HCO3", "PaCO2",                     # acid-base (recoverable)
           "BUN", "Creatinine",                       # renal (recoverable)
           "AST", "ALT",                              # liver (moderate)
           "Glucose", "Lactate", "Temp", "HR", "WBC", "Platelets"]  # ~independent controls


def signed_recoverability(X, groups, target_idx, sibs_idx):
    if not sibs_idx:
        return 0.0
    cc = []
    for s in sibs_idx:
        m = ~np.isnan(X[:, target_idx]) & ~np.isnan(X[:, s])
        if m.sum() > 30:
            cc.append(np.corrcoef(X[m, target_idx], X[m, s])[0, 1])
    return float(cc[int(np.argmax(np.abs(cc)))]) if cc else 0.0


def run_target(target, feats, groups, X, e_dim, cfg, device, seeds):
    j = NAMES.index(target)
    train_targets = [i for i in range(len(NAMES)) if i != j]
    recs = []
    for seed in seeds:
        rng = np.random.default_rng(1000 + seed)
        perm = rng.permutation(len(X)); cut = int(len(X) * 0.7)
        tr, te = perm[:cut], perm[cut:]
        # standardize per variable from TRAIN patients; median-impute context missingness
        mu = np.nanmean(X[tr], 0); sd = np.nanstd(X[tr], 0); sd = np.where(sd > 1e-6, sd, 1.0)
        med = np.nanmedian(X[tr], 0)
        def prep(rows):
            A = (X[rows] - mu) / sd
            A = np.where(np.isnan(A), 0.0, A).astype(np.float32)   # missing -> 0 (mean)
            return A
        Xtr = prep(tr); Xtr[:, j] = 0.0                            # strict: held var absent from input
        # eval only on test patients where the held variable is actually measured
        te_meas = te[~np.isnan(X[te, j])]
        if len(te_meas) < 30:
            continue
        Yte = prep(te_meas); true_j = ((X[te_meas, j] - mu[j]) / sd[j]).astype(np.float32)
        tables = build_nested_tables(feats, e_dim, seed=seed)
        for cond in CONDS:
            E = torch.tensor(np.stack([tables[cond][f] for f in tables[cond]]).astype(np.float32), device=device)
            model = SFTImputer(e_dim=e_dim, z_dim=cfg["z_dim"]).to(device)
            opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"]); lossf = nn.MSELoss(); bs = 256
            Xtr_t = torch.tensor(Xtr, device=device)
            r_idx = np.repeat(np.arange(len(Xtr)), 6); q_idx = rng.choice(train_targets, size=len(r_idx))
            for _ in range(cfg["epochs"]):
                o = rng.permutation(len(r_idx))
                for s0 in range(0, len(o), bs):
                    b = o[s0:s0 + bs]; vals = Xtr_t[r_idx[b]]; qb = torch.tensor(q_idx[b], device=device)
                    tgt = vals[torch.arange(len(b), device=device), qb]
                    mask = torch.ones(len(b), len(NAMES), device=device); mask[:, j] = 0.0
                    mask[torch.arange(len(b), device=device), qb] = 0.0
                    opt.zero_grad(); lossf(model(vals, E.unsqueeze(0).expand(len(b), -1, -1), qb, mask), tgt).backward(); opt.step()
            model.eval()
            with torch.no_grad():
                Xe = torch.tensor(Yte, device=device); qb = torch.full((len(Yte),), j, device=device)
                mask = torch.ones(len(Yte), len(NAMES), device=device); mask[:, j] = 0.0
                pred = model(Xe, E.unsqueeze(0).expand(len(Yte), -1, -1), qb, mask).cpu().numpy()
            recs.append(dict(target=target, group=groups[target], cond=cond, seed=seed,
                             skill=pearson(true_j, pred)))
            gc.collect()
    return recs


def analyze(X, groups):
    parts = sorted((ROOT / "results").glob("pn12_law_*.parquet"))
    df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    rows = []
    for t in df.target.unique():
        j = NAMES.index(t); sibs = [i for i in range(len(NAMES)) if groups[NAMES[i]] == groups[t] and i != j]
        rec = signed_recoverability(X, groups, j, sibs)
        r = {"target": t, "group": groups[t], "signed_rec": round(rec, 3)}
        for c in CONDS:
            v = df[(df.target == t) & (df.cond == c)]["skill"].values
            r[c + "_signed"] = round(float(np.mean(v)), 3)
            r[c + "_signcons"] = round(float(max((v > 0).mean(), (v < 0).mean())), 2)
        rows.append(r)
    R = pd.DataFrame(rows).sort_values("signed_rec")
    R.to_csv(ROOT / "results/pn12_law.csv", index=False)
    print(R.to_string(index=False))
    def sp(a, b): s = R[[a, b]].dropna(); return round(float(s[a].corr(s[b], method="spearman")), 3)
    out = {"spearman_signedrec_textskill": sp("signed_rec", "A5_text_signed"),
           "spearman_signedrec_metaskill": sp("signed_rec", "A4_metadata_signed"),
           "signcons_random": round(float(R["A1_random_signcons"].mean()), 3),
           "signcons_text": round(float(R["A5_text_signcons"].mean()), 3), "n": int(len(R))}
    (ROOT / "results/pn12_law_summary.json").write_text(json.dumps(out, indent=2))
    print("\n=== PhysioNet-2012 SIGNED-RECOVERABILITY LAW ===")
    print(f"  Spearman(signed_rec, text skill): {out['spearman_signedrec_textskill']}  (n={out['n']})")
    print(f"  Spearman(signed_rec, metadata skill): {out['spearman_signedrec_metaskill']}")
    print(f"  sign-consistency: random {out['signcons_random']}  text {out['signcons_text']}")
    print("wrote pn12_law.csv, pn12_law_summary.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs/phase1.json"))
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()
    feats = build_physionet_features()
    groups = {v[0]: v[2] for v in VARIABLES}
    X = load_matrix()
    if args.merge:
        analyze(X, groups); return
    cfg = json.loads(Path(args.config).read_text()); cfg["epochs"] = args.epochs
    torch.set_num_threads(max(1, min(3, torch.get_num_threads() or 3)))
    device = "cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu")
    e_dim = cfg["e_dim"]
    print(f"[pn12] device={device} patients={len(X)} vars={len(NAMES)} targets={len(TARGETS)}", flush=True)
    for t in TARGETS:
        pq = ROOT / f"results/pn12_law_{t}.parquet"
        if pq.exists():
            print(f"[pn12] {t} done, skipping", flush=True); continue
        recs = run_target(t, feats, groups, X, e_dim, cfg, device, range(args.seeds))
        pd.DataFrame(recs).to_parquet(pq)
        print(f"[pn12] {t} done", flush=True)
    analyze(X, groups)


if __name__ == "__main__":
    main()
