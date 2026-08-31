"""E1 (clean): the signed-recoverability law, one sensor held at a time.

For each target sensor (held ALONE, all siblings retained -> no co-holding confound), train the strict
imputer and record, per condition and seed: pooled signed skill, |skill|, and within-run skill. Then
per target compute SOURCE-ONLY signed recoverability = correlation of the target to its best-|corr|
same-type sibling on training data. The thesis:
  - |skill|  ~ |signed recoverability|   (magnitude is set by statistics)
  - sign-consistency across seeds rises random < metadata < text   (polarity is set by semantics)
  - signed skill ~ signed recoverability   (realized transfer = signed recoverability, source-only)
  - failures are PREDICTED: negative recoverability (anti-correlated siblings) -> negative skill.

Resumable: one parquet per target; rerun to fill missing targets, then --merge.
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
from ..datasets.tep import build_tep_features, COLUMNS
from ..embeddings.nested import build_nested_tables
from .run_imputation_strict import load_run, TRAIN_RUNS, TEST_RUNS, pearson

ROOT = Path(__file__).resolve().parents[3]
CONDS = ["A1_random", "A4_metadata", "A5_text", "A6_metaTopo"]
# targets span all types and the full recoverability range; each held ALONE keeps its siblings
TARGETS = ["XMEAS7", "XMEAS13", "XMEAS16",                 # pressure
           "XMEAS9", "XMEAS11", "XMEAS18", "XMEAS21", "XMEAS22",   # temperature
           "XMEAS8", "XMEAS12", "XMEAS15",                 # level
           "XMEAS5", "XMEAS10", "XMEAS14", "XMEAS17",      # flow
           "XMEAS25", "XMEAS30", "XMEAS37",                # composition
           "XMV6", "XMV9", "XMV11"]                        # actuator


def run_target(target, feats, e_dim, cfg, device, seeds, test_run_lens):
    j = COLUMNS.index(target)
    train_targets = [i for i in range(52) if i != j]
    tr = np.concatenate([load_run(k, i) for k, i in TRAIN_RUNS]).astype(np.float32)
    te = np.concatenate([load_run(k, i) for k, i in TEST_RUNS]).astype(np.float32)
    med = np.median(tr, 0); iqr = np.subtract(*np.percentile(tr, [75, 25], 0)); iqr = np.where(iqr > 1e-6, iqr, 1.0)
    Xtr = ((tr - med) / iqr).astype(np.float32); Yte = ((te - med) / iqr).astype(np.float32)
    Xtr[:, j] = 0.0                                        # strict: target absent from training input
    recs = []
    for seed in seeds:
        rng = np.random.default_rng(1000 + seed)
        tables = build_nested_tables(feats, e_dim, seed=seed)
        for cond in CONDS:
            E = torch.tensor(np.stack([tables[cond][f] for f in tables[cond]]).astype(np.float32), device=device)
            model = SFTImputer(e_dim=e_dim, z_dim=cfg["z_dim"]).to(device)
            opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"]); lossf = nn.MSELoss(); bs = 256
            Xtr_t = torch.tensor(Xtr, device=device)
            # training pairs: predict a random non-target sensor from the rest
            r_idx = np.repeat(np.arange(len(Xtr)), 6); q_idx = rng.choice(train_targets, size=len(r_idx))
            for _ in range(cfg["epochs"]):
                o = rng.permutation(len(r_idx))
                for s0 in range(0, len(o), bs):
                    b = o[s0:s0 + bs]; vals = Xtr_t[r_idx[b]]; qb = torch.tensor(q_idx[b], device=device)
                    tgt = vals[torch.arange(len(b), device=device), qb]
                    mask = torch.ones(len(b), 52, device=device); mask[:, j] = 0.0
                    mask[torch.arange(len(b), device=device), qb] = 0.0
                    opt.zero_grad(); lossf(model(vals, E.unsqueeze(0).expand(len(b), -1, -1), qb, mask), tgt).backward(); opt.step()
            model.eval()
            with torch.no_grad():
                Xc = torch.tensor(Yte, device=device)
                qb = torch.full((len(Yte),), j, device=device)
                mask = torch.ones(len(Yte), 52, device=device); mask[:, j] = 0.0
                pred = model(Xc, E.unsqueeze(0).expand(len(Yte), -1, -1), qb, mask).cpu().numpy()
            pooled = pearson(Yte[:, j], pred)
            # within-run skill: pearson per test trajectory, Fisher-z averaged
            zs, off = [], 0
            for L in test_run_lens:
                r = pearson(Yte[off:off + L, j], pred[off:off + L]); off += L
                if not np.isnan(r): zs.append(np.arctanh(np.clip(r, -0.999, 0.999)))
            within = float(np.tanh(np.mean(zs))) if zs else float("nan")
            recs.append(dict(target=target, condition=cond, seed=seed, pooled=pooled, within=within))
            gc.collect()
    return recs


def signed_recoverability(feats):
    types = dict(zip(feats["feature_id"].str.split(":").str[1], feats["measurement_type"]))
    tr = np.concatenate([load_run(k, i) for k, i in TRAIN_RUNS]).astype(np.float32)
    n2i = {c: i for i, c in enumerate(COLUMNS)}
    out = {}
    for t in TARGETS:
        sibs = [c for c in COLUMNS if types[c] == types[t] and c != t]
        cc = [np.corrcoef(tr[:, n2i[t]], tr[:, n2i[s]])[0, 1] for s in sibs] or [0.0]
        out[t] = float(cc[int(np.argmax(np.abs(cc)))])
    return out, types


def analyze():
    feats = build_tep_features()
    parts = sorted((ROOT / "results").glob("tep_law_*.parquet"))
    df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    rec, types = signed_recoverability(feats)
    rows = []
    for t in df.target.unique():
        d = df[df.target == t]
        r = {"target": t, "type": types[t], "signed_rec": round(rec[t], 3)}
        for c in CONDS:
            v = d[d.condition == c]["pooled"].values
            w = d[d.condition == c]["within"].values
            r[c + "_signed"] = round(float(np.mean(v)), 3)
            r[c + "_absr"] = round(float(np.mean(np.abs(v))), 3)
            r[c + "_signcons"] = round(float(max((v > 0).mean(), (v < 0).mean())), 2)
            r[c + "_within"] = round(float(np.nanmean(w)), 3)
        rows.append(r)
    R = pd.DataFrame(rows).sort_values("signed_rec")
    R.to_csv(ROOT / "results/tep_law.csv", index=False)
    print(R.to_string(index=False))
    def sp(a, b):
        s = R[[a, b]].dropna(); return round(float(s[a].corr(s[b], method="spearman")), 3), round(float(s[a].corr(s[b])), 3)
    law = {
        "spearman_signedrec_metaskill": sp("signed_rec", "A4_metadata_signed"),
        "spearman_signedrec_textskill": sp("signed_rec", "A5_text_signed"),
        "signcons_random": round(float(R["A1_random_signcons"].mean()), 3),
        "signcons_metadata": round(float(R["A4_metadata_signcons"].mean()), 3),
        "signcons_text": round(float(R["A5_text_signcons"].mean()), 3),
        "n_targets": int(len(R)),
    }
    (ROOT / "results/tep_law_summary.json").write_text(json.dumps(law, indent=2))
    print("\n=== SIGNED-RECOVERABILITY LAW ===")
    print(f"  Spearman(signed_rec, metadata signed skill): {law['spearman_signedrec_metaskill']}  (n={law['n_targets']})")
    print(f"  Spearman(signed_rec, text signed skill):     {law['spearman_signedrec_textskill']}")
    print(f"  mean sign-consistency: random {law['signcons_random']}  metadata {law['signcons_metadata']}  text {law['signcons_text']}")
    print("wrote tep_law.csv, tep_law_summary.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs/phase1.json"))
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()
    if args.merge:
        analyze(); return
    cfg = json.loads(Path(args.config).read_text()); cfg["epochs"] = args.epochs
    torch.set_num_threads(max(1, min(3, torch.get_num_threads() or 3)))
    device = "cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu")
    feats = build_tep_features(); e_dim = cfg["e_dim"]
    test_run_lens = [len(load_run(k, i)) for k, i in TEST_RUNS]
    for t in TARGETS:
        pq = ROOT / f"results/tep_law_{t}.parquet"
        if pq.exists():
            print(f"[law] {t} done, skipping", flush=True); continue
        recs = run_target(t, feats, e_dim, cfg, device, range(args.seeds), test_run_lens)
        pd.DataFrame(recs).to_parquet(pq)
        print(f"[law] {t} done (signed_rec computed at merge)", flush=True)
    analyze()


if __name__ == "__main__":
    main()
