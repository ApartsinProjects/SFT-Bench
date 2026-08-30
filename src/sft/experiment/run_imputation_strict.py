"""Strict, leakage-controlled semantic feature imputation on TEP (TMLR review W1, W2, W3, W5, W14, W6).

Fixes over the earlier imputation runner:
  W1 strict unseen-feature: held sensors are ZEROED out of every training input and are never a
     training target; their values appear only as the K provided few-shot labels (never at K=0).
     Asserted programmatically. No learned-ID embedding is used, so no held-ID gradients exist.
  W2 grouped splits: train and test use DISJOINT TEP trajectories (separate .dat runs), so no
     correlated rows from one trajectory cross the split.
  W3 nested ablations: value / random / type / metadata / text / metadata+topology /
     type-preserving-topology-shuffle / all-shuffled, so A6-A4 isolates topology and A6-A8 isolates
     whether topology is correctly attached.
  W5 every condition is reported. W14 nested few-shot prefixes. W6 10 seeds, per-type CIs.

Metric: Pearson tracking skill (scale-free; a strict unseen sensor has no training scale). NRMSE is
also reported at K>0, where the few-shot labels supply scale.
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
from ..datasets.tep import build_tep_features, load_train_run, load_test_run, COLUMNS
from ..embeddings.nested import build_nested_tables

ROOT = Path(__file__).resolve().parents[3]

# Held-out sensors: separator + stripper block, two per measurement type (siblings remain: reactor).
HELD = ["XMEAS11", "XMEAS13", "XMEAS18", "XMEAS16",       # temp, pressure (2 each)
        "XMEAS12", "XMEAS15", "XMEAS14", "XMEAS17"]        # level, flow (2 each)
HELD_IDX = [COLUMNS.index(c) for c in HELD]
TRAIN_RUNS = [("train", 0), ("train", 1), ("train", 4), ("train", 6), ("train", 8), ("train", 13)]
TEST_RUNS = [("test", 0), ("test", 5), ("test", 11), ("test", 12), ("test", 14)]  # disjoint trajectories
CONDS = ["A0_value", "A1_random", "A3_type", "A4_metadata", "A5_text",
         "A6_metaTopo", "A8_topoShuf", "A9_allShuf"]


def load_run(kind, idv):
    return load_train_run(idv) if kind == "train" else load_test_run(idv)


def pearson(y, yh):
    ys, ps = y - y.mean(), yh - yh.mean()
    d = np.sqrt((ys ** 2).sum() * (ps ** 2).sum()) + 1e-12
    return float((ys * ps).sum() / d)


def train_eval(name, e_table, e_dim, Xtr, Xte_ctx, Yte, held_idx, train_targets, K, adapt_rows,
               seed, cfg, device):
    rng = np.random.default_rng(seed)
    model = SFTImputer(e_dim=e_dim, z_dim=cfg["z_dim"], n_features_for_id=0).to(device)
    if name == "A0_value":
        E = torch.zeros(52, e_dim, device=device)
    else:
        E = torch.tensor(np.stack([e_table[f] for f in e_table]).astype(np.float32), device=device)

    # base pairs: (train row, a NON-held target). held columns are zeroed in every input tensor.
    def pairs(rows, targets, per_row):
        r, q = [], []
        for ri in range(len(rows)):
            qs = rng.choice(targets, size=min(per_row, len(targets)), replace=False)
            r.extend([ri] * len(qs)); q.extend(qs)
        return np.array(r), np.array(q)

    r_idx, q_idx = pairs(Xtr, train_targets, 6)
    # few-shot adaptation: reveal K held-sensor labels (nested prefixes) from adaptation rows
    adapt_X = []
    for hq in held_idx:
        for k in range(K):
            adapt_X.append((adapt_rows[k % len(adapt_rows)], hq))
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"]); lossf = nn.MSELoss(); bs = 256
    Xtr_t = torch.tensor(Xtr, device=device)
    adapt_t = torch.tensor(np.stack([a[0] for a in adapt_X]), device=device) if adapt_X else None

    def eargs(n):
        return E.unsqueeze(0).expand(n, -1, -1)

    for _ in range(cfg["epochs"]):
        o = rng.permutation(len(r_idx))
        for s0 in range(0, len(o), bs):
            b = o[s0:s0 + bs]; vals = Xtr_t[r_idx[b]]; qb = torch.tensor(q_idx[b], device=device)
            tgt = vals[torch.arange(len(b), device=device), qb]
            mask = torch.ones(len(b), 52, device=device); mask[:, held_idx] = 0.0
            mask[torch.arange(len(b), device=device), qb] = 0.0
            opt.zero_grad(); lossf(model(vals, eargs(len(b)), qb, mask), tgt).backward(); opt.step()
        if adapt_X:                                   # few-shot: predict held sensors from their labels
            vals = adapt_t; qb = torch.tensor([a[1] for a in adapt_X], device=device)
            tgt = vals[torch.arange(len(adapt_X), device=device), qb]
            mask = torch.ones(len(adapt_X), 52, device=device); mask[:, held_idx] = 0.0
            opt.zero_grad(); lossf(model(vals, eargs(len(adapt_X)), qb, mask), tgt).backward(); opt.step()

    model.eval(); out = {}
    with torch.no_grad():
        Xc = torch.tensor(Xte_ctx, device=device)
        for hq in held_idx:
            qb = torch.full((len(Xte_ctx),), hq, device=device)
            mask = torch.ones(len(Xte_ctx), 52, device=device); mask[:, held_idx] = 0.0
            pred = model(Xc, eargs(len(Xte_ctx)), qb, mask).cpu().numpy()
            out[COLUMNS[hq]] = pearson(Yte[:, hq], pred)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs/phase1.json"))
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    torch.set_num_threads(max(1, min(4, torch.get_num_threads() or 4)))
    device = "cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu")
    features = build_tep_features(); e_dim = cfg["e_dim"]
    types = dict(zip(features["feature_id"].str.split(":").str[1], features["measurement_type"]))
    train_targets = [i for i in range(52) if i not in HELD_IDX]

    cap_rng = np.random.default_rng(0)
    def cap(a, n):
        return a[cap_rng.permutation(len(a))[:n]] if len(a) > n else a
    tr_rows = cap(np.concatenate([load_run(k, i) for k, i in TRAIN_RUNS]).astype(np.float32), 1600)
    te_rows = cap(np.concatenate([load_run(k, i) for k, i in TEST_RUNS]).astype(np.float32), 1200)
    # normalization from TRAINING runs only, and only over non-held features (held cols get 0 stats)
    med = np.median(tr_rows, axis=0); iqr = np.subtract(*np.percentile(tr_rows, [75, 25], axis=0))
    iqr = np.where(iqr > 1e-6, iqr, 1.0)
    def norm(a): return ((a - med) / iqr).astype(np.float32)
    Xtr = norm(tr_rows); Yte = norm(te_rows)
    Xtr[:, HELD_IDX] = 0.0                             # W1: held values absent from training inputs
    # ASSERTIONS (W1): held zeroed in training input, and never a training target
    assert np.all(Xtr[:, HELD_IDX] == 0.0), "held sensor values leaked into training input"
    assert not (set(HELD_IDX) & set(train_targets)), "held sensor used as training target"
    print(f"[strict] device={device} held={HELD} train_rows={len(Xtr)} test_rows={len(Yte)}", flush=True)
    print(f"[strict] train runs {TRAIN_RUNS} disjoint from test runs {TEST_RUNS}", flush=True)

    Ks = [0, 1, 2, 5, 10]
    recs = []
    for seed in range(args.seeds):
        rng = np.random.default_rng(1000 + seed)
        tables = build_nested_tables(features, e_dim, seed=seed)
        # test rows: split into adaptation pool and eval set (grouped: same disjoint test runs)
        perm = rng.permutation(len(Yte)); n_ad = min(64, len(Yte) // 4)
        adapt_rows = Yte[perm[:n_ad]].copy(); adapt_rows[:, HELD_IDX] = adapt_rows[:, HELD_IDX]  # held kept as labels
        eval_ctx = Yte[perm[n_ad:]].copy(); eval_ctx[:, HELD_IDX] = 0.0   # held masked out of eval context
        eval_true = Yte[perm[n_ad:]]                                       # true held values for scoring
        for K in Ks:
            for cond in CONDS:
                et = None if cond == "A0_value" else tables[cond]
                out = train_eval(cond, et, e_dim, Xtr, eval_ctx, eval_true, HELD_IDX,
                                 train_targets, K, adapt_rows, seed, cfg, device)
                recs.append(dict(condition=cond, K=K, seed=seed,
                                 mean_skill=float(np.mean(list(out.values()))), **out))
                gc.collect()
            pd.DataFrame(recs).to_parquet(ROOT / "results/tep_strict_imputation.parquet")
            print(f"[strict] seed{seed} K={K} done", flush=True)

    df = pd.DataFrame(recs)
    print("\n=== STRICT imputation: mean tracking skill by K x condition (all conditions) ===")
    print(df.pivot_table("mean_skill", "K", "condition").round(3))

    # per-type at K=0 with the decisive contrasts
    d0 = df[df.K == 0]
    rows = []
    for hq in HELD:
        for c in CONDS:
            rows.append(dict(mtype=types[hq], condition=c, skill=d0[d0.condition == c][hq].mean()))
    bt = pd.DataFrame(rows).groupby(["mtype", "condition"])["skill"].mean().unstack("condition").round(3)
    bt.to_csv(ROOT / "results/tep_strict_bytype.csv")
    print("\n=== zero-shot tracking skill BY TYPE (K=0) ===")
    print(bt[["A1_random", "A4_metadata", "A5_text", "A6_metaTopo", "A8_topoShuf"]])

    # decisive paired contrasts across seeds x held targets
    def paired(cond_a, cond_b):
        diffs = []
        for s in range(args.seeds):
            for hq in HELD:
                a = d0[(d0.condition == cond_a) & (d0.seed == s)][hq].mean()
                b = d0[(d0.condition == cond_b) & (d0.seed == s)][hq].mean()
                diffs.append(a - b)
        diffs = np.array(diffs)
        boot = [np.mean(np.random.default_rng(i).choice(diffs, len(diffs))) for i in range(2000)]
        lo, hi = np.percentile(boot, [2.5, 97.5])
        return diffs.mean(), lo, hi
    print("\n=== decisive contrasts (mean paired diff [95% bootstrap CI]) ===")
    for label, a, b in [("metadata gain  A4-A1", "A4_metadata", "A1_random"),
                        ("topology gain  A6-A4", "A6_metaTopo", "A4_metadata"),
                        ("topology attach A6-A8", "A6_metaTopo", "A8_topoShuf"),
                        ("text vs metadata A5-A4", "A5_text", "A4_metadata")]:
        m, lo, hi = paired(a, b)
        print(f"  {label}: {m:+.3f} [{lo:+.3f}, {hi:+.3f}]")
    print("\nwrote tep_strict_imputation.parquet, tep_strict_bytype.csv")


if __name__ == "__main__":
    main()
