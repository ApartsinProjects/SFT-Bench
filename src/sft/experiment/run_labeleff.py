"""Label-efficiency / cold-start learning curve: does a known sensor embedding cut the data needed?

Pretrain a shared imputer on the training sensors (with fixed semantic context embeddings), then
COLD-START a held-out sensor j: adapt j's OWN embedding on K labeled samples and measure test skill
vs K. Two initializations, identical adaptation:
  - semantic : j's embedding starts at its metadata/text/KG embedding (a warm start from siblings);
  - fresh    : j's embedding starts random (a fresh per-sensor ID with no prior).
Claim: the semantic curve reaches a given skill with fewer K, and the two converge as K grows. The
area between them is the labeled-data saving from knowing the sensor's semantics. Reported for
recoverable and non-recoverable targets (the latter should show no benefit either way).

Resumable: one parquet per seed. python -m sft.experiment.run_labeleff --seeds 5 --cpu ; then --merge
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
TARGETS = ["XMEAS16", "XMEAS22", "XMV11", "XMEAS18"]   # 3 recoverable + 1 non-recoverable (temp)
KS = [0, 1, 2, 5, 10, 20]
INIT_EMB = "A4_metadata"           # semantic init source (metadata); text/KG variants easy to add


def pretrain(Xtr, E_ctx, train_targets, e_dim, cfg, device, rng):
    model = SFTImputer(e_dim=e_dim, z_dim=cfg["z_dim"]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"]); lossf = nn.MSELoss(); bs = 256
    Xtr_t = torch.tensor(Xtr, device=device)
    r_idx = np.repeat(np.arange(len(Xtr)), 6); q_idx = rng.choice(train_targets, size=len(r_idx))
    for _ in range(cfg["epochs"]):
        o = rng.permutation(len(r_idx))
        for s0 in range(0, len(o), bs):
            b = o[s0:s0 + bs]; vals = Xtr_t[r_idx[b]]; qb = torch.tensor(q_idx[b], device=device)
            tgt = vals[torch.arange(len(b), device=device), qb]
            mask = torch.ones(len(b), 52, device=device)
            mask[torch.arange(len(b), device=device), qb] = 0.0
            opt.zero_grad(); lossf(model(vals, E_ctx.unsqueeze(0).expand(len(b), -1, -1), qb, mask), tgt).backward(); opt.step()
    return model


def adapt_eval(model, E_ctx, j, e_init, K, adapt_X, eval_X, eval_true, e_dim, device, rng):
    """Freeze model; train ONLY j's embedding e_j (init = e_init) on K samples; eval signed skill."""
    e_j = torch.tensor(e_init.copy(), device=device, requires_grad=True)
    def Efull(bs):
        E = E_ctx.unsqueeze(0).expand(bs, -1, -1).clone()
        E[:, j, :] = e_j
        return E
    if K > 0:
        opt = torch.optim.Adam([e_j], lr=0.05); lossf = nn.MSELoss()
        Xa = torch.tensor(adapt_X[:K], device=device)
        qb = torch.full((K,), j, device=device); tgt = Xa[:, j]
        mask = torch.ones(K, 52, device=device); mask[:, j] = 0.0
        for _ in range(60):
            opt.zero_grad(); lossf(model(Xa, Efull(K), qb, mask), tgt).backward(); opt.step()
    model.eval()
    with torch.no_grad():
        Xe = torch.tensor(eval_X, device=device)
        qb = torch.full((len(eval_X),), j, device=device)
        mask = torch.ones(len(eval_X), 52, device=device); mask[:, j] = 0.0
        pred = model(Xe, Efull(len(eval_X)), qb, mask).cpu().numpy()
    return pearson(eval_true, pred)


def run_seed(seed, feats, e_dim, cfg, device):
    rng = np.random.default_rng(1000 + seed)
    tables = build_nested_tables(feats, e_dim, seed=seed)
    E_sem = np.stack([tables[INIT_EMB][f] for f in tables[INIT_EMB]]).astype(np.float32)
    E_ctx = torch.tensor(E_sem, device=device)
    tr = np.concatenate([load_run(k, i) for k, i in TRAIN_RUNS]).astype(np.float32)
    te = np.concatenate([load_run(k, i) for k, i in TEST_RUNS]).astype(np.float32)
    if len(tr) > 1500:                               # cap for tractable per-target pretraining
        tr = tr[rng.permutation(len(tr))[:1500]]
    if len(te) > 1200:
        te = te[rng.permutation(len(te))[:1200]]
    med = np.median(tr, 0); iqr = np.subtract(*np.percentile(tr, [75, 25], 0)); iqr = np.where(iqr > 1e-6, iqr, 1.0)
    Xtr = ((tr - med) / iqr).astype(np.float32); Yte = ((te - med) / iqr).astype(np.float32)
    recs = []
    for target in TARGETS:
        j = COLUMNS.index(target)
        train_targets = [i for i in range(52) if i != j]
        Xtr_j = Xtr.copy(); Xtr_j[:, j] = 0.0            # strict: target absent from pretraining input
        model = pretrain(Xtr_j, E_ctx, train_targets, e_dim, cfg, device, np.random.default_rng(seed))
        # adaptation pool vs eval set from disjoint test rows
        perm = rng.permutation(len(Yte)); n_ad = 40
        adapt_X = Yte[perm[:n_ad]].copy()                # keeps j value as the label
        eval_X = Yte[perm[n_ad:]].copy(); eval_X[:, j] = 0.0
        eval_true = Yte[perm[n_ad:], j]
        e_rand = rng.standard_normal(e_dim).astype(np.float32) * 0.1
        for K in KS:
            for cond, e_init in [("semantic", E_sem[j]), ("fresh", e_rand)]:
                sk = adapt_eval(model, E_ctx, j, e_init, K, adapt_X, eval_X, eval_true, e_dim, device, rng)
                recs.append(dict(target=target, cond=cond, K=K, seed=seed, skill=sk))
                gc.collect()
        print(f"[labeleff] seed{seed} {target} done", flush=True)
    return recs


def analyze():
    parts = sorted((ROOT / "results").glob("tep_labeleff_seed*.parquet"))
    df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    df.to_parquet(ROOT / "results/tep_labeleff.parquet")
    # use |skill| for the curve (magnitude reached); sign handled by K>=1 adaptation
    df["absk"] = df["skill"].abs()
    piv = df.pivot_table("absk", ["target", "K"], "cond").round(3)
    piv.to_csv(ROOT / "results/tep_labeleff_curve.csv")
    print("\n=== |skill| vs K (semantic warm-start vs fresh-ID), by target ===")
    print(piv)
    # label saving: smallest K where semantic and fresh reach 0.6 |skill| (recoverable ceiling proxy)
    print("\n=== K to reach |skill|>=0.6 (fewer = better) ===")
    for t in df.target.unique():
        d = df[df.target == t]
        def kreach(c):
            g = d[d.cond == c].groupby("K")["absk"].mean()
            hit = g[g >= 0.6]
            return int(hit.index.min()) if len(hit) else None
        print(f"  {t:9s} semantic K={kreach('semantic')}  fresh K={kreach('fresh')}")
    print("wrote tep_labeleff.parquet, tep_labeleff_curve.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs/phase1.json"))
    ap.add_argument("--seeds", type=int, default=5)
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
    for seed in range(args.seeds):
        pq = ROOT / f"results/tep_labeleff_seed{seed}.parquet"
        if pq.exists():
            print(f"[labeleff] seed{seed} done, skipping", flush=True); continue
        recs = run_seed(seed, feats, e_dim, cfg, device)
        pd.DataFrame(recs).to_parquet(pq)
    analyze()


if __name__ == "__main__":
    main()
