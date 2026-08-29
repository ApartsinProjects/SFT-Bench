"""Phase-3 (strengthened): semantic feature IMPUTATION on TEP.

Predict a held-out QUERY feature's value from the other features and the query's semantic embedding
e_q, via query-conditioned cross-attention (the model must attend, using e_q, to the right context
features). This is the task where meaning cannot be sidestepped. Held-out queries span every
measurement type that has same-type siblings, so we can read transfer BY TYPE. Data pools fault-free
and fault regions so variables actually move (steady-state d00 alone caps imputability).

Outputs (results/):
  tep_imputation.parquet  per (condition, K, seed) held-query R^2 and per-feature R^2.
  tep_impute_bytype.csv   per measurement-type transfer R^2 (winning vs shuffled) with CIs.
  tep_semantic_distance.csv  per held query: semantic similarity to nearest same-type sibling vs
                             zero-shot transfer R^2 (the semantic-distance-vs-transfer-gain scatter).
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
from ..datasets.tep import build_tep_features, load_train_run, load_test_run, COLUMNS, FAULT_ONSET
from ..embeddings.embedders import KGEmbedder
from .conditions import CONDITIONS, build_embedding_table

ROOT = Path(__file__).resolve().parents[3]

# Held-out query features spanning every type that keeps same-type siblings in training.
HELD_QUERIES = ["XMEAS16",          # Pressure   (siblings 7,13)
                "XMEAS18", "XMEAS22",# Temperature(siblings 9,11,21)
                "XMEAS15",          # Level      (siblings 8,12)
                "XMEAS17", "XMEAS14",# Flow       (many siblings)
                "XMEAS37", "XMEAS30",# Composition(siblings 23-41)
                "XMV9", "XMV11"]     # Actuator   (many siblings)
HELD_IDX = [COLUMNS.index(c) for c in HELD_QUERIES]
FAULT_RUNS = [1, 4, 5, 6, 8, 11, 12, 13, 14]     # pool fault regions so variables vary


def load_rows(cap=3500, seed=0) -> np.ndarray:
    rows = [load_train_run(0), load_test_run(0)]                 # fault-free
    for idv in FAULT_RUNS:
        rows.append(load_test_run(idv)[FAULT_ONSET:])           # post-onset (variables moving)
    X = np.concatenate(rows, axis=0).astype(np.float32)
    if len(X) > cap:
        X = X[np.random.default_rng(seed).permutation(len(X))[:cap]]
    return X


def r2(y, yhat):
    ss_res = np.sum((y - yhat) ** 2); ss_tot = np.sum((y - y.mean()) ** 2) + 1e-12
    return float(1 - ss_res / ss_tot)


def skill(y, yhat):
    """Robust, bounded metric: Pearson correlation between prediction and truth (how well the
    zero-shot imputation TRACKS the true sensor). Insensitive to the scale blow-ups that make R^2
    unstable when normal and fault regimes are pooled."""
    ys, ps = y - y.mean(), yhat - yhat.mean()
    d = np.sqrt((ys ** 2).sum() * (ps ** 2).sum()) + 1e-12
    return float((ys * ps).sum() / d)


def train_eval_impute(cond_kind, e_table, e_dim, Xtr, Xte, train_q, K, seed, cfg, device):
    n_features = 52
    rng = np.random.default_rng(seed)
    model = SFTImputer(e_dim=e_dim, z_dim=cfg["z_dim"],
                       n_features_for_id=(n_features if cond_kind == "learned_id" else 0)).to(device)
    if cond_kind not in ("learned_id", "zero"):
        E = torch.tensor(np.stack([e_table[f] for f in e_table]).astype(np.float32), device=device)
    elif cond_kind == "zero":
        E = torch.zeros(n_features, e_dim, device=device)
    else:
        E = None

    def pairs(rows_idx, queries, per_row):
        r, q = [], []
        for ri in rows_idx:
            qs = rng.choice(queries, size=min(per_row, len(queries)), replace=False)
            r.extend([ri] * len(qs)); q.extend(qs)
        return np.array(r), np.array(q)

    tr = np.arange(len(Xtr))
    r_idx, q_idx = pairs(tr, train_q, per_row=6)
    if K > 0:
        for hq in HELD_IDX:
            picks = rng.choice(tr, size=min(K, len(tr)), replace=False)
            r_idx = np.concatenate([r_idx, picks]); q_idx = np.concatenate([q_idx, [hq] * len(picks)])
    Xtr_t = torch.tensor(Xtr, device=device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"]); lossf = nn.MSELoss(); bs = 256
    o = rng.permutation(len(r_idx)); r_idx, q_idx = r_idx[o], q_idx[o]
    for _ in range(cfg["epochs"]):
        for s0 in range(0, len(r_idx), bs):
            rb, qb = r_idx[s0:s0 + bs], q_idx[s0:s0 + bs]
            vals = Xtr_t[rb]; qb_t = torch.tensor(qb, device=device)
            tgt = vals[torch.arange(len(qb), device=device), qb_t]
            mask = torch.ones(len(qb), n_features, device=device)
            mask[torch.arange(len(qb), device=device), qb_t] = 0.0
            eb = (E.unsqueeze(0).expand(len(qb), -1, -1) if E is not None
                  else torch.zeros(len(qb), n_features, e_dim, device=device))
            opt.zero_grad(); lossf(model(vals, eb, qb_t, mask), tgt).backward(); opt.step()

    probe = [COLUMNS.index("XMEAS7"), COLUMNS.index("XMEAS9")]
    model.eval(); Xte_t = torch.tensor(Xte, device=device); out = {}
    with torch.no_grad():
        for q, tag in [(h, COLUMNS[h]) for h in HELD_IDX] + [(p, "probe_" + COLUMNS[p]) for p in probe]:
            qb_t = torch.full((len(Xte),), q, device=device); tgt = Xte_t[:, q]
            mask = torch.ones(len(Xte), n_features, device=device); mask[:, q] = 0.0
            eb = (E.unsqueeze(0).expand(len(Xte), -1, -1) if E is not None
                  else torch.zeros(len(Xte), n_features, e_dim, device=device))
            yhat = model(Xte_t, eb, qb_t, mask).cpu().numpy(); yt = tgt.cpu().numpy()
            out[tag] = r2(yt, yhat)
            out["skill_" + tag] = skill(yt, yhat)      # robust bounded metric
    return out


def semantic_distance_table(features, train_q):
    """For each held query, cosine similarity of its raw KG embedding to the nearest SAME-TYPE
    training feature (the semantic proximity that should enable transfer)."""
    kg = KGEmbedder().embed(features)                      # raw multi-hot vectors
    fid = {r.feature_id.split(":")[1]: r.feature_id for _, r in features.iterrows()}
    types = dict(zip(features["feature_id"].str.split(":").str[1], features["measurement_type"]))
    def cos(a, b):
        return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
    rows = []
    train_names = {COLUMNS[i] for i in train_q}
    for hq in HELD_QUERIES:
        sib = [c for c in train_names if types.get(c) == types.get(hq)]
        sims = [cos(kg[fid[hq]], kg[fid[c]]) for c in sib] or [0.0]
        rows.append(dict(query=hq, mtype=types.get(hq), n_siblings=len(sib), max_sim=max(sims)))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs/phase1.json"))
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    torch.set_num_threads(max(1, min(4, torch.get_num_threads() or 4)))
    device = "cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu")
    features = build_tep_features(); e_dim = cfg["e_dim"]
    train_q = [i for i in range(52) if i not in HELD_IDX]
    types = dict(zip(features["feature_id"].str.split(":").str[1], features["measurement_type"]))
    print(f"[impute] device={device} held={HELD_QUERIES}", flush=True)

    Ks = [0, 1, 2, 5, 10, 20]
    recs = []
    for seed in range(args.seeds):
        X = load_rows(seed=seed)
        med = np.median(X, axis=0); iqr = np.subtract(*np.percentile(X, [75, 25], axis=0))
        iqr = np.where(iqr > 1e-6, iqr, 1.0); Xn = ((X - med) / iqr).astype(np.float32)
        perm = np.random.default_rng(seed).permutation(len(Xn)); cut = int(len(Xn) * 0.7)
        Xtr, Xte = Xn[perm[:cut]], Xn[perm[cut:]]
        e_tables = {c.cid: build_embedding_table(c, features, e_dim, seed=seed) for c in CONDITIONS}
        for K in Ks:
            for cond in CONDITIONS:
                out = train_eval_impute(cond.kind, e_tables[cond.cid], e_dim, Xtr, Xte,
                                        train_q, K, seed, cfg, device)
                held = float(np.mean([out[c] for c in HELD_QUERIES]))
                held_sk = float(np.mean([out["skill_" + c] for c in HELD_QUERIES]))
                probe = float(np.mean([out["skill_probe_" + COLUMNS[p]]
                                       for p in [COLUMNS.index("XMEAS7"), COLUMNS.index("XMEAS9")]]))
                recs.append(dict(condition=cond.cid, name=cond.name, K=K, seed=seed,
                                 mean_r2=held, mean_skill=held_sk, probe_skill=probe, **out))
                gc.collect()
            pd.DataFrame(recs).to_parquet(ROOT / "results/tep_imputation.parquet")
            print(f"[impute] seed{seed} K={K} done", flush=True)

    df = pd.DataFrame(recs)
    print("\n=== SANITY probe: train-query tracking (Pearson) by K x condition (must be >> 0) ===")
    print(df.pivot_table("probe_skill", "K", "condition").round(3))
    print("\n=== held-out-query imputation: mean tracking skill (Pearson) by K x condition ===")
    print(df.pivot_table("mean_skill", "K", "condition").round(3))

    # per-type transfer at K=0, robust skill metric (C4 metadata vs C5 KG vs C6 shuffled)
    d0 = df[df.K == 0]
    bytype = []
    for c in ("C4", "C5", "C6"):
        sub = d0[d0.condition == c]
        for hq in HELD_QUERIES:
            for _, row in sub.iterrows():
                bytype.append(dict(condition=c, mtype=types[hq], query=hq, skill=row["skill_" + hq]))
    bt = pd.DataFrame(bytype)
    piv = bt.groupby(["mtype", "condition"])["skill"].mean().unstack("condition").round(3)
    piv.to_csv(ROOT / "results/tep_impute_bytype.csv")
    print("\n=== zero-shot transfer TRACKING SKILL (Pearson) BY MEASUREMENT TYPE (K=0) ===")
    print(piv)

    # C4 vs C6 paired significance on skill across seeds x held queries
    c4 = np.array([d0[(d0.condition == "C4") & (d0.seed == s)]["mean_skill"].mean()
                   for s in range(args.seeds)])
    c6 = np.array([d0[(d0.condition == "C6") & (d0.seed == s)]["mean_skill"].mean()
                   for s in range(args.seeds)])
    print(f"\nC4(metadata) mean skill {c4.mean():.3f} vs C6(shuffled) {c6.mean():.3f}; "
          f"paired diff {(c4-c6).mean():.3f}, positive in {int((c4>c6).sum())}/{args.seeds} seeds")

    # semantic-distance vs transfer-gain scatter (skill, per query, K=0, C4)
    sd = semantic_distance_table(features, train_q)
    q_sk = {hq: d0[d0.condition == "C4"]["skill_" + hq].mean() for hq in HELD_QUERIES}
    sd["transfer_skill"] = sd["query"].map(q_sk)
    sd.to_csv(ROOT / "results/tep_semantic_distance.csv", index=False)
    rho = sd[["max_sim", "transfer_skill"]].corr(method="spearman").iloc[0, 1]
    print("\n=== semantic-distance vs transfer-gain (per held query, K=0, C4) ===")
    print(sd.round(3).to_string(index=False))
    print(f"Spearman(semantic similarity, transfer skill) = {rho:.3f}")
    print("\nwrote tep_imputation.parquet, tep_impute_bytype.csv, tep_semantic_distance.csv")


if __name__ == "__main__":
    main()
