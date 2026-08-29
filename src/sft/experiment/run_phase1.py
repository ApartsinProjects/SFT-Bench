"""Single-pass Phase-1 runner (docs/phase1_spec.md).

All conditions x splits x K x seeds are computed in ONE run and written to ONE artifact, so the
KG-vs-Text / KG-vs-Shuffled / KG-vs-Metadata comparisons are construct-matched. The sanity gate runs
at the end and its report is written next to the results.

Modes:
  --smoke   synthetic tiny data, no download; exercises the whole pipeline + sanity gate end-to-end.
  (default) real SDAHU windows; requires `python -m sft.datasets.lbnl_sdahu --download` first, then
            windows are assembled via datasets.iter_windows + splits.make_splits (see phase1_spec).
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

from ..model.sft_model import SFTModel
from ..build_features import build_sdahu_features
from .conditions import CONDITIONS, build_embedding_table
from .sanity import write_sanity

ROOT = Path(__file__).resolve().parents[3]


def auroc(y_true: np.ndarray, score: np.ndarray) -> float:
    """Rank-based AUROC, no sklearn dependency."""
    order = np.argsort(score)
    ranks = np.empty(len(score), dtype=float)
    ranks[order] = np.arange(1, len(score) + 1)
    pos = y_true == 1
    n_pos, n_neg = int(pos.sum()), int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return (ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def _device():
    try:
        if torch.cuda.is_available():
            torch.zeros(1, device="cuda")   # probe; falls back if CUDA DLLs/paging fail
            return "cuda"
    except Exception:  # noqa: BLE001
        pass
    return "cpu"


def train_eval(cond_kind, e_table, n_features, e_dim, Xtr, ytr, Xte, yte, cfg, device="cpu",
               batch_size=256) -> float:
    """One condition's train+eval. e_table is {fid: vec} for fixed-e conditions, else None.

    e_j is supplied three ways depending on condition kind:
      fixed       -> broadcast the precomputed (F, e_dim) table to every sample;
      zero        -> zeros (value-only, C0);
      learned_id  -> model's nn.Embedding over feature index (C2).
    """
    if len(ytr) == 0 or len(yte) == 0:
        return float("nan")
    n_id = n_features if cond_kind == "learned_id" else 0
    model = SFTModel(e_dim=e_dim, n_features_for_id=n_id,
                     h_hidden=cfg["h_hidden"], z_dim=cfg["z_dim"]).to(device)
    # per-sample e_j (fixed conditions) is the same (F, e_dim) table broadcast at batch time
    E = None
    if cond_kind not in ("learned_id", "zero"):
        E = torch.tensor(np.stack([e_table[f] for f in e_table]).astype(np.float32), device=device)

    def batch_e(bs):
        if cond_kind == "zero":
            return torch.zeros(bs, n_features, e_dim, device=device), None
        if cond_kind == "learned_id":
            return None, torch.arange(n_features, device=device).unsqueeze(0).repeat(bs, 1)
        return E.unsqueeze(0).expand(bs, -1, -1), None

    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    lossf = nn.BCEWithLogitsLoss()
    Xtr_t = torch.tensor(Xtr, device=device); ytr_t = torch.tensor(ytr, device=device)
    n = len(ytr_t)
    rng = np.random.default_rng(0)
    for _ in range(cfg["epochs"]):
        model.train()
        for s0 in range(0, n, batch_size):
            idx = slice(s0, min(s0 + batch_size, n))
            xb, yb = Xtr_t[idx], ytr_t[idx]
            e_b, f_b = batch_e(len(yb))
            opt.zero_grad()
            lossf(model(xb, e=e_b, feature_idx=f_b), yb).backward(); opt.step()
    model.eval()
    scores = []
    with torch.no_grad():
        Xte_t = torch.tensor(Xte, device=device)
        for s0 in range(0, len(yte), batch_size):
            xb = Xte_t[s0:s0 + batch_size]
            e_b, f_b = batch_e(len(xb))
            scores.append(torch.sigmoid(model(xb, e=e_b, feature_idx=f_b)).cpu().numpy())
    return auroc(yte, np.concatenate(scores))


# ------------------------- synthetic data for smoke mode -------------------------

def _synth(n_features, window, n_per_class, rng, signal_feats, signal_on=True):
    X, y = [], []
    for label in (0, 1):
        for _ in range(n_per_class):
            w = (rng.standard_normal((n_features, window)) * 0.5).astype(np.float32)
            if label == 1 and signal_on:
                w[signal_feats] += 1.2
            X.append(w); y.append(label)
    X, y = np.stack(X), np.array(y, dtype=np.float32)
    p = rng.permutation(len(y))
    return X[p], y[p]


def run_smoke(cfg: dict) -> pd.DataFrame:
    """PLUMBING CHECK ONLY. Uses one consistent learnable synthetic dataset (signal always present)
    so every condition should train to a high, similar AUROC. This validates that all 7 conditions
    wire up, the model trains, the artifact writes, and the sanity gate runs. The numbers are NOT
    findings and the cold-start science is NOT emulated here (that needs the real data)."""
    features = build_sdahu_features()
    n_features, e_dim, window = len(features), cfg["e_dim"], 12
    signal_feats = np.random.default_rng(0).choice(n_features, size=4, replace=False)
    Ks, splits, seeds = [0, 10], ["standard", "feature_cold_start"], [0]
    cfg = dict(cfg, epochs=6)

    rows = []
    for seed in seeds:
        # precompute each condition's embedding table ONCE per seed (independent of split/K)
        e_tables = {c.cid: build_embedding_table(c, features, e_dim, seed=seed) for c in CONDITIONS}
        rng = np.random.default_rng(1000 + seed)
        # single shared dataset; signal always on so results are boring and comparable
        Xtr, ytr = _synth(n_features, window, 100, rng, signal_feats, signal_on=True)
        Xte, yte = _synth(n_features, window, 60, rng, signal_feats, signal_on=True)
        for cond in CONDITIONS:
            a = train_eval(cond.kind, e_tables[cond.cid], n_features, e_dim, Xtr, ytr, Xte, yte, cfg)
            for split in splits:               # replicate the one number across the sweep axes
                for K in Ks:
                    rows.append(dict(condition=cond.cid, name=cond.name, split=split,
                                     K=K, seed=seed, auroc=a))
    return pd.DataFrame(rows)


def run_real(cfg: dict, seeds, Ks, cap_per_file: int, window: int, force_cpu: bool = False
             ) -> pd.DataFrame:
    """Real SDAHU run: all conditions x splits x K x seeds, single pass. Requires downloaded CSVs."""
    from .data_real import assemble, build_split
    torch.set_num_threads(max(1, min(4, (torch.get_num_threads() or 4))))
    ckpt = ROOT / "results/phase1_run.parquet"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    device = "cpu" if force_cpu else _device()
    print(f"[real] device={device}; assembling windows (cap {cap_per_file}/file, W={window}) ...",
          flush=True)
    features = build_sdahu_features()
    n_features, e_dim = len(features), cfg["e_dim"]
    cache: dict = {}
    byfam = assemble(window=window, cap_per_file=cap_per_file, cache=cache)
    counts = {f: len(byfam[f]["X"]) for f in byfam}
    print(f"[real] window counts by family: {counts}", flush=True)

    splits = cfg["splits"]
    rows = []
    for seed in seeds:
        e_tables = {c.cid: build_embedding_table(c, features, e_dim, seed=seed) for c in CONDITIONS}
        for split in splits:
            Ks_here = ["full"] if split == "standard" else Ks
            for K in Ks_here:
                Xtr, ytr, Xte, yte = build_split(split, byfam, seed=seed, K=K)
                for cond in CONDITIONS:
                    # C2 (learned-ID) cannot embed an unseen feature; here all 30 features are shared
                    # across splits, so C2 is defined, but it carries no cross-feature semantics.
                    a = train_eval(cond.kind, e_tables[cond.cid], n_features, e_dim,
                                   Xtr, ytr, Xte, yte, cfg, device=device)
                    kval = 999 if K == "full" else int(K)
                    rows.append(dict(condition=cond.cid, name=cond.name, split=split,
                                     K=kval, seed=seed, auroc=a))
                    gc.collect()
                    if device == "cuda":
                        torch.cuda.empty_cache()
                pd.DataFrame(rows).to_parquet(ckpt)   # checkpoint: survive a kill mid-run
                print(f"[real] seed{seed} {split} K={K} done "
                      f"(ntr={len(ytr)}, nte={len(yte)})", flush=True)
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs/phase1.json"))
    ap.add_argument("--smoke", action="store_true", help="synthetic pipeline test, no download")
    ap.add_argument("--seeds", type=int, default=3, help="real run: number of seeds")
    ap.add_argument("--cap", type=int, default=200, help="real run: max windows per file")
    ap.add_argument("--window", type=int, default=60)
    ap.add_argument("--cpu", action="store_true", help="force CPU (stable on low-memory hosts)")
    args = ap.parse_args()
    cfg = json.loads(Path(args.config).read_text())

    if args.smoke:
        print("[smoke] running synthetic end-to-end pipeline test ...", flush=True)
        df = run_smoke(cfg)
        out = ROOT / "results/phase1_smoke_run.parquet"
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out)
        rep = write_sanity(df, ROOT / "results/phase1_smoke_sanity.json")
        print("\nmean AUROC by split x condition:")
        print(df.groupby(["split", "condition"])["auroc"].mean().round(3).unstack("condition"))
        print(f"\nsanity: {rep['n_passed']}/{rep['n_checks']} invariants passed "
              f"({'ALL PASS' if rep['all_passed'] else 'SEE REPORT'})")
        print(f"wrote {out.name} and phase1_smoke_sanity.json", flush=True)
        return

    seeds = list(range(args.seeds))
    Ks = [0, 5, 20, 100, "full"]
    df = run_real(cfg, seeds=seeds, Ks=Ks, cap_per_file=args.cap, window=args.window,
                  force_cpu=args.cpu)
    out = ROOT / "results/phase1_run.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out)
    rep = write_sanity(df, ROOT / "results/phase1_sanity.json")
    print("\n=== mean AUROC by split x condition (across seeds, all K) ===")
    print(df.groupby(["split", "condition"])["auroc"].mean().round(3).unstack("condition"))
    print("\n=== few-shot: mean AUROC by K x condition on feature_cold_start ===")
    fc = df[df.split == "feature_cold_start"]
    print(fc.pivot_table("auroc", "K", "condition").round(3))
    print(f"\nsanity: {rep['n_passed']}/{rep['n_checks']} invariants passed "
          f"({'ALL PASS' if rep['all_passed'] else 'SEE REPORT'})")
    print(f"wrote {out.name} and phase1_sanity.json", flush=True)


if __name__ == "__main__":
    main()
