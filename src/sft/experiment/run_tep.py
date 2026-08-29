"""Phase-2 (TEP): headroom gate + held-out-column semantic transfer on Tennessee Eastman.

TEP faults are documented as feature-localized (unlike SDAHU's system-wide propagation), so this is
the domain where a held-out-feature transfer test can actually move. We first run the same headroom
gate: for each localized fault, does masking its primary columns hurt fault-vs-normal detection? A
fault that passes the gate is a valid transfer target.

Normal windows come from d00_te (fault-free) and the pre-onset region of every test file; fault
windows come from the post-onset region (sample >= 160) of that fault's test file. Robust per-feature
normalization from pooled unlabelled windows.
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
from ..datasets.tep import (build_tep_features, load_test_run, FAULTS, COLUMNS, FAULT_ONSET)
from .conditions import CONDITIONS, build_embedding_table
from .run_phase1 import auroc

ROOT = Path(__file__).resolve().parents[3]
GATE_FAULTS = [4, 11, 14, 5, 12, 6, 1, 13]   # localized cooling/feed faults + two plant-wide contrasts


def _windows(arr, lo, hi, window, stride):
    out = []
    for s in range(lo, hi - window + 1, stride):
        out.append(arr[s:s + window].T)      # (F, W)
    return np.asarray(out, dtype=np.float32) if out else np.empty((0, 52, window), np.float32)


def assemble_tep(window=20, stride=5, cap_per_class=400):
    normal, faults = [], {}
    for idv in sorted(set(GATE_FAULTS) | {0}):
        arr = load_test_run(idv)
        if idv == 0:
            normal.append(_windows(arr, 0, len(arr), window, stride))
        else:
            normal.append(_windows(arr, 0, FAULT_ONSET, window, stride))         # pre-onset normal
            faults[idv] = _windows(arr, FAULT_ONSET, len(arr), window, stride)    # post-onset faulty
    normal = np.concatenate(normal)
    def cap(a, seed=0):
        if len(a) > cap_per_class:
            a = a[np.random.default_rng(seed).permutation(len(a))[:cap_per_class]]
        return a
    normal = cap(normal)
    faults = {k: cap(v) for k, v in faults.items()}
    # No-leakage: robust stats from FAULT-FREE windows only (never fault/test-label regions).
    flat = normal.transpose(1, 0, 2).reshape(52, -1)
    med = np.median(flat, axis=1).astype(np.float32)
    q1, q3 = np.percentile(flat, [25, 75], axis=1)
    iqr = np.where((q3 - q1) > 1e-6, q3 - q1, 1.0).astype(np.float32)
    norm = lambda a: ((a - med[None, :, None]) / iqr[None, :, None]).astype(np.float32) if len(a) else a
    return norm(normal), {k: norm(v) for k, v in faults.items()}


def train_eval_bin(cond_kind, e_table, e_dim, Xtr, ytr, mtr, Xte, yte, mte, cfg, device):
    n_features = 52
    model = SFTModel(e_dim=e_dim, n_features_for_id=(n_features if cond_kind == "learned_id" else 0),
                     h_hidden=cfg["h_hidden"], z_dim=cfg["z_dim"], n_out=1).to(device)
    E = None
    if cond_kind not in ("learned_id", "zero"):
        E = torch.tensor(np.stack([e_table[f] for f in e_table]).astype(np.float32), device=device)
    def eargs(bs):
        if cond_kind == "zero": return dict(e=torch.zeros(bs, n_features, e_dim, device=device))
        if cond_kind == "learned_id":
            return dict(feature_idx=torch.arange(n_features, device=device).unsqueeze(0).repeat(bs, 1))
        return dict(e=E.unsqueeze(0).expand(bs, -1, -1))
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"]); lossf = nn.BCEWithLogitsLoss()
    Xtr_t = torch.tensor(Xtr, device=device); ytr_t = torch.tensor(ytr, device=device)
    mtr_t = torch.tensor(mtr, device=device); bs = cfg["batch_size"]
    for _ in range(cfg["epochs"]):
        model.train()
        for s0 in range(0, len(ytr_t), bs):
            sl = slice(s0, s0 + bs); opt.zero_grad()
            out = model(Xtr_t[sl], mask=mtr_t[sl], **eargs(len(ytr_t[sl])))
            lossf(out, ytr_t[sl]).backward(); opt.step()
    model.eval(); scores = []
    with torch.no_grad():
        Xte_t = torch.tensor(Xte, device=device); mte_t = torch.tensor(mte, device=device)
        for s0 in range(0, len(yte), bs):
            sl = slice(s0, s0 + bs)
            scores.append(torch.sigmoid(model(Xte_t[sl], mask=mte_t[sl], **eargs(sl.stop-sl.start if sl.stop<=len(yte) else len(yte)-s0))).cpu().numpy())
    return auroc(yte, np.concatenate(scores))


def split_bin(normal, fault, seed, held_idx=None, hold=False, test_frac=0.3):
    rng = np.random.default_rng(seed)
    def sp(a):
        p = rng.permutation(len(a)); c = int(len(a)*(1-test_frac)); return a[p[:c]], a[p[c:]]
    ntr, nte = sp(normal); ftr, fte = sp(fault)
    Xtr = np.concatenate([ntr, ftr]); ytr = np.r_[np.zeros(len(ntr)), np.ones(len(ftr))].astype(np.float32)
    Xte = np.concatenate([nte, fte]); yte = np.r_[np.zeros(len(nte)), np.ones(len(fte))].astype(np.float32)
    mtr = np.ones((len(ytr), 52), np.float32); mte = np.ones((len(yte), 52), np.float32)
    if hold and held_idx:
        mtr[:, held_idx] = 0.0; mte[:, held_idx] = 0.0
    ptr = rng.permutation(len(ytr))
    return Xtr[ptr], ytr[ptr], mtr[ptr], Xte, yte, mte


def split_transfer(normal, fault, seed, held_idx, K, test_frac=0.3):
    """Held-out-column few-shot split: held columns MASKED in training except for K revealed
    target-fault windows; all columns present at test. Tests whether a semantic e_j lets the shared
    encoder incorporate a column it never trained on."""
    rng = np.random.default_rng(seed)
    def sp(a):
        p = rng.permutation(len(a)); c = int(len(a) * (1 - test_frac)); return a[p[:c]], a[p[c:]]
    ntr, nte = sp(normal); ftr, fte = sp(fault)
    Xtr = np.concatenate([ntr, ftr]); ytr = np.r_[np.zeros(len(ntr)), np.ones(len(ftr))].astype(np.float32)
    mtr = np.ones((len(ytr), 52), np.float32); mtr[:, held_idx] = 0.0
    fault_pos = np.arange(len(ntr), len(ntr) + len(ftr))          # indices of fault-train windows
    reveal = fault_pos if K == "full" else fault_pos[:int(K)]
    mtr[reveal] = 1.0                                             # few-shot: unmask cols for K windows
    Xte = np.concatenate([nte, fte]); yte = np.r_[np.zeros(len(nte)), np.ones(len(fte))].astype(np.float32)
    mte = np.ones((len(yte), 52), np.float32)                    # test: all columns present
    p = rng.permutation(len(ytr))
    return Xtr[p], ytr[p], mtr[p], Xte, yte, mte


def run_transfer(cfg, features, device, seeds):
    """P1: 7-condition held-out-COLUMN transfer on reactor-cooling faults (hold XMEAS9/21+XMV10)."""
    e_dim = cfg["e_dim"]
    normal, faults = assemble_tep()
    target = np.concatenate([faults[i] for i in (4, 11, 14)])     # reactor-cooling super-class
    held = [COLUMNS.index(c) for c in ("XMEAS9", "XMEAS21", "XMV10")]
    print(f"[TEP-transfer] target(reactor_cooling)={len(target)} normal={len(normal)} "
          f"held_cols={held}", flush=True)
    Ks = [0, 1, 2, 5, 10, 20, "full"]
    rows = []
    for seed in seeds:
        e_tables = {c.cid: build_embedding_table(c, features, e_dim, seed=seed) for c in CONDITIONS}
        for K in Ks:
            Xtr, ytr, mtr, Xte, yte, mte = split_transfer(normal, target, seed, held, K)
            for cond in CONDITIONS:
                a = train_eval_bin(cond.kind, e_tables[cond.cid], e_dim, Xtr, ytr, mtr,
                                   Xte, yte, mte, cfg, device)
                rows.append(dict(condition=cond.cid, name=cond.name,
                                 K=(999 if K == "full" else int(K)), seed=seed, auroc=a))
                gc.collect()
            pd.DataFrame(rows).to_parquet(ROOT / "results/tep_transfer.parquet")
            print(f"[TEP-transfer] seed{seed} K={K} done", flush=True)
    df = pd.DataFrame(rows)
    print("\n=== reactor-cooling held-out-column transfer: AUROC by K x condition ===")
    print(df.pivot_table("auroc", "K", "condition").round(3))
    print("\nDecisive: C5(KG) vs C3(text) vs C1(random) at low K; C5 vs C6(shuffled) = semantic vs arch.")
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs/phase1.json"))
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--transfer", action="store_true", help="run P1 held-out-column transfer sweep")
    args = ap.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    torch.set_num_threads(max(1, min(4, torch.get_num_threads() or 4)))
    device = "cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu")
    features = build_tep_features(); e_dim = cfg["e_dim"]

    if args.transfer:
        run_transfer(cfg, features, device, seeds=list(range(args.seeds)))
        return

    e_rand = build_embedding_table(CONDITIONS[1], features, e_dim, seed=0)   # random e_j for the gate
    normal, faults = assemble_tep()
    print(f"[TEP] device={device} normal={len(normal)} "
          f"faults={ {k: len(v) for k,v in faults.items()} }", flush=True)

    print("\n=== HEADROOM GATE: fault-vs-normal AUROC, primary columns present vs masked ===")
    print(f"{'IDV':>4} {'name':22} {'local':6} {'present':>8} {'masked':>8} {'drop':>7}  verdict")
    results = []
    for idv in GATE_FAULTS:
        meta = FAULTS.get(idv, dict(name=f"idv{idv}", localized=False, primary=[]))
        held = [COLUMNS.index(c) for c in meta.get("primary", []) if c in COLUMNS]
        pres, mask = [], []
        for seed in range(args.seeds):
            Xtr, ytr, mtr, Xte, yte, mte = split_bin(normal, faults[idv], seed, held, hold=False)
            pres.append(train_eval_bin("fixed", e_rand, e_dim, Xtr, ytr, mtr, Xte, yte, mte, cfg, device))
            if held:
                Xtr, ytr, mtr, Xte, yte, mte = split_bin(normal, faults[idv], seed, held, hold=True)
                mask.append(train_eval_bin("fixed", e_rand, e_dim, Xtr, ytr, mtr, Xte, yte, mte, cfg, device))
            gc.collect()
        p = float(np.mean(pres)); m = float(np.mean(mask)) if mask else float("nan")
        drop = p - m if mask else float("nan")
        verdict = "-" if not held else ("HEADROOM" if drop > 0.05 else "redundant")
        results.append(dict(idv=idv, name=meta["name"], localized=meta["localized"],
                            present=p, masked=m, drop=drop, verdict=verdict, primary=meta.get("primary", [])))
        print(f"{idv:>4} {meta['name']:22} {str(meta['localized']):6} {p:8.3f} {m:8.3f} {drop:7.3f}  {verdict}")
    Path(ROOT / "results").mkdir(exist_ok=True)
    Path(ROOT / "results/tep_gate.json").write_text(json.dumps(results, indent=2, default=float))
    passing = [r for r in results if r["verdict"] == "HEADROOM"]
    print(f"\n{len(passing)} fault(s) pass the headroom gate: "
          f"{[r['name'] for r in passing]}")
    print("wrote results/tep_gate.json", flush=True)


if __name__ == "__main__":
    main()
