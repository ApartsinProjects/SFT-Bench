"""Phase-1b: multi-class fault-TYPE classification with held-out feature-COLUMN transfer.

This tests the plan's actual hypothesis (unseen FEATURE, not unseen fault family) and uses a task
with more headroom than binary fault-vs-normal. Design:

  Task     6-way fault-type classification: {fault_free, coi_bias, oa_bias, coi_leakage,
           coi_stuck, damper_stuck}. Metric: macro one-vs-rest AUROC, plus the OVR AUROC of the
           class whose evidence lives on the held-out columns.
  Transfer Hold out an entire component's SENSOR COLUMNS (e.g. Outdoor_Air_Damper -> OA_DMPR,
           OA_DMPR_DM) during training by MASKING those feature slots; present them at test. The
           held-out feature's semantic embedding e_j is still supplied (semantics known, values
           unseen in training). Few-shot K adds training windows where the held-out columns are
           unmasked. The clean question: does a semantic e_j for the held-out column (KG/text, close
           to trained sensors) let the shared encoder use it better than a random e_j (C1)?

HEADROOM GATE (printed first): compare the held-out class's AUROC with the columns masked in both
train and test vs. present in both. If masking the columns does NOT hurt that class, the held-out
columns are not necessary and no condition can demonstrate transfer -> SDAHU is unsuitable for this
test and we move to SWaT. This gate decides whether the full sweep is worth running.
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
from ..datasets.lbnl_sdahu import EXPECTED_POINTS
from .conditions import CONDITIONS, build_embedding_table
from .data_real import assemble, robust_stats
from .run_phase1 import auroc

ROOT = Path(__file__).resolve().parents[3]
FAMILIES = ["fault_free", "coi_bias", "oa_bias", "coi_leakage", "coi_stuck", "damper_stuck"]
HELDOUT_COMPONENT = "Outdoor_Air_Damper"      # columns OA_DMPR, OA_DMPR_DM; class damper_stuck


def macro_ovr_auroc(y, probs, n_classes):
    aucs = [auroc((y == c).astype(int), probs[:, c]) for c in range(n_classes)]
    aucs = [a for a in aucs if not np.isnan(a)]
    return float(np.mean(aucs)) if aucs else float("nan")


def heldout_col_indices(component: str) -> list[int]:
    feats = build_sdahu_features()
    pts = feats[feats["component"] == component]["feature_id"].str.split(":").str[1].tolist()
    return [EXPECTED_POINTS.index(p) for p in pts if p in EXPECTED_POINTS]


def build_multiclass(byfam, seed, per_class_cap=None, test_frac=0.3):
    rng = np.random.default_rng(seed)
    Xs, ys = [], []
    for ci, fam in enumerate(FAMILIES):
        X = byfam[fam]["X"]
        if per_class_cap and len(X) > per_class_cap:
            X = X[rng.permutation(len(X))[:per_class_cap]]
        Xs.append(X); ys.append(np.full(len(X), ci))
    X = np.concatenate(Xs); y = np.concatenate(ys)
    med, iqr = robust_stats(np.concatenate([byfam[f]["X"] for f in byfam]))
    X = ((X - med[None, :, None]) / iqr[None, :, None]).astype(np.float32)
    perm = rng.permutation(len(y)); X, y = X[perm], y[perm]
    cut = int(len(y) * (1 - test_frac))
    return X[:cut], y[:cut], X[cut:], y[cut:]


def train_eval_mc(cond_kind, e_table, n_features, e_dim, Xtr, ytr, mask_tr, Xte, yte, mask_te,
                  cfg, device, n_classes):
    model = SFTModel(e_dim=e_dim, n_features_for_id=(n_features if cond_kind == "learned_id" else 0),
                     h_hidden=cfg["h_hidden"], z_dim=cfg["z_dim"], n_out=n_classes).to(device)
    E = None
    if cond_kind not in ("learned_id", "zero"):
        E = torch.tensor(np.stack([e_table[f] for f in e_table]).astype(np.float32), device=device)

    def eargs(bs):
        if cond_kind == "zero":
            return dict(e=torch.zeros(bs, n_features, e_dim, device=device))
        if cond_kind == "learned_id":
            return dict(feature_idx=torch.arange(n_features, device=device).unsqueeze(0).repeat(bs, 1))
        return dict(e=E.unsqueeze(0).expand(bs, -1, -1))

    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    lossf = nn.CrossEntropyLoss()
    Xtr_t = torch.tensor(Xtr, device=device); ytr_t = torch.tensor(ytr, device=device).long()
    mtr = torch.tensor(mask_tr, device=device)
    bs = cfg["batch_size"]; n = len(ytr_t)
    for _ in range(cfg["epochs"]):
        model.train()
        for s0 in range(0, n, bs):
            sl = slice(s0, min(s0 + bs, n))
            opt.zero_grad()
            out = model(Xtr_t[sl], mask=mtr[sl], **eargs(len(ytr_t[sl])))
            lossf(out, ytr_t[sl]).backward(); opt.step()
    model.eval()
    probs = []
    with torch.no_grad():
        Xte_t = torch.tensor(Xte, device=device); mte = torch.tensor(mask_te, device=device)
        for s0 in range(0, len(yte), bs):
            sl = slice(s0, min(s0 + bs, len(yte)))
            out = model(Xte_t[sl], mask=mte[sl], **eargs(sl.stop - sl.start))
            probs.append(torch.softmax(out, dim=1).cpu().numpy())
    return np.concatenate(probs)


def masks(n, n_features, held_idx, hold: bool):
    m = np.ones((n, n_features), dtype=np.float32)
    if hold:
        m[:, held_idx] = 0.0
    return m


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs/phase1.json"))
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--cap", type=int, default=200)
    ap.add_argument("--window", type=int, default=60)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--gate-only", action="store_true", help="run only the headroom diagnostic")
    args = ap.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    torch.set_num_threads(max(1, min(4, torch.get_num_threads() or 4)))
    device = "cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu")

    features = build_sdahu_features()
    n_features, e_dim, n_classes = len(features), cfg["e_dim"], len(FAMILIES)
    held_idx = heldout_col_indices(HELDOUT_COMPONENT)
    held_class = FAMILIES.index("damper_stuck")
    print(f"[1b] device={device} heldout={HELDOUT_COMPONENT} cols={held_idx} "
          f"heldclass=damper_stuck", flush=True)

    cache: dict = {}
    byfam = assemble(window=args.window, cap_per_file=args.cap, cache=cache)
    print(f"[1b] window counts: { {f: len(byfam[f]['X']) for f in byfam} }", flush=True)

    # ---- HEADROOM GATE: does masking the held-out columns hurt the held-out class? ----
    e0 = build_embedding_table(CONDITIONS[1], features, e_dim, seed=0)   # random e_j, gate only
    gate = {"with_cols": [], "without_cols": []}
    for seed in range(max(2, args.seeds)):
        Xtr, ytr, Xte, yte = build_multiclass(byfam, seed, per_class_cap=args.cap)
        for tag, hold in [("with_cols", False), ("without_cols", True)]:
            mtr = masks(len(ytr), n_features, held_idx, hold)
            mte = masks(len(yte), n_features, held_idx, hold)
            probs = train_eval_mc("fixed", e0, n_features, e_dim, Xtr, ytr, mtr, Xte, yte, mte,
                                  cfg, device, n_classes)
            a = auroc((yte == held_class).astype(int), probs[:, held_class])
            gate[tag].append(a)
            gc.collect()
    gw, gwo = np.mean(gate["with_cols"]), np.mean(gate["without_cols"])
    print(f"\n=== HEADROOM GATE (damper_stuck one-vs-rest AUROC) ===")
    print(f"  columns PRESENT in train+test : {gw:.3f}")
    print(f"  columns MASKED  in train+test : {gwo:.3f}")
    print(f"  drop from masking             : {gw - gwo:+.3f}")
    verdict = ("HEADROOM: held-out columns matter -> transfer test is live"
               if (gw - gwo) > 0.05 else
               "NO HEADROOM: columns not needed -> SDAHU unsuitable, move to SWaT")
    print(f"  verdict: {verdict}\n", flush=True)
    Path(ROOT / "results").mkdir(exist_ok=True)
    Path(ROOT / "results/phase1b_gate.json").write_text(json.dumps(
        {"with_cols": gw, "without_cols": gwo, "drop": gw - gwo, "verdict": verdict,
         "per_seed": gate}, indent=2))
    if args.gate_only or (gw - gwo) <= 0.05:
        print("stopping after gate.", flush=True)
        return

    # ---- FULL SWEEP: 7 conditions x K x seeds on the held-out-column transfer ----
    Ks = [0, 5, 20, 100, "full"]
    rows = []
    for seed in range(args.seeds):
        e_tables = {c.cid: build_embedding_table(c, features, e_dim, seed=seed) for c in CONDITIONS}
        Xtr, ytr, Xte, yte = build_multiclass(byfam, seed, per_class_cap=args.cap)
        # identify training windows of the held-out class to reveal as few-shot
        held_train = np.where(ytr == held_class)[0]
        for K in Ks:
            mtr = masks(len(ytr), n_features, held_idx, hold=True)
            reveal = held_train if K == "full" else held_train[:int(K)]
            mtr[reveal] = 1.0                          # few-shot: unmask cols for K target windows
            mte = masks(len(yte), n_features, held_idx, hold=False)   # test: cols present
            for cond in CONDITIONS:
                probs = train_eval_mc(cond.kind, e_tables[cond.cid], n_features, e_dim,
                                      Xtr, ytr, mtr, Xte, yte, mte, cfg, device, n_classes)
                rows.append(dict(condition=cond.cid, name=cond.name, K=(999 if K == "full" else int(K)),
                                 seed=seed,
                                 macro_auroc=macro_ovr_auroc(yte, probs, n_classes),
                                 held_auroc=auroc((yte == held_class).astype(int), probs[:, held_class])))
                gc.collect()
            pd.DataFrame(rows).to_parquet(ROOT / "results/phase1b_run.parquet")
            print(f"[1b] seed{seed} K={K} done", flush=True)

    df = pd.DataFrame(rows)
    print("\n=== held-out-class (damper_stuck) OVR AUROC by K x condition ===")
    print(df.pivot_table("held_auroc", "K", "condition").round(3))
    print("\n=== macro AUROC by K x condition ===")
    print(df.pivot_table("macro_auroc", "K", "condition").round(3))
    print("\nKey comparison C5(KG) vs C1(random) vs C3(text) on held_auroc at low K decides transfer.")


if __name__ == "__main__":
    main()
