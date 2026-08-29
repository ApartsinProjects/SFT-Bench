# Phase 1 Experiment Spec — LBNL/Brick, decisive first experiment

**Goal.** Answer exactly one question before any second domain is touched:

> Under a rigorously controlled held-out-feature protocol on LBNL/Brick, does a KG-derived feature
> embedding produce a measurable few-shot label-efficiency advantage over a **strong text embedding of
> the same metadata** — and is that advantage semantic (Correct-KG > Shuffled-KG > Random) rather than
> architectural?

Continuation gate (pre-registered): **KG beats Text at K ∈ {1,2,5,10} with non-overlapping bootstrap
CIs on ≥2 of 3 transfer splits, AND Correct-KG > Shuffled-KG with non-overlapping CIs.** If either
fails, the graph contribution is not supported and the program pivots (see analysis.md).

---

## 1. Fixed configuration (one config, one pass, one artifact)

Everything below is held identical across conditions. Only `e_j` changes.

- **Dataset:** LBNL FDD **SDAHU** (single-duct AHU) subset. Source: DOI 10.25984/1881324, CC-BY 4.0,
  bulk download `fdddata.lbl.gov/data/Simulated_SDAHU/`. CSVs at 1-minute sampling, one file per
  (fault type × severity) plus a fault-free case; Brick `.ttl` ships with the subset. **Data is
  simulation (EnergyPlus–Modelica), not field sensors** — scope claims accordingly.
- **Task:** fault vs. normal (binary), point/window-level.
- **Window:** W = 60 samples (tune once, then freeze); per-feature temporal encoder shared across
  features.
- **Architecture:** shared `F_θ(h_j, e_j)` → attention-pool aggregator `A` → MLP head `G`. Identical
  weights init scheme, identical parameter count budget across conditions (pad `e_j` to a common dim D
  so no condition gets more parameters than another).
- **Normalization:** robust (median/IQR) per feature, stats from *unlabeled* target data only.
- **Seeds:** 5 seeds. Every metric reported as mean ± bootstrap 95% CI over seeds × held-out targets.
- **Splits, metrics, and conditions are all co-computed in a single run and saved as ONE artifact**
  (`results/phase1_run.parquet`) so the number-by-number audit is valid.

## 2. Conditions (identical architecture; `e_j` is the only difference)

| ID | Condition | `e_j` content | Controls for |
|----|-----------|---------------|--------------|
| C0 | Value-only | none (`e_j` = 0 / omitted) | floor |
| C1 | Random-emb | fixed random vector per feature | extra dims / identifier |
| C2 | Learned-ID | trainable per-feature embedding | in-distribution ceiling; cannot do zero-shot |
| C3 | Text-emb | sentence embedding of name+unit+component+description | **the real competitor** |
| C4 | Metadata-emb | measurement_type + component + unit, no topology | isolates topology's value |
| C5 | **KG-emb** | inductive graph encoder over Brick neighborhood | the proposed method |
| C6 | KG-shuffled | C5 with feature→node assignment permuted | **semantic vs. architectural** |

C5 vs C3 = novelty test. C5 vs C6 = semantic test. C5 vs C4 = topology test. All three must be read
off the *same* run.

## 3. Splits

- `standard.json` — conventional random split. Sanity only (Exp 1): confirms semantics don't degrade
  full-data performance.
- `feature_cold_start.json` — hold out specific sensor *types* entirely from predictive training; at
  test time values + KG metadata available, few-shot labels added at K ∈ {0,1,2,5,10,20,50,100,full}.
- `component_cold_start.json` — hold out an entire AHU instance. **This is the headline split.**

Report label-efficiency curves (performance vs. K) per split, and STE(q) = K_baseline(q) / K_KG(q)
at the q where curves are separable.

## 4. Pre-registered invariants (a violation = a bug, not a finding)

1. **Degenerate check:** C0 value-only under `standard.json` must match a plain fixed-column MLP
   trained on the same features within noise. If the set/attention model underperforms a vanilla MLP
   on full data, the architecture is broken — fix before interpreting any transfer result.
2. **Monotonicity:** performance must be non-decreasing in K in expectation for every condition. A
   curve that gets worse with more target labels signals a leakage/normalization bug.
3. **Ordering:** `Correct-KG ≥ Shuffled-KG ≥ Random-emb` at low K. If Shuffled ≈ Correct, benefit is
   architectural — report it plainly, do not dress it as a semantic win.
4. **No-leakage:** normalization and few-shot K-selection use only target-side unlabeled data + the K
   labeled examples; held-out target labels never touch training stats. Assert in code.
5. **Learned-ID cannot zero-shot:** C2 must fail the zero-shot feature cold-start (no embedding exists
   for an unseen feature). If it somehow "works," the split is leaking feature identity.

Each invariant is an assertion or an explicit check dumped to `results/phase1_sanity.json`, with the
expected outcome stated *before* the run.

## 5. Deliverables

- `results/phase1_run.parquet` — every (condition, split, K, seed) metric, one pass.
- `results/phase1_sanity.json` — invariant checks with pre-stated expectations.
- Figure: few-shot curves (C0/C1/C3/C5) on component_cold_start.
- Figure: C5 vs C6 vs C1 bar at K∈{1,5,10} (the semantic-vs-architectural panel).
- One-paragraph verdict against the continuation gate.

## 6. Order of work

1. LBNL AHU loader → unified schema (`observations.parquet`, `features.csv`, `graph.ttl`).
2. Brick `.ttl` parse → per-feature node mapping → inductive KG encoder (start with relation-aware
   neighborhood aggregation; GraphSAGE-style, inductive so unseen nodes get embeddings).
3. Text + metadata embedding baselines.
4. Shared temporal encoder + attention aggregator + head.
5. Split generators (standard / feature_cold_start / component_cold_start / few_shot).
6. Single-pass runner emitting the two artifacts above.
7. Sanity gate → verdict.

**In parallel (not blocking):** submit the SWaT/WADI iTrust data request and begin PhysioNet
credentialing, so Phases 2 and 5 are not gated on access later.
