# Stronger framing: "Semantics gate, statistics decide"

## The pivot

The weak paper: *"semantic feature embeddings enable transfer."* Not novel (CARTE, TransTab,
TabSTAR), and our effect is heterogeneous (pressure strong, others null), which reads as a weak
positive.

The strong paper: **a predictive account of WHEN zero-shot feature transfer works, and a source-only
rule that forecasts it before spending a single target label.** The heterogeneity stops being a
weakness and becomes the central phenomenon to explain.

Our data already sketches the law: the unseen pressure sensor transfers at 0.83 to 0.90 while flow,
level, and temperature do not, and semantic similarity does NOT predict which is which (Spearman
~0.06). The difference between pressure and the rest is not semantic; it is that pressure's same-type
siblings are strongly mutually predictable (r about 0.96) and the others' are not. That is the thesis:

> **External semantics identify which previously learned features are ELIGIBLE to transfer to a new
> sensor; whether transfer actually succeeds is determined by the STATISTICAL RECOVERABILITY of those
> semantically matched siblings, which is estimable from source data alone. Graph topology adds
> nothing because measurement type already fixes eligibility, and recoverability, not topology,
> decides the outcome.**

## Why this is novel and valuable

- Prior semantic-tabular work shows semantics help on average. Nobody has characterized the condition
  that separates success from failure, nor shown that semantic proximity FAILS to predict it while a
  cheap source-only statistic SUCCEEDS.
- It is actionable: a practitioner commissioning a new sensor can compute its recoverability from
  existing data and decide, before labeling, whether semantic transfer will save labels. That is a
  decision rule, not just a benchmark number.
- It converts three current "negatives" into results: the admission-gate null (SDAHU), the topology
  null, and the heterogeneity all become evidence for one coherent theory.

## Three-layer contribution

1. **Dataset admission gate** (have it): a dataset can test the hypothesis only if some fault/target
   is feature-localized and non-redundant. SDAHU fails, TEP passes. Screens the benchmark.
2. **Sensor-level recoverability predictor** (the new headline): for a candidate unseen sensor,
   `R_j = source-only predictability of the target from its semantically matched siblings`
   (measured with a ridge/kNN fit on SOURCE data, no target labels). Claim: `R_j` predicts realized
   zero-shot transfer gain, while semantic similarity does not.
3. **A commissioning decision rule**: transfer semantically only when `R_j` exceeds a threshold set on
   source/validation data; report the precision/recall of that a-priori call on held-out sensors.

## Additional experiments (prioritized), and what each buys

### E1. The recoverability law (headline, executable now on TEP)
Leave-one-feature-out over ~30-40 sensors (each held out singly, siblings retained). For every held
sensor compute, on SOURCE data only:
  - `R_j` = R^2 of a ridge/kNN predictor of sensor j from its same-type siblings;
  - realized transfer gain `G_j = skill(metadata) - skill(random)` at K=0.
Report `corr(R_j, G_j)` with CI, and `corr(semantic_similarity_j, G_j)` for contrast.
Expected and already-suggested outcome: recoverability predicts gain strongly; semantic similarity
does not. This single scatter is the paper's memorable figure and resolves the "one pressure target"
complaint, because the unit of analysis is now 30-40 (sensor, R_j, G_j) triples, not one type average.

### E2. Statistical power and heterogeneity (executable now)
The leave-one-out design gives many targets. Report a target-wise forest plot of G_j with hierarchical
bootstrap CIs (sensor, seed), and stratify by recoverability band (low/med/high). The claim becomes
"high-recoverability sensors transfer (CI excludes 0); low-recoverability do not," which is a clean,
significant, stratified result even though the pooled average is not significant.

### E3. Cross-domain replication (executable now: LBNL/Brick imputation; SWaT later)
Run the identical strict imputation + recoverability analysis on LBNL/Brick, using IMPUTATION (not the
failed fault-detection task): hold out a zone/air temperature sensor, predict it from the other Brick
points, siblings retained. If the recoverability law holds in HVAC too, the finding is cross-domain,
not a TEP artifact. This is the single biggest lift to acceptance. SWaT/WADI as a real-testbed third
domain once the iTrust data arrives.

### E4. Label-efficiency where it is valid (executable now)
For high-recoverability sensors, plot few-shot curves (K = 1,2,5,10,20,50) with nested adaptation
sets, and report semantic transfer efficiency STE(q) = K_baseline / K_semantic for reaching a
pre-registered quality q. This delivers the promised label-savings contribution, honestly restricted
to the regime where transfer is possible.

### E5. Strong baselines (executable now)
Oracle source selection (best single sibling, retrospective) and nearest-semantic-sibling copy bound
the exploitable structure; a text-embedding-of-metadata encoder (already A5) is the modern-semantic
competitor. Report standard imputers (mean, kNN, ridge) on KNOWN-channel imputation to show the task
itself is credible, then the unseen-channel regime where those cannot operate.

### E6. Topology as a principled boundary, not a footnote
Reframe the topology null: type already fixes eligibility, so topology could only help by
disambiguating same-type siblings with DIFFERENT recoverability. Test exactly that (targets with
multiple same-type siblings of differing recoverability) and report that topology still does not help,
which is a stated, tested boundary rather than an awkward negative.

## Revised title candidates

- **"Semantics Gate, Statistics Decide: Predicting Zero-Shot Feature Transfer for Unseen Sensors"**
- **"When Can a New Sensor Borrow a Model? Recoverability Predicts Semantic Feature Transfer"**

## Revised claim ladder (map each to its experiment)

- Supported now: a strict, leakage-controlled model imputes an unseen sensor from semantics where its
  siblings are recoverable (TEP pressure, 0.83-0.90 vs -0.35).
- After E1/E2: recoverability predicts transfer gain across sensors; semantic similarity does not.
- After E3: the law replicates in a second domain.
- After E4: semantic transfer reduces target labels by X-fold for recoverable sensors.
- After E6: graph topology adds no information beyond type on these processes, by a stated test.

## What NOT to claim
Do not claim "KG embeddings beat baselines" or "semantic transfer generalizes universally." The
defensible, novel, data-supported claim is the recoverability law plus the admission gate plus the
decision rule.
