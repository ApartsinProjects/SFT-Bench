# The paper's core: the signed-recoverability law (magnitude from statistics, polarity from semantics)

This supersedes framing_v2. A fable review plus direct verification on the 10-seed data uncovered a
metric structure that reorganizes the whole result into a clean, novel, quantitative law.

## The pathology that became the finding

Per-seed Pearson tracking skill is BIMODAL: for a given (sensor, condition) the MAGNITUDE |r| is
nearly deterministic across seeds, but the SIGN flips. Example, held pressure XMEAS16 at K=0, 10 seeds:
  - A1 random:   |r| = 0.90, but sign is +,-,+ ... (70% consistent) -> noisy mean -0.35
  - A4 metadata: |r| = 0.83, sign 100% consistent + -> mean +0.83
So a random embedding extracts the SAME 0.9 correlation with pressure; it just cannot decide the
polarity. Metadata does not increase the magnitude; it locks the sign.

## The decomposition (verified on 6 held sensors, all types)

  magnitude   |r|            <- set by STATISTICS   corr(|signed_rec|, |r|) = 0.976
  polarity    sign-consistency <- set by SEMANTICS   random 0.67 -> metadata 0.82 -> text 0.97
  realized    signed skill    = magnitude x correct-polarity ~ SIGNED recoverability
                              corr(signed_rec, signed_meta_skill) = 0.956

`signed_rec` = correlation of the target to its best-|corr| same-type sibling on SOURCE (training)
data only, sign kept. It predicts realized zero-shot transfer skill at r = 0.96 across 6 sensors.

## Why this is the strong paper

- **Novel.** Prior semantic-tabular work (CARTE, TransTab, TabSTAR) shows semantics help on average.
  Nobody has shown that (a) the semantic contribution is specifically POLARITY, not magnitude; (b) a
  source-only SIGNED recoverability statistic predicts realized transfer at r ~ 0.96; (c) semantic
  similarity does not. This is a mechanism, not a leaderboard number.
- **Valuable.** A pre-deployment decision rule: compute signed recoverability of a new sensor's
  semantic class from the historian; positive-and-large -> semantic transfer will work; near-zero ->
  no transferable signal; NEGATIVE -> semantics will confidently lock the WRONG sign, so do not deploy
  zero-shot (or supply one label to fix polarity). No prior method warns you of the negative case.
- **Data-supported.** Every previous "embarrassing negative" is now PREDICTED:
    - XMV9 actuator, metadata skill -0.47: its nearest same-type sibling XMV2 is anti-correlated
      (-0.63); the sign-blind imputer copies it -> negative. Predicted by signed_rec = -0.63.
    - Temperature XMEAS18 null: same-type temps are independently controlled (sibling |corr| 0.06);
      no transferable signal. Predicted by signed_rec ~ 0.
    - Pressure XMEAS16 success: siblings +0.99. Predicted.

## Failure mode as a first-class result: sign-blindness

Zero-shot semantic imputation is SIGN-BLIND. It attends, via the query embedding, to the
semantically eligible siblings and reproduces their signed correlation. When those siblings are
positively coherent it locks the correct sign; when anti-correlated it locks the wrong sign with high
confidence. This is the paper's mechanism section and its safety message.

## Sharpened thesis and title

> Semantic descriptors determine WHERE a zero-shot imputer looks; the SIGNED correlation structure of
> what it finds there determines what it achieves. Both are computable from source data before any
> target label. Semantics contributes polarity (sign-consistency rises random 0.67 -> text 0.97);
> statistics contributes magnitude; realized skill equals signed recoverability, and its failures
> (anti-correlated or cross-type-only predictors) are predicted, not merely observed.

Title: **"Semantics Gate, Statistics Decide: Predicting Zero-Shot Transfer to Unseen Sensors from
Source Data Alone."**
Fallback (if the 21-sensor law weakens): the mechanism paper, **"Sign-Blind Transfer: What Semantic
Embeddings Can and Cannot Tell a Model About an Unseen Sensor,"** built on the confirmed case studies
plus the sign-consistency decomposition; needs no cross-sensor correlation.

## Protocol fixes locked in (from the review)

1. Report |r|, sign-consistency, signed skill, and WITHIN-RUN r separately (pooled Pearson over 10
   fault regimes conflates regime-identification with tracking; the pressure claim must survive the
   within-run version).
2. Predictor is SIGNED recoverability (+ a sign-coherence covariate), not unsigned R^2. Similarity
   control uses TEXT-embedding cosine (continuous), not one-hot metadata cosine (a strawman).
3. Hold ONE sensor at a time (co-holding a target's cross-type predictor, e.g. XMV9 for XMEAS18,
   contaminates the dependent variable). One-at-a-time runs: run_recoverability_law.py.
4. Save raw predictions; regenerate all tables from ONE configuration in ONE pass; version the
   held-set in artifact filenames.
5. Retract the "topology-attach A6-A8 significant negative" claim: leave-one-sensor-out shows a single
   sensor (XMV9) drives the significance. Use per-sensor forest plots + hierarchical (sensor->seed)
   bootstrap.
6. K-shot: the mechanistically right adapter is an AFFINE recalibration a*yhat+b on the K labels (one
   label fixes polarity+scale), which should produce the dramatic K=1 jump the label-efficiency story
   needs.

## Experiments (status)

- E1 signed-recoverability law: 6 sensors done (r=0.96); expanding to 21 one-at-a-time
  (run_recoverability_law.py, resumable). DECISIVE headline figure.
- E3 LBNL/Brick replication with a PRE-REGISTERED dissociation: HVAC zone temps are cross-correlated
  (unlike TEP temps), so the same measurement type is predicted to transfer in HVAC and not in TEP,
  by the same recoverability statistic. Same type, different statistics, predicted different outcome.
- E4 label-efficiency with affine K-shot recalibration, only for high-recoverability sensors.
- E6 topology as a stated, LOSO-robust boundary test.

## Use case (selected)

Sensor-commissioning triage / virtual sensing: before installing or labeling a new point, compute the
signed recoverability of its semantic class from existing historian data; the rule returns
transfer / needs-labels / do-not-virtualize (negative recoverability), validated on held-out sensors
the rule never saw. This pre-deployment predictor is the concrete value over "semantics help on
average" prior work.
