# Semantic Feature Transfer with Knowledge-Graph Embeddings

## Project Summary

Modern predictive models usually treat input features as anonymous dimensions. A model may learn extensively from a feature such as **reactor pressure**, **supply-air temperature**, or **invasive systolic blood pressure**, yet it has no explicit mechanism for reusing that knowledge when a new but semantically related feature appears, such as **separator pressure**, **return-air temperature**, or **non-invasive systolic blood pressure**.

This project investigates whether **semantic representations of features derived from domain knowledge graphs (KGs)** can improve transfer to new or data-scarce features.

Instead of representing an observation only as a vector of numerical values,

\[
x = [x_1, x_2, \ldots, x_m],
\]

each input feature is represented as a pair

\[
(x_j, e_j),
\]

where:

- \(x_j\) is the observed numerical value or short temporal window for feature \(j\);
- \(e_j\) is a semantic embedding derived from a domain ontology or knowledge graph describing what the feature means.

The central hypothesis is that a model trained on such representations can reuse knowledge across semantically related features and therefore require fewer labeled examples when new sensors, variables, components, or measurement types are introduced.

The project will evaluate the same method across several unrelated domains and datasets rather than demonstrating it on a single ontology or application.

---

# 1. Research Question

## Primary Question

> Can knowledge-graph-derived feature embeddings improve zero-shot and few-shot generalization to previously unseen or sparsely observed features?

## Secondary Questions

1. Does semantic similarity between features predict actual transferability between their learned representations?
2. Are structured KG embeddings more useful than simple feature-name text embeddings?
3. Is the benefit mainly due to measurement type, component context, graph topology, or their combination?
4. Does semantic feature transfer remain useful when large amounts of target-domain data become available?
5. Can a single architecture operate across datasets with different feature sets and different numbers of variables?
6. Does incorrect or shuffled semantic metadata eliminate the transfer advantage?

---

# 2. Core Hypothesis

Conventional models effectively learn:

\[
f(x_1, x_2, \ldots, x_m) \rightarrow y
\]

where the identity of each feature is tied to a fixed input position.

The proposed approach instead represents a sample as a set of semantically described measurements:

\[
X =
\{(x_j,e_j)\}_{j=1}^{m}
\]

and learns a shared feature encoder:

\[
z_j = F_\theta(x_j,e_j).
\]

The encoded feature representations are then aggregated:

\[
z = A(z_1,\ldots,z_m)
\]

using attention, set aggregation, or a Transformer-style architecture, followed by:

\[
\hat y = G(z).
\]

Because the same \(F_\theta\) processes every feature, the model can learn rules associated with semantic concepts rather than fixed columns.

For a previously unseen feature \(j^\*\), an embedding \(e_{j^\*}\) can be generated from its KG neighborhood even when the model has never observed the corresponding feature identity during training.

---

# 3. Main Contribution

The intended contribution is not merely "adding KG embeddings to tabular data."

The project proposes and evaluates a general framework for:

> **Semantic feature transfer: transferring predictive knowledge from previously observed features to unseen or low-data features using external domain semantics.**

The empirical contribution will be a cross-domain benchmark testing the same idea on multiple datasets and ontologies.

A successful result should show that knowledge-graph semantics reduce the amount of labeled target data required to reach a given predictive performance.

---

# 4. Motivating Scenario

Consider an industrial monitoring company that has trained fault-detection models across many systems.

The existing model has seen:

- reactor pressure;
- separator temperature;
- pump vibration;
- coolant flow;
- valve position.

A new system is commissioned with a previously unseen sensor:

- stripper pressure.

A conventional model sees a new feature identity.

A semantically informed model knows that:

```text
StripperPressureSensor
    measures -> Pressure
    locatedAt -> Stripper
    partOf -> ProcessSystem

ReactorPressureSensor
    measures -> Pressure
    locatedAt -> Reactor
    partOf -> ProcessSystem

SeparatorPressureSensor
    measures -> Pressure
    locatedAt -> Separator
    partOf -> ProcessSystem
```

The model should therefore begin with useful prior knowledge about how pressure-like measurements behave, while still learning target-specific behavior from the available data.

Equivalent situations occur in:

- HVAC systems when a new building or AHU is commissioned;
- water infrastructure when a new pump, tank, or stage is introduced;
- chemical plants when similar sensors appear on different components;
- healthcare when related physiological measurements or laboratory tests are available under different modalities or locations.

---

# 5. Benchmark Strategy

The project should demonstrate the method across **multiple independent domains**, not merely multiple splits of one dataset.

The target benchmark suite will contain approximately:

- 3-4 application domains;
- 5-6 datasets;
- multiple independent ontologies;
- common transfer protocols.

Working benchmark name:

> **SFT-Bench: Semantic Feature Transfer Benchmark**

---

# 6. Candidate Domains and Datasets

## 6.1 HVAC / Smart Buildings

### Dataset
**LBNL Fault Detection and Diagnostics datasets**

Candidate subsets:

- single-duct AHU;
- dual-duct AHU;
- fan-coil unit;
- rooftop unit;
- chiller plant;
- boiler plant.

### Knowledge Graph
**Brick ontology** plus dataset-specific Brick `.ttl` graphs.

### Advantages

- rich labeled time series;
- multiple equipment types;
- repeated semantic measurement types;
- fault labels;
- explicit sensor metadata;
- graph structure distributed with the benchmark;
- ontology already designed for building systems.

### Example semantic relations

```text
SupplyAirTemperatureSensor
    subclassOf -> TemperatureSensor

Sensor_A
    type -> SupplyAirTemperatureSensor
    isPointOf -> AHU_1

AHU_1
    hasPart -> CoolingCoil
    feeds -> VAV_2
```

### Prediction Tasks

Primary:

- fault versus normal operation.

Secondary:

- fault-type diagnosis;
- fault severity.

### Transfer Tasks

- unseen sensor instance;
- unseen measurement subtype;
- unseen equipment instance;
- unseen equipment configuration;
- transfer between related HVAC systems.

---

# 6.2 Water Treatment / Industrial Control Systems

### Datasets
**SWaT** and **WADI**

### Knowledge Graph

Use:

- plant topology supplied by the datasets;
- sensor/actuator metadata;
- SOSA/SSN;
- SAREF/SAREF4WATR where appropriate.

### Semantic Structure

Features include:

- flow sensors;
- level sensors;
- pressure sensors;
- differential-pressure sensors;
- analyzers;
- pumps;
- motorized valves.

Example:

```text
FIT101
    type -> FlowSensor
    measures -> WaterFlow
    locatedAt -> Stage1

LIT301
    type -> LevelSensor
    measures -> WaterLevel
    locatedAt -> Stage3

Stage1
    feeds -> Stage2
```

### Prediction Tasks

- attack/anomaly detection;
- attack-type classification.

### Transfer Tasks

- one process stage to another;
- one sensor instance to another;
- one plant to another;
- SWaT -> WADI transfer;
- limited labeled attack data in the target plant.

---

# 6.3 Chemical Process Monitoring

### Dataset
**Tennessee Eastman Process (TEP)**

### Knowledge Graph

Use:

- OntoCAPE concepts;
- documented process topology;
- feature metadata from the process definition.

### Relevant Features

Repeated concepts include:

- reactor pressure;
- separator pressure;
- stripper pressure;
- temperatures;
- feed flows;
- product flows;
- cooling-water flows;
- valve positions;
- composition measurements.

### Example Structure

```text
Reactor
    downstreamOf -> FeedSystem

Separator
    downstreamOf -> Reactor

Stripper
    downstreamOf -> Separator

ReactorPressure
    measures -> Pressure
    locatedAt -> Reactor
```

### Prediction Task

- fault diagnosis.

### Transfer Tasks

- reactor -> separator;
- reactor -> condenser;
- separator -> stripper;
- corresponding cooling-system faults;
- withholding selected measurement classes during training.

TEP is especially useful because it provides controlled paired or analogous faults across related components.

---

# 6.4 Healthcare

### Datasets

Candidate pair:

- MIMIC-IV;
- eICU.

### Knowledge Graph / Ontology

Use combinations of:

- LOINC;
- SNOMED CT;
- FHIR Observation semantics.

### Candidate Features

Examples include:

- invasive systolic blood pressure;
- non-invasive systolic blood pressure;
- diastolic blood pressure;
- mean arterial pressure;
- heart rate;
- pulse rate;
- arterial oxygen saturation;
- pulse-oximeter oxygen saturation;
- temperature measurements;
- creatinine;
- BUN;
- urine output.

### Prediction Tasks

Possible common targets:

- mortality;
- shock;
- respiratory failure;
- deterioration.

### Transfer Tasks

- measurement modality transfer;
- MIMIC -> eICU;
- rare measurement -> common related measurement;
- feature cold start.

Healthcare should be treated as a later-stage validation because feature-to-ontology mapping requires more manual verification than the industrial datasets.

---

# 7. Unified Data Representation

Each dataset will be converted to a common schema.

Suggested structure:

```text
dataset/
    observations.parquet
    features.csv
    targets.parquet
    graph.ttl
    splits/
        standard.json
        feature_cold_start.json
        component_cold_start.json
        few_shot.json
```

## `features.csv`

Suggested fields:

```text
feature_id
feature_name
dataset
unit
measurement_type
component
component_type
ontology_id
ontology_source
description
```

## Observation Representation

For each observation window:

```text
[
    {
        feature_id: F1,
        values: [...],
        semantic_embedding: [...]
    },
    {
        feature_id: F2,
        values: [...],
        semantic_embedding: [...]
    }
]
```

This allows datasets with different feature counts to use the same architecture.

---

# 8. Knowledge-Graph Representation

The KG should describe at least four semantic dimensions when available.

## 8.1 Measurement Semantics

Examples:

```text
Pressure
Temperature
Flow
Level
Vibration
Concentration
Voltage
HeartRate
```

## 8.2 Component Semantics

Examples:

```text
Reactor
Pump
Valve
Tank
AHU
VAV
Heart
Artery
```

## 8.3 Structural Relations

Examples:

```text
hasPart
partOf
feeds
upstreamOf
downstreamOf
locatedAt
measures
controls
```

## 8.4 Hierarchical Relations

Example:

```text
SupplyAirTemperatureSensor
    subclassOf -> AirTemperatureSensor

AirTemperatureSensor
    subclassOf -> TemperatureSensor

TemperatureSensor
    subclassOf -> Sensor
```

---

# 9. Generating Feature Embeddings

Several approaches should be evaluated.

## Baseline A: Random Feature Embeddings

Random vector assigned to each feature.

Purpose:

- determines whether merely giving the model an additional identifier helps.

## Baseline B: Learned Feature-ID Embeddings

A trainable embedding for each feature identity.

Limitation:

- cannot naturally represent unseen features.

## Baseline C: Text Embeddings

Embed:

- feature name;
- description;
- unit;
- component description.

Example:

```text
"Supply air temperature sensor of AHU"
```

This is a strong baseline because modern text embeddings already contain substantial semantic knowledge.

## Method D: KG Embeddings

Candidate approaches:

- R-GCN;
- GraphSAGE-style relational encoder;
- ontology path aggregation;
- relation-aware Transformer;
- node embedding derived from semantic neighborhood.

The encoder should preferably be **inductive**, so an unseen feature node can receive an embedding from its relations without retraining the entire KG.

## Method E: KG + Text

Combine graph structure and textual node descriptions.

This may ultimately be the strongest model but should not replace the pure-KG comparison.

---

# 10. Predictive Architecture

## Static Features

For scalar feature \(j\):

\[
z_j =
F_\theta(
\tilde x_j,
e_j
)
\]

where \(\tilde x_j\) is normalized locally.

## Time-Series Features

For temporal problems:

\[
h_j =
T_\theta(
x_j(t-W:t)
)
\]

followed by:

\[
z_j =
F_\theta(
h_j,
e_j
).
\]

The same temporal encoder should be shared across features where possible.

## Aggregation

Candidate aggregation mechanisms:

- mean/sum pooling;
- Deep Sets;
- attention pooling;
- Set Transformer;
- Transformer without positional feature identities.

The preferred formulation is:

\[
z =
A(\{z_j\}_{j=1}^{m})
\]

so that input dimensionality can vary between systems.

---

# 11. Normalization

Raw numerical scales differ substantially between features.

Use label-independent normalization such as:

\[
\tilde{x}_{jt}
=
\frac{x_{jt}-\operatorname{median}(x_j)}
{\operatorname{IQR}(x_j)}.
\]

Alternative:

- robust z-score;
- quantile normalization.

Normalization statistics may be estimated from unlabeled target data because realistic deployment usually provides unlabeled observations before sufficient fault/outcome labels accumulate.

Units should remain available as semantic metadata.

---

# 12. Experimental Protocol

The same core experiments should be run in every domain.

## Experiment 1: Standard Full-Data Prediction

Train and test under conventional splits.

Purpose:

- ensure semantic representation does not substantially degrade ordinary predictive performance.

This is not the main experiment.

---

## Experiment 2: Few-Shot Feature Transfer

Choose target features or components and restrict labeled examples involving them.

Evaluate:

\[
K =
1, 2, 5, 10, 20, 50, 100, \text{full}
\]

target examples, trajectories, or fault episodes.

Measure predictive performance as a function of \(K\).

Primary expected effect:

\[
Performance_{KG}(K)
>
Performance_{baseline}(K)
\]

for small \(K\).

---

## Experiment 3: Zero-Shot Feature Cold Start

Withhold a feature type or sensor instance entirely during predictive-model training.

At test time:

- numerical observations become available;
- KG metadata is available;
- no labeled target examples are used.

Question:

> Can semantic metadata provide useful generalization to a completely unseen feature identity?

---

## Experiment 4: Component Cold Start

Hold out an entire component, subsystem, stage, equipment instance, or hospital/system.

Examples:

- AHU instance;
- SWaT process stage;
- TEP component;
- ICU dataset/site.

This is harder and more realistic than withholding individual columns.

---

## Experiment 5: Cross-Dataset Transfer

Examples:

\[
SWaT \rightarrow WADI
\]

and:

\[
MIMIC \rightarrow eICU.
\]

Where feasible, transfer HVAC knowledge between LBNL equipment classes or datasets.

---

# 13. Essential Baselines

The following baselines should be included.

1. **Value-only model**
   - numerical features only.

2. **Fixed-column model**
   - conventional MLP/Transformer architecture.

3. **Learned feature-ID embedding**
   - feature identity encoded but no external semantics.

4. **Random feature embedding**
   - controls for additional dimensions.

5. **Feature-name text embedding**
   - strong semantic baseline.

6. **Metadata embedding**
   - measurement type + component + unit without graph topology.

7. **KG embedding**
   - proposed semantic representation.

8. **KG + text**
   - combined model.

9. **Nearest-semantic-feature transfer**
   - explicitly copy/use the closest known feature.

10. **Oracle source selection**
    - retrospectively choose the source feature producing best target transfer.

The oracle baseline indicates how much exploitable transfer structure exists in the problem.

---

# 14. Critical Ablations

## 14.1 Shuffled KG

Randomly permute feature-to-KG assignments while leaving embedding dimensionality unchanged.

Expected result:

\[
KG_{\text{correct}}
>
KG_{\text{shuffled}}.
\]

If not, the model may simply be benefiting from additional parameters.

---

## 14.2 Remove Graph Topology

Retain only semantic class labels.

This separates:

- ontology labels;
- structural graph information.

---

## 14.3 Remove Measurement Type

Remove concepts such as `Pressure`, `Temperature`, or `Flow`.

Tests how much transfer comes from the obvious measurement category.

---

## 14.4 Remove Component Context

Keep measurement type but remove:

```text
locatedAt
isPointOf
hasPart
feeds
```

Tests whether physical context matters.

---

## 14.5 Text vs Graph

Compare KG embedding with text embedding of equivalent metadata.

This is essential because a reviewer may argue that an LLM/text encoder already captures the same semantics.

---

# 15. Key Scientific Analysis

Beyond aggregate predictive performance, test whether semantic distance predicts transferability.

For every feasible source-target feature pair:

1. compute semantic similarity:

\[
S_{ij} =
sim(e_i,e_j);
\]

2. train or evaluate transfer from \(i\) to \(j\);

3. compute transfer gain:

\[
T_{ij}
=
Performance_{transfer(i\rightarrow j)}
-
Performance_{scratch(j)};
\]

4. test:

\[
corr(S_{ij},T_{ij}).
\]

A positive relationship would support the deeper hypothesis:

> **semantic similarity predicts statistical transferability.**

This could become one of the strongest scientific findings of the project.

---

# 16. Primary Evaluation Metrics

Metrics depend on the prediction task.

Candidate classification metrics:

- AUROC;
- AUPRC;
- F1;
- sensitivity at fixed specificity;
- balanced accuracy.

For anomaly detection:

- point-level AUROC/AUPRC;
- event-level detection rate;
- detection delay where appropriate.

All comparisons should report confidence intervals over:

- repeated seeds;
- held-out features/components;
- target tasks.

---

# 17. Label-Efficiency Metric

The main contribution should be expressed in terms of **data efficiency**, not merely absolute accuracy.

Define:

\[
K_M(q)
\]

as the number of labeled target examples model \(M\) requires to reach performance \(q\).

Then define semantic transfer efficiency:

\[
STE(q)
=
\frac{K_{baseline}(q)}
{K_{KG}(q)}.
\]

Example interpretation:

```text
Baseline requires 100 labeled target episodes to achieve AUROC 0.90.
KG model requires 20.

STE(0.90) = 5.
```

This corresponds to a **5x reduction in target-label requirement**.

---

# 18. Statistical Evaluation

Use repeated experiments across:

- target features;
- target components;
- random seeds;
- datasets.

Recommended analyses:

- bootstrap confidence intervals;
- paired tests across identical source-target splits;
- effect sizes;
- aggregate rank across datasets;
- per-domain results rather than only pooled statistics.

Few-shot results should be shown as learning curves with confidence intervals.

---

# 19. Success Criteria

A convincing result would satisfy most of the following.

## Primary

1. KG semantics improve few-shot performance consistently across at least three domains.
2. Improvement is strongest at low target-data regimes.
3. Advantage decreases naturally as target data becomes abundant.
4. Correct KG mappings outperform shuffled KG mappings.

## Stronger Evidence

5. KG embeddings outperform feature-ID embeddings.
6. KG embeddings outperform simple metadata embeddings.
7. KG embeddings provide value beyond text embeddings.
8. Semantic similarity correlates with empirical transferability.
9. The model supports genuinely unseen feature identities.
10. Cross-dataset transfer works in at least one domain.

---

# 20. Failure Modes and Risks

## Risk 1: Text embeddings perform as well as KG embeddings

This is scientifically useful rather than fatal.

Possible reframing:

> Semantic feature representations enable transfer, while explicit graph structure provides value only in domains where topology matters.

The project should therefore treat text embeddings as a first-class baseline.

---

## Risk 2: KG helps only on HVAC

Interpretation:

- Brick may encode especially informative system structure;
- method may not generalize broadly.

Response:

- narrow claims;
- analyze why certain graph structures support transfer;
- identify necessary conditions.

---

## Risk 3: Semantic similarity does not predict statistical similarity

This may occur because two sensors measure the same physical property but have different dynamic roles.

Potential response:

Include:

- component context;
- topology;
- causal direction;
- operating regime.

The graph may need relations beyond measurement type.

---

## Risk 4: Feature cold-start task is too artificial

Mitigation:

Use realistic transfer units:

- new equipment instance;
- new process stage;
- new plant;
- new hospital;
- new building.

Feature-level withholding remains useful for controlled analysis but should not be the only evaluation.

---

## Risk 5: Target tasks differ too much across domains

The method does not require identical outcome labels across domains.

What must remain constant is:

\[
(x_j,e_j) \rightarrow F_\theta \rightarrow A \rightarrow y.
\]

The predictive head may remain dataset-specific while the feature-semantic mechanism stays identical.

---

# 21. Implementation Plan

## Phase 1: Proof of Concept

### Dataset
LBNL HVAC + Brick.

### Tasks

1. download selected LBNL subsets;
2. parse Brick `.ttl`;
3. map every feature to graph nodes;
4. create unified observation format;
5. implement value-only baseline;
6. implement feature-ID embedding baseline;
7. implement KG feature embeddings;
8. build shared feature encoder;
9. implement attention/set aggregation;
10. create feature cold-start splits;
11. create few-shot curves;
12. run shuffled-KG control.

### Decision Gate

Proceed if KG semantics provide reproducible improvement in low-data or cold-start settings.

---

# 22. Phase 2: Second Domain

### Dataset
SWaT.

### Tasks

1. obtain sensor metadata and process topology;
2. map sensors/actuators to SOSA/SSN/SAREF concepts;
3. create plant KG;
4. preserve exactly the same predictive architecture;
5. run standard, few-shot, and cold-start protocols;
6. compare text, metadata, and KG embeddings.

### Goal

Demonstrate that the method is not specific to Brick or HVAC.

---

# 23. Phase 3: Third Domain

### Dataset
Tennessee Eastman Process.

### Tasks

1. map variables to measurement semantics;
2. represent reactor/separator/stripper/cooling topology;
3. link concepts to OntoCAPE where practical;
4. exploit analogous component/fault pairs;
5. run component-level transfer;
6. test semantic-distance versus transfer-gain relationship.

### Goal

Provide a controlled process-engineering test with clear physical analogies.

---

# 24. Phase 4: Cross-Dataset Validation

Add:

- WADI after SWaT;
- possibly additional LBNL equipment classes.

Primary cross-dataset experiment:

\[
SWaT \rightarrow WADI.
\]

This tests whether semantic representations remain useful across independently collected systems in the same broad domain.

---

# 25. Phase 5: Healthcare Validation

### Datasets

- MIMIC-IV;
- eICU.

### Tasks

1. select a manageable common feature set;
2. map measurements to LOINC/SNOMED/FHIR concepts;
3. define matching prediction tasks;
4. create modality-level cold-start experiments;
5. test MIMIC -> eICU transfer;
6. audit all ontology mappings manually or semi-automatically.

Healthcare should be added only after the core approach is stable.

---

# 26. Suggested Work Packages

## WP1 — Benchmark Construction

Deliverables:

- common data schema;
- dataset converters;
- ontology mappings;
- cold-start splits;
- few-shot splits.

---

## WP2 — Semantic Representation

Deliverables:

- text embedding baseline;
- metadata baseline;
- KG encoder;
- inductive embedding mechanism.

---

## WP3 — Predictive Architecture

Deliverables:

- shared feature encoder;
- temporal encoder;
- permutation-invariant aggregation;
- dataset-specific output heads.

---

## WP4 — Cross-Domain Evaluation

Deliverables:

- learning curves;
- cold-start results;
- cross-dataset transfer;
- ablation studies.

---

## WP5 — Semantic Transfer Analysis

Deliverables:

- source-target transfer matrix;
- semantic similarity matrix;
- similarity-transfer correlation;
- negative-transfer analysis.

---

## WP6 — Benchmark and Paper Release

Deliverables:

- SFT-Bench;
- mappings;
- reproducible splits;
- code;
- experiment configuration;
- manuscript.

---

# 27. Recommended Experimental Order

Do not begin with all datasets simultaneously.

Recommended sequence:

```text
1. LBNL / Brick
        ↓
2. SWaT
        ↓
3. Tennessee Eastman
        ↓
4. WADI
        ↓
5. MIMIC + eICU
```

At each step, retain the same basic architecture and protocol.

If major architectural changes are required for every domain, that weakens the general-method claim.

---

# 28. Minimal Viable Paper

A credible initial paper could contain:

### Three domains

1. LBNL HVAC + Brick;
2. SWaT + SAREF/SOSA;
3. Tennessee Eastman + OntoCAPE/process topology.

### Required Experiments

- standard prediction;
- few-shot feature transfer;
- zero-shot feature cold start;
- component cold start;
- shuffled-KG control;
- text-embedding baseline;
- semantic-distance analysis.

This is sufficient to evaluate whether the main hypothesis is real.

---

# 29. Strong Journal Version

A stronger final benchmark would contain approximately:

### Domain 1 — HVAC
- multiple LBNL equipment datasets.

### Domain 2 — Water
- SWaT;
- WADI.

### Domain 3 — Chemical Process
- Tennessee Eastman.

### Domain 4 — Healthcare
- MIMIC-IV;
- eICU.

This would provide roughly six datasets across four unrelated domains.

---

# 30. Expected Paper Structure

## 1. Introduction

Motivate the problem:

> Models know numerical values but usually do not know what their features mean.

Introduce the new-feature cold-start problem.

---

## 2. Related Work

Cover:

- transfer learning;
- tabular transfer learning;
- schema matching;
- knowledge-graph embeddings;
- ontology-informed machine learning;
- semantic feature representations;
- inductive KG learning;
- set/feature-token architectures;
- metadata-aware models;
- industrial fault diagnosis;
- cross-dataset clinical prediction.

---

## 3. Problem Definition

Define:

- feature;
- semantic graph;
- source features;
- target features;
- zero-shot feature;
- few-shot feature;
- component transfer;
- semantic transfer efficiency.

---

## 4. Method

Describe:

\[
(x_j,e_j)
\rightarrow
F_\theta
\rightarrow
A
\rightarrow
\hat y.
\]

Explain KG construction and inductive feature embeddings.

---

## 5. SFT-Bench

Describe all datasets, ontologies, mappings, and transfer splits.

---

## 6. Experiments

Present:

- standard performance;
- few-shot curves;
- zero-shot experiments;
- cross-component transfer;
- cross-dataset transfer.

---

## 7. Ablations

Include:

- shuffled semantics;
- text only;
- class only;
- graph topology removed;
- measurement type removed;
- component context removed.

---

## 8. Semantic Transfer Analysis

Analyze:

\[
semantic\ similarity
\leftrightarrow
transfer\ gain.
\]

---

## 9. Discussion

Address:

- when semantics help;
- when they do not;
- negative transfer;
- ontology quality;
- deployment implications.

---

## 10. Conclusion

Main intended conclusion:

> External semantic knowledge can act as a prior over feature identity, allowing predictive models to reuse learned behavior when new but semantically related variables appear.

---

# 31. Central Figures

The final paper should likely contain the following figures.

## Figure 1 — Method Overview

```text
Domain KG
   │
   ├── feature A ──> embedding A ──┐
   ├── feature B ──> embedding B ──┤
   └── feature C ──> embedding C ──┘
                                  │
Observed values ──────────────────┤
                                  ↓
                         Shared feature encoder
                                  ↓
                         Set/attention aggregator
                                  ↓
                              Prediction
```

---

## Figure 2 — New-Feature Transfer

Illustrate:

```text
Previously trained:
Pressure@Reactor
Pressure@Separator
Temperature@Stripper

New:
Pressure@Stripper

KG semantics
      ↓
use existing pressure/component knowledge
      ↓
few target labels
```

---

## Figure 3 — Few-Shot Curves

Plot:

```text
Performance
   |
   |            KG
   |          /
   |       Text
   |      /
   |   Baseline
   |__|____|____|____
      1    5   20   100

      Target labels
```

---

## Figure 4 — Semantic Similarity vs Transfer Gain

Scatter plot:

\[
x = KG\ similarity
\]

\[
y = empirical\ transfer\ gain.
\]

This figure directly tests the underlying scientific hypothesis.

---

# 32. Core Claim to Aim For

The strongest defensible claim would be:

> **Across multiple unrelated domains and independent knowledge graphs, representing numerical features together with external semantic embeddings improves generalization to unseen and low-data features. The benefit is concentrated in the few-shot regime, and semantic similarity predicts which previously learned features transfer most effectively.**

A stronger quantitative version would report:

> The proposed method reduces the amount of target labeled data required to reach a fixed predictive-performance threshold by \(X\)-fold across \(N\) datasets and \(D\) domains.

---

# 33. Immediate Next Step

The first implementation milestone should answer only one question:

> **Does semantic feature representation produce a measurable few-shot advantage in LBNL/Brick under a rigorously controlled held-out-feature experiment?**

If yes, freeze the core architecture and replicate the protocol in SWaT and Tennessee Eastman before adding further complexity.

