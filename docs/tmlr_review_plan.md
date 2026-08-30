# Publication strategy and venue-specific review
## Semantic Feature Transfer with Knowledge-Graph Embeddings / SFT-Bench

**Paper reviewed:** https://apartsin.github.io/SFT-Bench/  
**Repository reviewed:** https://github.com/apartsin/SFT-Bench  
**Review date:** 2026-08-30  
**Recommended primary venue:** **Transactions on Machine Learning Research (TMLR)**  
**Current recommendation if submitted today:** **Reject, with a clear path to resubmission**  
**Target after revision:** **Accept / strong-accept-equivalent reviewer consensus**

---

# 1. Executive decision

## 1.1 Best venue: Transactions on Machine Learning Research (TMLR)

TMLR is the best home for the paper **if the paper is reframed as a controlled empirical study of semantic feature transfer and its boundary conditions**, rather than as a claim that knowledge-graph embeddings generally outperform alternatives.

This is unusually well aligned with TMLR's explicit acceptance standard. TMLR asks two principal questions:

1. Are the paper's claims supported by accurate, convincing, and clear evidence?
2. Would at least some members of the TMLR audience be interested in the findings?

TMLR explicitly says that lack of a new state of the art, modest significance, or a result that mainly clarifies when an idea works should not be used by itself as a rejection reason. That is important here because the scientifically interesting result in the current paper is **not** “KG wins.” The current evidence instead says:

- semantic information can support transfer to a prediction target that was not used as a training target;
- an admission test is needed because some sensor datasets make the putatively held-out variables redundant;
- simple semantic metadata can be as useful as, or better than, the current graph-derived encoding;
- raw semantic similarity is a poor predictor of transferability;
- statistical predictability between variables matters more than ontology proximity alone.

Those are potentially useful ML findings, and TMLR is structurally more receptive to such a carefully bounded result than venues that will demand a large algorithmic SOTA gain.

Official TMLR sources:

- Acceptance criteria: https://jmlr.org/tmlr/acceptance-criteria.html
- Reviewer guide: https://www.jmlr.org/tmlr/reviewer-guide.html
- Author guide: https://www.jmlr.org/tmlr/author-guide.html
- Editorial policies: https://www.jmlr.org/tmlr/editorial-policies.html
- TMLR homepage: https://jmlr.org/tmlr/

TMLR is also **no-fee open access** under its current editorial policy and uses rolling, double-blind OpenReview review.

### Why not make Knowledge-Based Systems the first target?

**Knowledge-Based Systems (KBS)** is an excellent thematic fit: it explicitly covers machine-learning methodology, knowledge representation/engineering, prediction systems, and knowledge-based AI. However, KBS would naturally invite the question “what does the knowledge graph contribute?” The current answer is uncomfortable:

- the implemented “KG embedding” is a hand-constructed relation-token multi-hot representation rather than a learned KGE;
- graph topology does not consistently beat simple metadata;
- the central reported table actually contains a case where metadata is much better than the graph representation;
- the current graph-vs-metadata comparison does not cleanly isolate topology.

A strong KBS paper is possible, but only if the revision produces a cleaner and stronger knowledge-representation contribution. If the final empirical truth remains “graph topology is useful only in certain transfer regimes, while typed metadata is often sufficient,” TMLR is the better editorial match.

KBS scope: https://shop.elsevier.com/journals/knowledge-based-systems/0950-7051

## 1.2 Ranked venue list

| Rank | Venue | Fit for the *revised* paper | Main reason | Current-paper outlook |
|---|---|---:|---|---|
| **1** | **TMLR** | **9.5/10** | Explicitly values technical correctness and informative empirical findings even without SOTA; ideal for a careful semantic-transfer study | Best route, but current validity and literature gaps still justify reject |
| **2** | **Knowledge-Based Systems** | **9/10** | Excellent semantic/KG/AI scope | Strong only if topology/KG contribution is made technically clean and materially useful |
| **3** | **Machine Learning (Springer)** | **8/10** | Transfer-learning methodology and generalization are in scope | Would expect broader methodological depth and stronger cross-domain evidence |
| **4** | **Data Mining and Knowledge Discovery / ACM TKDD** | **7.5/10** | Benchmark and knowledge-discovery framing can fit | Needs a much more mature multi-dataset benchmark and broader baselines |
| **5** | **Neural Networks / similar general ML journal** | **7/10** | Architecture and transfer mechanism fit | Current method is too simple relative to expected neural-method novelty |

### Practical decision

Submit to **TMLR only after the critical redesign below**. Do not submit the current web-paper version. The main blockers are not stylistic. At least four are construct-validity issues that a careful reviewer can identify directly from the public implementation.

---

# 2. What the paper actually establishes today

The paper's intended thesis is:

> A model can use external semantics attached to numerical features to transfer predictive behavior from previously learned features to semantically related new features, reducing target-label requirements.

The current experiments establish a narrower result:

> In a query-conditioned TEP imputation setup, semantic descriptors can help a model predict a feature that was never used as a prediction target during training. The effect is highly heterogeneous across sensor types. Simple typed metadata is already strong for some targets; the current graph-derived representation is not consistently superior. Fault-classification datasets can fail to expose semantic-feature transfer when the held columns are statistically redundant.

That narrower result can become a good TMLR paper. It is, however, materially different from several phrases in the current title, abstract, introduction, and benchmark description.

---

# 3. TMLR-style referee report

## Recommendation

**Reject in current form.**

I would encourage resubmission after a substantial experimental revision. I do not view the basic research question as the problem. The problem is that the strongest claims currently exceed what the experimental construct supports, several decisive baselines are omitted from the paper despite being implemented, the train/test construction is too weak for the strength of the generalization language, and the related-work section misses the literature most likely to determine novelty.

If the authors implement the critical changes in Sections 6–10 of this review and the main semantic effect survives, I would expect to move to **Accept** or **Leaning Accept** under TMLR's criteria.

## 3.1 Summary of contributions

The submission studies what it calls **semantic feature transfer**: attaching semantic representations to numerical features and using a shared feature encoder so knowledge can potentially transfer when a new sensor or variable is introduced. It proposes SFT-Bench, a protocol that holds the predictive architecture fixed while varying the feature-semantic representation, and introduces an admission/headroom check intended to reject datasets where the supposedly informative target feature is redundant. Experiments on LBNL HVAC and the Tennessee Eastman Process (TEP) suggest that the LBNL fault-detection setting does not provide adequate feature-level headroom, whereas TEP does. A strengthened TEP imputation experiment finds heterogeneous zero-shot tracking skill, including strong transfer for a held pressure target, and reports that simple metadata can be as effective as or better than the graph-derived feature representation.

The question is interesting and the attempt to distinguish semantic effects from architectural effects is commendable. However, the current implementation does not yet support the paper's strongest “unseen feature” and “knowledge-graph transfer” claims.

## 3.2 Criterion 1: Are the claims supported by accurate, convincing, and clear evidence?

**No, not yet.**

The evidence supports an **unseen prediction-target** result, but not yet a strict **unseen-feature cold-start** result. In the TEP imputation implementation, the ten held queries are removed from the list of features selected as prediction targets, but their values remain present as context while the model is trained to predict the other features. Consequently, the model has seen those sensors numerically during training. In the learned-ID condition, their ID embeddings can also receive gradients when the corresponding sensors act as context. This is inconsistent with the paper's stronger language that the model handles a feature whose predictive behavior/identity it has “never seen during training.”

Further, the TEP imputation code pools rows from a small number of process trajectories and randomly partitions rows into train/test sets. This permits strongly correlated observations from the same runs/regimes to appear on both sides. For a paper making transfer/generalization claims, evaluation needs to be grouped by independent run/trajectory or use the canonical TEP train/test separation.

The central ablation is also not clean. Metadata (C4) contains measurement type, component type, and unit, whereas the current KG representation (C5) contains measurement type plus relation/neighbor tokens but omits some of those metadata fields. Thus C5 versus C4 is not “same information plus topology” and cannot isolate the marginal value of topology. Likewise, shuffling the complete KG vector shuffles basic type semantics as well as graph structure, so correct-KG versus shuffled-KG is a broad semantic-assignment test, not a topology test.

Finally, the paper's zero-shot table excludes Value-only, Random, Learned-ID, and Text despite the code running all seven conditions and despite Text being explicitly described in the experimental specification as the decisive semantic competitor. This makes the main empirical claim incomplete.

## 3.3 Criterion 2: Would at least some TMLR readers be interested?

**Yes.**

There is clear current interest in cross-table transfer, schema-robust tabular learning, semantic column representations, tabular foundation models, and graph-based multivariate imputation. A result that cleanly distinguishes:

- feature-identity transfer,
- semantic metadata transfer,
- text-derived transfer,
- topology-derived transfer,
- and statistical cross-variable predictability,

would be useful even if the final conclusion is that graph topology helps only in particular structural regimes.

The key is that the paper must position itself relative to modern transferable/semantic tabular models and graph-based imputation rather than implying that semantic feature descriptions are largely absent from prior tabular ML.

## 3.4 Strengths

### S1. The paper asks a real and under-formalized deployment question

The difference between “same fixed schema forever” and “a new sensor arrives with known metadata but little or no labeled history” is meaningful in industrial and scientific ML.

### S2. The construct-matched instinct is correct

Keeping the predictive architecture fixed while varying only the feature representation is the right instinct. Random, ID, text, metadata, graph, and shuffled controls are much stronger than comparing unrelated end-to-end systems.

### S3. The admission/headroom concept is useful

The LBNL negative result is scientifically valuable. If removing a fault's nominally primary sensors does not meaningfully reduce performance, the dataset cannot support a claim that the model learned to exploit a newly introduced sensor. Turning that observation into an explicit dataset-admission rule is a contribution worth preserving.

### S4. The paper reports negative and boundary-condition results

The finding that semantic similarity has weak association with transferability, and that statistical predictability matters more, is more interesting than a simplistic “KG beats baseline” story.

### S5. The query-conditioned imputation task is better aligned with feature semantics than the original fault classifier

The imputer forces the model to reason about *which variable is being requested*. That is a substantially better mechanistic probe of semantic feature transfer than a classifier where many correlated sensors can reveal the label.

### S6. The implementation is compact enough to audit

The public code makes it possible to identify exactly what the controls do. With stronger artifact completeness, this could become a reproducible TMLR submission.

---

# 4. Blocking issues

## W1. The current experiment is not a strict unseen-feature cold start
**Severity: BLOCKING**

### Where

Paper, Introduction / Problem Definition:

> “a feature whose meaning is known from a domain ontology but whose predictive behaviour has not yet been observed”

and

> “an embedding ... is generated ... so the encoder produces a representation for it without ever having seen its identity during training.”

Implementation:

- `src/sft/experiment/run_imputation.py`
- `src/sft/model/sft_model.py`

### Problem

The held queries are excluded from the **target list**, not removed from the training data. When the model predicts another feature, all other sensor values are available as context, including the ten eventual held targets. Therefore the model has already observed those “held” sensors and can learn how their values relate to the rest of the system.

This matters especially for C2 Learned-ID. The paper/spec says a learned ID “cannot embed an unseen feature.” In the imputation architecture, an embedding exists for all 52 features. Even if a held feature is never selected as the target, its embedding participates as a context representation and can be trained through gradients from other targets.

Thus:

- **supported construct:** never-trained-as-target / target-head cold start;
- **claimed construct:** unseen feature / new sensor cold start.

These are not equivalent.

### Concrete fix

Create and report two explicitly different protocols:

**Protocol A: Unseen-target transfer**
- Current setup.
- Held variable can appear as context during source training.
- It is never used as an imputation target.
- This tests whether a shared target-conditioning mechanism can generalize its output role.

**Protocol B: Strict unseen-feature transfer — primary protocol**
- Remove every held target sensor from **all source-training inputs and targets**.
- Do not compute target-specific normalization statistics from labeled source training values.
- Do not instantiate/train a target ID embedding.
- At adaptation/test time provide:
  - the new sensor's external semantic descriptor;
  - the other observed context sensors;
  - K target labels/values only according to the few-shot condition.
- For K=0, the held sensor's values must never have been seen during training.
- Audit this with an assertion that the held sensor index is absent from every source-training tensor and every trainable feature-ID table.

A high-quality paper should show that Protocol A is easier and that the proposed mechanism still has measurable benefit under Protocol B.

---

## W2. Random row splits create correlated train/test leakage in TEP
**Severity: BLOCKING**

### Where

`src/sft/experiment/run_imputation.py`, `load_rows()` and `main()`.

The implementation pools rows from fault-free and post-fault trajectories, randomly caps them, then applies a random 70/30 row split.

The TEP fault-detection code also creates overlapping windows and randomly splits windows from the same trajectory.

### Problem

Rows/windows from the same physical simulation trajectory and operating episode are strongly autocorrelated. With window length 20 and stride 5 in the classifier, train and test windows can even overlap in time. A random split estimates interpolation within observed trajectories, not generalization to independent runs.

This affects both:
- the absolute imputation correlation;
- the certainty of differences between semantic conditions.

### Concrete fix

Use **grouped evaluation units**.

Preferred hierarchy:

1. Use canonical TEP training trajectories for model fitting and canonical `_te` trajectories as held-out evaluation whenever possible.
2. Where multiple fault runs are needed, hold out complete fault trajectories / run IDs.
3. Never allow overlapping windows from one trajectory to be split across train and test.
4. For each outer split, derive normalization statistics from training runs only, unless the experiment is explicitly labeled **transductive**.
5. Treat run/trajectory, not row, as the primary resampling unit for confidence intervals.

Add a table:

| Split | Training systems/runs | Validation systems/runs | Test systems/runs | Can target values occur in source training? |
|---|---|---|---|---|

A reviewer should be able to verify separation without reading code.

---

## W3. C4 versus C5 does not isolate graph topology
**Severity: BLOCKING**

### Where

Paper, Method and Evaluation Protocol:

> “Metadata ... without graph topology”

> “Knowledge graph ... type and relations”

Implementation: `src/sft/embeddings/embedders.py`.

### Problem

C4 Metadata uses:
- measurement type;
- component type;
- unit.

C5 KG currently uses:
- `self.type=measurement_type`;
- relation/neighbor tokens.

Therefore C5 is **not C4 plus topology**. It removes some metadata and adds some other fields. Any C5−C4 difference can be caused by lost unit/component information rather than topology.

The current C6 shuffle also permutes the whole semantic vector, including measurement-type information. It tests whether correct semantic assignment matters, but not whether graph topology matters beyond typed metadata.

### Concrete fix

Use a nested ablation:

- **M0 Value-only:** no semantic input.
- **M1 Random-ID control:** fixed random vector.
- **M2 Type only:** measurement type.
- **M3 Core metadata:** measurement type + unit + component type.
- **M4 Core metadata + component identity/class.**
- **M5 Core metadata + text description embedding.**
- **M6 Core metadata + explicit graph topology.**
- **M7 Core metadata + learned inductive graph encoder.**
- **M8 Topology-shuffled:** preserve M3 exactly, shuffle only topology/neighborhood assignment **within measurement type**.
- **M9 Full-semantics shuffled:** shuffle all semantic assignments.

The decisive graph-topology contrast becomes:

> M6 − M3, with identical metadata and model capacity.

The decisive “is the topology correctly attached?” contrast becomes:

> M6 − M8.

This one redesign would make the current paper far more defensible.

---

## W4. The paper calls a hand-built multi-hot representation a “knowledge-graph embedding” and inaccurately labels TEP as OntoCAPE
**Severity: BLOCKING FOR CURRENT TITLE/FRAMING**

### Where

Title:

> “Semantic Feature Transfer with Knowledge-Graph Embeddings”

SFT-Bench table:

> TEP ontology: “OntoCAPE, process topology”

Implementation:
- `src/sft/embeddings/embedders.py`
- `src/sft/datasets/tep.py`

The TEP loader explicitly states:

> “No formal TEP ontology exists, so the knowledge graph is built here from the Downs & Vogel variable table and process topology.”

### Problem

The implemented C5 representation is a deterministic multi-hot bag of relation/neighbor-name tokens. It is graph-derived, but it is not a learned KGE in the usual TransE/DistMult/ComplEx/R-GCN/GraphSAGE sense.

More importantly, the actual TEP graph is manually constructed in the code. It is not an OntoCAPE-derived graph. The web paper's ontology table is therefore inconsistent with the implementation.

### Concrete fix

Choose one of two honest directions.

**Direction 1: empirical semantics paper — recommended for TMLR**
- Rename C5 **graph-derived topology encoding**.
- Change title to avoid implying sophisticated KGE methodology.
- Change the TEP ontology row to **“hand-constructed process graph from the TEP variable definitions and process topology”**.
- Explain relation provenance.
- Treat learned KGE/GNN embeddings as an optional stronger baseline.

**Direction 2: actual KG-method paper — better for KBS**
- Map TEP concepts to a real ontology vocabulary where defensible.
- Build an RDF/typed graph with documented ontology alignment.
- Use an inductive relation-aware graph encoder, e.g. R-GCN/GraphSAGE-style message passing.
- Compare learned graph embeddings against the simple relation-token encoding.
- Demonstrate a consistent topology-specific gain.

Do not claim OntoCAPE unless the released artifact actually contains and uses an audited OntoCAPE mapping.

---

## W5. The main zero-shot result omits the paper's decisive controls
**Severity: BLOCKING**

### Where

Paper's zero-shot imputation table reports only:
- Metadata;
- KG;
- Shuffled KG.

But the implementation and experiment specification define and run:
- C0 Value-only;
- C1 Random;
- C2 Learned-ID;
- C3 Text;
- C4 Metadata;
- C5 KG;
- C6 KG-shuffled.

The spec itself calls C3 Text:

> “THE competitor.”

### Problem

A reviewer cannot tell from the paper whether:
- metadata beats no semantics;
- KG beats a random feature vector;
- text semantics equal or exceed the graph;
- learned IDs have residual transfer;
- shuffled semantics are worse than a true no-semantic baseline.

This is particularly serious because the paper's novelty depends on external semantics, and modern text embeddings of column descriptions are an obvious alternative.

### Concrete fix

Report **all conditions** for every primary K and every primary domain.

Minimum main-table columns:

| Condition | K=0 | K=1 | K=2 | K=5 | K=10 | K=20 | Full |
|---|---:|---:|---:|---:|---:|---:|---:|

For the main paper, plot:
- Value-only;
- Random;
- Text;
- Metadata;
- Metadata+Topology;
- Topology-shuffled.

Put Learned-ID in the same plot if interpretable under the strict protocol; otherwise explain why it is undefined at K=0.

The paper must also log the exact text encoder name/version and assert that the hashed-BoW fallback was **not** used for reported results.

---

## W6. The “five of five seeds” statement is statistically weak and the pressure claim is based on too few targets
**Severity: BLOCKING FOR THE CURRENT ABSTRACT WORDING**

### Where

Abstract:

> “correct semantics beating a shuffled-semantics control in five of five seeds and pressure sensors tracked at correlation 0.92.”

### Problem

Five initialization/data-split seeds are not five independent scientific replications. Under a simple sign test, 5/5 positive differences alone is not even below 0.05 two-sided (`2 / 2^5 = 0.0625`).

Also, the held-query list contains only one held pressure target (`XMEAS16`). The 0.919 pressure result is therefore not evidence across multiple held pressure sensors; it is one target averaged across seeds. The plural phrase “pressure sensors” overstates the breadth of evidence.

### Concrete fix

- Use at least 10–20 initialization seeds for model stochasticity.
- More importantly, increase the number of **independent held target features**.
- Make held feature / component and trajectory the main sampling units.
- Report paired effect size and 95% CI.
- Use hierarchical bootstrap:
  1. resample domain/split;
  2. resample held target/component;
  3. resample run;
  4. optionally resample initialization seed within each design cell.
- For per-type effects, report `n_targets` and `n_runs`.
- If making multiple type-specific claims, control family-wise error or FDR.

Rewrite the current statement, until more targets exist, as:

> “For the held stripper-pressure target, mean zero-shot tracking correlation was 0.92 across five model seeds.”

That is accurate.

---

## W7. “Cross-domain benchmark” is ahead of the evidence
**Severity: MAJOR**

### Where

Title block / abstract / Section 4:

> “SFT-Bench, a cross-domain benchmark”

Table lists:
- LBNL/Brick;
- SWaT/WADI;
- TEP;
- healthcare.

But the paper explicitly says only LBNL and TEP carry results, and LBNL is rejected by the admission gate for the headline transfer task.

### Problem

The actual positive semantic-transfer evidence is presently **one process domain**. The LBNL result is useful as a negative dataset-screening result, but it does not validate transfer across domains. SWaT/WADI and healthcare are future plans.

Calling the current artifact a cross-domain benchmark makes the paper look unfinished.

### Concrete fix

Two acceptable paths:

**Path A: Mature SFT-Bench**
- Add at least two admitted domains with real transfer results.
- Recommended:
  - TEP;
  - LBNL/Brick on **imputation**, not fault classification;
  - SWaT or WADI as a real cyber-physical testbed.
- Keep healthcare as future work unless fully audited.

**Path B: Controlled-study paper**
- Stop calling the current paper a cross-domain benchmark.
- Call SFT-Bench an **evaluation protocol/suite**.
- Present LBNL as a falsification/admission case and TEP as the main mechanistic case.
- Make the contribution the evaluation methodology plus the empirical boundary condition.

For strong acceptance, Path A is preferable.

---

## W8. STE is defined but not actually established, and the K=0 “limit” is mathematically problematic
**Severity: MAJOR**

### Where

Problem Definition:

\[
STE(q)=K_{\text{baseline}}(q)/K_{\text{semantic}}(q)
\]

then:

> “The zero-shot results ... correspond to the limit \(K_{\text{semantic}}=0\).”

### Problem

The paper frames label efficiency as a contribution but reports only the zero-shot rung. The code already defines multiple K values, so the absence of few-shot curves is conspicuous.

Moreover, at exactly \(K_{\text{semantic}}=0\), the proposed ratio is undefined/infinite. Calling the zero-shot case “the limit” does not make the metric mathematically well-defined.

### Concrete fix

Use two separate outcome families:

**Zero-shot transfer**
- skill at K=0;
- semantic gain at K=0:
  \[
  \Delta_0 = M_{\rm semantic}(0)-M_{\rm baseline}(0)
  \]

**Few-shot sample efficiency**
- K ∈ {1, 2, 5, 10, 20, 50, 100};
- threshold-based STE only where both denominators are positive and threshold crossing is observed;
- interpolate between K points with a stated method;
- report confidence intervals.

Add a robust aggregate such as:
- area under the transfer curve (AUTC);
- normalized area under performance-vs-log(K);
- labels saved to reach a pre-specified performance q.

Pre-register q using source/validation data, not after examining test curves.

---

## W9. The related-work section misses the literature that most directly challenges novelty
**Severity: BLOCKING FOR PUBLICATION QUALITY**

### Where

Section 8 currently compresses the literature into one short paragraph and only six references.

### Missing/required lines of work

#### A. Cross-table and variable-schema transfer

1. **TransTab: Learning Transferable Tabular Transformers Across Tables**, NeurIPS 2022.  
   https://proceedings.neurips.cc/paper_files/paper/2022/hash/1377f76686d56439a2bd7a91859972f5-Abstract-Conference.html  
   Directly relevant because it combines column descriptions with values and studies transfer/incremental feature scenarios.

2. **XTab: Cross-table Pretraining for Tabular Transformers**, ICML 2023.  
   https://proceedings.mlr.press/v202/zhu23k.html  
   Relevant to cross-table shared representations and variable schemas.

3. **CARTE: Pretraining and Transfer for Tabular Learning**, ICML 2024.  
   https://proceedings.mlr.press/v235/kim24d.html  
   Especially important: CARTE explicitly represents tabular/relational information as graphs, embeds column names, and transfers across unmatched columns.

4. **UniTabE**, ICLR 2024.  
   Must be discussed as a schema-flexible universal tabular representation line.

5. **TabSTAR**, NeurIPS 2025.  
   https://proceedings.neurips.cc/paper_files/paper/2025/hash/faf6e23e198314c7728eaa6ac44ae079-Abstract-Conference.html  
   Important because it makes semantic representations central and uses an architecture without dataset-specific parameters.

#### B. Text/LLM semantics for columns

6. **TabLLM: Few-shot Classification of Tabular Data with Large Language Models**, AISTATS 2023.  
   https://proceedings.mlr.press/v206/hegselmann23a.html

These papers make the current Introduction sentence

> “Predictive models usually treat input features as anonymous dimensions”

too broad for 2026. Rewrite it as:

> “Conventional fixed-schema predictors tie feature identity to column position, while recent cross-table and foundation-model approaches increasingly exploit column text or schema-flexible representations. We study a narrower question: whether *external structured domain semantics* can support transfer to a target sensor whose predictive role has not been trained.”

#### C. Multivariate time-series imputation

7. **GRIN: Filling the Gaps: Multivariate Time Series Imputation by Graph Neural Networks**, ICLR 2022.  
   https://openreview.net/pdf?id=kOu3-S3wJ7

8. **SAITS: Self-Attention-based Imputation for Time Series**, Expert Systems with Applications 2023.  
   https://doi.org/10.1016/j.eswa.2023.119619

9. **BRITS: Bidirectional Recurrent Imputation for Time Series**, NeurIPS 2018.

The key distinction must be explicit:

> GRIN/SAITS/BRITS learn to reconstruct missing values for known channels. SFT asks whether external semantics can enable prediction for a channel whose target role, and under the strict protocol whose values, were absent from source training.

That is a defensible novelty distinction if the strict protocol is actually implemented.

#### D. Ontology/KG sensor modeling

Add primary citations for:
- Brick;
- SOSA/SSN;
- SAREF;
- OntoCAPE only if actually used;
- any ontology-to-ML / knowledge-informed sensor modeling work directly used to motivate graph semantics.

### Required novelty paragraph

The paper needs one explicit paragraph saying:

> “Unlike TransTab/XTab, we do not primarily study transfer across tables using textual column descriptions or shared pretrained tabular parameters. Unlike CARTE, we do not rely on graphizing the table instance itself. Unlike GRIN, we do not assume that every target sensor/channel has been represented during source training. Our object of study is the incremental value of externally supplied domain semantics for a held sensor under a construct-matched architecture, with metadata-preserving graph ablations.”

Without such a paragraph, reviewers may conclude that the “semantic columns enable transfer” idea is already established.

---

## W10. “Task matters” is confounded with architecture
**Severity: MAJOR**

### Where

Abstract / Results:

> “the task matters: fault detection is value-evident ... whereas on zero-shot sensor imputation ...”

### Problem

The fault classifier uses `SFTModel`, while the imputation experiment uses `SFTImputer` with query-conditioned cross-attention. Thus task and architecture change together. The paper cannot causally infer that **task alone** explains the difference.

The more accurate conclusion is:

> the original classification formulation did not expose the intended construct, while the query-conditioned imputation formulation did.

### Concrete fix

Either:

1. Use a common semantic encoder and controlled task-specific heads, and avoid claiming a pure task effect; or
2. Run an architecture-matched experiment where the same query-conditioned representation is used for both tasks.

Preferred wording:

> “The classification formulation did not create identifiable dependence on the held feature, whereas query-conditioned imputation did.”

---

## W11. Semantic-distance analysis mixes representations and does not test the strongest mechanism
**Severity: MAJOR**

### Where

Paper:

> “semantic similarity alone does not predict transfer gain (Spearman 0.06)”

Implementation:
- semantic similarity is computed from raw KG-derived vectors;
- transfer skill is taken from the C4 Metadata condition.

The published value of approximately 0.06 is numerically consistent with the released ten-query table, but the causal interpretation is weak.

### Problem

The analysis correlates distance in one representation with performance of another representation. More importantly, raw cosine similarity between hand-built feature vectors is not the mechanism the query-conditioned network necessarily uses.

### Concrete fix

Report at least four predictors of transferability:

1. **typed-metadata similarity**;
2. **graph-topology similarity**;
3. **text-embedding similarity**;
4. **empirical source-only predictability**, e.g. best sibling correlation / mutual information / source-trained regression predictability.

Define the dependent variable as **transfer gain over the matched baseline**, not raw performance:

\[
G_j = M_{\rm semantic,j} - M_{\rm random,j}
\]

or, for topology:

\[
G^{topo}_j = M_{\rm metadata+topology,j} - M_{\rm metadata,j}.
\]

Then ask which predictor explains \(G_j\).

With sufficient target sensors, fit a simple mixed-effects or rank-regression model. The expected high-value finding is:

> structural semantics predict topology-specific gain only after statistical recoverability is established.

That is much stronger than a ten-point Spearman scatter.

---

## W12. The public artifact is incomplete relative to the paper's reproducibility claims
**Severity: MAJOR**

### Where

Paper, Evaluation Protocol:

> “every comparison is computed in a single pass and saved as one artifact.”

Repository `results/` currently exposes only a small subset of the artifacts described by the code/specification, including:
- `phase1_sanity.json`;
- `phase1_smoke_sanity.json`;
- `tep_impute_bytype.csv`;
- `tep_semantic_distance.csv`.

The raw main experiment parquet files and several gate/transfer result files referenced by the implementation are not currently in the public results directory.

### Problem

A reviewer cannot reconstruct:
- all seven-condition zero-shot results;
- per-seed raw results;
- confidence intervals;
- few-shot curves;
- exact gate values;
- raw predictions.

### Concrete fix

Release:

```text
results/
  manifest.json
  tep/
    split_manifest.json
    gate_by_run.csv
    imputation_all_conditions.parquet
    imputation_predictions.parquet
    transfer_curves.parquet
    semantic_analysis.csv
  lbnl/
    ...
  swat/
    ...
```

`manifest.json` should contain:
- Git commit;
- environment lock hash;
- dataset checksums;
- commands;
- random seeds;
- model configuration;
- text encoder version;
- graph-building version;
- split IDs;
- date.

Add one command that regenerates every main-paper table and figure from these raw files.

---

## W13. The hand-built TEP graph and “primary fault columns” require provenance/audit
**Severity: MAJOR**

### Where

`src/sft/datasets/tep.py`

The file manually specifies:
- component topology;
- feature semantic types;
- fault localization / “primary” affected variables.

### Problem

These annotations directly determine:
- the KG representation;
- the held-out feature sets;
- the headroom gate;
- the semantic transfer axes.

They therefore cannot be treated as innocuous implementation details.

### Concrete fix

For every semantic field, add provenance:

| Variable | Measurement type | Component | Unit | Graph edges | Source |
|---|---|---|---|---|---|

For every fault:
| IDV | Held primary variables | Literature/source | Rationale |
|---|---|---|---|

If manual interpretation was required:
- have a second annotator independently map at least the important relations;
- report agreement;
- publish disagreements and resolution rules.

For TEP, “process graph derived from the standard TEP process diagram and variable definitions” is sufficient if documented honestly. It does not need to be disguised as a formal external ontology.

---

## W14. Few-shot sets should be nested and label-efficiency curves need a fixed protocol
**Severity: MAJOR**

### Problem

For a sample-efficiency claim, K=1,2,5,10,... should represent nested prefixes of the same target adaptation set within each outer split. Otherwise differences across K mix label count with different sampled examples.

### Concrete fix

For each outer split and target:

1. Generate one deterministic target adaptation permutation.
2. Define:
   - K=1 = first 1;
   - K=2 = first 2;
   - K=5 = first 5;
   - etc.
3. Keep the held test set fixed across K and conditions.
4. Repeat across multiple independently generated outer adaptation sets.
5. Save all adaptation indices.

This allows meaningful monotonicity checks and paired K-wise comparisons.

---

# 5. Line-by-line scientific audit of the current paper

The table below follows the paper in order and identifies the changes most likely to matter to a TMLR reviewer.

| Location / current wording | Severity | Reviewer issue | Exact revision |
|---|---|---|---|
| **Title: “Semantic Feature Transfer with Knowledge-Graph Embeddings”** | Major | Overstates the implemented representation and makes KG superiority sound central, although KG is inconsistent vs metadata | If results remain as now: **“Semantic Feature Transfer for Unseen Sensors: What Metadata and Process Graphs Actually Transfer”**. If strict topology gain emerges: **“When Does Graph Semantics Enable Transfer to Unseen Sensor Features?”** |
| Meta: “SFT-Bench: a cross-domain benchmark” | Major | Only one admitted domain currently produces positive transfer evidence | Use “evaluation protocol” until ≥2 admitted domains have full transfer results |
| Abstract: “Predictive models usually treat input features as anonymous dimensions” | Major | Too broad after TransTab, XTab, CARTE, UniTabE, TabSTAR, TabLLM | Restrict to **conventional fixed-schema models** and immediately acknowledge semantic/schema-flexible tabular learning |
| Abstract: “KG ... lets a model reuse predictive behaviour across related features” | Major | Current experiment does not isolate KG topology and does not establish strict unseen-feature transfer | Say “we test whether externally supplied feature semantics support transfer” |
| Abstract: “cross-domain benchmark” | Major | Forward-looking rows are not evaluated | Either add domains or reduce claim |
| Abstract: “pre-registered invariants” | Major | The public spec contains a continuation gate that the current paper does not visibly satisfy as originally framed | State exactly which hypotheses/protocols were pre-registered, which were changed after gate failure, and mark post-hoc experiments |
| Abstract: “fault detection is value-evident and cannot exhibit the effect” | Moderate | Too absolute and conflates this dataset/task/architecture with fault detection generally | “The tested fault-classification formulation lacked feature-level headroom” |
| Abstract: “predict a sensor it was never trained to predict” | Accurate but incomplete | This is the supported construct; do not silently equate it to unseen sensor | Keep, then distinguish from strict unseen-feature in same sentence |
| Abstract: “correct semantics ... five of five seeds” | Major | Weak inference; seeds are not independent targets | Replace with effect + CI over targets/runs; remove 5/5 rhetoric |
| Abstract: “pressure sensors ... 0.92” | Major | Current held pressure target count is one | Singular and report n; expand target set before using plural |
| Intro lead: “Models know numerical values ... not what features mean” | Moderate | Good motivation but too generic | Follow immediately with prior semantic-table work |
| Intro: “conventional model ties identity ... fixed input position” | Minor | Correct for conventional models | Add “many conventional” and contrast variable-schema models |
| Intro: definition of new-feature cold-start | Blocking | Current experiment only holds target role out | Define two constructs: unseen-target and strict unseen-feature |
| Intro: “graph places sensor near others” | Moderate | Current encoder uses tokens, not a geometry learned from graph | Say “graph supplies typed relational descriptors” unless using learned KGE |
| Contributions bullet: formalization + STE | Major | STE not empirically reported and zero-shot makes ratio undefined | Add full K curves or drop STE as a headline contribution |
| Contributions: benchmark admission gate | Strength | Useful | Keep, formalize admission criterion quantitatively |
| Contributions: “evidence semantic embeddings enable zero-shot sensor imputation” | Major | True for unseen target, not strict sensor | Qualify construct |
| Problem definition set representation | Strength | Clean | Keep |
| “unseen feature embedding ... without ever having seen its identity” | Blocking | False for current imputation implementation | Only claim after strict protocol |
| STE formula | Major | Undefined at K_sem=0; threshold crossing details absent | Separate zero-shot gain from finite-K STE |
| Figure 1 | Moderate | Helpful but does not depict query-conditioned imputer actually used for key result | Add Figure 2 showing query sensor, masked target, context sensors, and semantic query attention |
| Method: “embedding sources evaluated inside one fixed architecture” | Major | True within a task, not across classification vs imputation | Say “fixed within each task” |
| “learned per-feature identity ... cannot embed unseen feature” | Blocking | Not true under current C2 implementation because held sensor can be context and receive gradient | Fix strict protocol and define C2 only where target ID truly absent |
| “KG embedding ... relation-typed encoding ... no training on graph” | Major | This is graph-derived encoding, not learned KGE | Rename or add actual KG encoder |
| SFT-Bench domain table: TEP “OntoCAPE” | Blocking factual | Implementation is hand-built TEP process graph | Correct the table or implement audited OntoCAPE alignment |
| SWaT/WADI rows | Moderate | Planned, not results | Visually mark “planned/not evaluated” |
| Healthcare row | Moderate | Makes artifact look over-scoped | Move to Future Work unless executed |
| LBNL paragraph | Strength | Simulation status is stated | Keep and emphasize that it is simulation |
| Admission gate explanation | Strength | Good construct check | Define exact threshold, uncertainty, and whether it was set before seeing outcomes |
| Evaluation: “only embedding changes” | Major | C4 and C5 carry different semantic fields | Redesign as nested semantic inputs |
| “all comparisons one artifact” | Major | Public results artifact incomplete | Release raw data and manifest |
| Conditions table | Strength | Good idea | Include it in the Results as well; do not omit rows |
| Pre-registered invariants | Strength/major | Valuable but need outcome table | Add an invariant pass/fail table |
| Experiment 1 fault classification | Moderate | Useful as diagnostic, not main transfer evidence | Shorten main text; move details to appendix |
| Experiment 2 TEP imputation | Blocking | Target sensors still present as training context | Add strict feature-cold-start experiment |
| “each condition is run on same held targets, seeds, splits” | Major | Random row split not adequate generalization | Use grouped run splits |
| semantic-distance formula | Moderate | Pairwise transfer score not clearly defined empirically | Define exact target-level gain and representation used |
| Results 7.1 gate values | Major | Need per-run CI and non-overlapping windows | Recompute with grouped runs |
| Results 7.1 “value-only reaches .94” | Strength | Important falsification | Use it to explain why classification is not diagnostic |
| Results 7.2 table | Blocking | Omits 4 of 7 conditions | Report all |
| Results 7.2 metadata .153 vs shuffled -.006 | Major | Need hierarchical CI; current averaging hides per-target heterogeneity | Report target-wise forest plot |
| Results 7.2 pressure .919 | Major | One held target | Show n and CI; broaden targets |
| Results 7.2 actuator KG .017 vs metadata .628 | Strength | Very informative negative result | Promote, because it shows KG topology can actively fail |
| Results 7.2 temperature metadata negative, KG positive | Strength | Potentially the clearest topology-use case | Analyze why: component/topology relation, statistical recoverability, target count |
| “KG gives no consistent advantage over metadata” | Strength | This is a credible boundary-condition result | Make it central rather than awkward |
| Results 7.3 Spearman .06 | Moderate | Ten points and cross-representation analysis are too weak | Increase targets and model transfer gain with statistical predictability |
| Related work | Blocking | Far too short and misses direct 2022–2025 literature | Replace with 4–5 subsections, ~25–40 relevant references |
| Conclusion: “first principles result” style | Major | Must match strict evidence | Distinguish observed target-role transfer from genuine new-feature transfer |
| References: 6 total | Blocking | Not archival-level literature coverage | Expand substantially |

---

# 6. The revision that would make the paper strong

## 6.1 Reframe the scientific question

The current paper asks something too broad:

> Do knowledge-graph feature embeddings enable transfer to new features?

The stronger, more defensible question is:

> **Under what conditions does external feature semantics enable transfer to a sensor whose predictive role, and under a strict protocol whose values, were absent from source training? How much additional value comes from graph topology beyond simple typed metadata and text?**

This framing has three advantages:

1. A metadata win is still a publishable finding.
2. A KG win is interpreted as an incremental topology effect rather than semantic magic.
3. Negative datasets such as LBNL fault classification become evidence about identifiability, not failed experiments.

## 6.2 Revised contribution set

A strong final paper should claim exactly four contributions:

### C1. Formal problem definition

Define:
- **target-role cold start**;
- **strict feature cold start**;
- zero-shot and K-shot adaptation.

### C2. A construct-valid evaluation protocol

Include:
- headroom/admission test;
- grouped splits;
- matched architectures;
- nested semantic ablations;
- negative controls;
- fixed K-shot adaptation sets.

### C3. Multi-domain empirical evidence

Show at least:
- TEP strict transfer;
- LBNL/Brick strict imputation transfer or another Brick-compatible task;
- one real CPS dataset, preferably SWaT/WADI.

### C4. Boundary-condition analysis

Quantify whether transfer gain is predicted by:
- metadata match;
- graph distance/topology;
- text similarity;
- source-only statistical recoverability.

The strongest likely conclusion is not “semantic similarity predicts transfer.” It is:

> **Semantics identifies which learned behavior is eligible to transfer; statistical recoverability determines whether there is useful behavior to transfer. Graph topology adds value only where it resolves ambiguity left by type-level metadata.**

That is a more interesting paper.

---

# 7. Experimental redesign

## Experiment 0. Dataset/target admission

For each candidate target feature or component:

1. Train a strong oracle model with target feature available.
2. Train/evaluate the same model with target feature removed.
3. Estimate:
   \[
   H_j=M_{\rm present}-M_{\rm removed}
   \]
4. Require a pre-specified minimum headroom and CI that excludes a practically negligible interval.
5. Separately measure whether the target itself is statistically recoverable from source features.

Do not use a fixed 0.05 rule without justification. Define a practical equivalence region or derive the threshold from metric noise.

Output:
- one headroom plot for every candidate target;
- one target-recoverability plot.

## Experiment 1. Strict TEP unseen-feature imputation

### Source training

For a held target component or sensor:
- remove held feature values from every training input;
- remove held feature from target choices;
- do not train target-specific ID embedding;
- train on independent source trajectories.

### Zero-shot test

Supply:
- external semantic descriptor for the new sensor;
- context sensors;
- no target values from training/adaptation.

Evaluate:
- Pearson correlation;
- normalized RMSE;
- MAE on standardized scale;
- calibration where relevant.

Correlation alone can reward scale/offset errors, so do not use it as the sole metric.

### Few-shot

K = {1, 2, 5, 10, 20, 50, 100} target observations.

Use nested K sets.

## Experiment 2. Component-level cold start

Hold out an entire physical component:
- e.g. stripper sensor set;
- train on reactor/separator analogues;
- test/adapt on stripper.

This is more deployment-realistic than withholding one column while retaining the rest of the component.

Primary endpoint:
- mean transfer curve over sensors in held component.

## Experiment 3. LBNL/Brick imputation

Do **not** reuse the failed fault-detection task as the only LBNL result.

Use the Brick graph for a task where feature identity is intrinsic:
- sensor imputation;
- virtual sensing;
- newly commissioned sensor reconstruction.

Hold out:
- a sensor class;
- a whole equipment instance;
- or a zone/equipment subsystem.

This would provide the cleanest test of whether a real ontology (Brick) adds information over typed metadata.

## Experiment 4. SWaT or WADI real testbed

Use a real cyber-physical system to reduce simulation-only validity concerns.

Protocol:
- train on stages/components;
- hold out one stage or semantically matched sensor family;
- use SOSA/SSN/SAREF-compatible metadata if mapping is defensible;
- compare metadata vs metadata+topology.

This is the domain that can turn the study from a TEP case study into a generalizable paper.

## Experiment 5. Semantic representation ablation

Every domain should use the same conceptual hierarchy:

| ID | Representation | Purpose |
|---|---|---|
| A0 | none | no-semantics floor |
| A1 | random fixed | vector-capacity control |
| A2 | learned source feature ID | closed-schema reference |
| A3 | measurement type | minimal semantics |
| A4 | type + unit + component type | core metadata |
| A5 | text description embedding | strong unstructured semantic baseline |
| A6 | A4 + graph topology encoding | incremental topology test |
| A7 | A4 + learned inductive graph encoder | stronger graph model |
| A8 | A4 + type-preserving shuffled topology | topology-specific negative control |
| A9 | fully shuffled semantics | semantic-assignment negative control |

This table should become the conceptual center of the paper.

## Experiment 6. Mechanism analysis

For each target sensor j, compute before looking at target labels where possible:

- number of semantically matched source siblings;
- maximum/mean text similarity;
- graph distance to closest same-type source;
- relation overlap;
- source-only statistical predictability;
- source-only correlation/MI to matched sibling;
- target headroom.

Then model transfer gain:

\[
G_j = M_{A6,j}-M_{A1,j}
\]

and topology-specific gain:

\[
G^{topo}_j = M_{A6,j}-M_{A4,j}.
\]

This can yield a valuable practical decision rule:

> use graph-semantic transfer only when the target has both semantic support and sufficient statistical recoverability.

---

# 8. Baselines a TMLR reviewer will expect

## 8.1 Semantic/schema-transfer baselines

At least discuss and, where feasible, compare with:

- TransTab;
- XTab;
- CARTE;
- UniTabE;
- TabSTAR;
- a strong sentence-embedding semantic feature encoder.

It may not be possible to plug every system directly into multivariate sensor imputation. If so, state exactly which comparison is conceptual and which is empirical.

The minimum empirical competitor is a **text-semantic version of exactly the same architecture**.

## 8.2 Imputation baselines

Include ordinary imputation competence baselines so the task itself is credible:

- mean/median;
- KNN imputation;
- linear/ridge multivariate regression;
- MLP;
- SAITS;
- GRIN or another graph-aware multivariate imputer.

For strict unseen-feature transfer, conventional imputers may not directly support a never-seen target channel. That is useful: define two evaluation regimes.

1. **Known-channel missingness**: compare raw imputation quality to standard methods.
2. **Unseen-channel transfer**: compare semantic transfer methods under the new construct.

This prevents a reviewer from saying the paper created a custom task to avoid standard baselines.

## 8.3 Process-specific sanity baselines

For each target:
- best same-type sibling;
- best source-feature linear predictor;
- source-only ridge using all context;
- oracle target-trained model.

These establish:
- available statistical signal;
- lower bound;
- ceiling.

---

# 9. Statistical plan

## 9.1 Primary unit of replication

Not row. Not overlapping window. Not random seed.

Use:
- held system/component/feature;
- independent trajectory/run;
- domain.

Random initialization is a lower-level repeated measure.

## 9.2 Primary endpoint

Pre-specify one endpoint, for example:

> Mean topology-specific zero-shot gain \(A6-A4\) over admitted held components, aggregated hierarchically across domains.

If topology is not the thesis, use:

> Mean semantic gain \(A4-A1\).

## 9.3 Confidence intervals

Use paired hierarchical bootstrap.

Recommended hierarchy:
- domain;
- held component;
- held target sensor;
- run;
- seed.

Report:
- mean paired difference;
- median paired difference;
- 95% CI;
- probability of positive improvement / sign consistency as descriptive only.

## 9.4 Number of seeds

10 seeds is a reasonable minimum for model-initialization stability after the larger sources of variation are correctly grouped. More seeds do not compensate for having one pressure target or one process trajectory.

## 9.5 Multiple comparisons

For per-type tables:
- designate them secondary;
- report adjusted p-values or bootstrap intervals with Holm correction;
- include `n` beside every type.

## 9.6 Metrics

For imputation, report at least:
- Pearson r;
- standardized RMSE or NRMSE;
- MAE.

Why: correlation can be high even if predictions are badly biased in scale or offset.

---

# 10. Reproducibility plan

A strong TMLR submission should be one command away from regenerating the paper.

## Required repository additions

### A. Frozen experiment configuration

```yaml
paper_version: 1.0
commit: ...
datasets:
  tep:
    source: ...
    checksums: ...
splits:
  strict_feature_cold_start: ...
embeddings:
  text_model: ...
  graph_encoder: ...
seeds: [...]
```

### B. Split manifests

For every target:
- source runs;
- adaptation rows;
- test runs;
- excluded feature indices.

### C. Assertions

Add programmatic assertions:

```python
assert held_idx not in source_input_feature_indices
assert held_idx not in source_target_indices
assert held_idx not in trainable_id_indices
assert no_train_test_run_overlap
assert no_train_test_window_overlap
assert text_encoder_is_real_model_not_hash_fallback
```

### D. Raw outputs

Store:
- per-run;
- per-target;
- per-seed;
- per-K metrics;
- raw predictions.

### E. Figure scripts

One script per figure/table, reading only frozen result files.

### F. Environment

Use:
- locked Python dependencies;
- exact PyTorch version;
- hardware log;
- deterministic settings where possible.

---

# 11. Writing and structure plan

## Proposed title

### Best if the empirical conclusion remains mixed

**Semantic Feature Transfer for Unseen Sensors: A Controlled Study of Metadata and Process-Graph Semantics**

### Best if graph topology shows a clear conditional advantage

**When Does Graph Semantics Enable Transfer to Unseen Sensor Features?**

### Best if SFT-Bench becomes a genuine multi-domain benchmark

**SFT-Bench: Evaluating Semantic Transfer to Unseen Features Across Sensor Domains**

Do not keep “Knowledge-Graph Embeddings” in the title unless the final paper genuinely studies a proper graph representation and gives graph-specific evidence.

## Proposed paper structure

1. Introduction
2. Related Work
   - variable-schema and cross-table transfer
   - semantic tabular representations
   - multivariate time-series imputation
   - graph/ontology-informed sensor learning
3. Problem Definition
   - unseen-target vs strict unseen-feature
   - zero-shot and K-shot transfer
4. Evaluation Principles
   - admission/headroom
   - construct-matched ablations
   - grouped splits
5. Semantic Representations
   - metadata
   - text
   - topology
   - graph encoder
6. Datasets and Semantic Mapping
7. Experiments
   - TEP
   - Brick/LBNL
   - SWaT/WADI
8. Results
9. What Predicts Transferability?
10. Threats to Validity
11. Conclusion

## What to move out of the main text

Move to appendix:
- full hyperparameters;
- all per-fault gate tables;
- all non-primary K values if plots become crowded;
- ontology mapping tables;
- per-seed tables;
- implementation details.

Keep in main text:
- strict construct definition;
- all decisive controls;
- main transfer curves;
- metadata-vs-topology result;
- domain generalization;
- mechanism analysis.

---

# 12. Figures and tables that would materially improve acceptance odds

## Figure 1. Construct diagram

Show three cases side by side:

1. normal closed-schema prediction;
2. unseen-target cold start;
3. strict unseen-feature cold start.

This will prevent reviewer confusion and force the experiment to match the terminology.

## Figure 2. Semantic ablation ladder

A diagram:

```text
Type
  + unit/component
      + text
      + topology
          + learned graph encoder
```

with arrows showing the exact paired comparisons.

## Figure 3. Main zero/few-shot transfer curves

x-axis: K target observations, log scale plus K=0.

Lines:
- random;
- text;
- metadata;
- metadata+topology;
- topology-shuffled.

One panel per domain.

## Figure 4. Target-wise forest plot

For each held target:
- semantic gain vs random;
- topology gain vs metadata;
- 95% CI.

This will expose heterogeneity honestly and make the pressure/actuator/temperature story much clearer than a type-average table.

## Figure 5. Transferability mechanism

x-axis: source-only statistical recoverability.  
y-axis: semantic transfer gain.  
Point color/shape: whether topology disambiguates same-type sensors.

This may become the paper's most memorable figure.

## Table 1. Domain maturity

Clearly mark:
- evaluated;
- admitted;
- transfer experiment complete;
- ontology provenance;
- real vs simulated.

## Table 2. Primary quantitative comparison

Every condition, not a selected subset.

## Table 3. Ablation of topology

Core metadata vs +topology vs topology-shuffled, with paired CI.

---

# 13. Exact claim discipline for the revised paper

Use the following claim ladder.

## Claim that can already be supported with modest rewriting

> A query-conditioned model can use semantic feature descriptors to generalize to a sensor that was not used as an imputation target during training.

## Claim that requires strict feature removal

> External semantics enable zero-shot prediction for a sensor whose values and target role were absent from source training.

## Claim that requires nested topology ablation

> Process-graph topology provides transfer information beyond measurement type, unit, and component metadata.

## Claim that requires multiple domains

> Semantic feature transfer generalizes across domains.

## Claim that requires K curves

> Semantic representations reduce target-label requirements by X×.

## Claim that requires more targets

> Semantic similarity/statistical recoverability predicts transfer gain.

Do not move a claim one rung higher than the corresponding experiment.

---

# 14. Suggested revised abstract skeleton

Do not freeze numerical values until the redesigned experiments are complete. The abstract should follow this logic:

> Fixed-schema models typically bind numerical variables to dataset-specific columns, while recent schema-flexible models increasingly use textual column semantics. We study a narrower problem: whether external domain semantics can transfer predictive behavior to a sensor that was absent from source training. We distinguish target-role cold start from strict unseen-feature cold start and introduce a construct-matched protocol that holds the predictive architecture fixed while varying only the feature-semantic representation. The protocol first tests whether a candidate target has sufficient feature-level headroom, then compares value-only, random, text, typed-metadata, metadata-plus-topology, and topology-shuffled controls under grouped system-level splits. Across [N domains / N held components], semantic metadata improves zero- and few-shot transfer by [...], while graph topology provides additional benefit only when [condition]. Transfer gain is better predicted by source-only statistical recoverability than by raw semantic similarity. These results identify when external feature semantics can reduce the data required to commission new predictive variables and when ontology proximity alone is insufficient.

This framing remains publishable even if topology is useful only conditionally.

---

# 15. Prioritized improvement plan to reach acceptance

## Priority 1 — Fix the construct
**Impact: critical | Effort: medium**

- Implement strict unseen-feature removal.
- Separate it from unseen-target transfer.
- Add assertions proving no held feature values/IDs enter source training.

**Why first:** without this, the title problem is not the experiment actually run.

## Priority 2 — Fix the split
**Impact: critical | Effort: low-medium**

- Group by independent TEP run/trajectory.
- Eliminate overlapping-window train/test contamination.
- Recompute all reported numbers.

**Why second:** every downstream comparison depends on credible test separation.

## Priority 3 — Redesign semantic ablations
**Impact: critical | Effort: medium**

- Core metadata.
- Core metadata + topology.
- Type-preserving topology shuffle.
- Strong text baseline.
- Optional learned graph encoder.

**Why:** this creates a scientifically interpretable answer about what the graph contributes.

## Priority 4 — Report all seven/current conditions
**Impact: critical | Effort: low**

- Publish C0–C6 current results immediately for internal diagnosis.
- Do not select only C4–C6.
- Verify real sentence-transformer execution.

**Why:** this may reveal the correct final thesis before spending more compute.

## Priority 5 — Add full K-shot curves
**Impact: high | Effort: medium**

- K=0,1,2,5,10,20,50,100.
- Nested adaptation sets.
- Replace zero-denominator STE language.

**Why:** label efficiency is currently promised but not demonstrated.

## Priority 6 — Add second and preferably third admitted domain
**Impact: very high | Effort: high**

Preferred order:
1. LBNL/Brick imputation;
2. SWaT or WADI;
3. healthcare only if it can be done rigorously without delaying the paper.

**Why:** this converts a TEP study into a general ML result.

## Priority 7 — Strengthen statistical inference
**Impact: high | Effort: medium**

- hierarchical bootstrap;
- run/target as replication units;
- 10+ model seeds;
- n per type;
- multiple-comparison discipline.

## Priority 8 — Rewrite Related Work before rewriting prose elsewhere
**Impact: high | Effort: low-medium**

The related-work rewrite will determine the actual novelty sentence. Do it before polishing the Introduction.

## Priority 9 — Complete the artifact
**Impact: high | Effort: low-medium**

- raw result parquet;
- split manifests;
- predictions;
- exact config;
- environment;
- one-command figure regeneration.

## Priority 10 — Final narrative compression
**Impact: medium | Effort: low**

Once the evidence is fixed:
- remove future benchmark rows that remain unexecuted;
- shorten fault-detection details;
- promote strict transfer curves;
- promote topology-vs-metadata boundary condition;
- make mechanism analysis the final result section.

---

# 16. Acceptance checklist

I would not submit to TMLR until every **critical** item below is true.

## Critical

- [ ] Held test sensor values are absent from source training in the strict protocol.
- [ ] Held feature IDs cannot receive gradients in the strict protocol.
- [ ] Train/test separation is by run/system, not random correlated rows/windows.
- [ ] No overlapping temporal windows cross train/test.
- [ ] Text baseline is reported and its exact encoder is recorded.
- [ ] Value-only and random controls are reported.
- [ ] Metadata+topology is nested over the exact same metadata baseline.
- [ ] Topology-only shuffle preserves basic metadata.
- [ ] TEP is no longer falsely labeled as OntoCAPE unless an actual mapping is implemented.
- [ ] Full per-target/per-seed results are public.
- [ ] Related work covers TransTab, XTab, CARTE, UniTabE, TabLLM, TabSTAR, GRIN, SAITS/BRITS.
- [ ] Abstract no longer claims more than the strictest completed experiment.

## Strong-accept level

- [ ] ≥2 admitted domains show semantic transfer.
- [ ] At least one domain is real rather than simulation-only.
- [ ] Component-level cold start is included.
- [ ] Full K-shot curves demonstrate label savings.
- [ ] Primary effect has hierarchical 95% CI.
- [ ] ≥10 meaningful held targets, preferably substantially more.
- [ ] Target-wise heterogeneity is analyzed rather than hidden by type averages.
- [ ] Statistical recoverability vs semantic/topological similarity is evaluated mechanistically.
- [ ] A useful decision rule or boundary condition emerges.
- [ ] Raw results and split manifests reproduce every paper number.

---

# 17. What would change my TMLR recommendation to Accept

I would recommend **Accept** if a revision demonstrated the following package:

1. **Construct validity:** strict unseen-feature transfer is genuinely implemented.
2. **Leakage-free evaluation:** complete-run/system holdouts.
3. **Semantic causality:** core metadata beats random/no semantics and correctly attached topology is compared against a metadata-preserving topology shuffle.
4. **Strong competitor:** text semantics is included and honestly reported.
5. **Breadth:** the effect is reproduced in at least two admitted domains.
6. **Uncertainty:** target/run-level confidence intervals support the main effect.
7. **Claim discipline:** the paper openly reports when topology does not help.
8. **Literature:** novelty is positioned against modern semantic and cross-table tabular learning.
9. **Artifact:** every result is reproducible from the released files.

A result in which **metadata consistently helps, topology helps only for structurally ambiguous targets, and raw KG similarity is weak unless the target is statistically recoverable** would be entirely acceptable scientifically. The paper does not need a universal KG win. It needs a clean answer to *when semantics transfers and why*.

---

# 18. Estimated venue outlook

These are judgment bands for this specific manuscript, not venue-wide acceptance rates.

| Venue | As current draft | After critical fixes only | After full strong-accept plan |
|---|---:|---:|---:|
| **TMLR** | **Low (≈10–20%)** | **Moderate (≈40–55%)** | **Good (≈65–80%)** |
| **Knowledge-Based Systems** | **Low (≈10–15%)** | **Moderate (≈30–45%)** | **Moderate-good (≈45–65%)**, especially if topology has a clear gain |
| **Machine Learning** | **Very low-low** | **Low-moderate** | **Moderate**, if cross-domain generality is strong |
| **DMKD / TKDD** | **Low** | **Low-moderate** | **Moderate**, especially if SFT-Bench becomes a genuine benchmark |

The largest increase in acceptance probability will come from **construct validity and cross-domain replication**, not from making the neural architecture larger.

---

# 19. Final publication recommendation

**Primary target: Transactions on Machine Learning Research.**

The paper should be rebuilt around the question:

> **When can external semantics transfer a learned predictive role to a genuinely unseen sensor, and when does graph topology add information beyond ordinary metadata?**

The current draft contains the seed of that paper, particularly the admission-gate idea and the negative finding that graph structure is not automatically superior. But a TMLR reviewer reading the implementation can currently reject the strongest claims for concrete reasons: held sensors are not truly absent from source training, splits are not grouped by independent trajectories, topology is not isolated by the ablations, decisive controls are omitted from the paper, and the literature framing predates the modern semantic-tabular transfer literature.

Fix those issues before investing effort in prose polish. If the strict experiment survives, add at least one more admitted domain and full few-shot curves. At that point, the paper no longer depends on “KG beats everything”; it becomes a rigorous and useful study of **semantic transfer, its identifiable causal ingredients, and its limits**, which is a much stronger TMLR submission.
