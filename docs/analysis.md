# Critical Analysis — Semantic Feature Transfer with KG Embeddings

Source plan: [research_plan.md](research_plan.md). This document records the critique; the actionable
first experiment lives in [phase1_spec.md](phase1_spec.md).

## Summary judgment

The core idea — represent each feature as `(x_j, e_j)` and share one encoder `F_θ` across all
features so knowledge is keyed on *meaning*, not column position — is novel enough to publish and
well-motivated. The label-efficiency framing (semantic transfer efficiency, STE) is the correct way
to state the contribution. The program is over-scoped for a first paper; the science is decided by two
comparisons that the current plan files under "ablations."

## The two make-or-break comparisons (promote to primary results)

1. **KG embedding vs. strong text embedding of the same metadata.** A modern text encoder fed
   `"supply air temperature sensor of AHU, unit °C"` already encodes measurement type, component, and
   unit. If KG does not beat text, the KG contribution collapses to "text embeddings help," which is
   not novel. Expected outcome: text matches KG on measurement-type transfer; KG wins only where
   **graph topology** (`feeds`, upstream/downstream, multi-hop component structure) carries signal
   text cannot easily linearize. Therefore the defensible headline is *"graph topology helps where
   topology is predictive,"* not *"KGs beat everything."* Reframe now, not as a fallback.

2. **Correct-KG vs. Shuffled-KG vs. Random-embedding on an identical pooled architecture.** The
   confound the plan under-controls: the "advantage" may come from the shared/pooled architecture
   (parameter sharing across features) regardless of what `e_j` contains. Pre-registered invariant:
   `Correct-KG > Shuffled-KG > Random-emb`, same dims, same seeds, same splits, co-computed in one
   pass. If `Correct ≈ Shuffled`, the benefit is architectural, not semantic — and the paper has no
   claim.

## Conceptual soft spots

- **"Semantic similarity predicts transferability" may be false for a physical reason.** Two sensors
  can measure the same property yet play opposite dynamic roles (reactor pressure vs. downstream
  separator pressure). Cosine similarity of `e_i, e_j` will not capture this unless the graph encodes
  causal direction/regime. Prediction: the raw similarity↔transfer correlation is weak by measurement
  type alone and strengthens only once topology/direction is in the embedding — which is *also* the
  evidence that graph structure matters. Design the two analyses together.

- **Zero-shot single-feature cold-start is artificial.** Withholding one column is a clean
  mechanistic probe but not a realistic deployment unit. Component cold-start (hold out a whole AHU /
  process stage / plant) is what reviewers believe. Lead with component cold-start; keep
  feature-column withholding as the microscope.

## Execution risk is data access, not modeling

| Domain | Data access | Ontology mapping | Verdict |
|---|---|---|---|
| LBNL / Brick | Public, CC-BY 4.0, DOI 10.25984/1881324, `fdddata.lbl.gov/data`; Brick `.ttl` **confirmed shipped** per equipment type | Low (graph given) | Correct MVP |
| SWaT / WADI | iTrust request form (`sutd.edu.sg/itrust/request-for-datasets`), **≤3 working days**, institutional email required | Medium (build KG from P&IDs) | Submit request now |
| TEP | Public simulator/datasets | Medium (OntoCAPE mostly manual) | Good controlled test |
| MIMIC-IV / eICU | Credentialed PhysioNet + CITI training | High (LOINC/SNOMED audit) | Defer to last |

Kick off the SWaT/WADI request and PhysioNet credentialing in parallel with Phase 1, or they become
the critical path.

### Confirmed data facts (web-verified) and a framing correction

- LBNL FDD ships **CSV time series at 1-minute sampling**, one file per (fault type × severity), plus
  a fault-free case, plus a Brick `.ttl` per equipment subset. Subsets present: SDAHU, DDAHU, FCU,
  FPU, RTU, Chiller Plant, Boiler Plant.
- **Framing correction:** the public release is **simulation-only** (EnergyPlus–Modelica
  co-simulation; every folder prefixed `Simulated_`). The §4 "new sensor commissioning on a real
  plant" motivation is narrative, not what Phase 1 tests. Scope the Phase-1 claim to simulated HVAC
  and lean on SWaT (real testbed) for the real-data leg. Do not let the abstract imply the HVAC
  evidence is from field sensors.
- Confirm at implementation: (a) which Brick version the shipped `.ttl` imports (open file, check
  ontology version — current stable Brick is v1.4.4); (b) exact CSV column schema + severity
  enumeration, documented in each subset's PDF.

## Scope recommendation

The §28 minimal-viable cut (HVAC + SWaT + TEP) is right, but even that is large for a first pass.
Sharpen the first milestone to a single decisive experiment on LBNL/Brick before touching a second
domain. Continuation is gated on one number: does KG beat text at low K with non-overlapping CIs?
