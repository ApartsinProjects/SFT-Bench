# GraphTransfer — Semantic Feature Transfer with KG Embeddings

Represent each input feature as a pair `(x_j, e_j)` — its observed value/window `x_j` plus a semantic
embedding `e_j` derived from a domain knowledge graph — and share one encoder `F_θ(x_j, e_j)` across
all features, so predictive knowledge transfers to unseen or data-scarce features by *meaning* rather
than by column position.

- Research plan: [docs/research_plan.md](docs/research_plan.md)
- Critique / analysis: [docs/analysis.md](docs/analysis.md)
- **Phase-1 experiment spec (start here):** [docs/phase1_spec.md](docs/phase1_spec.md)
- Locked dataset schema: [docs/dataset_notes_sdahu.md](docs/dataset_notes_sdahu.md)

## Status

Phase 1 (LBNL/Brick SDAHU) scaffold. The decisive first experiment compares 7 feature-embedding
conditions (value-only, random, learned-ID, text, metadata, KG, shuffled-KG) on one frozen
architecture, one pass, one artifact, gated by pre-registered invariants.

## Layout

```
docs/                 plan, analysis, spec, dataset notes
data/raw/SDAHU/       LBNL FDD SDAHU: PDF doc + Brick .ttl (CSV corpus fetched on demand, 579 MB)
src/sft/              the package
  schema.py           unified feature/observation schema
  graph/brick_parse.py    parse Brick .ttl -> feature metadata + topology  [working]
  build_features.py   emit features.csv from the Brick graph               [working]
  datasets/lbnl_sdahu.py  fault-file inventory, label parsing, CSV loader  [working]
  embeddings/         random / text / metadata / kg feature embeddings     [stubs]
  model/              shared encoder, temporal encoder, aggregator, head   [stubs]
  splits/             standard / feature-cold-start / component-cold-start [stub]
  experiment/         7 conditions, single-pass runner, sanity gate        [stubs]
configs/phase1.json   frozen Phase-1 configuration
```

## Setup

Python 3.14 (`/c/Python314/python`) or 3.11. Deps: `pandas numpy torch pyarrow rdflib` (all present;
`rdflib` installed for Brick parsing). Configs are JSON to avoid a yaml dependency.

## Get the data

The 579 MB CSV corpus is not committed. Fetch it with:

```bash
python -m sft.datasets.lbnl_sdahu --download
```

The Brick `.ttl` and the documentation PDF are already in `data/raw/SDAHU/`.

Source: LBNL FDD Data Sets, DOI 10.25984/1881324, CC-BY 4.0, https://fdddata.lbl.gov/data/
(simulation data; EnergyPlus–Modelica co-simulation).
