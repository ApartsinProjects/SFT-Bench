"""Unified feature / observation schema shared across all SFT datasets.

The whole point of Semantic Feature Transfer is that datasets with different feature sets and
different feature counts use the *same* representation: each feature carries its own semantic
metadata, so a shared encoder can process it regardless of which dataset it came from. Every
dataset converter must emit `FeatureMeta` records and observation windows in the form below.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class FeatureMeta:
    """Semantic description of a single feature (one sensor / point / variable).

    Fields mirror the `features.csv` schema in the research plan (section 7). `component_path` and
    the relation lists come from the knowledge graph and are what the KG embedding consumes;
    `name`, `unit`, `description` are what the text/metadata baselines consume. Keeping both on one
    record is what makes the KG-vs-text comparison construct-matched.
    """

    feature_id: str                      # stable id, unique within a dataset (e.g. "SDAHU:SA_TEMP")
    name: str                            # human name, e.g. "AHU Supply Air Temperature"
    dataset: str                         # e.g. "LBNL_SDAHU"
    unit: str = ""                       # e.g. "degF", "CFM", "W", "frac", "inH2O", ""
    measurement_type: str = ""           # KG class of the quantity, e.g. "Temperature"
    component: str = ""                   # owning component instance, e.g. "Supply_Air_Fan"
    component_type: str = ""             # KG class of the component, e.g. "Fan"
    ontology_id: str = ""                # node iri/curie in the source graph, e.g. "bldg:SA_TEMP"
    ontology_source: str = ""            # e.g. "Brick-1.2"
    description: str = ""                # free text for the text embedding baseline
    # graph neighbourhood (for the KG encoder); each entry is (relation, target_node_id)
    relations: tuple = field(default_factory=tuple)

    def text_blob(self) -> str:
        """Canonical string fed to the text-embedding baseline (condition C3)."""
        parts = [self.name]
        if self.component:
            parts.append(f"of {self.component.replace('_', ' ')}")
        if self.unit:
            parts.append(f"in {self.unit}")
        if self.description and self.description.lower() not in self.name.lower():
            parts.append(f"- {self.description}")
        return " ".join(parts)

    def metadata_tuple(self) -> tuple[str, str, str]:
        """Fields fed to the metadata baseline (condition C4): type + component + unit, no topology."""
        return (self.measurement_type, self.component_type, self.unit)

    def as_row(self) -> dict:
        d = asdict(self)
        d["relations"] = ";".join(f"{r}->{t}" for r, t in self.relations)
        return d


@dataclass
class Observation:
    """One prediction sample: a set of (value-window, feature) pairs plus a label.

    `values[j]` is the length-W window for feature `feature_ids[j]`; the aggregator is
    permutation-invariant over j, so datasets with different feature counts share the architecture.
    """

    values: np.ndarray                   # shape (n_features, window) float32
    feature_ids: Sequence[str]           # length n_features, indexes into the FeatureMeta table
    label: int                           # task target (binary fault flag in Phase 1)
    meta: dict = field(default_factory=dict)   # provenance: source file, window start, fault info

    def __post_init__(self) -> None:
        if self.values.shape[0] != len(self.feature_ids):
            raise ValueError(
                f"values has {self.values.shape[0]} rows but {len(self.feature_ids)} feature_ids"
            )
