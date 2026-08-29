"""Transfer-split definitions for SDAHU (research plan section 12; docs/phase1_spec.md section 3).

A split is a JSON object naming which fault families/components are held out of predictive training
and the K schedule for adding target-labeled episodes back. Splits select on the FaultCase.family /
FaultCase.component fields parsed in datasets/lbnl_sdahu.py, so no file re-reading is needed to build
them. The actual window sampling given a split lives in the runner.

Three splits, matching the spec:
  standard              conventional random split; sanity that semantics don't hurt full-data.
  feature_cold_start    hold out a measurement/location family entirely (e.g. oa_bias), then add
                        K target episodes. Tests transfer of a fault CONCEPT across location.
  component_cold_start  hold out an entire component's faults (e.g. Outdoor_Air_Damper), the
                        headline realistic unit.
"""
from __future__ import annotations

import json
from pathlib import Path

K_SCHEDULE = [0, 1, 2, 5, 10, 20, 50, 100]   # "full" appended by the runner

# The two transfer axes justified in docs/dataset_notes_sdahu.md.
SPLITS = {
    "standard": {
        "kind": "random",
        "test_frac": 0.3,
        "note": "Experiment 1 sanity; all families in train and test.",
    },
    "feature_cold_start": {
        "kind": "holdout_family",
        "holdout_family": "oa_bias",
        "sibling_in_train": "coi_bias",
        "K": K_SCHEDULE,
        "note": "Transfer temperature-sensor-bias concept: cooling-coil sensor -> outdoor-air sensor.",
    },
    "component_cold_start": {
        "kind": "holdout_component",
        "holdout_component": "Outdoor_Air_Damper",
        "sibling_in_train": "Cooling_Coil",
        "K": K_SCHEDULE,
        "note": "Transfer actuator-stuck behaviour Cooling_Coil->OA_Damper (headline split).",
    },
}

ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = ROOT / "data/processed/SDAHU/splits"


def write_splits() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, spec in SPLITS.items():
        (OUT_DIR / f"{name}.json").write_text(json.dumps(spec, indent=2))
    print(f"wrote {len(SPLITS)} split definitions to {OUT_DIR}")


if __name__ == "__main__":
    write_splits()
