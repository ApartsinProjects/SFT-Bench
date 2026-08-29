"""Tennessee Eastman Process dataset: 52-variable semantics, hand-built process graph, and loader.

Source: classic Downs & Vogel / Braatz simulation set (d00..d21 train, d00_te..d21_te test),
52 columns = [XMEAS(1..41), XMV(1..11)], whitespace .dat. Sampling 3 min. Test files are 960 rows
with the fault injected at sample 160; d00* are fault-free. d00.dat ships TRANSPOSED (52x500) and is
corrected on load.

No formal TEP ontology exists, so the knowledge graph is built here from the Downs & Vogel variable
table and process topology (feeds -> reactor -> condenser -> separator -> {compressor recycle, purge,
stripper} -> product, with reactor and condenser cooling loops). Each variable carries a measurement
type, a component, a component type, and graph relations, exactly the fields the embedders consume,
so the text / metadata / KG conditions run unchanged. Reactor / separator / stripper each carry
analogous pressure, temperature, level, and flow, which is the canonical semantic-transfer axis.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
TEP_DIR = ROOT / "data/raw/TEP"

# (var, description, unit, component, component_type, measurement_type)
VARIABLES: list[tuple] = [
    ("XMEAS1", "A feed (stream 1)", "kscmh", "Feed_A", "Feed", "Flow"),
    ("XMEAS2", "D feed (stream 2)", "kg/hr", "Feed_D", "Feed", "Flow"),
    ("XMEAS3", "E feed (stream 3)", "kg/hr", "Feed_E", "Feed", "Flow"),
    ("XMEAS4", "A and C feed (stream 4)", "kscmh", "Feed_AC", "Feed", "Flow"),
    ("XMEAS5", "recycle flow (stream 8)", "kscmh", "Compressor", "Compressor", "Flow"),
    ("XMEAS6", "reactor feed rate (stream 6)", "kscmh", "Reactor", "Reactor", "Flow"),
    ("XMEAS7", "reactor pressure", "kPa", "Reactor", "Reactor", "Pressure"),
    ("XMEAS8", "reactor level", "%", "Reactor", "Reactor", "Level"),
    ("XMEAS9", "reactor temperature", "degC", "Reactor", "Reactor", "Temperature"),
    ("XMEAS10", "purge rate (stream 9)", "kscmh", "Purge", "Purge", "Flow"),
    ("XMEAS11", "product separator temperature", "degC", "Separator", "Separator", "Temperature"),
    ("XMEAS12", "product separator level", "%", "Separator", "Separator", "Level"),
    ("XMEAS13", "product separator pressure", "kPa", "Separator", "Separator", "Pressure"),
    ("XMEAS14", "product separator underflow (stream 10)", "m3/hr", "Separator", "Separator", "Flow"),
    ("XMEAS15", "stripper level", "%", "Stripper", "Stripper", "Level"),
    ("XMEAS16", "stripper pressure", "kPa", "Stripper", "Stripper", "Pressure"),
    ("XMEAS17", "stripper underflow (stream 11)", "m3/hr", "Stripper", "Stripper", "Flow"),
    ("XMEAS18", "stripper temperature", "degC", "Stripper", "Stripper", "Temperature"),
    ("XMEAS19", "stripper steam flow", "kg/hr", "Stripper", "Stripper", "Flow"),
    ("XMEAS20", "compressor work", "kW", "Compressor", "Compressor", "Power"),
    ("XMEAS21", "reactor cooling water outlet temperature", "degC", "Reactor_Cooling", "CoolingLoop", "Temperature"),
    ("XMEAS22", "separator cooling water outlet temperature", "degC", "Condenser_Cooling", "CoolingLoop", "Temperature"),
    *[(f"XMEAS{22+i}", f"reactor feed component {c} mole fraction", "mol%", "Reactor", "Reactor", "Composition")
      for i, c in enumerate("ABCDEF", start=1)],
    *[(f"XMEAS{28+i}", f"purge gas component {c} mole fraction", "mol%", "Purge", "Purge", "Composition")
      for i, c in enumerate("ABCDEFGH", start=1)],
    *[(f"XMEAS{36+i}", f"product component {c} mole fraction", "mol%", "Stripper", "Stripper", "Composition")
      for i, c in enumerate("DEFGH", start=1)],
    ("XMV1", "D feed flow valve", "%", "Feed_D", "Feed", "Actuator"),
    ("XMV2", "E feed flow valve", "%", "Feed_E", "Feed", "Actuator"),
    ("XMV3", "A feed flow valve", "%", "Feed_A", "Feed", "Actuator"),
    ("XMV4", "A and C feed flow valve", "%", "Feed_AC", "Feed", "Actuator"),
    ("XMV5", "compressor recycle valve", "%", "Compressor", "Compressor", "Actuator"),
    ("XMV6", "purge valve", "%", "Purge", "Purge", "Actuator"),
    ("XMV7", "separator pot liquid flow valve", "%", "Separator", "Separator", "Actuator"),
    ("XMV8", "stripper liquid product flow valve", "%", "Stripper", "Stripper", "Actuator"),
    ("XMV9", "stripper steam valve", "%", "Stripper", "Stripper", "Actuator"),
    ("XMV10", "reactor cooling water flow valve", "%", "Reactor_Cooling", "CoolingLoop", "Actuator"),
    ("XMV11", "condenser cooling water flow valve", "%", "Condenser_Cooling", "CoolingLoop", "Actuator"),
]
COLUMNS = [v[0] for v in VARIABLES]           # 52, in file column order
assert len(COLUMNS) == 52, len(COLUMNS)

# component -> structural edges (relation, target). Upstream/downstream process flow + cooling.
COMPONENT_EDGES: dict[str, list[tuple]] = {
    "Feed_A": [("feeds", "Reactor")], "Feed_D": [("feeds", "Reactor")],
    "Feed_E": [("feeds", "Reactor")], "Feed_AC": [("feeds", "Reactor")],
    "Reactor": [("downstreamOf", "Feed_A"), ("feeds", "Condenser"), ("cooledBy", "Reactor_Cooling")],
    "Condenser": [("downstreamOf", "Reactor"), ("feeds", "Separator"), ("cooledBy", "Condenser_Cooling")],
    "Separator": [("downstreamOf", "Condenser"), ("feeds", "Stripper"),
                  ("feeds", "Compressor"), ("feeds", "Purge")],
    "Compressor": [("downstreamOf", "Separator"), ("feeds", "Reactor")],
    "Purge": [("downstreamOf", "Separator")],
    "Stripper": [("downstreamOf", "Separator"), ("feeds", "Product")],
    "Reactor_Cooling": [("cools", "Reactor")],
    "Condenser_Cooling": [("cools", "Condenser")],
}

# Fault metadata. localized -> primary affected variables (Downs & Vogel + FDD literature).
FAULTS: dict[int, dict] = {
    0:  dict(name="fault_free", component="", localized=False, primary=[]),
    4:  dict(name="reactor_cooling_step", component="Reactor_Cooling", localized=True,
             primary=["XMEAS9", "XMEAS21", "XMV10"]),
    11: dict(name="reactor_cooling_random", component="Reactor_Cooling", localized=True,
             primary=["XMEAS9", "XMEAS21", "XMV10"]),
    14: dict(name="reactor_cooling_valve", component="Reactor_Cooling", localized=True,
             primary=["XMV10", "XMEAS21", "XMEAS9"]),
    5:  dict(name="condenser_cooling_step", component="Condenser_Cooling", localized=True,
             primary=["XMEAS22", "XMEAS11", "XMV11"]),
    12: dict(name="condenser_cooling_random", component="Condenser_Cooling", localized=True,
             primary=["XMEAS22", "XMEAS11", "XMV11"]),
    6:  dict(name="feedA_loss", component="Feed_A", localized=True, primary=["XMEAS1", "XMV3"]),
    1:  dict(name="feed_ratio_AC", component="Feed_AC", localized=False, primary=["XMEAS1", "XMEAS4"]),
    13: dict(name="reaction_kinetics", component="Reactor", localized=False, primary=[]),
}


def build_tep_features() -> pd.DataFrame:
    rows = []
    for var, desc, unit, comp, comp_type, meas in VARIABLES:
        rels = [("isPartOf", comp)]
        for r, t in COMPONENT_EDGES.get(comp, []):
            rels.append((f"component.{r}", t))
        rows.append(dict(
            feature_id=f"TEP:{var}", name=f"{var} {desc}", dataset="TEP", unit=unit,
            measurement_type=meas, component=comp, component_type=comp_type,
            ontology_id=f"tep:{var}", ontology_source="TEP-DownsVogel",
            description=f"{desc} ({meas.lower()} on {comp.replace('_',' ')})",
            relations=";".join(f"{r}->{t}" for r, t in rels),
        ))
    return pd.DataFrame(rows)


def load_test_run(idv: int) -> np.ndarray:
    """Return the (960, 52) test run for fault idv (rows 0..159 normal, 160.. faulty)."""
    a = np.loadtxt(TEP_DIR / f"d{idv:02d}_te.dat", dtype=np.float32)
    if a.shape[1] != 52:               # guard against a transposed mirror
        a = a.T
    return a


def load_train_run(idv: int) -> np.ndarray:
    a = np.loadtxt(TEP_DIR / f"d{idv:02d}.dat", dtype=np.float32)
    if a.shape != (480, 52) and a.shape[0] == 52:   # d00.dat ships transposed (52x500)
        a = a.T
    return a


FAULT_ONSET = 160   # sample index at which the fault begins in *_te.dat files


if __name__ == "__main__":
    f = build_tep_features()
    print(f"{len(f)} TEP features")
    print("components:", sorted(f["component"].unique()))
    print("measurement types:", sorted(f["measurement_type"].unique()))
    print("\nlocalized faults:", [ (k, v["name"], v["primary"]) for k, v in FAULTS.items() if v["localized"]])
    te = load_test_run(4); tr0 = load_train_run(0)
    print(f"\nd04_te shape {te.shape}, d00 train shape {tr0.shape}")
