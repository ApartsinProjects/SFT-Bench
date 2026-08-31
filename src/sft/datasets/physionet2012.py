"""PhysioNet/CinC Challenge 2012 loader: per-patient variable means + variable semantics.

Open-access ICU dataset (4000 records in set-a). Each record is a .txt of Time,Parameter,Value rows.
We aggregate each time-series variable to its per-patient mean (ignoring the -1 missing sentinel),
giving a (patients x variables) matrix; this is the cross-sectional analog of the TEP sensor matrix.
Variables carry physiological GROUPS so that clinically coupled variables are same-group siblings:
the blood-pressure triad (Sys/Dias/MAP, invasive and non-invasive) and the acid-base trio
(pH/PaCO2/HCO3) are near-deterministically recoverable; renal (BUN/Creatinine) is strongly coupled;
Glucose/Lactate/Temp are near-independent negative controls. This is the same recoverable-vs-not
structure as TEP, for a cross-domain replication of the signed-recoverability law.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
REC_DIR = ROOT / "data/raw/physionet2012/set-a"

# (variable, unit, physiological group). Group = the "measurement type" for sibling recoverability.
VARIABLES: list[tuple] = [
    ("HR", "bpm", "HeartRate"),
    ("Temp", "degC", "Temperature"),
    ("RespRate", "bpm", "Respiration"),
    ("SysABP", "mmHg", "BloodPressure"), ("DiasABP", "mmHg", "BloodPressure"), ("MAP", "mmHg", "BloodPressure"),
    ("NISysABP", "mmHg", "BloodPressure"), ("NIDiasABP", "mmHg", "BloodPressure"), ("NIMAP", "mmHg", "BloodPressure"),
    ("SaO2", "%", "Oximetry"), ("GCS", "score", "Neuro"), ("Urine", "mL", "Renal"),
    ("pH", "pH", "AcidBase"), ("PaCO2", "mmHg", "AcidBase"), ("HCO3", "mmol/L", "AcidBase"),
    ("PaO2", "mmHg", "BloodGas"), ("FiO2", "frac", "BloodGas"),
    ("Na", "mEq/L", "Electrolyte"), ("K", "mEq/L", "Electrolyte"), ("Mg", "mmol/L", "Electrolyte"),
    ("Glucose", "mg/dL", "Metabolite"), ("Lactate", "mmol/L", "Metabolite"),
    ("BUN", "mg/dL", "RenalChem"), ("Creatinine", "mg/dL", "RenalChem"),
    ("Albumin", "g/dL", "Protein"), ("Bilirubin", "mg/dL", "Liver"),
    ("ALP", "IU/L", "Liver"), ("ALT", "IU/L", "Liver"), ("AST", "IU/L", "Liver"),
    ("HCT", "%", "Hematology"), ("WBC", "cells/nL", "Hematology"), ("Platelets", "cells/nL", "Hematology"),
    ("Weight", "kg", "Anthropometric"),
]
NAMES = [v[0] for v in VARIABLES]
DESC = {
    "HR": "heart rate", "Temp": "body temperature", "RespRate": "respiration rate",
    "SysABP": "invasive systolic arterial blood pressure", "DiasABP": "invasive diastolic arterial blood pressure",
    "MAP": "invasive mean arterial blood pressure", "NISysABP": "non-invasive systolic blood pressure",
    "NIDiasABP": "non-invasive diastolic blood pressure", "NIMAP": "non-invasive mean arterial pressure",
    "SaO2": "arterial oxygen saturation", "GCS": "Glasgow coma score", "Urine": "urine output",
    "pH": "arterial pH", "PaCO2": "arterial partial pressure of carbon dioxide", "HCO3": "serum bicarbonate",
    "PaO2": "arterial partial pressure of oxygen", "FiO2": "fraction of inspired oxygen",
    "Na": "serum sodium", "K": "serum potassium", "Mg": "serum magnesium",
    "Glucose": "serum glucose", "Lactate": "serum lactate", "BUN": "blood urea nitrogen",
    "Creatinine": "serum creatinine", "Albumin": "serum albumin", "Bilirubin": "serum bilirubin",
    "ALP": "alkaline phosphatase", "ALT": "alanine transaminase", "AST": "aspartate transaminase",
    "HCT": "hematocrit", "WBC": "white blood cell count", "Platelets": "platelet count",
    "Weight": "body weight",
}


def build_physionet_features() -> pd.DataFrame:
    rows = []
    for name, unit, group in VARIABLES:
        rows.append(dict(
            feature_id=f"PN12:{name}", name=f"{name} {DESC[name]}", dataset="PhysioNet2012",
            unit=unit, measurement_type=group, component=group, component_type=group,
            ontology_id=f"pn12:{name}", ontology_source="PhysioNet2012",
            description=f"{DESC[name]} ({unit})", relations=f"isPartOf->{group}",
        ))
    return pd.DataFrame(rows)


def load_matrix(cap: int | None = None) -> np.ndarray:
    """Return (n_patients, n_variables) matrix of per-patient variable means; NaN where unmeasured."""
    files = sorted(REC_DIR.glob("*.txt"))
    if cap:
        files = files[:cap]
    idx = {n: i for i, n in enumerate(NAMES)}
    out = np.full((len(files), len(NAMES)), np.nan, dtype=np.float32)
    for r, f in enumerate(files):
        sums = np.zeros(len(NAMES)); cnts = np.zeros(len(NAMES))
        for line in f.read_text().splitlines()[1:]:
            parts = line.split(",")
            if len(parts) != 3:
                continue
            _, param, val = parts
            if param in idx and val not in ("", "-1", "-1.0"):
                try:
                    v = float(val)
                except ValueError:
                    continue
                if v >= 0:
                    sums[idx[param]] += v; cnts[idx[param]] += 1
        m = cnts > 0
        out[r, m] = (sums[m] / cnts[m]).astype(np.float32)
    return out


if __name__ == "__main__":
    X = load_matrix(cap=500)
    print(f"matrix {X.shape}; measured fraction per var:")
    frac = 1 - np.isnan(X).mean(0)
    for n, fr in sorted(zip(NAMES, frac), key=lambda x: -x[1]):
        print(f"  {n:11s} {fr:.2f}")
