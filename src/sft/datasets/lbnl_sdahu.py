"""LBNL FDD SDAHU dataset: fault-file inventory, per-file labels, download, and a windowing loader.

Label convention (Phase 1, binary fault-vs-normal): the fault-free file is label 0; every faulted
file is label 1. Fault family and intensity are parsed from the filename and carried on each window's
`meta` so component/measurement-type cold-start splits can select on them without re-reading files.

The 579 MB CSV corpus is fetched on demand (`--download`); the Brick .ttl and PDF are committed.
"""
from __future__ import annotations

import argparse
import io
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
RAW = ROOT / "data/raw/SDAHU"
CSV_DIR = RAW / "csv"
CSV_ZIP_URL = ("https://fdddata.lbl.gov/data/Simulated_LBNL_FDD_Data_Sets_SDAHU/"
               "LBNL_FDD_Data_Sets_SDAHU.zip")

# 30 expected data columns (order per documentation Table 2); validated at load time.
EXPECTED_POINTS = [
    "SA_TEMP", "SA_TEMPSPT", "OA_TEMP", "MA_TEMP", "RA_TEMP", "SF_SPD_DM", "RF_SPD_DM",
    "OA_CFM", "RA_CFM", "SA_CFM", "SF_CS", "SF_SPD", "RF_CS", "RF_SPD", "SF_WAT", "RF_WAT",
    "OA_DMPR_DM", "OA_DMPR", "RA_DMPR_DM", "RA_DMPR", "CHWC_VLV_DM", "CHWC_VLV",
    "SA_SP", "SA_SPSPT", "SYS_CTL",
    "ZONE_TEMP_1", "ZONE_TEMP_2", "ZONE_TEMP_3", "ZONE_TEMP_4", "ZONE_TEMP_5",
]


@dataclass(frozen=True)
class FaultCase:
    filename: str
    family: str        # "fault_free" | "sa_bias" | "oa_bias" | "coi_leakage" | "coi_stuck" | "damper_stuck"
    component: str     # affected component, "" for fault-free
    intensity: float   # signed magnitude (°C bias) or percent; 0.0 for fault-free
    label: int         # 0 normal, 1 faulted


def _parse_case(filename: str) -> FaultCase:
    stem = filename[:-4] if filename.endswith(".csv") else filename
    if stem == "AHU_annual":
        return FaultCase(filename, "fault_free", "", 0.0, 0)
    # Temperature-sensor bias faults. NOTE: the actual release ships coi_bias (cooling-coil temp
    # sensor) in place of the PDF's sa_bias. Bias faults are logged as the TRUE value and manifest
    # as system-wide control-loop shifts, not a clean offset on one logged column (verified 2026-08).
    m = re.match(r"(coi|oa)_bias_(-?\d+)_annual", stem)
    if m:
        fam = f"{m.group(1)}_bias"
        comp = "Cooling_Coil" if m.group(1) == "coi" else "Outdoor_Air_Damper"
        return FaultCase(filename, fam, comp, float(m.group(2)), 1)
    m = re.match(r"coi_(leakage|stuck)_(\d+)_annual", stem)
    if m:
        return FaultCase(filename, f"coi_{m.group(1)}", "Cooling_Coil", float(m.group(2)), 1)
    m = re.match(r"damper_stuck_(\d+)_annual", stem)  # matches ..._annual and ..._annual_short
    if m:
        return FaultCase(filename, "damper_stuck", "Outdoor_Air_Damper", float(m.group(1)), 1)
    raise ValueError(f"unrecognised SDAHU filename: {filename}")


# Full inventory — the ACTUAL shipped file set (differs from the 2022 PDF: coi_bias, not sa_bias).
INVENTORY: list[FaultCase] = [_parse_case(f) for f in [
    "AHU_annual.csv",
    "coi_bias_-2_annual.csv", "coi_bias_-4_annual.csv", "coi_bias_2_annual.csv", "coi_bias_4_annual.csv",
    "oa_bias_-2_annual.csv", "oa_bias_-4_annual.csv", "oa_bias_2_annual.csv", "oa_bias_4_annual.csv",
    "coi_leakage_010_annual.csv", "coi_leakage_025_annual.csv", "coi_leakage_040_annual.csv",
    "coi_leakage_050_annual.csv",
    "coi_stuck_010_annual.csv", "coi_stuck_025_annual.csv", "coi_stuck_050_annual.csv",
    "coi_stuck_075_annual.csv",
    "damper_stuck_010_annual.csv", "damper_stuck_025_annual.csv", "damper_stuck_075_annual.csv",
    "damper_stuck_100_annual_short.csv",
]]


def download(force: bool = False) -> None:
    """Fetch and unzip the SDAHU CSV corpus (~579 MB) into data/raw/SDAHU/csv/."""
    CSV_DIR.mkdir(parents=True, exist_ok=True)
    if not force and any(CSV_DIR.glob("*.csv")):
        print(f"CSVs already present in {CSV_DIR}; use force=True to re-download.")
        return
    print(f"downloading {CSV_ZIP_URL} ...")
    with urlopen(CSV_ZIP_URL) as resp:                       # nosec - canonical LBNL host
        blob = resp.read()
    print(f"  got {len(blob)/1e6:.0f} MB; extracting ...")
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        for n in zf.namelist():
            if n.endswith(".csv"):
                (CSV_DIR / Path(n).name).write_bytes(zf.read(n))
    print(f"  extracted {len(list(CSV_DIR.glob('*.csv')))} CSVs into {CSV_DIR}")


def _read_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    cols = {c.strip(): c for c in df.columns}
    missing = [p for p in EXPECTED_POINTS if p not in cols]
    if missing:
        raise ValueError(f"{path.name}: missing expected points {missing}; header={list(df.columns)}")
    return df.rename(columns={cols[p]: p for p in EXPECTED_POINTS})


def iter_windows(case: FaultCase, window: int, stride: int, points=EXPECTED_POINTS):
    """Yield (values[n_features, window] float32, meta) windows from one fault file.

    Requires the CSVs to be downloaded. Windows are non-overlapping when stride==window.
    """
    path = CSV_DIR / case.filename
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run `python -m sft.datasets.lbnl_sdahu --download`")
    df = _read_csv(path)
    arr = df[points].to_numpy(dtype=np.float32)              # (T, n_features)
    T = arr.shape[0]
    for start in range(0, T - window + 1, stride):
        win = arr[start:start + window].T                    # (n_features, window)
        yield win, {"file": case.filename, "family": case.family, "component": case.component,
                    "intensity": case.intensity, "label": case.label, "start": start}


def main() -> None:
    ap = argparse.ArgumentParser(description="LBNL SDAHU dataset utility")
    ap.add_argument("--download", action="store_true", help="fetch + unzip the 579 MB CSV corpus")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--inventory", action="store_true", help="print the fault-file inventory")
    args = ap.parse_args()
    if args.download:
        download(force=args.force)
    if args.inventory or not args.download:
        print(f"{len(INVENTORY)} cases ({sum(c.label==0 for c in INVENTORY)} normal, "
              f"{sum(c.label==1 for c in INVENTORY)} faulted)")
        for c in INVENTORY:
            print(f"  {c.filename:34s} {c.family:13s} {c.component:24s} "
                  f"int={c.intensity:>6}  label={c.label}")


if __name__ == "__main__":
    main()
