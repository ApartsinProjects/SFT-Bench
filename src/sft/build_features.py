"""Build the unified `features.csv` for a dataset by merging the Brick graph metadata with the
dataset's own unit table. Run:  python -m sft.build_features
Writes data/processed/SDAHU/features.csv.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from sft.graph.brick_parse import parse_brick
from sft.schema import FeatureMeta

# Units per point, from the official SDAHU documentation Table 2 (Brick v1.2 carries no units).
SDAHU_UNITS: dict[str, str] = {
    "SA_TEMP": "degF", "SA_TEMPSPT": "degF", "OA_TEMP": "degF", "MA_TEMP": "degF", "RA_TEMP": "degF",
    "SF_SPD_DM": "", "RF_SPD_DM": "", "OA_CFM": "CFM", "RA_CFM": "CFM", "SA_CFM": "CFM",
    "SF_CS": "frac", "SF_SPD": "frac", "RF_CS": "frac", "RF_SPD": "frac",
    "SF_WAT": "W", "RF_WAT": "W",
    "OA_DMPR_DM": "frac", "OA_DMPR": "frac", "RA_DMPR_DM": "frac", "RA_DMPR": "frac",
    "CHWC_VLV_DM": "frac", "CHWC_VLV": "frac",
    "SA_SP": "inH2O", "SA_SPSPT": "inH2O", "SYS_CTL": "",
    "ZONE_TEMP_1": "degF", "ZONE_TEMP_2": "degF", "ZONE_TEMP_3": "degF",
    "ZONE_TEMP_4": "degF", "ZONE_TEMP_5": "degF",
}

ROOT = Path(__file__).resolve().parents[2]
SDAHU_TTL = ROOT / "data/raw/SDAHU/ttl/LBNL_FDD_Data_Sets_SDAHU_ttl.ttl"


def build_sdahu_features() -> pd.DataFrame:
    rows = parse_brick(SDAHU_TTL, dataset="LBNL_SDAHU")
    metas: list[FeatureMeta] = []
    for r in rows:
        point = r["feature_id"].split(":", 1)[1]
        r = dict(r, unit=SDAHU_UNITS.get(point, ""))
        metas.append(FeatureMeta(**r))
    df = pd.DataFrame(m.as_row() for m in metas)
    return df


def main() -> None:
    df = build_sdahu_features()
    out = ROOT / "data/processed/SDAHU/features.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    missing = df[df["unit"] == ""]["feature_id"].tolist()
    print(f"wrote {out}  ({len(df)} features)")
    print(f"measurement types: {sorted(df['measurement_type'].unique())}")
    print(f"components: {sorted(df['component'].unique())}")
    print(f"blank-unit (dimensionless/status) features: {[m.split(':')[1] for m in missing]}")


if __name__ == "__main__":
    main()
