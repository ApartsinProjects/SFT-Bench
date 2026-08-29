# LBNL FDD SDAHU — locked schema (from the official PDF + Brick .ttl)

Source: *LBNL Fault Detection and Diagnostics Data Sets: Single Duct Air Handling Unit*, Granderson
et al., LBNL/PNNL, Sept 2022. DOI 10.25984/1881324, CC-BY 4.0.
Download: `https://fdddata.lbl.gov/data/Simulated_LBNL_FDD_Data_Sets_SDAHU/`.

- **System:** single-duct VAV AHU serving the middle floor (1 interior + 4 perimeter zones) of the
  DOE large-office reference building. Chilled water from a central chiller; hot water from a gas
  boiler. **Simulation** (EnergyPlus–Modelica), Chicago IL TMY weather.
- **Sampling:** 1-minute. Each file is one year (~525,600 rows) except one short case.
- **Files:** 21 CSVs — 1 fault-free + 20 faulted. Fault label is per-file (whole file is that
  fault/intensity). CSV corpus zip ≈ 579 MB.
- **Brick model:** ships as `LBNL_FDD_Data_Sets_SDAHU_ttl.ttl`, Brick **v1.2**, namespace
  `bldg: <bldg-59#>`. Contains measurement class per point, `hasPart`/`hasPoint` component structure,
  and `feeds` topology (AHU → Zones).

## Data points (30) — CSV columns = "abbreviation"

| # | abbreviation | description | unit | class (measurement/type) | component |
|---|---|---|---|---|---|
| 1 | SA_TEMP | supply air temperature | °F | Supply_Air_Temperature_Sensor | AHU/supply |
| 2 | SA_TEMPSPT | supply air temp setpoint | °F | Supply_Air_Temperature_Setpoint | AHU/supply |
| 3 | OA_TEMP | outdoor air temperature | °F | Outside_Air_Temperature_Sensor | AHU/outdoor |
| 4 | MA_TEMP | mixed air temperature | °F | Mixed_Air_Temperature_Sensor | AHU/mixing |
| 5 | RA_TEMP | return air temperature | °F | Return_Air_Temperature_Sensor | AHU/return |
| 6 | SF_SPD_DM | supply fan status (0/1) | – | Fan_On_Off_Status | Supply_Air_Fan |
| 7 | RF_SPD_DM | return fan status (0/1) | – | Fan_On_Off_Status | Return_Air_Fan |
| 8 | OA_CFM | outdoor airflow | CFM | Outside_Air_Flow_Sensor | AHU/outdoor |
| 9 | RA_CFM | return airflow | CFM | Return_Air_Flow_Sensor | AHU/return |
| 10 | SA_CFM | supply airflow | CFM | Supply_Air_Flow_Sensor | AHU/supply |
| 11 | SF_CS | supply fan speed control signal | 0-1 | Speed_Setpoint | Supply_Air_Fan |
| 12 | SF_SPD | supply fan speed position | 0-1 | Speed_status | Supply_Air_Fan |
| 13 | RF_CS | return fan speed control signal | 0-1 | Speed_Setpoint | Return_Air_Fan |
| 14 | RF_SPD | return fan speed position | 0-1 | Speed_status | Return_Air_Fan |
| 15 | SF_WAT | supply fan power | W | Electrical_Power_Sensor | Supply_Air_Fan |
| 16 | RF_WAT | return fan power | W | Electrical_Power_Sensor | Return_Air_Fan |
| 17 | OA_DMPR_DM | OA damper control signal | 0-1 | Damper_Position_Command | Outdoor_Air_Damper |
| 18 | OA_DMPR | OA damper position | 0-1 | Damper_Position_Sensor | Outdoor_Air_Damper |
| 19 | RA_DMPR_DM | RA damper control signal | 0-1 | Damper_Position_Command | Return_Air_Damper |
| 20 | RA_DMPR | RA damper position | 0-1 | Damper_Position_Sensor | Return_Air_Damper |
| 21 | CHWC_VLV_DM | cooling coil valve control signal | 0-1 | Valve_Position_Command | Cooling_Coil |
| 22 | CHWC_VLV | cooling coil valve position | 0-1 | Valve_Position_Sensor | Cooling_Coil |
| 23 | SA_SP | supply duct static pressure | inH2O | Supply_Air_Static_Pressure_Sensor | AHU/supply |
| 24 | SA_SPSPT | supply duct static pressure setpoint | inH2O | Supply_Air_Static_Pressure_Setpoint | AHU/supply |
| 25 | SYS_CTL | occupancy mode (0/1) | – | Occupancy_Status | AHU |
| 26-30 | ZONE_TEMP_1..5 | zone air temperature | °F | Zone_Air_Temperature_Sensor | Zone_1..5 |

Note: exact CSV header (datetime column name, exact casing) is confirmed at load time by
`datasets/lbnl_sdahu.py`, which validates the 30 expected abbreviations are present.

## Fault inventory (Table 4) — label parsed from filename

| file | fault type | intensity |
|---|---|---|
| `AHU_annual.csv` | fault-free | – (label 0) |
| `sa_bias_{-2,-4,2,4}_annual.csv` | supply-air temp sensor bias | ±2/±4 °C |
| `oa_bias_{-2,-4,2,4}_annual.csv` | outdoor-air temp sensor bias | ±2/±4 °C |
| `coi_leakage_{010,025,040,050}_annual.csv` | cooling-coil valve leaking | 10/25/40/50 % |
| `coi_stuck_{010,025,050,075}_annual.csv` | cooling-coil valve stuck | 10/25/50/75 % |
| `damper_stuck_{010,025,075}_annual.csv` | OA damper stuck | 10/25/75 % |
| `damper_stuck_100_annual_short.csv` | OA damper stuck (Apr1–Nov1 only) | 100 % |

## Why this maps to the semantic-transfer hypothesis

Fault families give ready-made transfer axes keyed on Brick semantics:

- **Measurement-type / location transfer:** `sa_bias` ↔ `oa_bias` — same mechanism (temperature-sensor
  bias), different location (`Supply_Air_*` vs `Outside_Air_*`). Held-out `oa_bias` tests whether KG
  transfers the "temperature sensor bias" concept across component context. Text embeddings should do
  well here; the interesting question is whether KG adds anything.
- **Component / actuator transfer:** `coi_stuck` ↔ `damper_stuck` — actuator-stuck faults on different
  components (`Cooling_Coil` valve vs `Outdoor_Air_Damper`). Here topology (`hasPart`, position vs
  command point pairing) is more likely to carry signal text cannot linearize — the KG-favouring case.

These two axes are the concrete instantiation of the KG-vs-Text headline in
[analysis.md](analysis.md).
