# -*- coding: utf-8 -*-
"""
Baseline stats for ALL metals (ng/m³), no filtering.
- Auto-detects concentration columns (ends with " (ng/m3)" and not "uncert")
- Excludes instrument-down intervals
- Optional TIME_RANGE (set to None to use all remaining data)
- Computes mean/median/std/min/max + counts
- Outputs two CSVs:
    1) stats_by_column.csv   (each instrument column)
    2) stats_by_element.csv  (collapsed per element token)
"""

import os
import pandas as pd
import numpy as np

# ----------------------- CONFIG -----------------------
FILE_PATH = "Xact_EST_May2023_July2025_combined.csv"
TZ = "US/Eastern"

# Intervals to EXCLUDE entirely (local time)
EXCLUDE_INTERVALS = [
    ("2024-01-09", "2024-02-13"),
    ("2024-07-02", "2024-08-08"),
]

# Optional time range (local time). Use None for all data outside excludes.
# TIME_RANGE = ("2024-01-01 00:00", "2024-12-31 23:59")
TIME_RANGE = None

# Output
OUTDIR = "./baseline_stats_all_metals"
OUTCSV_BY_COLUMN  = "stats_by_column.csv"
OUTCSV_BY_ELEMENT = "stats_by_element.csv"

# ----------------------- LOAD -----------------------
df = pd.read_csv(FILE_PATH)
df["TIME"] = pd.to_datetime(df["TIME"], utc=True).dt.tz_convert(TZ)
df = df.set_index("TIME").sort_index()

# ----------------------- APPLY EXCLUDES -----------------------
df_work = df.copy()

# Drop rows within any exclude interval
for start_s, end_s in EXCLUDE_INTERVALS:
    s = pd.Timestamp(start_s, tz=TZ)
    e = pd.Timestamp(end_s,   tz=TZ)
    df_work = df_work.loc[(df_work.index < s) | (df_work.index > e)]

# Optional time window
if TIME_RANGE is not None:
    t0 = pd.Timestamp(TIME_RANGE[0], tz=TZ)
    t1 = pd.Timestamp(TIME_RANGE[1], tz=TZ)
    df_work = df_work.loc[(df_work.index >= t0) & (df_work.index <= t1)]
    print(f"Applied TIME_RANGE: {t0} → {t1} ({len(df_work)} rows)")

# ----------------------- PICK METAL COLUMNS -----------------------
all_cols = df_work.columns.tolist()
metal_cols = [c for c in all_cols if " (ng/m3)" in c and "uncert" not in c.lower()]
if not metal_cols:
    raise SystemExit("No metal concentration columns found (ending with ' (ng/m3)')")

print(f"Found {len(metal_cols)} metal concentration columns.")

# Ensure numeric (coerce bad strings to NaN)
df_numeric = df_work[metal_cols].apply(pd.to_numeric, errors="coerce")

# ----------------------- STATS (BY COLUMN) -----------------------
def compute_stats_frame(df_sub: pd.DataFrame) -> pd.DataFrame:
    """
    Compute summary stats for each column in df_sub.
    NOTE: Use list-of-funcs in .agg to avoid dict misinterpretation.
    """
    # Compute core stats; result index=metric, columns=original columns
    stats_core = df_sub.agg(['mean', 'median', 'std', 'min', 'max']).T
    stats_core.rename(columns={
        "mean": "Mean_ng_m3",
        "median": "Median_ng_m3",
        "std": "Std_ng_m3",
        "min": "Min_ng_m3",
        "max": "Max_ng_m3",
    }, inplace=True)

    # counts
    n_total = df_sub.shape[0]
    n_valid_series = df_sub.notna().sum(axis=0)

    stats_core["N_valid"] = n_valid_series.values
    stats_core["N_total"] = n_total
    stats_core["Valid_fraction"] = stats_core["N_valid"] / stats_core["N_total"]

    stats_core.reset_index(inplace=True)
    stats_core.rename(columns={"index": "Column"}, inplace=True)

    # Element token = first whitespace-delimited token from the column name
    stats_core["Element"] = stats_core["Column"].str.split().str[0]

    # order columns
    stats_core = stats_core[[
        "Element", "Column",
        "Mean_ng_m3", "Median_ng_m3", "Std_ng_m3", "Min_ng_m3", "Max_ng_m3",
        "N_valid", "N_total", "Valid_fraction"
    ]].sort_values(["Element", "Column"]).reset_index(drop=True)
    return stats_core

stats_by_col = compute_stats_frame(df_numeric)

# ----------------------- COLLAPSE BY ELEMENT -----------------------
def collapse_by_element(stats_col: pd.DataFrame) -> pd.DataFrame:
    # Average stats across columns for the same element; sum counts.
    grp = stats_col.groupby("Element", as_index=False).agg({
        "Mean_ng_m3": "mean",
        "Median_ng_m3": "mean",
        "Std_ng_m3": "mean",
        "Min_ng_m3": "mean",
        "Max_ng_m3": "mean",
        "N_valid": "sum",
        "N_total": "sum"
    })
    grp["Valid_fraction"] = grp["N_valid"] / grp["N_total"]
    return grp.sort_values("Element").reset_index(drop=True)

stats_by_elem = collapse_by_element(stats_by_col)

# ----------------------- OUTPUT -----------------------
os.makedirs(OUTDIR, exist_ok=True)
path_col  = os.path.join(OUTDIR, OUTCSV_BY_COLUMN)
path_elem = os.path.join(OUTDIR, OUTCSV_BY_ELEMENT)

stats_by_col.to_csv(path_col, index=False)
stats_by_elem.to_csv(path_elem, index=False)

print("\n=== Stats by COLUMN (first 20 rows) ===")
print(stats_by_col.head(20).to_string(index=False, float_format=lambda x: f"{x:.5g}"))

print("\n=== Stats by ELEMENT ===")
print(stats_by_elem.to_string(index=False, float_format=lambda x: f"{x:.5g}"))

print(f"\nSaved:\n  - {path_col}\n  - {path_elem}")
