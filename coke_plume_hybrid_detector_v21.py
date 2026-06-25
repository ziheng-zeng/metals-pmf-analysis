# -*- coding: utf-8 -*-
"""
Coke Plume Nighttime Detector — Hybrid Workflow (threshold=2× background)
Version: v2.1 (duplicate-index safe; uses '1h' resample)
Author: ChatGPT (for Ziheng)
Date: 2025-10-21

Fixes vs v2:
- Use '1h' instead of '1H' to avoid FutureWarning.
- Collapse duplicate timestamps by averaging (groupby index) before resampling.
- Remove uncert_df.reindex(...).resample sequence; resample each frame directly.
- Add a small diagnostic print for number of duplicate timestamps collapsed.

Everything else is the same workflow:
- TIME parsed as: UTC -> US/Eastern -> naive
- Exclusion windows
- Auto conc/uncert pairing
- ER >= 2 AND conc > 3×uncert (≥ MIN_METALS_EXCEEDING metals) at night
- Merge consecutive hours into events (min duration configurable)
"""

import os
import sys
import warnings
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

# ----------------------- USER CONFIG -----------------------
FILE_PATH = r"Xact_EST_May2023_Oct2025_combined.csv"  # <-- path to your CSV
OUTPUT_DIR = r"./coke_plume_flags"

TARGET_METALS = ["Pb", "Zn", "Se", "As"]

# Night definition (24h clock, inclusive bounds): hour >= NIGHT_START or <= NIGHT_END
NIGHT_START_HOUR = 20
NIGHT_END_HOUR   = 6

# Rolling background
ROLLING_WINDOW_HOURS = 24
ENRICHMENT_THRESHOLD = 2.0
MIN_METALS_EXCEEDING = 3
MIN_EVENT_DURATION_HOURS = 2

# Exclusion windows (inclusive between [start, end])
EXCLUSIONS = [
    ("2024-01-09", "2024-02-13"),
    ("2024-07-02", "2024-08-08"),
]
# -----------------------------------------------------------


@dataclass
class Event:
    start: pd.Timestamp
    end: pd.Timestamp
    duration_hours: int
    hours_count: int
    max_metals_exceeding: int
    mean_metals_exceeding: float
    peak_ERs: Dict[str, float]
    mean_ERs: Dict[str, float]
    median_ERs: Dict[str, float]


def is_night_hour(ts: pd.Timestamp) -> bool:
    h = ts.hour
    return (h >= NIGHT_START_HOUR) or (h <= NIGHT_END_HOUR)


def rolling_background(series: pd.Series, hours: int) -> pd.Series:
    return series.rolling(f"{hours}h", min_periods=max(6, hours//4), center=False).median()


def find_conc_uncert_pairs(df: pd.DataFrame, target_metals: List[str]) -> Dict[str, Tuple[str, Optional[str]]]:
    """
    For each element symbol in target_metals, search for its concentration column and uncertainty column.
    - Concentration column heuristic: contains "{M} " and "(ng/m3)", and not "uncert" (case-insensitive).
    - Uncertainty column heuristic: any column with "uncert" and the element symbol present.

    Returns dict mapping element -> (conc_col, uncert_col or None).
    Raises if no concentration column is found for a requested element.
    """
    headers = df.columns.tolist()
    conc_cols = [c for c in headers if (" (ng/m3)" in c) and ("uncert" not in c.lower())]
    uncert_cols = [c for c in headers if ("uncert" in c.lower())]

    pairs = {}
    for m in target_metals:
        cand = [c for c in conc_cols if (f"{m} " in c)]
        if not cand:
            raise ValueError(f"Could not find concentration column for metal '{m}'. "
                             f"Looked for columns like '\"{m} ... (ng/m3)\"' not containing 'uncert'.")
        conc_col = cand[0]
        u_cand = next((u for u in uncert_cols if m in u), None)
        pairs[m] = (conc_col, u_cand)
    return pairs


def detect_events(detail_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    runs, current = [], []
    for ts, row in detail_df.iterrows():
        if row["is_candidate"]:
            current.append(ts)
        else:
            if current:
                runs.append(current)
                current = []
    if current:
        runs.append(current)

    events = []
    for r in runs:
        r_sorted = sorted(r)
        blocks, buf = [], [r_sorted[0]]
        for a, b in zip(r_sorted, r_sorted[1:]):
            if (b - a) == pd.Timedelta(hours=1):
                buf.append(b)
            else:
                blocks.append(buf)
                buf = [b]
        blocks.append(buf)

        for block in blocks:
            if len(block) >= MIN_EVENT_DURATION_HOURS:
                seg = detail_df.loc[block]
                ER_cols = [c for c in seg.columns if c.startswith("ER_")]
                peak_ERs = {c.replace("ER_", ""): float(seg[c].max(skipna=True)) for c in ER_cols}
                mean_ERs = {c.replace("ER_", ""): float(seg[c].mean(skipna=True)) for c in ER_cols}
                median_ERs = {c.replace("ER_", ""): float(seg[c].median(skipna=True)) for c in ER_cols}

                events.append(dict(
                    start=block[0],
                    end=block[-1],
                    duration_hours=len(block),
                    hours_count=int(seg.shape[0]),
                    max_metals_exceeding=int(seg["metals_exceeding"].max()),
                    mean_metals_exceeding=float(seg["metals_exceeding"].mean()),
                    **{f"peak_ER_{k}": v for k, v in peak_ERs.items()},
                    **{f"mean_ER_{k}": v for k, v in mean_ERs.items()},
                    **{f"median_ER_{k}": v for k, v in median_ERs.items()},
                ))

    if events:
        events_df = pd.DataFrame(events).sort_values("start").reset_index(drop=True)
    else:
        events_df = pd.DataFrame(columns=[
            "start","end","duration_hours","hours_count",
            "max_metals_exceeding","mean_metals_exceeding"
        ] + [f"{s}_{m}" for s in ["peak_ER","mean_ER","median_ER"] for m in TARGET_METALS])
    return events_df, detail_df


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = pd.read_csv(FILE_PATH)

    # --- TIME parsing as specified ---
    if "TIME" not in df.columns:
        raise ValueError("Expected a 'TIME' column in the CSV.")
    df["TIME"] = pd.to_datetime(df["TIME"], utc=True).dt.tz_convert("US/Eastern").dt.tz_localize(None)

    # --- Apply exclusion windows ---
    for start_str, end_str in EXCLUSIONS:
        start = pd.Timestamp(start_str)
        end   = pd.Timestamp(end_str)
        df = df[(df["TIME"] < start) | (df["TIME"] > end)]

    df = df.set_index("TIME").sort_index()

    # --- Collapse duplicate timestamps by averaging ---
    if df.index.duplicated().any():
        dup_count = int(df.index.duplicated().sum())
        print(f"[info] Collapsing {dup_count} duplicate timestamp rows by hourly mean.")
        df = df.groupby(level=0).mean(numeric_only=True)

    # --- Find conc/uncert pairs for target metals ---
    pairs = find_conc_uncert_pairs(df, TARGET_METALS)

    # Build conc/uncert frames (numeric)
    conc_df = pd.DataFrame(index=df.index)
    uncert_df = pd.DataFrame(index=df.index)
    for m, (c_col, u_col) in pairs.items():
        conc_df[m] = pd.to_numeric(df[c_col], errors="coerce")
        if u_col is not None:
            uncert_df[m] = pd.to_numeric(df[u_col], errors="coerce")
        else:
            uncert_df[m] = np.nan

    # --- Resample to hourly grid directly ---
    conc_df = conc_df.resample("1h").mean()
    uncert_df = uncert_df.resample("1h").mean().reindex(conc_df.index)

    # --- Night mask ---
    is_night = pd.Series([is_night_hour(ts) for ts in conc_df.index], index=conc_df.index, name="is_night")

    # --- Rolling background & ER ---
    bg = pd.DataFrame(index=conc_df.index)
    for m in TARGET_METALS:
        bg[m] = rolling_background(conc_df[m], ROLLING_WINDOW_HOURS)

    ER = pd.DataFrame(index=conc_df.index)
    for m in TARGET_METALS:
        valid_bg = bg[m] > 0
        ER[f"ER_{m}"] = np.where(valid_bg, conc_df[m] / bg[m], np.nan)

    # --- Threshold exceedance & absolute floor ---
    exceed_mask = pd.DataFrame(index=conc_df.index)
    abs_floor_mask = pd.DataFrame(index=conc_df.index)
    for m in TARGET_METALS:
        exceed_mask[m]    = ER[f"ER_{m}"] >= ENRICHMENT_THRESHOLD
        abs_floor_mask[m] = conc_df[m] > (3.0 * uncert_df[m])

    both_mask = exceed_mask & abs_floor_mask
    metals_exceeding = both_mask.sum(axis=1)
    is_candidate = is_night & (metals_exceeding >= MIN_METALS_EXCEEDING)

    # --- Detail DF (night only) ---
    detail = pd.concat([conc_df.add_prefix("C_"), uncert_df.add_prefix("U_"), ER, is_night, metals_exceeding.rename("metals_exceeding"), is_candidate.rename("is_candidate")], axis=1)
    detail_night = detail[detail["is_night"]].copy()

    # --- Detect events ---
    events_df, detail_df = detect_events(detail_night)

    # --- Output ---
    base = os.path.splitext(os.path.basename(FILE_PATH))[0]
    events_path = os.path.join(OUTPUT_DIR, f"{base}_coke_events_thresh{ENRICHMENT_THRESHOLD:.1f}_min{MIN_METALS_EXCEEDING}_minDur{MIN_EVENT_DURATION_HOURS}_v21.csv")
    hours_path  = os.path.join(OUTPUT_DIR, f"{base}_coke_hours_thresh{ENRICHMENT_THRESHOLD:.1f}_min{MIN_METALS_EXCEEDING}_minDur{MIN_EVENT_DURATION_HOURS}_v21.csv")

    events_df.to_csv(events_path, index=False)
    detail_df.to_csv(hours_path, index=True)

    # Console summary
    n_events = events_df.shape[0]
    total_flagged_hours = int(detail_df["is_candidate"].sum())
    print(f"\nDetected {n_events} nighttime events (>= {MIN_EVENT_DURATION_HOURS} consecutive hours).")
    print(f"Total flagged nighttime hours: {total_flagged_hours}")
    if n_events > 0 and "peak_ER_Zn" in events_df.columns:
        print("\nTop 5 events by peak_ER_Zn:")
        print(events_df.sort_values("peak_ER_Zn", ascending=False).head(5).to_string(index=False))
    print(f"\nWrote:\n  - {events_path}\n  - {hours_path}\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("ERROR:", e)
        sys.exit(1)
