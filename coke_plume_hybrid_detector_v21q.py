# -*- coding: utf-8 -*-
"""
Coke Plume Nighttime Detector — v21q
Baseline v2.1 with two changes:
  (1) Night/day via Astral sunrise/sunset (true night), with fallback to fixed hours if Astral unavailable.
  (2) Rolling background uses a quantile (default 30th percentile) instead of median.

What remains the same as v2.1:
- ER = conc / rolling background
- Candidate hour: night AND (>= MIN_METALS_EXCEEDING metals have ER >= ENRICHMENT_THRESHOLD and conc > 3×uncert)
- Events: consecutive candidate hours, min duration = MIN_EVENT_DURATION_HOURS (default 2)
- Duplicate index safe; '1h' resampling
"""

import os
import sys
import numpy as np
import pandas as pd

# ----------------------- USER CONFIG -----------------------
FILE_PATH = r"Xact_EST_May2023_Oct2025_combined.csv"
OUTPUT_DIR = r"./coke_plume_flags"

TARGET_METALS = ["Pb", "Zn", "Se", "As"]

# Astral site info (Pittsburgh by default)
TIMEZONE = "US/Eastern"
SITE_NAME = "Pittsburgh"
REGION = "USA"
LATITUDE = 40.4406
LONGITUDE = -79.9959

# Fallback fixed-night definition (in case Astral not installed)
FALLBACK_NIGHT_START_HOUR = 20
FALLBACK_NIGHT_END_HOUR   = 6

# Background window and quantile
ROLLING_WINDOW_HOURS = 24
BG_QUANTILE = 0.25
ENRICHMENT_THRESHOLD = 2.0
MIN_METALS_EXCEEDING = 3

# Absolute floor
FLOOR_MULTIPLIER = 3.0

# Event rule
MIN_EVENT_DURATION_HOURS = 2

# Exclusions
EXCLUSIONS = [
    ("2024-01-09", "2024-02-13"),
    ("2024-07-02", "2024-08-08"),
]
# -----------------------------------------------------------

def has_astral():
    try:
        import astral  # noqa: F401
        return True
    except Exception:
        return False

def is_night_mask_astral(index: pd.DatetimeIndex) -> pd.Series:
    try:
        from astral import LocationInfo
        from astral.sun import sun
        import pytz
    except Exception as e:
        print(f"[warn] Astral not available ({e}); using fallback fixed hours.")
        return is_night_mask_fallback(index)

    tz = pytz.timezone(TIMEZONE)
    loc = LocationInfo(name=SITE_NAME, region=REGION, timezone=TIMEZONE, latitude=LATITUDE, longitude=LONGITUDE)

    dates = pd.to_datetime(index.date).unique()
    rows = []
    for d in dates:
        s = sun(loc.observer, date=pd.Timestamp(d).date(), tzinfo=tz)
        rise = pd.Timestamp(s["sunrise"]).tz_convert(tz).tz_localize(None)
        set_ = pd.Timestamp(s["sunset"]).tz_convert(tz).tz_localize(None)
        rows.append({"date": pd.Timestamp(d), "sunrise": rise, "sunset": set_})
    day_tbl = pd.DataFrame(rows).set_index("date")

    def _is_night(ts):
        d = pd.Timestamp(ts.date())
        if d not in day_tbl.index:
            return False
        sr = day_tbl.loc[d, "sunrise"]
        ss = day_tbl.loc[d, "sunset"]
        return (ts < sr) or (ts >= ss)

    return pd.Series([_is_night(ts) for ts in index], index=index, name="is_night")

def is_night_mask_fallback(index: pd.DatetimeIndex) -> pd.Series:
    return pd.Series([(ts.hour >= FALLBACK_NIGHT_START_HOUR) or (ts.hour <= FALLBACK_NIGHT_END_HOUR)
                      for ts in index], index=index, name="is_night")

def rolling_quantile(series: pd.Series, hours: int, q: float) -> pd.Series:
    return series.rolling(f"{hours}h", min_periods=max(6, hours//4)).quantile(q)

def detect_events(detail_df: pd.DataFrame) -> pd.DataFrame:
    # Build events from consecutive True in 'is_candidate', requiring exact 1h steps
    runs, current = [], []
    for ts, row in detail_df.iterrows():
        if row["is_candidate"]:
            current.append(ts)
        else:
            if current:
                runs.append(current); current = []
    if current:
        runs.append(current)

    # Split into strictly consecutive blocks
    blocks = []
    for r in runs:
        r = sorted(r)
        buf = [r[0]]
        for a, b in zip(r, r[1:]):
            if (b - a) == pd.Timedelta(hours=1):
                buf.append(b)
            else:
                blocks.append(buf); buf = [b]
        blocks.append(buf)

    # Keep only blocks of sufficient length
    kept = []
    for block in blocks:
        if len(block) >= MIN_EVENT_DURATION_HOURS:
            kept.append(block)

    # Summaries
    events = []
    for block in kept:
        seg = detail_df.loc[block]
        row = {
            "start": block[0],
            "end": block[-1],
            "duration_hours": len(block),
            "max_metals_exceeding": int(seg["metals_exceeding"].max()),
            "mean_metals_exceeding": float(seg["metals_exceeding"].mean()),
        }
        # Peak/mean/median ERs
        er_cols = [c for c in seg.columns if c.startswith("ER_")]
        for c in er_cols:
            m = c.replace("ER_", "")
            row[f"peak_ER_{m}"] = float(seg[c].max(skipna=True))
            row[f"mean_ER_{m}"] = float(seg[c].mean(skipna=True))
            row[f"median_ER_{m}"] = float(seg[c].median(skipna=True))
        events.append(row)

    events_df = pd.DataFrame(events).sort_values("start").reset_index(drop=True)
    return events_df

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = pd.read_csv(FILE_PATH)

    # TIME parsing
    if "TIME" not in df.columns:
        raise ValueError("Expected a 'TIME' column in the CSV.")
    df["TIME"] = pd.to_datetime(df["TIME"], utc=True).dt.tz_convert(TIMEZONE).dt.tz_localize(None)

    # Exclusions
    for start_str, end_str in EXCLUSIONS:
        start = pd.Timestamp(start_str); end = pd.Timestamp(end_str)
        df = df[(df["TIME"] < start) | (df["TIME"] > end)]

    df = df.set_index("TIME").sort_index()

    # Collapse duplicates
    if df.index.duplicated().any():
        dup_count = int(df.index.duplicated().sum())
        print(f"[info] Collapsing {dup_count} duplicate timestamp rows by mean.")
        df = df.groupby(level=0).mean(numeric_only=True)

    # Auto-detect conc & uncertainty columns
    headers = df.columns.tolist()
    conc_cols = [c for c in headers if (" (ng/m3)" in c) and ("uncert" not in c.lower())]
    uncert_cols = [c for c in headers if ("uncert" in c.lower())]

    pairs = {}
    for m in TARGET_METALS:
        cand = [c for c in conc_cols if (f"{m} " in c)]
        if not cand:
            raise ValueError(f"Could not find concentration column for metal '{m}'")
        c_col = cand[0]
        u_col = next((u for u in uncert_cols if m in u), None)
        pairs[m] = (c_col, u_col)

    # Build frames; resample to hourly
    conc = pd.DataFrame(index=df.index)
    unc  = pd.DataFrame(index=df.index)
    for m, (c_col, u_col) in pairs.items():
        conc[m] = pd.to_numeric(df[c_col], errors="coerce")
        if u_col is not None:
            unc[m] = pd.to_numeric(df[u_col], errors="coerce")
        else:
            unc[m] = np.nan

    conc = conc.resample("1h").mean()
    unc  = unc.resample("1h").mean().reindex(conc.index)

    # Night mask via Astral (fallback if needed)
    if has_astral():
        is_night = is_night_mask_astral(conc.index)
    else:
        print("[warn] 'astral' not installed; using fallback fixed hours for night.")
        is_night = is_night_mask_fallback(conc.index)

    # Rolling quantile background & ER
    bg = pd.DataFrame(index=conc.index)
    ER = pd.DataFrame(index=conc.index)
    for m in TARGET_METALS:
        bg[m] = rolling_quantile(conc[m], ROLLING_WINDOW_HOURS, BG_QUANTILE)
        ER[f"ER_{m}"] = np.where(bg[m] > 0, conc[m] / bg[m], np.nan)

    # Threshold + absolute floor (3× uncert) — same as v2.1
    exceed_mask = pd.DataFrame(index=conc.index)
    abs_floor_mask = pd.DataFrame(index=conc.index)
    for m in TARGET_METALS:
        exceed_mask[m] = ER[f"ER_{m}"] >= ENRICHMENT_THRESHOLD
        abs_floor_mask[m] = conc[m] > (FLOOR_MULTIPLIER * unc[m])

    both_mask = exceed_mask & abs_floor_mask
    metals_exceeding = both_mask.sum(axis=1)
    is_candidate = is_night & (metals_exceeding >= MIN_METALS_EXCEEDING)

    # Night detail
    detail = pd.concat([
        conc.add_prefix("C_"), unc.add_prefix("U_"), ER,
        is_night.rename("is_night"),
        metals_exceeding.rename("metals_exceeding"),
        is_candidate.rename("is_candidate"),
    ], axis=1)
    detail_night = detail[detail["is_night"]].copy()

    # Events
    events_df = detect_events(detail_night)

    # Output
    base = os.path.splitext(os.path.basename(FILE_PATH))[0]
    events_path = os.path.join(OUTPUT_DIR, f"{base}_events_v21q.csv")
    hours_path  = os.path.join(OUTPUT_DIR, f"{base}_hours_v21q.csv")

    events_df.to_csv(events_path, index=False)
    detail_night.to_csv(hours_path, index=True)

    # Summary
    print(f"\nDetected {events_df.shape[0]} events (min {MIN_EVENT_DURATION_HOURS} h).")
    print(f"Total flagged nighttime hours: {int(detail_night['is_candidate'].sum())}")
    if not events_df.empty and 'mean_ER_Zn' in events_df.columns:
        print("\nTop 5 by mean_ER_Zn:")
        print(events_df.sort_values('mean_ER_Zn', ascending=False).head(5).to_string(index=False))
    print(f"\nWrote:\n  - {events_path}\n  - {hours_path}\n")

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print('ERROR:', e)
        sys.exit(1)
