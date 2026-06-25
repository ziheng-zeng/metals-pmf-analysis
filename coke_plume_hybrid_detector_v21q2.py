# -*- coding: utf-8 -*-
"""
Coke Plume Nighttime Detector — v21q2
v21q + date-range filtering + full-hours CSV output.

Changes vs v21q:
- Optional date filter: START_DATE / END_DATE (inclusive bounds on TIME after tz conversion).
- Outputs BOTH:
   * *_hours_full_v21q2.csv   (all hours with ER, is_night, metals_exceeding, is_candidate)
   * *_hours_night_v21q2.csv  (night-only subset)
- Events logic unchanged: consecutive candidate hours at night, min duration=2h.

Core logic recap:
- TIME parsed as UTC -> US/Eastern -> naive.
- Night via Astral sunrise/sunset (fallback fixed hours if Astral missing).
- Background = rolling 30th percentile over 24h.
- Candidate hour = night AND (>=3 metals have ER>=2 and conc > 3×uncert).
- Events = 2+ consecutive candidate hours.
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

# Fallback fixed-night definition
FALLBACK_NIGHT_START_HOUR = 20
FALLBACK_NIGHT_END_HOUR   = 6

# Background window and quantile
ROLLING_WINDOW_HOURS = 24
BG_QUANTILE = 0.25
ENRICHMENT_THRESHOLD = 2.0
MIN_METALS_EXCEEDING = 3
FLOOR_MULTIPLIER = 1.0

# Event rule
MIN_EVENT_DURATION_HOURS = 2

# Optional date filter (set to None to disable). Use 'YYYY-MM-DD' strings.
START_DATE = "2024-09-01"   # e.g., "2024-09-01"
END_DATE   = "2024-09-30"   # e.g., "2024-09-30"

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
    runs, current = [], []
    for ts, row in detail_df.iterrows():
        if row["is_candidate"]:
            current.append(ts)
        else:
            if current:
                runs.append(current); current = []
    if current:
        runs.append(current)

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

    kept = []
    for block in blocks:
        if len(block) >= MIN_EVENT_DURATION_HOURS:
            kept.append(block)

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
        er_cols = [c for c in seg.columns if c.startswith("ER_")]
        for c in er_cols:
            m = c.replace("ER_", "")
            row[f"peak_ER_{m}"] = float(seg[c].max(skipna=True))
            row[f"mean_ER_{m}"] = float(seg[c].mean(skipna=True))
            row[f"median_ER_{m}"] = float(seg[c].median(skipna=True))
        events.append(row)

    return pd.DataFrame(events).sort_values("start").reset_index(drop=True)

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = pd.read_csv(FILE_PATH)

    if "TIME" not in df.columns:
        raise ValueError("Expected a 'TIME' column in the CSV.")
    df["TIME"] = pd.to_datetime(df["TIME"], utc=True).dt.tz_convert(TIMEZONE).dt.tz_localize(None)

    # Date filter (after tz conversion)
    if START_DATE is not None:
        df = df[df["TIME"] >= pd.Timestamp(START_DATE)]
    if END_DATE is not None:
        df = df[df["TIME"] <= pd.Timestamp(END_DATE) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)]

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

    # Auto-detect columns
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

    # Night mask
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

    # Threshold + absolute floor (3× uncert)
    exceed_mask = pd.DataFrame(index=conc.index)
    abs_floor_mask = pd.DataFrame(index=conc.index)
    for m in TARGET_METALS:
        exceed_mask[m] = ER[f"ER_{m}"] >= ENRICHMENT_THRESHOLD
        abs_floor_mask[m] = conc[m] > (FLOOR_MULTIPLIER * unc[m])

    both_mask = exceed_mask & abs_floor_mask
    metals_exceeding = both_mask.sum(axis=1)
    is_candidate = is_night & (metals_exceeding >= MIN_METALS_EXCEEDING)

    # Full-hours detail (for plotting/QA)
    detail_full = pd.concat([
        conc.add_prefix("C_"), unc.add_prefix("U_"), ER,
        is_night.rename("is_night"),
        metals_exceeding.rename("metals_exceeding"),
        is_candidate.rename("is_candidate"),
    ], axis=1)

    # Night-only detail (compact)
    detail_night = detail_full[detail_full["is_night"]].copy()

    # Events from night-only candidates
    events_df = detect_events(detail_night)

    # Output
    base = os.path.splitext(os.path.basename(FILE_PATH))[0]
    suffix = ""
    if START_DATE or END_DATE:
        suffix = "_rng"
        if START_DATE: suffix += f"_{str(START_DATE)}"
        if END_DATE:   suffix += f"_{str(END_DATE)}"

    events_path      = os.path.join(OUTPUT_DIR, f"{base}_events_v21q2{suffix}.csv")
    hours_full_path  = os.path.join(OUTPUT_DIR, f"{base}_hours_full_v21q2{suffix}.csv")
    hours_night_path = os.path.join(OUTPUT_DIR, f"{base}_hours_night_v21q2{suffix}.csv")

    events_df.to_csv(events_path, index=False)
    detail_full.to_csv(hours_full_path, index=True)
    detail_night.to_csv(hours_night_path, index=True)

    print(f"\nDetected {events_df.shape[0]} events (min {MIN_EVENT_DURATION_HOURS} h).")
    print(f"Total flagged nighttime hours: {int(detail_night['is_candidate'].sum())}")
    print(f"Wrote:\n  - {events_path}\n  - {hours_full_path}\n  - {hours_night_path}\n")

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print('ERROR:', e)
        sys.exit(1)
