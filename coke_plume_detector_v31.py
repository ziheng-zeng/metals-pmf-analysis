# -*- coding: utf-8 -*-
"""
Coke Plume Detector — v3.1 (Astral day/night + Quantile background + Jump→Sustain hybrid)
Author: ChatGPT (for Ziheng)
Date: 2025-10-23

New in v3.1
-----------
- Uses Astral's sunrise/sunset for *actual* day/night classification at your site.
  (Falls back to fixed hours if Astral is not available.)

Core features (from v3)
-----------------------
- Background = rolling 30th percentile (configurable) over 24h.
- Hybrid detection:
  • Seed on instantaneous jump C(t)/C(t-1) for >= MIN_METALS_JUMP metals + 3×uncert (now & prev).
  • Sustain while >= MIN_METALS_SUSTAIN metals have ER >= SUSTAIN_ER vs rolling quantile + 3×uncert.
  • Night-only enforced using Astral day/night.
- Event building with GAP_TOL to bridge short dips; min duration = 2h.
"""

import os
import sys
import numpy as np
import pandas as pd

# ----------------------- USER CONFIG -----------------------
FILE_PATH = r"Xact_EST_May2023_Oct2025_combined.csv"
OUTPUT_DIR = r"./coke_plume_flags"

TARGET_METALS = ["Pb", "Zn", "Se", "As"]

# Site info for Astral (Pittsburgh by default)
SITE_NAME = "Pittsburgh"
REGION = "USA"
TIMEZONE = "US/Eastern"
LATITUDE = 40.4406
LONGITUDE = -79.9959

# Fallback fixed-night definition if Astral is missing (inclusive)
FALLBACK_NIGHT_START_HOUR = 20
FALLBACK_NIGHT_END_HOUR   = 6

# Background window and quantile
ROLLING_WINDOW_HOURS = 24
BG_QUANTILE = 0.30          # 30th percentile background

# Jump (seed) thresholds
JUMP_RATIO = 2.0
MIN_METALS_JUMP = 3

# Sustain thresholds
SUSTAIN_ER = 2.0
MIN_METALS_SUSTAIN = 3

# Absolute floors
FLOOR_MULTIPLIER = 3.0
APPLY_ABS_FLOOR = True

# Event building
MIN_EVENT_DURATION_HOURS = 2
GAP_TOL = 1

# Exclusion windows
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
    """Return a boolean Series marking true night hours using Astral sunrise/sunset.
    Assumes index is *naive* local time in TIMEZONE (US/Eastern by default).
    """
    try:
        from astral import LocationInfo
        from astral.sun import sun
        import pytz
    except Exception as e:
        # Should not happen if has_astral() was checked, but fallback just in case
        print(f"[warn] Astral not available ({e}); falling back to fixed hours.")
        return is_night_mask_fallback(index)

    tz = pytz.timezone(TIMEZONE)
    # Build a sunrise/sunset table per date
    dates = pd.to_datetime(index.date).unique()
    rows = []
    loc = LocationInfo(name=SITE_NAME, region=REGION, timezone=TIMEZONE, latitude=LATITUDE, longitude=LONGITUDE)
    for d in dates:
        # Astral expects a date; returns timezone-aware datetimes
        s = sun(loc.observer, date=pd.Timestamp(d).date(), tzinfo=tz)
        # Convert to naive local for comparison since our index is naive local
        rise = pd.Timestamp(s["sunrise"]).tz_convert(tz).tz_localize(None)
        set_  = pd.Timestamp(s["sunset"]).tz_convert(tz).tz_localize(None)
        rows.append({"date": pd.Timestamp(d), "sunrise": rise, "sunset": set_})
    day_tbl = pd.DataFrame(rows).set_index("date")

    # Map each timestamp to its date's sunrise/sunset
    # Night if t < sunrise on that date OR t >= sunset on that date
    # Handle times after local midnight but before sunrise: date alignment is correct
    def _is_night(ts):
        d = pd.Timestamp(ts.date())
        if d not in day_tbl.index:
            return False
        sr = day_tbl.loc[d, "sunrise"]
        ss = day_tbl.loc[d, "sunset"]
        return (ts < sr) or (ts >= ss)

    return pd.Series([_is_night(ts) for ts in index], index=index, name="is_night")

def is_night_mask_fallback(index: pd.DatetimeIndex) -> pd.Series:
    """Fallback fixed night hours if Astral is missing."""
    return pd.Series([(ts.hour >= FALLBACK_NIGHT_START_HOUR) or (ts.hour <= FALLBACK_NIGHT_END_HOUR)
                      for ts in index], index=index, name="is_night")

def rolling_quantile(series: pd.Series, hours: int, q: float) -> pd.Series:
    return series.rolling(f"{hours}h", min_periods=max(6, hours//4)).quantile(q)

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = pd.read_csv(FILE_PATH)

    # TIME parsing per your convention
    if "TIME" not in df.columns:
        raise ValueError("Expected a 'TIME' column in the CSV.")
    df["TIME"] = pd.to_datetime(df["TIME"], utc=True).dt.tz_convert(TIMEZONE).dt.tz_localize(None)

    # Exclusions
    for start_str, end_str in EXCLUSIONS:
        start = pd.Timestamp(start_str)
        end   = pd.Timestamp(end_str)
        df = df[(df["TIME"] < start) | (df["TIME"] > end)]

    df = df.set_index("TIME").sort_index()

    # Collapse duplicates
    if df.index.duplicated().any():
        dup_count = int(df.index.duplicated().sum())
        print(f"[info] Collapsing {dup_count} duplicate timestamp rows by mean.")
        df = df.groupby(level=0).mean(numeric_only=True)

    # Auto-detect concentration & uncertainty columns
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

    # Jump ratios vs t-1
    ratio = pd.DataFrame(index=conc.index)
    for m in TARGET_METALS:
        prev = conc[m].shift(1)
        ratio[m] = conc[m] / prev

    # Absolute floors
    if APPLY_ABS_FLOOR:
        abs_now = conc > (FLOOR_MULTIPLIER * unc)
        abs_prev = conc.shift(1) > (FLOOR_MULTIPLIER * unc.shift(1))
    else:
        abs_now = pd.DataFrame(True, index=conc.index, columns=conc.columns)
        abs_prev = pd.DataFrame(True, index=conc.index, columns=conc.columns)

    # SEED: jump condition
    jump_mask = (ratio >= JUMP_RATIO) & abs_now & abs_prev
    n_metals_jump = jump_mask[TARGET_METALS].sum(axis=1)
    is_seed = is_night & (n_metals_jump >= MIN_METALS_JUMP)

    # SUSTAIN: ER vs background quantile
    sustain_mask = pd.DataFrame(index=conc.index)
    for m in TARGET_METALS:
        sustain_mask[m] = (ER[f"ER_{m}"] >= SUSTAIN_ER) & abs_now[m]
    n_metals_sustain = sustain_mask[TARGET_METALS].sum(axis=1)
    is_sustain = is_night & (n_metals_sustain >= MIN_METALS_SUSTAIN)

    # Build events from seeds with GAP_TOL
    used = pd.Series(False, index=conc.index)
    events = []
    membership = pd.Series(False, index=conc.index, name="in_event")

    ts_list = conc.index.tolist()
    for t in ts_list:
        if not is_seed.get(t, False) or used.get(t, False):
            continue

        block = [t]
        used[t] = True
        gap_count = 0
        k = 1
        while True:
            nxt = t + pd.Timedelta(hours=1 * k)
            if nxt not in is_sustain.index or used.get(nxt, False):
                break
            if is_sustain.get(nxt, False):
                block.append(nxt); used[nxt] = True; gap_count = 0; k += 1
            else:
                if gap_count < GAP_TOL:
                    block.append(nxt); used[nxt] = True; gap_count += 1; k += 1
                else:
                    break

        if len(block) >= MIN_EVENT_DURATION_HOURS:
            membership.loc[block] = True
            seg = ER.loc[block]
            row = {
                "start": block[0],
                "end": block[-1],
                "duration_hours": len(block),
                "n_metals_jump_seed": int(n_metals_jump.loc[block[0]]),
                "max_metals_sustain": int(n_metals_sustain.loc[block].max()),
                "mean_metals_sustain": float(n_metals_sustain.loc[block].mean()),
            }
            for m in TARGET_METALS:
                er_col = f"ER_{m}"
                row[f"peak_ER_{m}"]   = float(seg[er_col].max(skipna=True))
                row[f"mean_ER_{m}"]   = float(seg[er_col].mean(skipna=True))
                row[f"median_ER_{m}"] = float(seg[er_col].median(skipna=True))
            events.append(row)

    events_df = pd.DataFrame(events).sort_values("start").reset_index(drop=True)

    # Nighttime detail export
    detail = pd.concat([
        conc.add_prefix("C_"),
        unc.add_prefix("U_"),
        ER,
        ratio.add_prefix("R_"),
        is_night.rename("is_night"),
        n_metals_jump.rename("n_metals_jump"),
        n_metals_sustain.rename("n_metals_sustain"),
        is_seed.rename("is_seed"),
        is_sustain.rename("is_sustain"),
        membership.rename("in_event"),
    ], axis=1)

    detail_night = detail[detail["is_night"]].copy()

    # Output
    base = os.path.splitext(os.path.basename(FILE_PATH))[0]
    events_path = os.path.join(OUTPUT_DIR, f"{base}_events_v31_astral.csv")
    hours_path  = os.path.join(OUTPUT_DIR, f"{base}_hours_v31_astral.csv")

    events_df.to_csv(events_path, index=False)
    detail_night.to_csv(hours_path, index=True)

    # Console summary
    print(f"\nDetected {events_df.shape[0]} events (min {MIN_EVENT_DURATION_HOURS} h, GAP_TOL={GAP_TOL}).")
    if not events_df.empty:
        sort_col = "mean_ER_Zn" if "mean_ER_Zn" in events_df.columns else events_df.columns[0]
        print("Top 5 events by mean_ER_Zn:")
        print(events_df.sort_values(sort_col, ascending=False).head(5).to_string(index=False))
    print(f"\nWrote:\n  - {events_path}\n  - {hours_path}\n")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("ERROR:", e)
        sys.exit(1)
