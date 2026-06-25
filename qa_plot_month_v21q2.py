# -*- coding: utf-8 -*-
"""
QA Plot for Coke Plume Detector (v21q2-compatible)
- Time range filter
- 4 metals (Pb, Zn, Se, As) with hourly uncertainty bars
- Night shading via Astral (fallback to fixed hours)
- Event shading from events CSV
- Day boundaries at 16:00 local

Usage:
  1) Run detector to produce:
     - *_hours_full_v21q2[...].csv
     - *_events_v21q2[...].csv
  2) Set INPUT_* paths and date range below, then:
     python qa_plot_month_v21q2.py
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta

# -------------------- USER CONFIG --------------------
INPUT_HOURS_FULL = r"./coke_plume_flags/Xact_EST_May2023_Oct2025_combined_hours_full_v21q2_rng_2024-10-01_2024-10-31.csv"  # adjust
INPUT_EVENTS     = r"./coke_plume_flags/XXact_EST_May2023_Oct2025_combined_events_v21q2_rng_2024-10-01_2024-10-31.csv"       # adjust
OUTPUT_PNG       = r"./coke_plume_flags/qa_plot_v21q2.png"

TIMEZONE   = "US/Eastern"
LATITUDE   = 40.4406
LONGITUDE  = -79.9959
SITE_NAME  = "Pittsburgh"
REGION     = "USA"

# Optional date filter for plotting (inclusive)
START_DATE = "2024-10-01"   # e.g., "2024-09-01"
END_DATE   = "2024-10-31"   # e.g., "2024-09-30"

TARGET_METALS = ["Pb", "Zn", "Se", "As"]
# ----------------------------------------------------

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
    return pd.Series([(ts.hour >= 20) or (ts.hour <= 6) for ts in index], index=index, name="is_night")

def main():
    # Load hours_full (must include C_* and U_* columns for TARGET_METALS)
    df = pd.read_csv(INPUT_HOURS_FULL, parse_dates=True, index_col=0)
    df.index = pd.to_datetime(df.index)

    # Date filter for plotting
    if START_DATE is not None:
        df = df[df.index >= pd.Timestamp(START_DATE)]
    if END_DATE is not None:
        df = df[df.index <= pd.Timestamp(END_DATE) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)]

    # Night mask (recompute to ensure shading even if file lacks it)
    if has_astral():
        is_night = is_night_mask_astral(df.index)
    else:
        print("[warn] 'astral' not installed; using fallback fixed hours for night.")
        is_night = is_night_mask_fallback(df.index)

    # Load events for shading
    ev = pd.read_csv(INPUT_EVENTS, parse_dates=['start','end']) if os.path.exists(INPUT_EVENTS) else pd.DataFrame()

    # Prepare figure
    fig, axes = plt.subplots(len(TARGET_METALS), 1, figsize=(14, 9), sharex=True, constrained_layout=True)
    if len(TARGET_METALS) == 1:
        axes = [axes]

    # Shade night
    # Build contiguous night segments
    night_segments = []
    prev = None
    for t, val in is_night.items():
        if val and prev is None:
            start = t
            prev = True
        elif not val and prev:
            night_segments.append((start, t))
            prev = False
        elif val and prev and (t - last_t) > pd.Timedelta(hours=1):
            # handle gaps
            night_segments.append((start, last_t + pd.Timedelta(hours=1)))
            start = t
        last_t = t
    if is_night.iloc[-1]:
        night_segments.append((start, is_night.index[-1] + pd.Timedelta(hours=1)))

    # Shade events
    event_segments = []
    if not ev.empty:
        for _, r in ev.iterrows():
            # clip to plot range
            s = max(r['start'], df.index.min())
            e = min(r['end'] + pd.Timedelta(hours=1), df.index.max() + pd.Timedelta(hours=1))
            if s < e:
                event_segments.append((s, e))

    # Plot metals with error bars
    for ax, m in zip(axes, TARGET_METALS):
        c = df.get(f"C_{m}")
        u = df.get(f"U_{m}")
        if c is None:
            ax.set_visible(False)
            continue

        # Night shading
        for (s, e) in night_segments:
            ax.axvspan(s, e, alpha=0.08, label=None)

        # Event shading (slightly darker)
        for (s, e) in event_segments:
            ax.axvspan(s, e, alpha=0.15, label=None)

        # Error bars
        if u is not None:
            ax.errorbar(df.index, c, yerr=u, fmt='.', markersize=3, linewidth=0.6, ecolor='gray', alpha=0.7)
        else:
            ax.plot(df.index, c, '.', markersize=3, alpha=0.7)

        ax.set_ylabel(f"{m} (ng/m³)")
        ax.grid(True, which="both", axis="y", alpha=0.2)

    # 4 PM day boundaries
    # Create ticks at every day 16:00 in the range
    start = df.index.min().floor('D') + pd.Timedelta(hours=16)
    end   = df.index.max().ceil('D') + pd.Timedelta(hours=16)
    cur = start
    while cur <= end:
        for ax in axes:
            ax.axvline(cur, linestyle='--', alpha=0.2)
        cur += pd.Timedelta(days=1)

    axes[0].set_title("QA Plot — Metals with night & event shading (day boundary at 4 PM)")

    axes[-1].set_xlabel("Local time")
    fig.savefig(OUTPUT_PNG, dpi=160)
    print(f"Saved plot: {OUTPUT_PNG}")

if __name__ == "__main__":
    main()
