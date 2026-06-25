
# -*- coding: utf-8 -*-
"""
Month QA Plotter — 4 PM → 4 PM windows with Astral night shading + Event overlay + Legend
- Input: Raw Xact CSV (TIME in UTC); auto-pairs metal conc & uncertainty columns
- Select a month (YYYY-MM), e.g., "2024-10"
- For each 4 PM → next day 4 PM window within that month:
    * Plot Zn, As, Se, Pb together on one panel with uncertainty bars
    * Shade true night (Astral sunrise/sunset; fallback to fixed 20:00–06:00 if Astral missing)
    * Overlay events from v21q2 events CSV
    * Legend shown INSIDE (top-right), including shaded patch entries

Saves one PNG per window in an output folder.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# ----------------------- USER CONFIG -----------------------
FILE_PATH = r"Xact_EST_May2023_Oct2025_combined.csv"  # raw CSV
OUTDIR = r"./qa_month_4pm_windows"

# Choose which month to plot (YYYY-MM)
MONTH = "2024-09"   # e.g., "2024-09", "2024-10"

# Timezone & site for Astral
TIMEZONE = "US/Eastern"
LATITUDE = 40.4406
LONGITUDE = -79.9959
SITE_NAME = "Pittsburgh"
REGION    = "USA"

# Metals and visuals
TARGET_METALS = ["Zn", "As", "Se", "Pb"]
SCALE_ZN_BY_10 = True   # divide Zn & its uncertainty by 10 for visual balance
USE_LOG_Y = False       # set True to use log-y axis

# Event file (detector output)
EVENTS_CSV = r"./coke_plume_flags/Xact_EST_May2023_Oct2025_combined_events_v21q2_rng_2024-09-01_2024-09-30.csv"

# Optional exclusions (same as detector); empty list if none
EXCLUSIONS = [
    ("2024-01-09", "2024-02-13"),
    ("2024-07-02", "2024-08-08"),
]

# Shading colors
NIGHT_COLOR = "#9ecae1"   # light blue
NIGHT_ALPHA = 0.08
EVENT_COLOR = "#fdae6b"   # light orange
EVENT_ALPHA = 0.18
# -----------------------------------------------------------

def has_astral():
    try:
        import astral  # noqa: F401
        return True
    except Exception:
        return False

def night_mask_astral(index: pd.DatetimeIndex, tz_str: str) -> pd.Series:
    try:
        from astral import LocationInfo
        from astral.sun import sun
        import pytz
    except Exception:
        return night_mask_fallback(index)

    tz = pytz.timezone(tz_str)
    loc = LocationInfo(name=SITE_NAME, region=REGION, timezone=tz_str, latitude=LATITUDE, longitude=LONGITUDE)
    dates = pd.to_datetime(index.date).unique()
    rows = []
    for d in dates:
        s = sun(loc.observer, date=pd.Timestamp(d).date(), tzinfo=tz)
        rise = pd.Timestamp(s["sunrise"]).tz_convert(tz).tz_localize(None)
        set_ = pd.Timestamp(s["sunset"]).tz_convert(tz).tz_localize(None)
        rows.append({"date": pd.Timestamp(d), "sunrise": rise, "sunset": set_})
    tbl = pd.DataFrame(rows).set_index("date")

    def _is_night(ts):
        d = pd.Timestamp(ts.date())
        if d not in tbl.index:
            return False
        return (ts < tbl.at[d, "sunrise"]) or (ts >= tbl.at[d, "sunset"])

    return pd.Series([_is_night(ts) for ts in index], index=index, name="is_night")

def night_mask_fallback(index: pd.DatetimeIndex) -> pd.Series:
    return pd.Series([(ts.hour >= 20) or (ts.hour <= 6) for ts in index], index=index, name="is_night")

def pair_conc_unc(df: pd.DataFrame, target_metals):
    headers = df.columns.tolist()
    conc_cols = [c for c in headers if (" (ng/m3)" in c) and ("uncert" not in c.lower())]
    unc_cols  = [c for c in headers if ("uncert" in c.lower())]
    conc_map, unc_map = {}, {}
    for m in target_metals:
        cand = [c for c in conc_cols if (f"{m} " in c)]
        if cand:
            conc_map[m] = cand[0]
            unc_map[m]  = next((u for u in unc_cols if m in u), None)
    return conc_map, unc_map

def windows_4pm(month_str: str, tz: str):
    # compute first day and next month in local naive time
    start = pd.Timestamp(f"{month_str}-01", tz=tz).tz_localize(None)
    y, m = start.year, start.month
    next_month = pd.Timestamp(f"{y+1}-01-01", tz=tz).tz_localize(None) if m == 12 else pd.Timestamp(f"{y}-{m+1:02d}-01", tz=tz).tz_localize(None)
    # 4pm boundaries
    cur = start.replace(hour=16, minute=0, second=0, microsecond=0)
    if cur < start:
        cur += pd.Timedelta(days=1)
    stops = []
    while cur <= next_month:
        stops.append(cur)
        cur += pd.Timedelta(days=1)
    wins = [(s, min(s + pd.Timedelta(hours=24), next_month)) for s in stops]
    return wins

def main():
    os.makedirs(OUTDIR, exist_ok=True)

    # Load raw CSV
    df = pd.read_csv(FILE_PATH)
    if "TIME" not in df.columns:
        raise ValueError("Expected a 'TIME' column.")
    df["TIME"] = pd.to_datetime(df["TIME"], utc=True).dt.tz_convert(TIMEZONE).dt.tz_localize(None)
    # exclusions
    for s, e in EXCLUSIONS:
        df = df[(df["TIME"] < pd.Timestamp(s)) | (df["TIME"] > pd.Timestamp(e))]
    df = df.set_index("TIME").sort_index()

    # Pair columns
    conc_map, unc_map = pair_conc_unc(df, TARGET_METALS)
    present = [m for m in TARGET_METALS if m in conc_map]
    if not present:
        raise RuntimeError("None of the TARGET_METALS were found in the file.")

    # Build a working frame with only present metals (conc + unc)
    cols = []
    for m in present:
        cols += [conc_map[m]]
        if unc_map[m]:
            cols += [unc_map[m]]
    d = df[cols].copy()

    # Make 4pm windows for this month
    wins = windows_4pm(MONTH, TIMEZONE)
    print(f"Will plot {len(wins)} windows for {MONTH} (4 PM → 4 PM).")

    # Load events (optional but encouraged)
    ev = None
    if EVENTS_CSV and os.path.exists(EVENTS_CSV):
        ev = pd.read_csv(EVENTS_CSV, parse_dates=["start","end"])
    else:
        print(f"[info] EVENTS_CSV not found or not set: {EVENTS_CSV}")

    # Precompute night mask at the dataframe's time index
    idx = d.index
    if has_astral():
        night = night_mask_astral(idx, TIMEZONE)
    else:
        print("[warn] 'astral' not installed; using fixed 20:00–06:00 night.")
        night = night_mask_fallback(idx)

    month_dir = os.path.join(OUTDIR, MONTH.replace(":", "-"))
    os.makedirs(month_dir, exist_ok=True)

    for s, e in wins:
        sub = d.loc[(d.index >= s) & (d.index <= e)].copy()
        if sub.empty:
            continue

        fig, ax = plt.subplots(figsize=(12, 4.4))

        # Shade night segments (blue tint)
        nsub = night.loc[sub.index]
        if not nsub.empty:
            in_seg = False
            seg_start = None
            for t, is_n in nsub.items():
                if is_n and not in_seg:
                    in_seg = True; seg_start = t
                elif not is_n and in_seg:
                    ax.axvspan(seg_start, t, color=NIGHT_COLOR, alpha=NIGHT_ALPHA)
                    in_seg = False
            if in_seg:
                ax.axvspan(seg_start, sub.index[-1] + pd.Timedelta(hours=1), color=NIGHT_COLOR, alpha=NIGHT_ALPHA)

        # Overlay events (orange tint)
        if ev is not None and not ev.empty:
            for _, r in ev.iterrows():
                es = max(r["start"], s)
                ee = min(r["end"] + pd.Timedelta(hours=1), e)
                if es < ee:
                    ax.axvspan(es, ee, color=EVENT_COLOR, alpha=EVENT_ALPHA)

        # Plot metals with uncertainty bars
        line_handles = []
        line_labels = []
        for m in present:
            ccol, ucol = conc_map[m], unc_map[m]
            c = sub[ccol].astype(float)
            u = sub[ucol].astype(float) if ucol in sub.columns else None
            if m == "Zn" and SCALE_ZN_BY_10:
                c = c / 10.0
                if u is not None: u = u / 10.0
                label = "Zn (÷10)"
            else:
                label = m
            if u is not None:
                lh = ax.errorbar(sub.index, c, yerr=u, fmt='.-', ms=2.8, elinewidth=0.8, capsize=2, label=label, alpha=0.9)
                line_handles.append(lh)
                line_labels.append(label)
            else:
                lh = ax.plot(sub.index, c, '.-', ms=2.8, label=label, alpha=0.9)[0]
                line_handles.append(lh)
                line_labels.append(label)

        if USE_LOG_Y:
            ax.set_yscale("log")

        ax.set_title(f"{MONTH} | {s:%Y-%m-%d 16:00} → {e:%Y-%m-%d 16:00} ({TIMEZONE})")
        ax.set_ylabel("Concentration (ng/m³)")
        ax.grid(True, axis="y", alpha=0.25)
        ax.set_xlim(s, e)

        # 4 PM boundaries
        ax.axvline(s, linestyle='--', alpha=0.25)
        ax.axvline(e, linestyle='--', alpha=0.25)

        # Build legend (top-right) with shaded patches
        patch_night = Patch(facecolor=NIGHT_COLOR, edgecolor='none', alpha=NIGHT_ALPHA, label="Nighttime")
        patch_event = Patch(facecolor=EVENT_COLOR, edgecolor='none', alpha=EVENT_ALPHA, label="Detected event")
        handles = line_handles + [patch_night, patch_event]
        labels  = line_labels  + ["Nighttime", "Detected event"]
        leg = ax.legend(handles, labels, loc='upper right', framealpha=0.85, fontsize=9)
        for lh in leg.legendHandles:
            try:
                lh.set_linewidth(2.0)
            except Exception:
                pass

        # Save
        fname = f"{s:%Y%m%d_1600}__to__{e:%Y%m%d_1600}.png"
        path = os.path.join(month_dir, fname)
        fig.tight_layout()
        fig.savefig(path, dpi=170, facecolor="white", bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {path}")

    print("Done.")

if __name__ == "__main__":
    main()
