# -*- coding: utf-8 -*-
"""
Make time-series plots for Zn, As, Se, Pb over specific 24h windows (4pm-3pm).
- Individual plots: one metal per figure (dots with error bars)
- Combined plots: Zn/As/Se/Pb together (with error bars)
- 24-hour average displayed as tick mark on secondary y-axis
"""

import os
import warnings
from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ----------------------- USER CONFIG -----------------------
FILE_PATH = "Xact_EST_May2023_Oct2025_combined.csv"
TZ = "US/Eastern"
OUTDIR = "./weird_peaks_plots"
APPLY_3SIGMA_FILTER = False  # keep False to avoid hiding subtle peaks

# Metals of interest
TARGET_METALS = ["Zn", "As", "Se", "Pb"]

# Requested 24h windows (start, end) in local time - 4pm to 3pm next day
WINDOWS = [
    ("2024-10-22 16:00", "2024-10-23 15:00"),
    # ("2025-03-18 16:00", "2025-03-19 15:00"),
    # ("2025-03-10 16:00", "2025-03-11 15:00"),
    # ("2024-09-08 16:00", "2024-09-09 15:00"),
    # ("2024-10-04 16:00", "2024-10-05 15:00"),
    # ("2023-09-05 16:00", "2023-09-06 15:00"),
]

# ----------------------- LOAD & PAIR -----------------------
df = pd.read_csv(FILE_PATH)
df["TIME"] = pd.to_datetime(df["TIME"], utc=True).dt.tz_convert(TZ)
df = df.set_index("TIME").sort_index()

headers = df.columns.tolist()
conc_cols = [c for c in headers if " (ng/m3)" in c and "uncert" not in c.lower()]
unc_cols = [c for c in headers if "uncert" in c.lower()]

# map element -> (conc_col, uncert_col)
metals_cols = {}
uncert_cols = {}

for conc_col in conc_cols:
    element = conc_col.split()[0]  # "Zn 30 (ng/m3)" -> "Zn"
    ucol = next((u for u in unc_cols if u.startswith(element)), None)
    if ucol:
        metals_cols[element] = conc_col
        uncert_cols[element] = ucol

# manual sulfur fix if relevant (safe no-op otherwise)
if "S 16 (ng/m3)" in df.columns and "S Uncert (ng/m3)" in df.columns:
    metals_cols["S"] = "S 16 (ng/m3)"
    uncert_cols["S"] = "S Uncert (ng/m3)"

# keep only targets that exist
present = [m for m in TARGET_METALS if m in metals_cols]
missing = [m for m in TARGET_METALS if m not in metals_cols]
if missing:
    print(f"⚠️ Missing metal columns for: {', '.join(missing)} (skipping)")

# optional 3×σ filter
df_work = df.copy()
if APPLY_3SIGMA_FILTER:
    print("Applying 3× uncertainty filtering …")
    for m in present:
        ccol, ucol = metals_cols[m], uncert_cols[m]
        c = df_work[ccol]
        u = df_work[ucol]
        mask = (c > 3.0 * u) | c.isna()
        df_work[ccol] = c.where(mask)
else:
    print("3× uncertainty filtering DISABLED.")

os.makedirs(OUTDIR, exist_ok=True)


# ----------------------- HELPERS -----------------------
def slice_window(dfw: pd.DataFrame, start_s: str, end_s: str):
    st = pd.Timestamp(start_s, tz=TZ)
    et = pd.Timestamp(end_s, tz=TZ)
    return dfw.loc[(dfw.index >= st) & (dfw.index <= et)].copy(), st, et


def calculate_window_avg(win: pd.DataFrame, metal: str):
    """Calculate the average concentration for the entire window."""
    ccol = metals_cols[metal]
    values = win[ccol].astype(float).dropna()
    if len(values) > 0:
        return values.mean()
    return np.nan


# ---------- Plot: per-metal (raw with error bars + 24h avg tick) ----------
def plot_one_metal_with_avg(win: pd.DataFrame, metal: str, outdir_win: str):
    ccol, ucol = metals_cols[metal], uncert_cols[metal]
    if ccol not in win or ucol not in win:
        return None
    vmask = ~(win[ccol].isna() | win[ucol].isna())
    if not vmask.any():
        return None

    times = win.index[vmask]
    conc = win.loc[vmask, ccol].astype(float)
    unc = win.loc[vmask, ucol].astype(float)

    # Calculate 24h average
    avg_24h = calculate_window_avg(win, metal)

    # consistent color
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    color = color_cycle[0]

    fig, ax = plt.subplots(figsize=(8, 4.5))

    # --- hourly data: connected line + markers + error bars ---
    ax.errorbar(
        times, conc, yerr=unc,
        fmt='o-', ms=3.5, lw=1.2, elinewidth=1, capsize=2,
        color=color, alpha=1.0,
        label="Hourly"
    )

    ax.set_ylabel("Concentration (ng/m³)", color=color)
    ax.tick_params(axis='y', labelcolor=color)
    ax.set_title(f"{metal} (ng/m³) — 24h average: {avg_24h:.2f}",
                 fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis="x", rotation=25)
    if np.isfinite(conc.min()) and conc.min() >= 0:
        ax.set_ylim(bottom=0)

    # --- Add secondary y-axis with tick mark for 24h average ---
    if np.isfinite(avg_24h):
        ax2 = ax.twinx()
        ax2.set_ylim(ax.get_ylim())  # Match primary axis limits
        ax2.set_yticks([avg_24h])
        ax2.set_yticklabels([f'{avg_24h:.2f}'], color='red', fontweight='bold')
        ax2.tick_params(axis='y', colors='red', length=10, width=2)
        ax2.set_ylabel('24h Avg', color='red', fontweight='bold')

    fname = f"{metal}_with_avg.png"
    path = os.path.join(outdir_win, fname)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def plot_combined_with_avg(win: pd.DataFrame, metals: List[str],
                           outdir_win: str, title_suffix: str):
    fig, ax = plt.subplots(figsize=(6, 2.5))
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    color_map = {m: color_cycle[i % len(color_cycle)] for i, m in enumerate(metals)}

    # Calculate averages for all metals
    averages = {}
    for m in metals:
        avg = calculate_window_avg(win, m)
        if m == "Zn":
            avg = avg / 10.0 if np.isfinite(avg) else np.nan
        averages[m] = avg

    for m in metals:
        ccol, ucol = metals_cols[m], uncert_cols[m]
        s = win[ccol].astype(float)
        u = win[ucol].astype(float)
        clr = color_map[m]

        # scale Zn
        s_plot = s / 10.0 if m == "Zn" else s
        u_plot = u / 10.0 if m == "Zn" else u

        # --- hourly data: line + markers + error bars ---
        raw_label = f"{m}" if m != "Zn" else "Zn (÷10)"
        # if np.isfinite(averages[m]):
        #     raw_label += f" (24h avg: {averages[m]:.2f})"

        ax.errorbar(
            win.index, s_plot, yerr=u_plot,
            fmt='o-', ms=3, lw=1.0, elinewidth=1, capsize=2,
            color=clr, alpha=0.8,
            label=raw_label
        )

    ax.set_ylabel("Concentration (ng/m³)")
    ax.set_xlabel("Local Time (US Eastern)")
    # ax.set_title(f"Hourly measurements — {title_suffix}",
    #              fontsize=12, fontweight="bold")
    # ax.grid(True, alpha=0.25)
    ax.tick_params(axis="x", rotation=25)
    ax.legend(ncol=1, fontsize=9, frameon=True)

    # --- Add secondary y-axis with tick marks for all 24h averages ---
    ax2 = ax.twinx()
    ax2.set_ylim(ax.get_ylim())

    valid_avgs = [avg for avg in averages.values() if np.isfinite(avg)]
    if valid_avgs:
        ax2.set_yticks(valid_avgs)
        labels = []
        for m in metals:
            if np.isfinite(averages[m]):
                label = f"{m}" if m != "Zn" else "Zn(÷10)"
                labels.append(f"{label}: {averages[m]:.1f}")
        ax2.set_yticklabels(labels, fontsize=7,fontweight='bold')

        # Color the tick labels by metal
        for tick_label, m in zip(ax2.get_yticklabels(),
                                 [m for m in metals if np.isfinite(averages[m])]):
            tick_label.set_color(color_map[m])
            # Move Pb label up slightly to avoid overlap with Se
            if m == "Pb":
                tick_label.set_verticalalignment('bottom')

        ax2.set_ylabel('24h Avg (ng/m³)', fontsize=10)
        ax2.tick_params(axis='y', length=5, width=1.5)  # Normal horizontal ticks

    fname = "combined_Zn_As_Se_Pb_with_avg.png"
    path = os.path.join(outdir_win, fname)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


# ----------------------- RUN -----------------------
for (start_s, end_s) in WINDOWS:
    win, st, et = slice_window(df_work, start_s, end_s)

    label = f"{st:%Y%m%d_%H%M}_to_{et:%Y%m%d_%H%M}"
    outdir_win = os.path.join(OUTDIR, label)
    os.makedirs(outdir_win, exist_ok=True)

    print(f"\n🗓️ {st} → {et}  ({len(win)} rows) → {outdir_win}")

    # Per-metal plots with 24h average tick
    for m in present:
        p_avg = plot_one_metal_with_avg(win, m, outdir_win)
        if p_avg:
            print(f"  • saved {os.path.basename(p_avg)}")

    # Combined plot with 24h average ticks
    title_suffix = f"{st:%Y-%m-%d %H:%M} → {et:%Y-%m-%d %H:%M} {TZ}"
    p_comb = plot_combined_with_avg(win, present, outdir_win, title_suffix)
    print(f"  • saved {os.path.basename(p_comb)}")

print("\n✅ All plots generated!")