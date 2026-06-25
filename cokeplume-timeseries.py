# -*- coding: utf-8 -*-
"""
Make time-series plots for Zn, As, Se, Pb over specific 24h windows.
- Individual plots: one metal per figure (dots with error bars)
- Combined plots: Zn/As/Se/Pb together on log-y (with error bars)
"""

import os
import warnings
from typing import List, Tuple

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

# Requested 24h windows (start, end) in local time
WINDOWS = [
    ("2024-08-10 16:00", "2024-08-11 18:00"),
    # ("2023-11-02 18:00", "2023-11-03 16:00"),
    # ("2025-09-20 18:00", "2025-09-21 16:00"),
    # ("2024-05-10 18:00", "2024-05-11 16:00"),
    # ("2023-07-10 18:00", "2023-07-11 16:00"),

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
    # find uncertainty column containing the element token
    ucol = next((u for u in unc_cols if element in u), None)
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
def slice_window(dfw: pd.DataFrame, start_s: str, end_s: str) -> pd.DataFrame:
    st = pd.Timestamp(start_s, tz=TZ)
    et = pd.Timestamp(end_s,   tz=TZ)
    return dfw.loc[(dfw.index >= st) & (dfw.index <= et)].copy(), st, et

def plot_one_metal(win: pd.DataFrame, metal: str, outdir_win: str):
    """Plot a single metal as dots with error bars."""
    ccol, ucol = metals_cols[metal], uncert_cols[metal]
    if ccol not in win or ucol not in win:
        return None
    vmask = ~(win[ccol].isna() | win[ucol].isna())
    if not vmask.any():
        return None

    times = win.index[vmask]
    conc  = win.loc[vmask, ccol].astype(float)
    unc   = win.loc[vmask, ucol].astype(float)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.errorbar(times, conc, yerr=unc, fmt='o', ms=3.5, elinewidth=1, capsize=2)
    ax.set_title(f"{metal} (ng/m³)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Concentration (ng/m³)")
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis="x", rotation=25)
    if conc.min() >= 0:
        ax.set_ylim(bottom=0)

    fname = f"{metal}.png"
    path = os.path.join(outdir_win, fname)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path

def plot_combined_linear_scaled(win: pd.DataFrame, metals: List[str], outdir_win: str, title_suffix: str):
    """Combined linear plot of multiple metals (Zn scaled down by 10)."""
    fig, ax = plt.subplots(figsize=(7, 3))
    plotted = 0

    for m in metals:
        ccol, ucol = metals_cols[m], uncert_cols[m]
        if ccol not in win or ucol not in win:
            continue
        vmask = ~(win[ccol].isna() | win[ucol].isna())
        if not vmask.any():
            continue

        times = win.index[vmask]
        conc = win.loc[vmask, ccol].astype(float)
        unc  = win.loc[vmask, ucol].astype(float)

        # scale Zn
        if m == "Zn":
            conc = conc / 10.0
            unc  = unc / 10.0
            label = "Zn (÷10)"
        else:
            label = m

        ax.errorbar(times, conc, yerr=unc, fmt='o-', ms=3, elinewidth=1, capsize=2, label=label)
        plotted += 1

    if plotted == 0:
        plt.close(fig)
        return None

    ax.set_ylabel("Concentration (ng/m³)")
    ax.set_title(f"Zn, As, Se, Pb — {title_suffix}", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.25)
    ax.tick_params(axis="x", rotation=25)
    ax.legend(ncol=1, fontsize=9)

    fname = f"combined_Zn_As_Se_Pb_scaled.png"
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

    # Per-metal plots
    for m in present:
        p = plot_one_metal(win, m, outdir_win)
        if p:
            print(f"  • saved {os.path.basename(p)}")

    # Combined y
    title_suffix = f"{st:%Y-%m-%d %H:%M} → {et:%Y-%m-%d %H:%M} {TZ}"
    p = plot_combined_linear_scaled(win, present, outdir_win, title_suffix)
    if p:
        print(f"  • saved {os.path.basename(p)}")

print("\n✅ Done.")
