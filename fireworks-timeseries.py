# -*- coding: utf-8 -*-
"""
Time-series with error bars for K, Cu, Ba, Sr over two 7-day windows.
- Per-metal plots: dots + error bars
- Combined plot per window: all 4 metals (K shown ÷100 by default)
- No masking/filtering (to preserve small/weird peaks)

Outputs:
  ./metal_timeseries/<window_label>/
    - K.png, Cu.png, Ba.png, Sr.png
    - combined_K_Cu_Ba_Sr_scaled.png
"""

import os
import warnings
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ----------------------- CONFIG -----------------------
FILE_PATH = "Xact_EST_May2023_Oct2025_combined.csv"   # adjust if needed
TZ = "US/Eastern"
OUTDIR = "./fireworks_timeseries"

# Metals of interest
TARGET_METALS = ["K", "Cu", "Ba", "Sr"]

# Two 7-day windows (inclusive). End at 23:59:59 for full-day coverage.
WINDOWS: List[Tuple[str, str]] = [
    ("2023-06-27 00:00:00", "2023-07-07 23:59:59"),
    ("2025-07-01 00:00:00", "2025-07-07 23:59:59"),
]

# Scaling (apply as value * SCALE[metal]). K ÷100 => 0.01
# You can add per-window overrides below if you want different scales by window.
SCALE: Dict[str, float] = {"K": 0.1, "Cu": 1.0, "Ba": 1.0, "Sr": 1.0}

# Optional per-window overrides, keyed by window label (see label construction below).
# Example:
# WINDOW_SCALE_OVERRIDES = {
#     "20230701_0000_to_20230707_235959": {"Cu": 0.5, "Ba": 2.0}
# }
WINDOW_SCALE_OVERRIDES: Dict[str, Dict[str, float]] = {}

# Plot aesthetics
POINT_SIZE = 3.5
ELINEWIDTH = 1.0
CAPSIZE = 2
LINEWIDTH = 1.4

# ----------------------- LOAD & PAIR -----------------------
df = pd.read_csv(FILE_PATH)
df["TIME"] = pd.to_datetime(df["TIME"], utc=True).dt.tz_convert(TZ)
df = df.set_index("TIME").sort_index()

headers = df.columns.tolist()
conc_cols = [c for c in headers if " (ng/m3)" in c and "uncert" not in c.lower()]
unc_cols = [c for c in headers if "uncert" in c.lower()]

# Map element -> (conc_col, uncert_col) by token match
metals_cols: Dict[str, str] = {}
uncert_cols: Dict[str, str] = {}

for conc_col in conc_cols:
    element = conc_col.split()[0]  # e.g., "K 19 (ng/m3)" -> "K"
    ucol = next((u for u in unc_cols if element in u), None)
    if ucol:
        metals_cols[element] = conc_col
        uncert_cols[element] = ucol

# Manual sulfur fix (safe no-op if absent)
if "S 16 (ng/m3)" in df.columns and "S Uncert (ng/m3)" in df.columns:
    metals_cols["S"] = "S 16 (ng/m3)"
    uncert_cols["S"] = "S Uncert (ng/m3)"

present = [m for m in TARGET_METALS if m in metals_cols]
missing = [m for m in TARGET_METALS if m not in metals_cols]
if missing:
    print(f"⚠️ Missing columns for: {', '.join(missing)} (skipping them)")
if not present:
    raise SystemExit("None of the target metals were found in the dataset.")

os.makedirs(OUTDIR, exist_ok=True)

# ----------------------- HELPERS -----------------------
def slice_window(dfw: pd.DataFrame, start_s: str, end_s: str) -> tuple[pd.DataFrame, pd.Timestamp, pd.Timestamp]:
    st = pd.Timestamp(start_s, tz=TZ)
    et = pd.Timestamp(end_s, tz=TZ)
    return dfw.loc[(dfw.index >= st) & (dfw.index <= et)].copy(), st, et

def scale_for_label(m: str, sc: float) -> str:
    """Return nice legend label based on scale factor."""
    if np.isclose(sc, 1.0):
        return m
    # express scale as ÷ or × where clean
    if sc < 1.0:
        inv = 1.0 / sc
        if abs(inv - round(inv)) < 1e-9:   # integer divisor like 10, 100, etc.
            return f"{m} (÷{int(round(inv))})"
    return f"{m} (×{sc:g})"

def plot_one_metal(win: pd.DataFrame, metal: str, outdir_win: str, scale: float = 1.0):
    """Single-metal plot: dots + error bars (scaled)."""
    ccol, ucol = metals_cols[metal], uncert_cols[metal]
    if ccol not in win or ucol not in win:
        return None
    vmask = ~(win[ccol].isna() | win[ucol].isna())
    if not vmask.any():
        return None

    times = win.index[vmask]
    conc  = win.loc[vmask, ccol].astype(float) * scale
    unc   = win.loc[vmask, ucol].astype(float) * scale

    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.errorbar(times, conc, yerr=unc, fmt='o-', ms=POINT_SIZE,
                elinewidth=ELINEWIDTH, capsize=CAPSIZE, linewidth=LINEWIDTH)
    ax.set_title(f"{scale_for_label(metal, scale)} (ng/m³)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Concentration (scaled ng/m³)")
    # ax.grid(True, alpha=0.3)
    ax.tick_params(axis="x", rotation=25)
    if conc.min() >= 0:
        ax.set_ylim(bottom=0)

    fname = f"{metal}.png"
    path = os.path.join(outdir_win, fname)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path

def plot_combined(win: pd.DataFrame, metals: List[str], outdir_win: str, title_suffix: str, scales: Dict[str, float]):
    """Combined plot: dots + error bars for multiple metals with per-metal scaling."""
    fig, ax = plt.subplots(figsize=(7, 3))
    plotted = 0
    colors = plt.cm.tab10(np.linspace(0, 1, max(4, len(metals))))

    for i, m in enumerate(metals):
        ccol, ucol = metals_cols[m], uncert_cols[m]
        if ccol not in win or ucol not in win:
            continue
        vmask = ~(win[ccol].isna() | win[ucol].isna())
        if not vmask.any():
            continue

        times = win.index[vmask]
        sc    = float(scales.get(m, 1.0))
        conc  = win.loc[vmask, ccol].astype(float) * sc
        unc   = win.loc[vmask, ucol].astype(float) * sc

        c = colors[i % len(colors)]
        ax.errorbar(times, conc, yerr=unc, fmt='o-', ms=POINT_SIZE,
                    elinewidth=ELINEWIDTH, capsize=CAPSIZE, linewidth=LINEWIDTH,
                    color=c, label=scale_for_label(m, sc))
        plotted += 1

    if plotted == 0:
        plt.close(fig)
        return None

    ax.set_ylabel("Concentration (scaled ng/m³)")
    ax.set_title(f"K, Cu, Ba, Sr — {title_suffix}", fontsize=12, fontweight="bold")
    # ax.grid(False, alpha=0.25)
    ax.tick_params(axis="x", rotation=25)
    ax.legend(ncol=1, fontsize=9)

    fname = f"combined_K_Cu_Ba_Sr_scaled.png"
    path = os.path.join(outdir_win, fname)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path

# ----------------------- RUN -----------------------
for (start_s, end_s) in WINDOWS:
    win, st, et = slice_window(df, start_s, end_s)
    label = f"{st:%Y%m%d_%H%M%S}_to_{et:%Y%m%d_%H%M%S}"
    outdir_win = os.path.join(OUTDIR, label)
    os.makedirs(outdir_win, exist_ok=True)

    print(f"\n🗓️ {st} → {et}  ({len(win)} rows) → {outdir_win}")

    # Resolve scales (apply per-window overrides if provided)
    scales = SCALE.copy()
    if label in WINDOW_SCALE_OVERRIDES:
        scales.update(WINDOW_SCALE_OVERRIDES[label])

    # Per-metal plots
    for m in present:
        p = plot_one_metal(win, m, outdir_win, scale=scales.get(m, 1.0))
        if p:
            print(f"  • saved {os.path.basename(p)}")

    # Combined plot
    title_suffix = f"{st:%Y-%m-%d %H:%M} → {et:%Y-%m-%d %H:%M} {TZ}"
    p = plot_combined(win, present, outdir_win, title_suffix, scales)
    if p:
        print(f"  • saved {os.path.basename(p)}")

print("\n✅ Done.")
