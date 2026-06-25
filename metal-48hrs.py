# -*- coding: utf-8 -*-
"""
Minimal 48-hour metals browser (with uncertainties)
- No baseline/event/enhancement calculations
- Optional 3×-uncertainty filtering (default OFF for sensitivity to small peaks)
- Walks through the entire time range in 48 h windows (configurable)
- One figure per window; each subplot is one metal with ±uncert band

Outputs: PNGs to OUTDIR (and optional multi-page PDF)
"""

import os
import warnings
from datetime import timedelta

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, MaxNLocator
from matplotlib.dates import DateFormatter, HourLocator
import matplotlib.dates as mdates



warnings.filterwarnings("ignore")

# ----------------------- USER CONFIG -----------------------
FILE_PATH = "Xact_EST_May2023_Oct2025_combined.csv"
TZ = "US/Eastern"
OUTDIR = "./metals_48h_plots_selected"
WINDOW_HOURS = 48
STEP_HOURS = 48           # use 24 for overlapping windows; 48 for non overlapping
APPLY_3SIGMA_FILTER = False   # <- keep False to catch small/weird peaks
EXCLUDE_INTERVALS = [          # instrument-down (optional)
    ("2024-01-09", "2024-02-13"),
    ("2024-07-02", "2024-08-08"),
]
# --- layout controls ---
ONE_COLUMN_IF_AT_MOST = 16     # use 1-column + shared x if n_metals <= this
ROW_HEIGHT_IN = 1.6            # figure height per row in one-column mode
ONECOL_FIGWIDTH_IN = 8        # figure width in one-column mode

# You can limit to a subset, e.g., METALS_INCLUDE = ["K", "Cu", "Ba", "Ti"]
METALS_INCLUDE = ["Al", "S", "Cl", "K", "Ti", "Cr", "Mn", "Cu",
                  "Zn", "As", "Se", "Sr", "Ba", "Pb", "Bi"]
# METALS_INCLUDE =["K", "S", "Al", "Ca", "Zn", "Br", "Fe", "Se", "As"]

SAVE_PDF = False                  # also save a combined PDF of all windows
PDF_NAME = "metals_48h_browser.pdf"
N_COLS = 3                        # grid layout for subplots
FIGSIZE = (18, 10)                # per-window figure size
LINE_ALPHA = 0.9
BAND_ALPHA = 0.25

# Limit the dataset to a specific time range (optional)
TIME_RANGE = ("2025-07-04 00:00", "2025-07-06 00:00")  # or None for all data

# ----------------------- LOAD & PREP -----------------------
df = pd.read_csv(FILE_PATH)
# time handling
df["TIME"] = pd.to_datetime(df["TIME"], utc=True).dt.tz_convert(TZ)
df = df.set_index("TIME").sort_index()

# pair concentration and uncertainty columns
headers = df.columns.tolist()
conc_cols = [c for c in headers if " (ng/m3)" in c and "uncert" not in c.lower()]
unc_cols = [c for c in headers if "uncert" in c.lower()]

metal_uncert_pairs = {}
for conc_col in conc_cols:
    # element name assumed to be the first token (e.g., "K 19 (ng/m3)")
    element = conc_col.split()[0]
    # find the first uncertainty col that contains the element token
    ucol = next((u for u in unc_cols if element in u), None)
    if ucol:
        metal_uncert_pairs[conc_col] = ucol

# manual fix for sulfur naming if needed
if "S 16 (ng/m3)" in conc_cols and "S Uncert (ng/m3)" in df.columns:
    metal_uncert_pairs["S 16 (ng/m3)"] = "S Uncert (ng/m3)"

# make element->column maps
metals_cols = {}
uncert_cols = {}
for conc_col, ucol in metal_uncert_pairs.items():
    element = conc_col.split()[0]
    if conc_col in df.columns and ucol in df.columns:
        metals_cols[element] = conc_col
        uncert_cols[element] = ucol

if METALS_INCLUDE is not None:
    # keep only requested metals that exist
    metals_cols = {m: metals_cols[m] for m in METALS_INCLUDE if m in metals_cols}
    uncert_cols = {m: uncert_cols[m] for m in metals_cols}

print(f"Paired {len(metals_cols)} metals with uncertainties.")

# optional 1×-uncertainty filtering (mask values <= 3σ)
df_work = df.copy()
if APPLY_3SIGMA_FILTER:
    print("Applying 1× uncertainty filtering…")
    for m in metals_cols:
        conc = metals_cols[m]
        unc = uncert_cols[m]
        try:
            c = df_work[conc]
            u = df_work[unc]
            mask = (c > 1.0 * u) | c.isna()   # <- correct 3× filter
            df_work[conc] = c.where(mask)
        except Exception as e:
            print(f"  warn: could not filter {m}: {e}")
else:
    print("1× uncertainty filtering DISABLED (using raw concentrations).")

# optional excludes
if EXCLUDE_INTERVALS:
    for (s, e) in EXCLUDE_INTERVALS:
        s = pd.Timestamp(s, tz=TZ)
        e = pd.Timestamp(e, tz=TZ)
        df_work = df_work.loc[(df_work.index < s) | (df_work.index > e)]
    print(f"Applied {len(EXCLUDE_INTERVALS)} exclude intervals.")

# --- optional global time filter ---
if TIME_RANGE is not None:
    tstart = pd.Timestamp(TIME_RANGE[0], tz=TZ)
    tend   = pd.Timestamp(TIME_RANGE[1], tz=TZ)
    df_work = df_work.loc[(df_work.index >= tstart) & (df_work.index <= tend)]
    print(f"Applied TIME_RANGE: {tstart} → {tend} ({len(df_work)} rows)")

# ----------------------- WINDOWING -----------------------
if df_work.empty:
    raise SystemExit("No data after filtering/exclusion.")

t0 = df_work.index.min()
t1 = df_work.index.max()

os.makedirs(OUTDIR, exist_ok=True)
pdf = None
if SAVE_PDF:
    from matplotlib.backends.backend_pdf import PdfPages
    pdf = PdfPages(os.path.join(OUTDIR, PDF_NAME))

def _nice_yfmt(x, pos):
    # compact human-friendly ticks: 0–2 decimals depending on magnitude
    ax_abs = abs(x)
    if ax_abs >= 1000:
        return f"{x:,.0f}"           # 1,000; 10,000
    if ax_abs >= 100:
        return f"{x:.0f}"            # 100
    if ax_abs >= 10:
        return f"{x:.1f}"            # 10.5
    return f"{x:.2f}"                # 0.12

# add once near your imports:
# from matplotlib.dates import DateFormatter
# import matplotlib.dates as mdates

def plot_window(dfw, ts_start, ts_end, idx):
    """Plot only metals that have valid conc+unc within [ts_start, ts_end]."""
    win = dfw.loc[(dfw.index >= ts_start) & (dfw.index <= ts_end)]
    if win.empty:
        return None

    # which metals have any valid data in this window?
    valid_metals = []
    for m, conc_col in metals_cols.items():
        unc_col = uncert_cols[m]
        if conc_col not in win or unc_col not in win:
            continue
        vmask = ~(win[conc_col].isna() | win[unc_col].isna())
        if vmask.any():
            valid_metals.append(m)
    if not valid_metals:
        return None

    n_metals = len(valid_metals)

    # layout
    use_onecol = n_metals <= ONE_COLUMN_IF_AT_MOST
    if use_onecol:
        n_rows = n_metals
        fig, axes = plt.subplots(
            n_rows, 1,
            figsize=(ONECOL_FIGWIDTH_IN, ROW_HEIGHT_IN * n_rows),
            sharex=True,
            squeeze=False
        )
        axes = axes.flatten()
    else:
        n_cols = N_COLS
        n_rows = int(np.ceil(n_metals / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=FIGSIZE, squeeze=False)
        axes = axes.flatten()

    color_cycle = plt.cm.tab20(np.linspace(0, 1, max(n_metals, 2)))

    # precompute midnight ticks covering [ts_start, ts_end]
    # ensures first/last day labels are included even at the edges
    # Ensure timestamps are timezone-aware in the right zone
    ts_start_local = ts_start if ts_start.tzinfo else pd.Timestamp(ts_start, tz=TZ)
    ts_end_local = ts_end if ts_end.tzinfo else pd.Timestamp(ts_end, tz=TZ)

    midnights = pd.date_range(
        start=ts_start_local.floor("D"),
        end=ts_end_local.ceil("D"),
        freq="D",
        tz=ts_start_local.tz
    )

    midnight_ticks = [mdates.date2num(d.to_pydatetime()) for d in midnights]

    # --------- PLOT EACH METAL ---------
    for i, m in enumerate(valid_metals):
        ax = axes[i]
        conc_col = metals_cols[m]
        unc_col  = uncert_cols[m]

        vmask = ~(win[conc_col].isna() | win[unc_col].isna())
        times = win.index[vmask]
        conc  = win.loc[vmask, conc_col].astype(float)
        unc   = win.loc[vmask, unc_col].astype(float)

        c = color_cycle[i % len(color_cycle)]
        ax.plot(times, conc, lw=2.0, alpha=LINE_ALPHA, color=c)
        ax.fill_between(times, conc - unc, conc + unc,
                        alpha=BAND_ALPHA, step="mid", color=c)

        # metal label
        ax.text(0.02, 0.96, m, transform=ax.transAxes,
                ha="left", va="top", fontsize=22, fontweight="bold",)
                # bbox=dict(boxstyle="round,pad=0.25",
                #           facecolor="white", edgecolor="none", alpha=0.8))

        # y-axis formatting
        sf = ScalarFormatter(useMathText=False)
        sf.set_scientific(False)
        sf.set_useOffset(False)
        ax.yaxis.set_major_formatter(sf)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.tick_params(axis="x", labelsize=16)
        ax.tick_params(axis="y", labelsize=16)
        ax.grid(True, alpha=0.25)

        # x-axis: ticks at daily midnight, labeled mm/dd/yyyy (force inclusion)
        ax.set_xticks(midnight_ticks)
        ax.xaxis.set_major_formatter(DateFormatter("%m/%d/%Y"))
        ax.xaxis.set_minor_locator(HourLocator(interval=4))

        ax.tick_params(axis="x")

        # y label
        ax.set_ylabel("(ng/m³)", fontsize=16, labelpad=16)

        if conc.min() >= 0:
            ax.set_ylim(bottom=0)

    # shared-x cleanup for one-column stack: only bottom shows x tick labels
    if use_onecol:
        for k, a in enumerate(axes[:n_metals]):
            if k < n_metals - 1:
                a.tick_params(labelbottom=False)

    # hide unused axes
    for j in range(n_metals, len(axes)):
        axes[j].set_visible(False)

    # align y labels in one-column mode after plotting
    if use_onecol:
        fig.align_ylabels(axes[:n_metals])

    fig.tight_layout(rect=[0, 0, 1, 0.97])

    png_name = f"metals_{ts_start:%Y%m%d_%H%M}_to_{ts_end:%Y%m%d_%H%M}.png"
    out_path = os.path.join(OUTDIR, png_name)
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    if pdf is not None:
        pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ----------------------- RUN -----------------------
win  = pd.Timedelta(hours=WINDOW_HOURS)
step = pd.Timedelta(hours=STEP_HOURS)

t_start = df_work.index.min()
t_end   = df_work.index.max()

os.makedirs(OUTDIR, exist_ok=True)
pdf = None
if SAVE_PDF:
    from matplotlib.backends.backend_pdf import PdfPages
    pdf = PdfPages(os.path.join(OUTDIR, PDF_NAME))

idx = 0
saved = []
ts = t_start

while ts + win <= t_end:
    te = ts + win
    p = plot_window(df_work, ts, te, idx)
    if p:
        saved.append(p)
        print(f"[{idx:03d}] {os.path.basename(p)}")
    idx += 1
    ts = ts + step  # <- advance by 24 h for overlap

if pdf is not None:
    pdf.close()
    print(f"Saved PDF: {os.path.join(OUTDIR, PDF_NAME)}")

print(f"✅ Done — saved {len(saved)} overlapping 48 h figures to {OUTDIR}")