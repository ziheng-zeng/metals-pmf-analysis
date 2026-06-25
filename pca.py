# -*- coding: utf-8 -*-
"""
Nighttime Metals PCA (all-metals, season-normalized, robust)
- 3×-uncertainty filtering
- 12h day/night blocks
- season-wise normalization (relative enrichment)
- log1p + RobustScaler
- PCA per season (night-only by default)
- anomaly detection on PCs (MAD threshold)
- clustering extreme nights (auto-K via silhouette)
Outputs: CSVs + diagnostic plots

Requires: pandas, numpy, matplotlib, scikit-learn, scipy
"""

import os
import warnings
from dataclasses import dataclass
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.signal import find_peaks

warnings.filterwarnings("ignore")

# ----------------------- USER CONFIG -----------------------
FILE_PATH = "Xact_EST_May2023_Oct2025_combined.csv"
TZ = "US/Eastern"
OUTDIR = "./pca_day_night_outputs"
USE_NIGHT_ONLY = False          # PCA on night blocks only; set False to include day too
BLOCK_HOURS = 12               # 12-hr blocks (night/day)
NIGHT_START = 19               # 7 PM local start of night block
MAD_Z_THRESH = 3.5             # anomaly z-threshold for PC scores
TOP_PC_FOR_EVENTS = 3          # examine first N PCs for peaks
EVENT_MIN_SEPARATION_H = 6     # min separation between detected peaks (hours)
EVENT_WINDOW_H = 4             # +/- window around PC-score peak (hours)
PC_MAX_PLOT = 5                # plot first N PCs
SEED = 42
# -----------------------------------------------------------

np.random.seed(SEED)
os.makedirs(OUTDIR, exist_ok=True)

# -------------------- Helper dataclasses -------------------
@dataclass
class PCAResult:
    season: str
    block_df: pd.DataFrame     # block-level normalized data used for PCA
    scaled_df: pd.DataFrame    # scaled features sent to PCA
    pca_scores: pd.DataFrame
    loadings: pd.DataFrame
    explained: np.ndarray
    cumulative: np.ndarray
    metals: list

# -------------------- Load & pair columns ------------------
print("Loading data...")
df = pd.read_csv(FILE_PATH)
if "TIME" not in df.columns:
    raise ValueError("Expected a TIME column in the input CSV.")

# timezone
df["TIME"] = pd.to_datetime(df["TIME"], utc=True).dt.tz_convert(TZ)
df = df.set_index("TIME").sort_index()

# auto-detect concentration/uncertainty columns
headers = df.columns.tolist()
conc_cols = [c for c in headers if "(ng/m3)" in c and "uncert" not in c.lower()]
unc_cols = [c for c in headers if "uncert" in c.lower()]

# map element symbol as key (best-effort)
def element_from_conc(col):
    # typical format: "Fe 26 (ng/m3)" -> "Fe"
    return col.split()[0]

metal_to_conc = {}
metal_to_unc = {}

for c in conc_cols:
    el = element_from_conc(c)
    match = None
    for u in unc_cols:
        if u.startswith(el + " ") or u.startswith(el+"(") or el+" " in u:
            match = u
            break
    if match:
        metal_to_conc[el] = c
        metal_to_unc[el] = match

# Manual fix for S if needed
if "S" in metal_to_conc and "S" not in metal_to_unc:
    # try a common uncertainty name
    for u in unc_cols:
        if u.lower().startswith("s ") or u.lower().startswith("s(") or "s uncert" in u.lower():
            metal_to_unc["S"] = u
            break

metals_all = sorted(metal_to_conc.keys())
print(f"Paired {len(metals_all)} metals with uncertainties.")

# ------------------ Metal whitelist ------------------
WHITELIST = ["Fe", "Zn", "K", "Ca", "Ti", "As", "Cu", "Mn", "Ba", "Se", "Cr", "Pb", "Ni"]

metals_all = [m for m in metals_all if m in WHITELIST]
print(f"Using {len(metals_all)} metals after whitelist filter: {metals_all}")

# ---------------- 1× uncertainty filtering -----------------
print("Applying 1×-uncertainty filter...")
df_filt = df.copy()
for m in metals_all:
    ccol = metal_to_conc[m]
    ucol = metal_to_unc.get(m, None)
    if ucol is None or ucol not in df_filt.columns:
        # if no uncertainty column found, keep as-is but warn
        print(f"  [WARN] No uncert for {m}; skipping filter.")
        continue
    conc = df_filt[ccol]
    unc = df_filt[ucol]
    # keep values where conc > 3*unc; else set NaN
    mask = (conc > 1.0 * unc) | conc.isna()
    df_filt[ccol] = conc.where(mask, np.nan)

# ---- Exclude known downtime (optional; adjust if needed) --
downtimes = [
    ("2024-01-09", "2024-02-13"),
    ("2024-07-02", "2024-08-08"),
]
mask_ok = pd.Series(True, index=df_filt.index)
for start, end in downtimes:
    s = pd.Timestamp(start, tz=TZ)
    e = pd.Timestamp(end, tz=TZ)
    mask_ok &= ~((df_filt.index >= s) & (df_filt.index <= e))
df_filt = df_filt.loc[mask_ok]

# -------------------- Day/Night labeling -------------------
def is_night(ts):
    h = ts.hour
    return (h >= NIGHT_START) or (h < (NIGHT_START + 12) % 24)

def season_of(ts):
    m = ts.month
    if m in (12, 1, 2): return "Winter"
    if m in (3, 4, 5):  return "Spring"
    if m in (6, 7, 8):  return "Summer"
    return "Fall"

df_f = df_filt.copy()
df_f["period"] = ["night" if is_night(t) else "day" for t in df_f.index]
df_f["season"] = [season_of(t) for t in df_f.index]
df_f["date"] = df_f.index.date

# ----------------- Build 12h blocks (features) -------------
# For each date & period, aggregate metals using a robust statistic.
# Median is good; 95th percentile can emphasize spikes — we’ll do both and choose MEDIAN as default.
AGG = "median"  # "median" or "p95"
def agg_func(x):
    if AGG == "p95":
        return np.nanpercentile(x, 95)
    return np.nanmedian(x)

block_rows = []
grouped = df_f.groupby(["date", "period"])
for (d, p), g in grouped:
    row = {"date": pd.to_datetime(str(d)).tz_localize(TZ),
           "period": p,
           "season": season_of(pd.to_datetime(str(d)).tz_localize(TZ))}
    for m in metals_all:
        ccol = metal_to_conc[m]
        row[m] = agg_func(g[ccol]) if ccol in g.columns else np.nan
        # for potential impute, we could also aggregate uncert:
        ucol = metal_to_unc.get(m, None)
        if ucol in g.columns:
            row[m+"_UNC"] = agg_func(g[ucol])
    block_rows.append(row)

block_df = pd.DataFrame(block_rows).sort_values("date").set_index("date")
print(f"Block-level rows: {len(block_df)} (12h day/night blocks).")

# -------------- Season-wise normalization ------------------
# For each season and metal, compute seasonal median across *daytime* blocks as baseline
# (daytime baseline suppresses fireworks/wildfire night spikes). Then normalize block value / seasonal_median.
norm_blocks = []
for season, sub in block_df.groupby("season"):
    sub = sub.copy()
    # baseline medians from DAY blocks within the season
    day_baseline = sub[sub["period"] == "day"][metals_all].median()

    # baseline medians from all blocks
    # day_baseline = sub[metals_all].median()

    # fallback to season medians if day-baseline is NaN
    fallback = sub[metals_all].median()
    base = day_baseline.fillna(fallback).replace(0, np.nan)

    # normalize
    for m in metals_all:
        sub[m+"_REL"] = sub[m] / base.get(m, np.nan)

    # simple imputation for missing normalized values:
    # if REL is NaN, try using half of normalized uncertainty; else small epsilon (0.1)
    for m in metals_all:
        rel = sub[m+"_REL"]
        # if uncertainty exists, estimate REL via (0.5 * UNC) / baseline
        ucol = m + "_UNC"
        if ucol in sub.columns and ucol.replace("_UNC","") in metals_all:
            est_rel = (0.5 * sub[ucol]) / base.get(m, np.nan)
            rel = rel.where(~rel.isna(), est_rel)
        # still NaN? set to small epsilon (keeps feature but tiny)
        rel = rel.fillna(0.1)
        sub[m+"_REL"] = rel

    norm_blocks.append(sub)

block_rel = pd.concat(norm_blocks).sort_index()

# Optional: restrict to night only for PCA
if USE_NIGHT_ONLY:
    pca_base = block_rel[block_rel["period"] == "night"].copy()
else:
    pca_base = block_rel.copy()

# ----------------- NO NORMALIZATION VERSION -----------------
block_nonorm = block_df.copy()

# Optional: small impute for NaNs to keep PCA happy
for m in metals_all:
    block_nonorm[m] = block_nonorm[m].fillna(0.1)


# -------------------- PCA per season -----------------------
def run_pca_for_season(season_name, df_blocks, max_pc=10):
    # pick REL columns
    rel_cols = [m+"_REL" for m in metals_all if m+"_REL" in df_blocks.columns]
    X = df_blocks[rel_cols].copy()

    # log1p + RobustScaler
    X_log = np.log1p(X)
    scaler = RobustScaler()
    X_sc = pd.DataFrame(scaler.fit_transform(X_log), index=X_log.index, columns=X_log.columns)

    # components
    ncomp = min(max_pc, X_sc.shape[0]-1, X_sc.shape[1])
    ncomp = max(2, ncomp)

    pca = PCA(n_components=ncomp, random_state=SEED)
    scores = pd.DataFrame(pca.fit_transform(X_sc), index=X_sc.index,
                          columns=[f"PC{i+1}" for i in range(ncomp)])
    loadings = pd.DataFrame(pca.components_.T, index=rel_cols,
                            columns=[f"PC{i+1}" for i in range(ncomp)])
    explained = pca.explained_variance_ratio_
    cumulative = np.cumsum(explained)

    # add tags back
    meta = df_blocks[["period","season"]].copy()
    out_scores = meta.join(scores)

    return PCAResult(
        season=season_name,
        block_df=df_blocks.copy(),
        scaled_df=X_sc,
        pca_scores=out_scores,
        loadings=loadings,
        explained=explained,
        cumulative=cumulative,
        metals=metals_all
    )

pca_results = []
for season, sub in pca_base.groupby("season"):
    if len(sub) < 30:
        print(f"[{season}] too few blocks for PCA, skipping.")
        continue
    print(f"Running PCA for {season} ({len(sub)} blocks)...")
    pca_results.append(run_pca_for_season(season, sub))

# Also optional: all seasons combined PCA
if len(pca_base) >= 60:
    print("Running PCA for ALL seasons combined...")
    pca_results.append(run_pca_for_season("ALL", pca_base))

# ----------------- Detect nighttime anomalies --------------
def mad_z(x):
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    return (x - med) / (1.4826 * (mad + 1e-12))

def detect_pc_peaks(scores_df, pc_name, z_thresh=MAD_Z_THRESH,
                    min_sep_h=EVENT_MIN_SEPARATION_H, event_win_h=EVENT_WINDOW_H):
    series = scores_df[pc_name]
    z = mad_z(series.values)
    # consider abs z since ± can matter
    z_abs = np.abs(z)
    # use a high quantile cutoff or MAD threshold
    # convert MAD threshold to absolute values
    peaks, props = find_peaks(z_abs, height=np.nanpercentile(z_abs, 95), distance=min_sep_h)
    # Build events
    events = []
    for idx in peaks:
        t = series.index[idx]
        events.append({
            "pc": pc_name,
            "peak_time": t,
            "start_time": t - pd.Timedelta(hours=event_win_h//2),
            "end_time": t + pd.Timedelta(hours=event_win_h//2),
            "abs_z": float(z_abs[idx]),
            "score": float(series.iloc[idx]),
        })
    return events

all_events = []
for res in pca_results:
    sc = res.pca_scores.copy()
    pcs = [c for c in sc.columns if c.startswith("PC")][:TOP_PC_FOR_EVENTS]
    evts = []
    for pc in pcs:
        evts += detect_pc_peaks(sc, pc)
    # de-duplicate by peak_time within +/- 2h, keep larger |z|
    evts = sorted(evts, key=lambda e: e["abs_z"], reverse=True)
    kept = []
    for e in evts:
        if all(abs((e["peak_time"] - k["peak_time"]).total_seconds())/3600.0 >= 2 for k in kept):
            e["season"] = res.season
            kept.append(e)
    all_events += kept

events_df = pd.DataFrame(all_events).sort_values(["season","peak_time"])
events_path = os.path.join(OUTDIR, "night_events_from_pcs.csv")
events_df.to_csv(events_path, index=False)
print(f"Detected {len(events_df)} PC-peak nighttime events → {events_path}")

# -------------- Cluster extreme nights (scores) ------------
cluster_rows = []
for res in pca_results:
    sc = res.pca_scores.copy()
    pc_cols = [c for c in sc.columns if c.startswith("PC")]
    # Select extreme nights across first few PCs (top 10% by |PC| in any of first 3 PCs)
    mask_extreme = np.zeros(len(sc), dtype=bool)
    for pc in pc_cols[:TOP_PC_FOR_EVENTS]:
        thr = np.nanpercentile(np.abs(sc[pc]), 90)
        mask_extreme |= (np.abs(sc[pc]) >= thr)
    X = sc.loc[mask_extreme, pc_cols].copy()
    if len(X) < 10:
        continue

    # choose K by silhouette over K=2..6
    best_k, best_s = None, -1
    best_labels = None
    for k in range(2, min(7, len(X))):
        km = KMeans(n_clusters=k, random_state=SEED, n_init="auto")
        labels = km.fit_predict(X.values)
        # some tiny sets can break silhouette
        if len(np.unique(labels)) < 2:
            continue
        s = silhouette_score(X.values, labels)
        if s > best_s:
            best_s, best_k, best_labels = s, k, labels
    if best_labels is None:
        continue

    Xc = X.copy()
    Xc["cluster"] = best_labels
    Xc["season"] = res.season
    cluster_rows.append(Xc)

if cluster_rows:
    clusters_df = pd.concat(cluster_rows).reset_index().rename(columns={"index":"date"})
    clusters_path = os.path.join(OUTDIR, "night_extreme_clusters.csv")
    clusters_df.to_csv(clusters_path, index=False)
    print(f"Clustered extreme nights → {clusters_path}")
else:
    clusters_df = pd.DataFrame()

# -------------- Save PCA scores & loadings per season ------
for res in pca_results:
    season_tag = res.season.lower()
    scores_out = os.path.join(OUTDIR, f"pca_scores_{season_tag}.csv")
    loads_out = os.path.join(OUTDIR, f"pca_loadings_{season_tag}.csv")
    res.pca_scores.to_csv(scores_out)
    res.loadings.to_csv(loads_out)
    print(f"[{res.season}] Saved scores → {scores_out}")
    print(f"[{res.season}] Saved loadings → {loads_out}")

# ----------------- Loadings plotting helpers -----------------
def _strip_rel(idx):
    # turn "Ni_REL" -> "Ni"
    return [s.replace("_REL","") for s in idx]

def plot_loadings_per_pc(res: PCAResult, max_pcs=3, top_n=25):
    """
    For each PC (up to max_pcs), plot a horizontal bar chart of loadings,
    sorted by absolute loading. Saves PNGs + a CSV of the ranked table.
    """
    L = res.loadings.copy()
    L.index = _strip_rel(L.index)
    n_pc = min(max_pcs, L.shape[1])

    for i in range(n_pc):
        pc = f"PC{i+1}"
        ser = L[pc].copy()
        ranked = ser.reindex(ser.abs().sort_values(ascending=False).index)
        if top_n:
            ranked = ranked.iloc[:top_n]

        # save ranked table
        tbl_path = os.path.join(OUTDIR, f"loadings_ranked_{res.season.lower()}_{pc}.csv")
        ranked.to_csv(tbl_path, header=[pc])

        # plot
        fig, ax = plt.subplots(figsize=(8, max(4, 0.3*len(ranked))))
        ax.barh(ranked.index, ranked.values)
        ax.invert_yaxis()
        ax.set_xlabel("Loading")
        ax.set_title(f"{res.season} {pc} loadings (top {len(ranked)})")
        ax.grid(axis="x", alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(OUTDIR, f"loadings_{res.season.lower()}_{pc}.png"), dpi=200)
        plt.close(fig)

def plot_loadings_grouped(res: PCAResult, pcs=("PC1","PC2","PC3"), top_n=20):
    """
    Grouped bar of loadings for multiple PCs on the same axes.
    Picks metals with largest max |loading| across the requested PCs.
    """
    L = res.loadings.copy()
    L.index = _strip_rel(L.index)
    pcs = [p for p in pcs if p in L.columns]
    if not pcs:
        return

    # choose metals with largest max |loading| across the chosen PCs
    keep = (L[pcs].abs().max(axis=1).sort_values(ascending=False)).index[:top_n]
    sub = L.loc[keep, pcs]

    # plotting
    fig, ax = plt.subplots(figsize=(10, max(4, 0.35*len(keep))))
    x = np.arange(len(keep))
    width = 0.8 / len(pcs)

    for j, pc in enumerate(pcs):
        ax.bar(x + j*width - (len(pcs)-1)*width/2, sub[pc].values, width=width, label=pc)

    ax.set_xticks(x)
    ax.set_xticklabels(sub.index, rotation=45, ha="right")
    ax.set_ylabel("Loading")
    ax.set_title(f"{res.season} loadings (top {top_n} metals by max |loading| across {', '.join(pcs)})")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, f"loadings_grouped_{res.season.lower()}_{pcs[0].lower()}_{len(pcs)}pc.png"), dpi=200)
    plt.close(fig)


# ----------------- Quick diagnostic plots ------------------
def plot_explained(res: PCAResult):
    exp = res.explained * 100
    cum = res.cumulative * 100
    n = min(PC_MAX_PLOT, len(exp))
    fig, ax = plt.subplots(figsize=(6,4))
    ax.bar(range(1, n+1), exp[:n], alpha=0.7)
    ax.plot(range(1, n+1), cum[:n], marker="o")
    ax.set_xlabel("PC")
    ax.set_ylabel("Variance explained (%)")
    ax.set_title(f"{res.season} PCA variance")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, f"explained_{res.season.lower()}.png"), dpi=200)
    plt.close(fig)

def plot_pc_scatter(res: PCAResult, pcx="PC1", pcy="PC2"):
    sc = res.pca_scores.copy()
    fig, ax = plt.subplots(figsize=(6,5))
    for lbl, g in sc.groupby("season" if res.season=="ALL" else "period"):
        ax.scatter(g[pcx], g[pcy], s=18, alpha=0.6, label=lbl)
    ax.set_xlabel(pcx)
    ax.set_ylabel(pcy)
    ax.set_title(f"{res.season} {pcx} vs {pcy}")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, f"scatter_{res.season.lower()}_{pcx}_{pcy}.png"), dpi=200)
    plt.close(fig)

def plot_pc_time(res: PCAResult, pc="PC1"):
    sc = res.pca_scores.copy()
    fig, ax = plt.subplots(figsize=(10,3))
    ax.plot(sc.index, sc[pc], lw=0.9)
    # overlay detected events of this season and pc
    if len(events_df):
        for _, r in events_df[(events_df["season"]==res.season) & (events_df["pc"]==pc)].iterrows():
            ax.axvspan(r["start_time"], r["end_time"], color="tomato", alpha=0.2)
    ax.set_ylabel(pc)
    ax.set_title(f"{res.season} {pc} score (events shaded)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, f"pc_time_{res.season.lower()}_{pc}.png"), dpi=200)
    plt.close(fig)

for res in pca_results:
    plot_explained(res)
    plot_pc_scatter(res, "PC1", "PC2")
    for i in range(min(PC_MAX_PLOT, res.pca_scores.filter(like="PC").shape[1])):
        plot_pc_time(res, f"PC{i+1}")

    plot_loadings_per_pc(res, max_pcs=3, top_n=25)          # one PNG per PC + ranked CSV
    plot_loadings_grouped(res, pcs=("PC1","PC2","PC3"), top_n=20)  # grouped comparison

print("\nDone. Outputs:")
print(f"  • Events CSV: {events_path}")
if len(clusters_df):
    print(f"  • Clustered nights CSV: {clusters_path}")
print(f"  • Per-season scores/loadings CSVs + PNGs in: {OUTDIR}")

"""
How to read results quickly:
- pca_loadings_*.csv: columns are PCs, rows are METAL_REL. Large positive loadings on a PC
  = metals that define that source-axis (e.g., Ni_REL, V_REL, Se_REL).
- pca_scores_*.csv: each 12-hr block's coordinates on those PCs. Large ± scores at night
  = nights dominated by that component. Cross-check dates with events CSV.
- night_events_from_pcs.csv: list of detected PC-score peaks (season, pc, peak_time).
- night_extreme_clusters.csv: clusters of extreme nights in PC space (helps find recurrent
  nighttime fingerprints). Inspect each cluster's mean loadings by joining to loadings.
"""
