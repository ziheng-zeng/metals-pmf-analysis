# -*- coding: utf-8 -*-
"""
Single-Day PCA (RAW, no imputation/normalization)
- 24h local window (midnight→midnight)
- Astral sunrise/sunset (with robust date pinning), export flags
- 3×-uncertainty mask -> NaN (keep ALL timestamps)
- Pairwise-complete covariance PCA (handles NaNs)
- Scores = projection using only available metals (missing contribute 0 after centering)
- Exactly 3 PCs (or fewer if <3 metals survive)
"""

import os, datetime as dt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from numpy.linalg import eigh  # eigen-decomp for symmetric cov

# ----------------------- USER CONFIG -----------------------
FILE_PATH = "Xact_EST_May2023_Oct2025_combined.csv"
TZ = "US/Eastern"
OUTDIR = "./pca_day_outputs"
UNC_FILTER_MULT = 3.0
DAY_LOCAL = dt.date(2023, 10, 12)     # pick your day here
SEED = 50
np.random.seed(SEED)
os.makedirs(OUTDIR, exist_ok=True)

print("Loading data...")
df = pd.read_csv(FILE_PATH)
if "TIME" not in df.columns:
    raise ValueError("Expected a TIME column in the input CSV.")
df["TIME"] = pd.to_datetime(df["TIME"], utc=True).dt.tz_convert(TZ)
df = df.set_index("TIME").sort_index()

# -------------------- Pair conc/uncert columns --------------
headers = df.columns.tolist()
conc_cols = [c for c in headers if "(ng/m3)" in c and "uncert" not in c.lower()]
unc_cols  = [c for c in headers if "uncert" in c.lower()]

def element_from_conc(col):  # "Fe (ng/m3)" -> "Fe"
    return col.split()[0]

metal_to_conc, metal_to_unc = {}, {}
for c in conc_cols:
    el = element_from_conc(c)
    match = None
    for u in unc_cols:
        if u.startswith(el + " ") or u.startswith(el+"(") or (el+" " in u):
            match = u; break
    if match:
        metal_to_conc[el] = c
        metal_to_unc[el]  = match

# Fix S uncertainty if odd name
if "S" in metal_to_conc and "S" not in metal_to_unc:
    for u in unc_cols:
        if u.lower().startswith("s ") or u.lower().startswith("s(") or "s uncert" in u.lower():
            metal_to_unc["S"] = u; break

metals_all = sorted(metal_to_conc.keys())
if not metals_all:
    raise ValueError("No paired metals found.")
print(f"Paired metals detected: {len(metals_all)}")

# -------------------- Select 24h window ---------------------
start_local = pd.Timestamp(dt.datetime.combine(DAY_LOCAL, dt.time(0,0)), tz=TZ)
end_local   = start_local + pd.Timedelta(days=1)
dfd = df.loc[start_local:end_local - pd.Timedelta(seconds=1)].copy()
if dfd.empty:
    raise ValueError("Selected 24h window contains no data.")

# --------------- Sunrise / Sunset & day/night ---------------
def get_sun_times_for_pittsburgh(date_obj: dt.date):
    try:
        from astral.sun import sun
        from astral import LocationInfo
        city = LocationInfo("Pittsburgh", "USA", "US/Eastern", 40.4406, -79.9959)
        s = sun(city.observer, date=date_obj)
        sr = pd.Timestamp(s["sunrise"])
        ss = pd.Timestamp(s["sunset"])
    except Exception:
        # July PIT fallback ~05:55/20:54 (close enough)
        sr = pd.Timestamp(dt.datetime(date_obj.year, date_obj.month, date_obj.day, 5, 55))
        ss = pd.Timestamp(dt.datetime(date_obj.year, date_obj.month, date_obj.day, 20, 54))
    # ensure both are tz-aware and pinned to DAY_LOCAL date
    sr = (sr.tz_convert(TZ) if sr.tzinfo else sr.tz_localize(TZ))
    ss = (ss.tz_convert(TZ) if ss.tzinfo else ss.tz_localize(TZ))
    sr_time, ss_time = sr.time(), ss.time()
    # pin times to target date to avoid "sunset from previous day"
    sr = pd.Timestamp.combine(pd.Timestamp(DAY_LOCAL), sr_time).tz_localize(TZ)
    ss = pd.Timestamp.combine(pd.Timestamp(DAY_LOCAL), ss_time).tz_localize(TZ)
    if ss <= sr:  # ultra-rare, but guard anyway
        ss = ss + pd.Timedelta(days=1)
    return sr, ss

sunrise_ts, sunset_ts = get_sun_times_for_pittsburgh(DAY_LOCAL)

def is_night(ts):
    # True if before sunrise or at/after sunset on the target date
    return (ts < sunrise_ts) or (ts >= sunset_ts)

dfd["is_night"] = dfd.index.map(is_night)

sun_flags_path = os.path.join(OUTDIR, f"sun_times_and_flags_{DAY_LOCAL.isoformat()}.csv")
with open(sun_flags_path, "w", encoding="utf-8") as fh:
    pd.DataFrame({"sunrise_local":[sunrise_ts], "sunset_local":[sunset_ts]}).to_csv(fh, index=False)
    fh.write("\n")
    dfd[["is_night"]].to_csv(fh)
print(f"Saved: {sun_flags_path}")
print("Using sunrise:", sunrise_ts, " sunset:", sunset_ts)

# ------------------ Build RAW matrix (no log/scale) --------
X = pd.DataFrame(index=dfd.index, dtype=float)
for el in metals_all:
    X[el] = dfd[metal_to_conc[el]].astype(float)

# Apply 3×-uncertainty mask (values failing -> NaN). Keep ALL rows.
for el in metals_all:
    if el in metal_to_unc:
        unc = dfd[metal_to_unc[el]].astype(float)
        bad = X[el] < (UNC_FILTER_MULT * unc)
        X.loc[bad, el] = np.nan

# Drop metals that are entirely NaN after mask (nothing to contribute)
all_nan_cols = [el for el in X.columns if X[el].isna().all()]
if all_nan_cols:
    print("Dropping metals with all-NaN after 3× uncertainty filter:", all_nan_cols)
    X = X.drop(columns=all_nan_cols)

metals_used = X.columns.tolist()
p = len(metals_used)
if p == 0:
    raise ValueError("All metals became NaN after filtering.")
print(f"Using {p} metals for PCA.")

# ---------------- Pairwise-complete covariance PCA ---------
# (a) column means using available values
col_means = np.array([np.nanmean(X[c].values) for c in metals_used])

# (b) center (without touching NaNs)
X_centered = X.copy()
for j, c in enumerate(metals_used):
    X_centered[c] = X_centered[c] - col_means[j]

# (c) pairwise covariance matrix (NaN-safe)
def pairwise_cov(df_centered: pd.DataFrame) -> np.ndarray:
    A = df_centered.values  # shape n x p with NaNs
    n, p = A.shape
    cov = np.zeros((p, p), dtype=float)
    for i in range(p):
        ai = A[:, i]
        mask_i = ~np.isnan(ai)
        for j in range(i, p):
            aj = A[:, j]
            mask = mask_i & (~np.isnan(aj))
            n_ij = mask.sum()
            if n_ij >= 2:
                cov_ij = np.dot(ai[mask], aj[mask]) / (n_ij - 1)
            else:
                cov_ij = 0.0
            cov[i, j] = cov_ij
            cov[j, i] = cov_ij
    return cov

Cov = pairwise_cov(X_centered)

# (d) eigen-decomposition (symmetric)
eigvals, eigvecs = eigh(Cov)                # ascending
order = np.argsort(eigvals)[::-1]           # descending
eigvals = eigvals[order]
eigvecs = eigvecs[:, order]

# choose up to 3 PCs (or fewer if <3 metals)
ncomp = min(3, p)
eigvals_sel = eigvals[:ncomp]
eigvecs_sel = eigvecs[:, :ncomp]            # columns are eigenvectors (loadings)

# explained variance ratio (clip negatives to 0 due to pairwise noise)
eigvals_pos = np.clip(eigvals, a_min=0, a_max=None)
den = eigvals_pos.sum() if eigvals_pos.sum() > 0 else 1.0
explained = np.clip(eigvals_sel, 0, None) / den

# (e) scores: project each row onto eigenvectors using available metals
# Treat missing entries as 0 *after centering* (i.e., as "mean" for that metal)
Xc_vals = X_centered.values  # n x p with NaNs
Xc_vals = np.nan_to_num(Xc_vals, nan=0.0)  # missing metals contribute 0
scores = Xc_vals @ eigvecs_sel             # n x ncomp

# --------------- Pack results ------------------------------
pc_cols = [f"PC{i+1}" for i in range(ncomp)]
df_scores = pd.DataFrame(scores, index=X.index, columns=pc_cols)
df_scores["is_night"] = dfd["is_night"].astype(bool)

loadings = pd.DataFrame(eigvecs_sel, index=metals_used, columns=pc_cols)

# ----------------------- Plots -----------------------------
# PC1 vs PC2
if ncomp >= 2:
    plt.figure(figsize=(7.2,6.4))
    colors = df_scores["is_night"].map({True: "navy", False: "orange"})
    plt.scatter(df_scores["PC1"], df_scores["PC2"], c=colors, alpha=0.85, s=48, edgecolor="none")
    from matplotlib.lines import Line2D
    legend_elems = [
        Line2D([0],[0], marker='o', color='w', label='Day', markerfacecolor='orange', markersize=9),
        Line2D([0],[0], marker='o', color='w', label='Night', markerfacecolor='navy', markersize=9),
    ]
    plt.legend(handles=legend_elems, frameon=False, loc="best")
    plt.xlabel(f"PC1 ({explained[0]*100:.1f}% var)")
    plt.ylabel(f"PC2 ({explained[1]*100:.1f}% var)")
    plt.title(f"PCA (RAW) — {DAY_LOCAL.isoformat()}  [no log/scale, no impute]")
    plt.tight_layout()
    out_scatter = os.path.join(OUTDIR, f"pca_raw_scatter_daynight_{DAY_LOCAL.isoformat()}.png")
    plt.savefig(out_scatter, dpi=220); plt.close()
    print(f"Saved: {out_scatter}")

# Loadings (PC1 & PC2 if available)
top_k = min(20, len(metals_used))
order_load = loadings["PC1"].abs().sort_values(ascending=False).index[:top_k]
plt.figure(figsize=(10.0,5.2))
cols_to_plot = ["PC1"] + (["PC2"] if ncomp >= 2 else [])
loadings.loc[order_load, cols_to_plot].plot(kind="bar", width=0.85)
plt.ylabel("Loading")
plt.title("PCA Loadings (RAW)")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
out_load = os.path.join(OUTDIR, f"pca_raw_loadings_{DAY_LOCAL.isoformat()}.png")
plt.savefig(out_load, dpi=220); plt.close()
print(f"Saved: {out_load}")

# Explained variance
plt.figure(figsize=(6.2,4.0))
plt.bar(range(1, ncomp+1), explained*100)
plt.xticks(range(1, ncomp+1), [f"PC{i}" for i in range(1, ncomp+1)])
plt.ylabel("Explained variance (%)")
plt.title("PCA Explained Variance (RAW, pairwise cov)")
plt.tight_layout()
out_expl = os.path.join(OUTDIR, f"pca_raw_explained_{DAY_LOCAL.isoformat()}.png")
plt.savefig(out_expl, dpi=220); plt.close()
print(f"Saved: {out_expl}")

# Export scores (optional)
scores_csv = os.path.join(OUTDIR, f"pca_raw_scores_{DAY_LOCAL.isoformat()}.csv")
df_scores.to_csv(scores_csv)
print(f"Saved: {scores_csv}")

print("\nTop |loadings| PC1:")
print(loadings["PC1"].abs().sort_values(ascending=False).head(10))
if ncomp >= 2:
    print("\nTop |loadings| PC2:")
    print(loadings["PC2"].abs().sort_values(ascending=False).head(10))
print("\nDone.")
