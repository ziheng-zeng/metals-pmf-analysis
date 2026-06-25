# -*- coding: utf-8 -*-
"""
PCA with TWO points per day: Day vs Night (RAW), over multiple days
- Date range: inclusive DATE_START → DATE_END (local)
- Sunrise/sunset via Astral (per-day, with time pinning)
- 3×-uncertainty mask -> NaN (no row drops)
- Aggregate per group (Day, Night): mean/median/pXX
- Drop metals missing in ANY sample across the whole range
- PCA on stacked matrix: (2 * n_days) × P (no log, no scaling)
"""

import os, datetime as dt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.linalg import eigh

# ----------------------- USER CONFIG -----------------------
FILE_PATH   = "Xact_EST_May2023_Oct2025_combined.csv"
TZ          = "US/Eastern"
OUTDIR      = "./pca_day_outputs_multi"
DATE_START  = dt.date(2023, 7, 1)    # inclusive
DATE_END    = dt.date(2023, 7, 7)    # inclusive
UNC_FILTER_MULT = 3.0
AGG         = "mean"                 # "median", "mean", or like "p95"
SITE_NAME   = "Pittsburgh"           # for Astral label only
LAT, LON    = 40.4406, -79.9959      # PIT
SEED        = 50
np.random.seed(SEED)
os.makedirs(OUTDIR, exist_ok=True)

# ---------------------- Load & pair ------------------------
df = pd.read_csv(FILE_PATH)
if "TIME" not in df.columns:
    raise ValueError("Expected a TIME column in the input CSV.")
df["TIME"] = pd.to_datetime(df["TIME"], utc=True).dt.tz_convert(TZ)
df = df.set_index("TIME").sort_index()

headers   = df.columns.tolist()
conc_cols = [c for c in headers if "(ng/m3)" in c and "uncert" not in c.lower()]
unc_cols  = [c for c in headers if "uncert" in c.lower()]

def element_from_conc(col):
    # robust split on space or '('
    raw = col.split("(ng/m3)")[0].strip()
    return raw.split()[0].strip("()")

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

# Fix S uncertainty if oddly named
if "S" in metal_to_conc and "S" not in metal_to_unc:
    for u in unc_cols:
        if u.lower().startswith("s ") or u.lower().startswith("s(") or "s uncert" in u.lower():
            metal_to_unc["S"] = u; break

metals_all = sorted(metal_to_conc.keys())
if not metals_all:
    raise ValueError("No paired metals found.")

# -------------------------------------------------------------
# Optional inclusion/exclusion lists
# -------------------------------------------------------------
# Define the metals you explicitly want to INCLUDE or EXCLUDE.
# Leave as empty lists if not using.
METALS_INCLUDE = [
    # e.g. "K", "Fe", "Zn", "Ca", "Pb"
]
METALS_EXCLUDE = ["K"
    # e.g. "Na", "Cl", "S"
]

# If you specify INCLUDE, use intersection (only those)
if METALS_INCLUDE:
    metals_all = [m for m in metals_all if m in METALS_INCLUDE]

# If you specify EXCLUDE, remove those
if METALS_EXCLUDE:
    metals_all = [m for m in metals_all if m not in METALS_EXCLUDE]

print(f"Metals to use (after include/exclude filtering): {metals_all}")

# --------------- Sunrise / Sunset helper -------------------
def sun_times_pinned(date_obj: dt.date):
    try:
        from astral.sun import sun
        from astral import LocationInfo
        city = LocationInfo(SITE_NAME, "USA", TZ, LAT, LON)
        s = sun(city.observer, date=date_obj)
        sr, ss = pd.Timestamp(s["sunrise"]), pd.Timestamp(s["sunset"])
    except Exception:
        # Fallback approx for PIT (typical summer values)
        sr = pd.Timestamp(dt.datetime(date_obj.year, date_obj.month, date_obj.day, 5, 55))
        ss = pd.Timestamp(dt.datetime(date_obj.year, date_obj.month, date_obj.day, 20, 54))
    sr = (sr.tz_convert(TZ) if sr.tzinfo else sr.tz_localize(TZ))
    ss = (ss.tz_convert(TZ) if ss.tzinfo else ss.tz_localize(TZ))
    # Pin times to the given date (ignore DST drift artifacts)
    sr = pd.Timestamp.combine(pd.Timestamp(date_obj), sr.time()).tz_localize(TZ)
    ss = pd.Timestamp.combine(pd.Timestamp(date_obj), ss.time()).tz_localize(TZ)
    if ss <= sr:
        ss = ss + pd.Timedelta(days=1)
    return sr, ss

# ---------------- Aggregation helper -----------------------
def reduce_block(block: pd.DataFrame, how: str):
    h = how.lower()
    if h == "median": return block.median(skipna=True)
    if h == "mean":   return block.mean(skipna=True)
    if h.startswith("p") and h[1:].isdigit():
        q = int(h[1:]) / 100.0
        return block.quantile(q=q, interpolation="linear", numeric_only=True)
    raise ValueError("AGG must be 'median', 'mean', or like 'p95'.")

# --------------- Build stacked Day/Night matrix ------------
rows = []                 # will hold Series for each Day/Night per date
row_index = []            # e.g., "2023-07-01 Day", "2023-07-01 Night"
sun_meta = []             # keep for export

cur_date = DATE_START
while cur_date <= DATE_END:
    # Select 24h window for the day
    start_local = pd.Timestamp(dt.datetime.combine(cur_date, dt.time(0,0)), tz=TZ)
    end_local   = start_local + pd.Timedelta(days=1)
    dfd = df.loc[start_local:end_local - pd.Timedelta(seconds=1)].copy()
    if dfd.empty:
        cur_date += dt.timedelta(days=1)
        continue

    # Day/Night flags for that day
    sunrise_ts, sunset_ts = sun_times_pinned(cur_date)
    is_day = (dfd.index >= sunrise_ts) & (dfd.index < sunset_ts)

    # RAW matrix for this day
    X = pd.DataFrame(index=dfd.index, dtype=float)
    for el in metals_all:
        X[el] = dfd[metal_to_conc[el]].astype(float)

    # 3× uncertainty mask -> NaN
    for el in metals_all:
        if el in metal_to_unc:
            unc = dfd[metal_to_unc[el]].astype(float)
            X.loc[X[el] < (UNC_FILTER_MULT * unc), el] = np.nan

    # Aggregate Day/Night
    vec_day   = reduce_block(X.loc[ is_day], AGG)
    vec_night = reduce_block(X.loc[~is_day], AGG)

    # Stash
    rows.append(vec_day);   row_index.append(f"{cur_date.isoformat()}__Day")
    rows.append(vec_night); row_index.append(f"{cur_date.isoformat()}__Night")
    sun_meta.append({"date": cur_date.isoformat(),
                     "sunrise_local": sunrise_ts, "sunset_local": sunset_ts})

    cur_date += dt.timedelta(days=1)

if not rows:
    raise ValueError("No data found in the requested date range.")

X_dn_all = pd.DataFrame(rows, index=row_index)  # (2*D) × P
# Keep only metals present across ALL rows (no NaNs columnwise)
X_dn_all = X_dn_all.dropna(axis=1)
metals_used = X_dn_all.columns.tolist()
if len(metals_used) == 0:
    raise ValueError("After masking/aggregation, all metals are missing in at least one sample.")

# Save aggregated matrix + sun times
agg_path = os.path.join(OUTDIR, f"aggregated_day_night_{DATE_START.isoformat()}_to_{DATE_END.isoformat()}_{AGG}.csv")
X_dn_all.to_csv(agg_path)

sun_path = os.path.join(OUTDIR, f"sun_times_{DATE_START.isoformat()}_to_{DATE_END.isoformat()}.csv")
pd.DataFrame(sun_meta).to_csv(sun_path, index=False)
print(f"Saved:\n  {agg_path}\n  {sun_path}")

# ---------------- PCA on stacked matrix --------------------
# center by feature mean (no scaling)
Xc   = X_dn_all - X_dn_all.mean(axis=0, skipna=True)
n, p = Xc.shape
Cov  = (Xc.T @ Xc) / (n - 1)           # p×p
eigvals, eigvecs = eigh(Cov)           # ascending
order = np.argsort(eigvals)[::-1]
eigvals, eigvecs = eigvals[order], eigvecs[:, order]

ncomp = min(2, p)
eigvals_sel = eigvals[:ncomp]
eigvecs_sel = eigvecs[:, :ncomp]
explained = np.clip(eigvals_sel, 0, None) / max(np.sum(np.clip(eigvals, 0, None)), 1e-12)

# Scores: n × ncomp
scores = (Xc.values) @ eigvecs_sel
df_scores = pd.DataFrame(scores, index=X_dn_all.index, columns=[f"PC{i+1}" for i in range(ncomp)])
loadings  = pd.DataFrame(eigvecs_sel, index=metals_used, columns=df_scores.columns)

# ----------------------- Plots -----------------------------
# Color by Day/Night, label by date
def label_parts(idx):
    date_str, dn = idx.split("__")
    return date_str, dn

colors_map = {"Day": "orange", "Night": "navy"}
point_colors = [colors_map[label_parts(i)[1]] for i in df_scores.index]
sizes = [90]*len(df_scores)

plt.figure(figsize=(7.3,6.2))
x = df_scores["PC1"].values
y = df_scores["PC2"].values if ncomp > 1 else np.zeros(len(df_scores))
plt.scatter(x, y, c=point_colors, s=sizes, alpha=0.9, edgecolor="white", linewidths=0.6)
for i, idx in enumerate(df_scores.index):
    date_str, dn = label_parts(idx)
    plt.annotate(f"{dn[:1]} {date_str[5:]}", (x[i], y[i]), textcoords="offset points", xytext=(6,6), fontsize=8)
plt.xlabel(f"PC1 ({explained[0]*100:.1f}% var)")
plt.ylabel(f"PC2 ({explained[1]*100:.1f}% var)" if ncomp>1 else "PC2")
plt.title(f"Day vs Night PCA (RAW, {AGG}) — {DATE_START.isoformat()} → {DATE_END.isoformat()}")
# Legend
for k,v in colors_map.items():
    plt.scatter([], [], c=v, s=90, label=k)
plt.legend(frameon=True, title="Group")
plt.tight_layout()
out_sc = os.path.join(OUTDIR, f"pca_2pt_scatter_MULTI_{DATE_START.isoformat()}_{DATE_END.isoformat()}_{AGG}.png")
plt.savefig(out_sc, dpi=220); plt.close(); print(f"Saved: {out_sc}")

# Loadings bar (PC1; PC2 if present)
cols_to_plot = ["PC1"] + (["PC2"] if ncomp>1 else [])
top_k = min(25, len(metals_used))
order_ld = loadings["PC1"].abs().sort_values(ascending=False).index[:top_k]
plt.figure(figsize=(10.8,5.2))
loadings.loc[order_ld, cols_to_plot].plot(kind="bar", width=0.85)
plt.ylabel("Loading")
plt.title(f"PCA Loadings (RAW) — metals used: {len(metals_used)}")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
out_ld = os.path.join(OUTDIR, f"pca_2pt_loadings_MULTI_{DATE_START.isoformat()}_{DATE_END.isoformat()}_{AGG}.png")
plt.savefig(out_ld, dpi=220); plt.close(); print(f"Saved: {out_ld}")

# Explained variance
plt.figure(figsize=(5.6,3.8))
plt.bar(range(1, ncomp+1), explained*100)
plt.xticks(range(1, ncomp+1), [f"PC{i}" for i in range(1, ncomp+1)])
plt.ylabel("Explained variance (%)")
plt.title("PCA Explained Variance (stacked RAW)")
plt.tight_layout()
out_ex = os.path.join(OUTDIR, f"pca_2pt_explained_MULTI_{DATE_START.isoformat()}_{DATE_END.isoformat()}_{AGG}.png")
plt.savefig(out_ex, dpi=220); plt.close(); print(f"Saved: {out_ex}")

print("\nTop |loadings| PC1:")
print(loadings['PC1'].abs().sort_values(ascending=False).head(10))
if ncomp>1:
    print("\nTop |loadings| PC2:")
    print(loadings['PC2'].abs().sort_values(ascending=False).head(10))
print("\nDone.")
