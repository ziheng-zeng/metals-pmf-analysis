import numpy as np
import pandas as pd
from sklearn.decomposition import NMF

# -------------------
# Step 1: Load + slice
# -------------------

FILE_PATH = "Xact_EST_May2023_Oct2025_combined.csv"
TIME_COL = "TIME"
TZ = "US/Eastern"

EXCLUDE_METALS = ["Nb", "Pd", "In", "Cs", "I", "La", "Tl", "Pt", "Rh", "Ru", "Re", "Pr"]

# ---- helpers (same as before, trimmed) ----
def load_xact_csv(file_path: str, tz: str = "US/Eastern", time_col: str = "TIME") -> pd.DataFrame:
    df = pd.read_csv(file_path)
    df[time_col] = pd.to_datetime(df[time_col], utc=True).dt.tz_convert(tz)
    return df.set_index(time_col).sort_index()

def _element_from_conc(col: str) -> str:
    if "(ng/m3)" in col:
        raw = col.split("(ng/m3)")[0].strip()
        return raw.split()[0].strip("()")
    return col.split()[0].strip()

def pair_metals_with_uncert(df: pd.DataFrame):
    headers = df.columns.tolist()
    conc_cols = [c for c in headers if "(ng/m3)" in c and "uncert" not in c.lower()]
    unc_cols  = [c for c in headers if "uncert" in c.lower()]
    metal_to_conc, metal_to_unc = {}, {}

    for c in conc_cols:
        el = _element_from_conc(c)
        match = None
        for u in unc_cols:
            if u.startswith(el + " ") or u.startswith(el + "(") or (el + " " in u):
                match = u
                break
        if match:
            metal_to_conc[el] = c
            metal_to_unc[el]  = match

    if "S" in metal_to_conc and "S" not in metal_to_unc:
        for u in unc_cols:
            if u.lower().startswith("s ") or u.lower().startswith("s(") or "s uncert" in u.lower():
                metal_to_unc["S"] = u
                break

    metals_all = sorted(metal_to_conc.keys())
    return metals_all, metal_to_conc, metal_to_unc

def build_X_U(df: pd.DataFrame, metals: list, metal_to_conc: dict, metal_to_unc: dict):
    X = pd.DataFrame(index=df.index, columns=metals, dtype=float)
    U = pd.DataFrame(index=df.index, columns=metals, dtype=float)
    for m in metals:
        X[m] = df[metal_to_conc[m]].astype(float)
        U[m] = df[metal_to_unc[m]].astype(float) if m in metal_to_unc else np.nan
    return X, U

def apply_unc_filter(X: pd.DataFrame, U: pd.DataFrame, mult: float = 1.0):
    mask = (X >= (mult * U)) | U.isna()
    Xf = X.where(mask, 0.0).clip(lower=0)
    return Xf

from sklearn.cluster import KMeans
import hdbscan
import numpy as np

def cluster_W(
    W,
    method="kmeans",
    n_clusters=3,
    min_cluster_size=10,
    min_samples=5,
    random_state=0
):
    """
    Cluster rows of W (time × factors).
    """
    X = W.values

    if method == "kmeans":
        km = KMeans(
            n_clusters=n_clusters,
            random_state=random_state,
            n_init=20
        )
        labels = km.fit_predict(X)

    elif method == "hdbscan":
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples
        )
        labels = clusterer.fit_predict(X)

    else:
        raise ValueError("method must be 'kmeans' or 'hdbscan'")

    return labels

def summarize_cluster_profiles(X_raw: pd.DataFrame, labels: np.ndarray, topn: int = 12):
    """
    Mean metal profile per cluster label.
    """
    s = pd.Series(labels, index=X_raw.index, name="cluster")
    out = {}
    for lab in sorted(s.unique()):
        seg = X_raw.loc[s == lab]
        if len(seg) == 0:
            continue
        out[lab] = seg.mean().sort_values(ascending=False).head(topn)
    return out


def find_cluster_events(index: pd.DatetimeIndex, labels: np.ndarray,
                        min_hours: int = 3, ignore_label: int = -1):
    """
    Find contiguous runs of the same cluster label.
    """
    s = pd.Series(labels, index=index, name="cluster")
    events = []
    start = None
    current = None

    for t, lab in s.items():
        if lab == ignore_label:
            if start is not None:
                duration = int((t - start) / pd.Timedelta(hours=1))
                if duration >= min_hours:
                    events.append((start, t, current, duration))
                start, current = None, None
            continue

        if current is None:
            start, current = t, lab
        elif lab != current:
            duration = int((t - start) / pd.Timedelta(hours=1))
            if duration >= min_hours:
                events.append((start, t, current, duration))
            start, current = t, lab

    if start is not None:
        duration = int((s.index[-1] - start) / pd.Timedelta(hours=1))
        if duration >= min_hours:
            events.append((start, s.index[-1], current, duration))

    return pd.DataFrame(events, columns=["start", "end", "cluster", "duration_hours"])

# ---- run Step 1 ----
df = load_xact_csv(FILE_PATH, tz=TZ, time_col=TIME_COL)
metals_all, metal_to_conc, metal_to_unc = pair_metals_with_uncert(df)
metals = [m for m in metals_all if m not in EXCLUDE_METALS]

# # Slice to a simple window
# t0 = "2023-06-25"
# t1 = "2023-07-10"
# dfw = df.loc[t0:t1].copy()

# Slice to October 2024 (full month)
t0 = "2024-10-01"
t1 = "2024-10-31 23:59:59"
dfw = df.loc[t0:t1].copy()


X, U = build_X_U(dfw, metals, metal_to_conc, metal_to_unc)
Xf = apply_unc_filter(X, U, mult=1.0)

# Check for NaNs
print("Rows in window:", len(Xf))
print("Metals used:", len(metals))
print("Top medians (sanity):")
print(Xf.median().sort_values(ascending=False).head(12))

nan_by_metal = Xf.isna().mean().sort_values(ascending=False)
print("Metals with most NaNs:")
print(nan_by_metal.head(15))

# If any metal is mostly NaN, drop it for now
drop_m = nan_by_metal[nan_by_metal > 0.2].index.tolist()  # >20% NaN in this short window
print("Dropping metals:", drop_m)
Xf = Xf.drop(columns=drop_m).fillna(0.0)
print("Metals kept:", Xf.shape[1])

# Simple check on coke metals
COKE_METALS = ["Zn", "Pb", "As", "Se"]
available = [m for m in COKE_METALS if m in Xf.columns]
print("Coke metals available:", available)

dfw["coke_score"] = Xf[available].sum(axis=1)

print("\nTop 25 coke_score hours:")
print(dfw["coke_score"].sort_values(ascending=False).head(25))


# Fie a small NMF (K=4)
X_nmf = Xf.copy()  # raw, already nonnegative
N_FACTORS = 4
nmf = NMF(n_components=N_FACTORS, init="nndsvda", random_state=0, max_iter=2000)

W = nmf.fit_transform(X_nmf.values)
H = nmf.components_

W = pd.DataFrame(W, index=dfw.index, columns=[f"F{i+1}" for i in range(N_FACTORS)])
H = pd.DataFrame(H, index=W.columns, columns=X_nmf.columns)

def print_top_metals(H, topn=10):
    """
    Print top metals for each NMF factor.
    H: DataFrame (factors x metals)
    """
    for f in H.index:
        print(f"\n {f} top metals:")
        print(
            H.loc[f]
            .sort_values(ascending=False)
            .head(topn)
        )

print_top_metals(H, topn=12)

corrs = W.apply(lambda col: col.corr(dfw["coke_score"]))
print("\nCorrelation of factors with coke_score:")
print(corrs.sort_values(ascending=False))

coke_factor = corrs.sort_values(ascending=False).index[0]
print("\nBest coke factor:", coke_factor)

print("\nTop 25 hours for best coke factor:")
print(W[coke_factor].sort_values(ascending=False).head(25))

labels_h = cluster_W(
    W,
    method="hdbscan",
    min_cluster_size=12,   # half-day-ish
    min_samples=4
)
dfw["cluster_h"] = labels_h

print("\nHDBSCAN cluster sizes:")
print(dfw["cluster_h"].value_counts())
top_mask = dfw["coke_score"] >= dfw["coke_score"].quantile(0.95)  # top 5%
print("\nTop-5% coke hours by HDBSCAN cluster:")
print(pd.crosstab(dfw["cluster_h"], top_mask))

profiles = summarize_cluster_profiles(Xf, dfw["cluster_h"].values, topn=12)
for lab in sorted(profiles.keys()):
    print(f"\nCluster {lab} mean top metals:")
    print(profiles[lab])

events = find_cluster_events(dfw.index, dfw["cluster_h"].values, min_hours=3, ignore_label=-1)
events = events.sort_values("duration_hours", ascending=False)
print("\nLongest cluster runs:")
print(events.head(20))

# =========================
# PLOTTING # ============================================================
# Option A: Time series of each NMF factor (focus on Zn-rich)
# ============================================================

import matplotlib.pyplot as plt

plt.figure(figsize=(12, 4))
for col in W.columns:
    plt.plot(W.index, W[col], label=col, alpha=0.8)

plt.title("NMF factor time series (October 2024)")
plt.ylabel("Factor strength")
plt.xlabel("Time")
plt.legend()
plt.tight_layout()
plt.show()


# ============================================================
# Option B: Factor–metal scatter plots (Zn, Pb, As)
# ============================================================

KEY_METALS = [m for m in ["Zn", "Pb", "As"] if m in Xf.columns]

for metal in KEY_METALS:
    plt.figure(figsize=(5, 4))
    plt.scatter(
        Xf[metal],
        W["F2"],   # Zn-rich factor
        s=20,
        alpha=0.6
    )
    plt.xlabel(f"{metal} concentration (ng/m³)")
    plt.ylabel("F2 factor strength")
    plt.title(f"F2 vs {metal}")
    plt.tight_layout()
    plt.show()


# ============================================================
# Option C: Factor profile bar charts (normalized)
# ============================================================

TOPN = 12

for f in H.index:
    prof = H.loc[f].sort_values(ascending=False).head(TOPN)
    prof_norm = prof / prof.sum()

    plt.figure(figsize=(6, 3))
    prof_norm[::-1].plot(kind="barh")
    plt.xlabel("Fraction of factor profile")
    plt.title(f"{f} normalized metal profile (top {TOPN})")
    plt.tight_layout()
    plt.show()

# # ----------------------------------------
# # Step 2: Fit a tiny NMF (K=3) and inspect H
# # ----------------------------------------
# # Uncomment when Step 1 looks sane
# print("NaN count before fill:", int(Xf.isna().sum().sum()))
# Xf = Xf.fillna(0.0)
# print("NaN count after fill:", int(Xf.isna().sum().sum()))
#
# K = 3
# nmf = NMF(n_components=K, init="nndsvda", random_state=0, max_iter=2000)
# W = nmf.fit_transform(Xf.values)
# H = nmf.components_
# W = pd.DataFrame(W, index=Xf.index, columns=[f"F{k+1}" for k in range(K)])
# H = pd.DataFrame(H, index=W.columns, columns=Xf.columns)
#
# for f in H.index:
#     print("\n", f, "top metals:")
#     print(H.loc[f].sort_values(ascending=False).head(10))
# # #
# # ----------------------------------------
# # Step 3: Identify "fireworks hours" by top factor score
# # ----------------------------------------
# # Uncomment after Step 2
#
# top_factor = W.idxmax(axis=1)  # factor with largest contribution each hour
# print("\nFactor dominance counts:")
# print(top_factor.value_counts())
#
# # Show the 20 hours with biggest value of F? (pick the factor you think is fireworks)
# # Example: choose the factor that has K/Ba/Cu high in H
# fw_factor = "F1"  # <-- update after you inspect H
# print("\nTop hours for", fw_factor)
# print(W[fw_factor].sort_values(ascending=False).head(30))
# print("\nTop hours for F2 (fireworks factor):")
# print(W["F2"].sort_values(ascending=False).head(30))
#
# # Show the actual metal concentraions during top hours
# key = ["K","Ba","Cu","Sr","Bi","S","Al","Cl"]
# key = [m for m in key if m in Xf.columns]
#
# top_f2_times = W["F2"].sort_values(ascending=False).head(10).index
# print("\nKey metals during top F2 hours:")
# print(Xf.loc[top_f2_times, key])
#
# # import matplotlib.pyplot as plt
# #
# # W[["F1","F2","F3"]].plot(figsize=(12,4))
# # plt.ylabel("NMF factor score (W)")
# # plt.title("Factor contributions over time")
# # plt.tight_layout()
# # plt.show()
#
# # ----------------------------------------
# # Step 4: Only now add clustering
# # ----------------------------------------
#
# from sklearn.preprocessing import StandardScaler
# from sklearn.cluster import KMeans
#
# Xw = StandardScaler().fit_transform(W.values)
#
# km = KMeans(n_clusters=3, random_state=0, n_init=50)
# labels = km.fit_predict(Xw)
#
# dfw = Xf.copy()
# dfw["cluster"] = labels
#
# print("Cluster sizes:")
# print(dfw["cluster"].value_counts().sort_index())
#
# # Mean metal profile per cluster (in original Xf space)
# for c in sorted(dfw["cluster"].unique()):
#     mean_prof = Xf.loc[dfw["cluster"] == c].mean().sort_values(ascending=False).head(12)
#     print(f"\nCluster {c} mean top metals:")
#     print(mean_prof)
#
# fw = W["F2"] > 5.0   # very conservative; adjust after seeing distribution
# print(pd.crosstab(dfw["cluster"], fw))
#
# # Extract cluster 1 as fireworks
# # Find contiguous runs of cluster 1
# lab = dfw["cluster"]
#
# is_fw = (lab == 1)
# fw_runs = []
# start = None
#
# for t, v in is_fw.items():
#     if v and start is None:
#         start = t
#     if (not v) and (start is not None):
#         fw_runs.append((start, t))
#         start = None
#
# if start is not None:
#     fw_runs.append((start, is_fw.index[-1]))
#
# print("Fireworks runs:")
# for (a, b) in fw_runs:
#     dur = (b - a) / pd.Timedelta(hours=1)
#     print(a, "->", b, f"({dur:.0f} h)")
#
# # Now add HDBSCAN clustering
# from sklearn.preprocessing import StandardScaler
# import hdbscan
#
# Xw = StandardScaler().fit_transform(W.values)
#
# clusterer = hdbscan.HDBSCAN(
#     min_cluster_size=6,   # ~6 hours minimum cluster "mass"
#     min_samples=4
# )
# labels_h = clusterer.fit_predict(Xw)
#
# dfh = Xf.copy()
# dfh["cluster_h"] = labels_h
#
# print("HDBSCAN cluster sizes:")
# print(dfh["cluster_h"].value_counts().head(15))
#
# # Mean metal profiles per cluster (skip -1 noise)
# for lab in sorted(dfh["cluster_h"].unique()):
#     if lab == -1:
#         continue
#     mean_prof = Xf.loc[dfh["cluster_h"] == lab].mean().sort_values(ascending=False).head(12)
#     print(f"\nCluster {lab} mean top metals:")
#     print(mean_prof)
#
# # Where did fireworks go?
# fw = W["F2"] > 5.0
# print("\nFireworks (F2>5) by HDBSCAN cluster:")
# print(pd.crosstab(dfh["cluster_h"], fw))
#
# def contiguous_runs(index, labels, target_label):
#     s = pd.Series(labels, index=index)
#     is_t = (s == target_label)
#     runs = []
#     start = None
#     for t, v in is_t.items():
#         if v and start is None:
#             start = t
#         if (not v) and (start is not None):
#             runs.append((start, t))
#             start = None
#     if start is not None:
#         runs.append((start, is_t.index[-1]))
#     return runs
#
# labs = sorted(set(labels_h) - {-1})
# for lab in labs:
#     runs = contiguous_runs(W.index, labels_h, lab)
#     if runs:
#         print(f"\nCluster {lab} runs:")
#         for a,b in runs:
#             print(a, "->", b, f"({(b-a)/pd.Timedelta(hours=1):.0f} h)")
#

# # ============================
# # SAVE OUTPUTS (paste at end)
# # ============================
# import os
# import json
# import matplotlib.pyplot as plt
# import pandas as pd
# import numpy as np
#
# def _safe_write_text(path: str, text: str):
#     with open(path, "w", encoding="utf-8") as f:
#         f.write(text)
#
# def _to_csv(obj, path: str):
#     if obj is None:
#         return
#     if isinstance(obj, (pd.Series, pd.DataFrame)):
#         obj.to_csv(path)
#     else:
#         # try to coerce
#         pd.DataFrame(obj).to_csv(path, index=False)
#
# def build_fireworks_runs(index: pd.DatetimeIndex, is_fireworks: pd.Series):
#     """
#     Returns a DataFrame of contiguous True runs with start/end/duration_hours.
#     """
#     if not isinstance(is_fireworks, pd.Series):
#         is_fireworks = pd.Series(is_fireworks, index=index)
#
#     runs = []
#     start = None
#     for t, v in is_fireworks.items():
#         if v and start is None:
#             start = t
#         if (not v) and (start is not None):
#             end = t
#             dur = (end - start) / pd.Timedelta(hours=1)
#             runs.append((start, end, float(dur)))
#             start = None
#     if start is not None:
#         end = index[-1]
#         dur = (end - start) / pd.Timedelta(hours=1)
#         runs.append((start, end, float(dur)))
#
#     return pd.DataFrame(runs, columns=["start", "end", "duration_hours"])
#
# def top_metals_per_factor(H: pd.DataFrame, topn: int = 10) -> pd.DataFrame:
#     """
#     Returns a long-form table: factor, rank, metal, loading
#     """
#     rows = []
#     for f in H.index:
#         s = H.loc[f].sort_values(ascending=False).head(topn)
#         for r, (m, val) in enumerate(s.items(), start=1):
#             rows.append({"factor": f, "rank": r, "metal": m, "loading": float(val)})
#     return pd.DataFrame(rows)
#
# def mean_profile_by_cluster(X: pd.DataFrame, labels: np.ndarray, topn: int = 12) -> pd.DataFrame:
#     """
#     Returns a long-form table: cluster, rank, metal, mean_value
#     """
#     lab_series = pd.Series(labels, index=X.index, name="cluster")
#     rows = []
#     for lab in sorted(lab_series.unique()):
#         seg = X.loc[lab_series == lab]
#         if len(seg) == 0:
#             continue
#         s = seg.mean().sort_values(ascending=False).head(topn)
#         for r, (m, val) in enumerate(s.items(), start=1):
#             rows.append({"cluster": int(lab), "rank": r, "metal": m, "mean_value": float(val)})
#     return pd.DataFrame(rows)
#
# def save_run_outputs(
#     outdir: str,
#     settings: dict,
#     metals_used: list,
#     Xf: pd.DataFrame,
#     W: pd.DataFrame,
#     H: pd.DataFrame,
#     labels_kmeans=None,
#     labels_hdbscan=None,
#     fireworks_factor: str = None,
#     fireworks_threshold: float = None,
# ):
#     os.makedirs(outdir, exist_ok=True)
#
#     # ---- settings ----
#     _safe_write_text(os.path.join(outdir, "settings.json"), json.dumps(settings, indent=2))
#
#     # ---- metals used ----
#     pd.Series(metals_used, name="metals_used").to_csv(os.path.join(outdir, "metals_used.csv"), index=False)
#
#     # ---- core matrices ----
#     Xf.to_csv(os.path.join(outdir, "Xf_filtered_concentrations.csv"))
#     W.to_csv(os.path.join(outdir, "W_factor_timeseries.csv"))
#     H.to_csv(os.path.join(outdir, "H_factor_profiles.csv"))
#
#     # ---- readable summaries ----
#     topH = top_metals_per_factor(H, topn=10)
#     topH.to_csv(os.path.join(outdir, "H_top_metals_long.csv"), index=False)
#
#     dominant = W.idxmax(axis=1).value_counts()
#     dominant.to_csv(os.path.join(outdir, "factor_dominance_counts.csv"))
#
#     # ---- clusters ----
#     df_labels = pd.DataFrame(index=W.index)
#     if labels_kmeans is not None:
#         df_labels["cluster_kmeans"] = labels_kmeans
#         prof = mean_profile_by_cluster(Xf, np.asarray(labels_kmeans), topn=12)
#         prof.to_csv(os.path.join(outdir, "kmeans_cluster_mean_profiles_long.csv"), index=False)
#
#     if labels_hdbscan is not None:
#         df_labels["cluster_hdbscan"] = labels_hdbscan
#         profh = mean_profile_by_cluster(Xf, np.asarray(labels_hdbscan), topn=12)
#         profh.to_csv(os.path.join(outdir, "hdbscan_cluster_mean_profiles_long.csv"), index=False)
#
#     if df_labels.shape[1] > 0:
#         df_labels.to_csv(os.path.join(outdir, "hourly_cluster_labels.csv"))
#
#     # ---- fireworks hours + runs (optional but very useful) ----
#     if (fireworks_factor is not None) and (fireworks_factor in W.columns) and (fireworks_threshold is not None):
#         is_fw = W[fireworks_factor] > fireworks_threshold
#         fw_hours = pd.DataFrame({
#             "W_fireworks": W[fireworks_factor],
#             "is_fireworks": is_fw.astype(int),
#         }, index=W.index)
#         fw_hours.to_csv(os.path.join(outdir, "fireworks_hours.csv"))
#
#         fw_runs = build_fireworks_runs(W.index, is_fw)
#         fw_runs.to_csv(os.path.join(outdir, "fireworks_runs.csv"), index=False)
#
#     # ---- plots ----
#     # W timeseries plot
#     ax = W.plot(figsize=(12, 4))
#     ax.set_ylabel("NMF factor score (W)")
#     ax.set_title("NMF factor contributions over time")
#     plt.tight_layout()
#     plt.savefig(os.path.join(outdir, "W_factors_timeseries.png"), dpi=150)
#     plt.close()
#
#     # Optional: fireworks factor plot (if provided)
#     if (fireworks_factor is not None) and (fireworks_factor in W.columns):
#         ax = W[fireworks_factor].plot(figsize=(12, 3))
#         ax.set_ylabel(f"{fireworks_factor} score")
#         ax.set_title(f"Fireworks factor ({fireworks_factor}) over time")
#         if fireworks_threshold is not None:
#             ax.axhline(fireworks_threshold, linestyle="--")
#         plt.tight_layout()
#         plt.savefig(os.path.join(outdir, f"{fireworks_factor}_timeseries.png"), dpi=150)
#         plt.close()
#
#     print(f"\n✅ Saved run outputs to: {outdir}")
#
#
# # ----------------------------
# # CALL THE SAVER (edit these)
# # ----------------------------
# # Choose a human-readable run id that encodes your choices
# RUN_ID = "2023-06-25_to_2023-07-10_K3_raw_unc1_kmeans3_hdbscan"
# OUTDIR = os.path.join("outputs", RUN_ID)
#
# # Build settings dict from variables you already have (edit keys as needed)
# _settings = {
#     "run_id": RUN_ID,
#     "time_start": str(Xf.index.min()),
#     "time_end": str(Xf.index.max()),
#     "rows": int(len(Xf)),
#     "n_factors": int(W.shape[1]),
#     "preprocess_mode": globals().get("PREPROCESS_MODE", "unknown"),
#     "unc_mult": float(globals().get("UNC_MULT", np.nan)) if "UNC_MULT" in globals() else None,
#     "tz": globals().get("TZ", "unknown"),
#     "excluded_metals": globals().get("EXCLUDE_METALS", []),
#     "notes": "Toy window run; validated fireworks cluster; save for reference.",
# }
#
# # Pick fireworks factor + threshold for saving (for your K=3 example)
# FIREWORKS_FACTOR = "F2"
# FIREWORKS_THRESH = 5.0
#
# # Provide cluster labels if you have them; otherwise set to None
# LABELS_KMEANS = globals().get("labels", None)     # your kmeans labels variable name
# LABELS_HDBSCAN = globals().get("labels_h", None)  # your hdbscan labels variable name
#
# save_run_outputs(
#     outdir=OUTDIR,
#     settings=_settings,
#     metals_used=list(Xf.columns),
#     Xf=Xf,
#     W=W,
#     H=H,
#     labels_kmeans=LABELS_KMEANS,
#     labels_hdbscan=LABELS_HDBSCAN,
#     fireworks_factor=FIREWORKS_FACTOR,
#     fireworks_threshold=FIREWORKS_THRESH,
# )
# import numpy as np
# import matplotlib.pyplot as plt
#
# def plot_H_heatmap(H: pd.DataFrame, outpath: str, top_m: int = 25):
#     """
#     Heatmap of H (factors x metals). We keep the top_m metals by total loading across factors.
#     Uses log1p scaling so big metals don't crush small ones.
#     """
#     # pick metals that matter most overall
#     metal_score = H.sum(axis=0).sort_values(ascending=False)
#     keep_metals = metal_score.head(top_m).index.tolist()
#     Hs = H[keep_metals].copy()
#
#     Z = np.log1p(Hs.values)
#
#     fig, ax = plt.subplots(figsize=(0.45*len(keep_metals) + 4, 0.55*Hs.shape[0] + 2))
#     im = ax.imshow(Z, aspect="auto")
#     ax.set_yticks(range(Hs.shape[0]))
#     ax.set_yticklabels(Hs.index)
#     ax.set_xticks(range(len(keep_metals)))
#     ax.set_xticklabels(keep_metals, rotation=90)
#     ax.set_title(f"H factor profiles (log1p), top {top_m} metals")
#     fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02, label="log1p(loading)")
#     plt.tight_layout()
#     plt.savefig(outpath, dpi=160)
#     plt.close()
#
# # usage:
# plot_H_heatmap(H, f"{OUTDIR}/H_profiles_heatmap.png", top_m=25)
#
# def plot_W_timeseries(W: pd.DataFrame, outpath: str, fireworks_factor: str = "F2", thr: float = 5.0):
#     fig, ax = plt.subplots(figsize=(12, 4))
#     W.plot(ax=ax, linewidth=1)
#     ax.set_ylabel("Factor score (W)")
#     ax.set_title("NMF factor contributions over time")
#
#     # highlight fireworks hours
#     if fireworks_factor in W.columns:
#         fw = W[fireworks_factor] > thr
#         if fw.any():
#             ax.fill_between(W.index, 0, 1, where=fw.values,
#                             transform=ax.get_xaxis_transform(),
#                             alpha=0.15, label="fireworks hours")
#             ax.legend(loc="upper right")
#
#     plt.tight_layout()
#     plt.savefig(outpath, dpi=160)
#     plt.close()
#
# plot_W_timeseries(W, f"{OUTDIR}/W_timeseries.png", fireworks_factor="F2", thr=5.0)
#
# def plot_cluster_profiles(
#     X_raw: pd.DataFrame,
#     labels: np.ndarray,
#     outpath: str,
#     top_m: int = 12,
#     ignore_label=None
# ):
#     lab = pd.Series(labels, index=X_raw.index, name="cluster")
#     clusters = sorted(lab.unique())
#
#     # compute mean profiles
#     profs = {}
#     for c in clusters:
#         if ignore_label is not None and c == ignore_label:
#             continue
#         seg = X_raw.loc[lab == c]
#         if len(seg) < 3:
#             continue
#         profs[c] = seg.mean()
#
#     if len(profs) == 0:
#         print("No clusters to plot.")
#         return
#
#     # choose common metals to display: top_m by max mean across clusters
#     prof_df = pd.DataFrame(profs).T  # cluster x metal
#     keep = prof_df.max(axis=0).sort_values(ascending=False).head(top_m).index
#     prof_df = prof_df[keep]
#
#     # normalize rows to fractions so shapes compare
#     prof_frac = prof_df.div(prof_df.sum(axis=1), axis=0).fillna(0)
#
#     fig, ax = plt.subplots(figsize=(10, 4 + 0.35*len(prof_frac)))
#     prof_frac.plot(kind="barh", stacked=True, ax=ax)
#     ax.set_xlabel("Fraction of mean profile (top metals)")
#     ax.set_ylabel("Cluster")
#     ax.set_title(f"Cluster mean profiles (normalized), top {top_m} metals")
#     ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", title="Metal")
#     plt.tight_layout()
#     plt.savefig(outpath, dpi=160)
#     plt.close()
#
# # KMeans:
# if LABELS_KMEANS is not None:
#     plot_cluster_profiles(Xf, np.asarray(LABELS_KMEANS), f"{OUTDIR}/kmeans_cluster_profiles.png", top_m=12)
#
# # HDBSCAN (no noise in your run, but if you get -1 later pass ignore_label=-1)
# if LABELS_HDBSCAN is not None:
#     plot_cluster_profiles(Xf, np.asarray(LABELS_HDBSCAN), f"{OUTDIR}/hdbscan_cluster_profiles.png", top_m=12, ignore_label=-1)
#
# from sklearn.preprocessing import StandardScaler
# from sklearn.decomposition import PCA
#
# def plot_factor_space_pca(W: pd.DataFrame, labels: np.ndarray, outpath: str):
#     Xw = StandardScaler().fit_transform(W.values)
#     Z = PCA(n_components=2, random_state=0).fit_transform(Xw)
#
#     fig, ax = plt.subplots(figsize=(6, 5))
#     sc = ax.scatter(Z[:, 0], Z[:, 1], c=labels, s=12, alpha=0.8)
#     ax.set_title("Hours in factor space (PCA of W)")
#     ax.set_xlabel("PC1")
#     ax.set_ylabel("PC2")
#     plt.colorbar(sc, ax=ax, fraction=0.03, pad=0.02, label="cluster")
#     plt.tight_layout()
#     plt.savefig(outpath, dpi=160)
#     plt.close()
#
# if LABELS_KMEANS is not None:
#     plot_factor_space_pca(W, np.asarray(LABELS_KMEANS), f"{OUTDIR}/kmeans_factor_space_pca.png")
# if LABELS_HDBSCAN is not None:
#     plot_factor_space_pca(W, np.asarray(LABELS_HDBSCAN), f"{OUTDIR}/hdbscan_factor_space_pca.png")
