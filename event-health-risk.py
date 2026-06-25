import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pytz
import matplotlib as mpl

# === Load and preprocess data ===
df = pd.read_csv("Xact_EST_May2023_Oct2025_combined.csv")
df['TIME'] = pd.to_datetime(df['TIME'], utc=True).dt.tz_convert('US/Eastern')
df.set_index('TIME', inplace=True)

# === Define exclusion periods ===
exclude_periods = [
    (pd.Timestamp('2024-01-09', tz='US/Eastern'), pd.Timestamp('2024-02-13', tz='US/Eastern')),
    (pd.Timestamp('2024-07-02', tz='US/Eastern'), pd.Timestamp('2024-08-08', tz='US/Eastern')),
]

# Create a mask for valid data outside exclusion windows
global_valid_mask = pd.Series(True, index=df.index)
for start, end in exclude_periods:
    global_valid_mask &= ~((df.index >= start) & (df.index <= end))

# === Define NATTS thresholds ===
natts_thresholds = {
    "As": {"cancer": 0.00023, "noncancer": 0.003},
    "Cr": {"cancer": 0.00008, "noncancer": 0.01},
    "Pb": {"cancer": None, "noncancer": 0.015},
    "Se": {"cancer": None, "noncancer": 2.0}

}
# natts_thresholds = {
#     "As": {"cancer": 0.00023, "noncancer": 0.003},
#     "Cr": {"cancer": 0.00008, "noncancer": 0.01},
#     "Ni": {"cancer": 0.0021, "noncancer": 0.009},
#     "Pb": {"cancer": None, "noncancer": 0.015},
#     # "Cd": {"cancer": 0.00056, "noncancer": 0.001},
#     "Mn": {"cancer": None, "noncancer": 0.03},
#     # "Be": {"cancer": 0.00042, "noncancer": 0.002},
#
#     # "Sb": {"cancer": None, "noncancer": 0.02},
#
#     # "Co": {"cancer": None, "noncancer": 0.01},
#     # "Se": {"cancer": None, "noncancer": 2.0}
# }

# === Define event periods ===
eastern = pytz.timezone('US/Eastern')
event_periods = {
    "Fireworks": [
        (pd.Timestamp('2023-06-30 17:00', tz=eastern), pd.Timestamp('2023-07-01 12:00', tz=eastern)),
        (pd.Timestamp('2023-07-01 17:00', tz=eastern), pd.Timestamp('2023-07-02 12:00', tz=eastern)),
        (pd.Timestamp('2023-07-03 17:00', tz=eastern), pd.Timestamp('2023-07-04 16:00', tz=eastern)),
        (pd.Timestamp('2023-07-04 17:00', tz=eastern), pd.Timestamp('2023-07-05 12:00', tz=eastern)),
        (pd.Timestamp('2023-07-05 17:00', tz=eastern), pd.Timestamp('2023-07-06 12:00', tz=eastern)),
        (pd.Timestamp('2024-06-29 20:00', tz=eastern), pd.Timestamp('2024-06-29 23:59', tz=eastern))
    ],
    "Wildfire": [
        (pd.Timestamp('2023-06-05 12:00', tz=eastern), pd.Timestamp('2023-06-08', tz=eastern)),
        (pd.Timestamp('2023-06-28', tz=eastern), pd.Timestamp('2023-06-30', tz=eastern)),
        (pd.Timestamp('2023-07-16', tz=eastern), pd.Timestamp('2023-07-18', tz=eastern)),
    ],
    "Background": [
        (pd.Timestamp('2023-12-15 00:00', tz=eastern), pd.Timestamp('2023-12-22 23:59', tz=eastern)),
        (pd.Timestamp('2024-04-15 00:00', tz=eastern), pd.Timestamp('2024-04-22 23:59', tz=eastern)),
        (pd.Timestamp('2024-09-13 00:00', tz=eastern), pd.Timestamp('2024-09-20 23:59', tz=eastern))
    ]
}
# --- Add Coke Plume periods (Eastern time) ---
coke_plume_periods = [
    (pd.Timestamp('2024-10-22 23:00', tz=eastern), pd.Timestamp('2024-10-23 09:00', tz=eastern)),
    (pd.Timestamp('2025-03-18 23:00', tz=eastern), pd.Timestamp('2025-03-19 11:00', tz=eastern)),
    (pd.Timestamp('2025-03-11 03:00', tz=eastern), pd.Timestamp('2025-03-11 09:00', tz=eastern)),
    (pd.Timestamp('2024-09-08 21:00', tz=eastern), pd.Timestamp('2024-09-09 09:00', tz=eastern)),
]
event_periods["Coke Plume"] = coke_plume_periods

# === Helper function: compute exceedance % for a metal ===
def compute_event_exceedance_stats(metal_short, conc_col, uncert_col, cancer_thresh, noncancer_thresh):
    results = []
    for event_name, periods in event_periods.items():
        period_mask = pd.Series(False, index=df.index)
        for start, end in periods:
            period_mask |= (df.index >= start) & (df.index <= end)

        combined_mask = period_mask & global_valid_mask & (df[conc_col] > 3 * df[uncert_col])
        total_valid = combined_mask.sum()
        if total_valid == 0:
            results.append((metal_short, event_name, "Cancer", np.nan))
            results.append((metal_short, event_name, "Noncancer", np.nan))
            continue

        cancer_exceed = ((df[conc_col] / 1000) > cancer_thresh) & combined_mask if cancer_thresh else None
        noncancer_exceed = ((df[conc_col] / 1000) > noncancer_thresh) & combined_mask if noncancer_thresh else None

        results.append((metal_short, event_name, "Cancer", 100 * cancer_exceed.sum() / total_valid if cancer_exceed is not None else np.nan))
        results.append((metal_short, event_name, "Noncancer", 100 * noncancer_exceed.sum() / total_valid if noncancer_exceed is not None else np.nan))
    return results

# === Compute stats for As and Cr ===
records = []
for metal_short, limits in natts_thresholds.items():
    conc_col = next((c for c in df.columns if c.startswith(metal_short + " ") and "(ng/m3)" in c and "uncert" not in c.lower()), None)
    uncert_col = next((c for c in df.columns if c.startswith(metal_short + " ") and "uncert" in c.lower()), None)
    if conc_col and uncert_col:
        records.extend(compute_event_exceedance_stats(metal_short, conc_col, uncert_col, limits["cancer"], limits["noncancer"]))

# === Make long DataFrame ===
long_df = pd.DataFrame(records, columns=["Metal", "Event", "Risk Type", "Exceedance (%)"])
long_df.dropna(inplace=True)

# --- Setup style ---
plt.rcParams.update({'font.size': 14})
colors = {"Cancer": "red", "Noncancer": "orange"}

# === Prepare data ===
pivot_df = long_df.pivot(index="Event", columns=["Metal", "Risk Type"], values="Exceedance (%)")

# optional but recommended: enforce a readable event order if present
desired_order = ["Fireworks", "Wildfire", "Coke Plume", "Background"]
present = [e for e in desired_order if e in pivot_df.index]
others  = [e for e in pivot_df.index if e not in present]
pivot_df = pivot_df.loc[present + others]

events = pivot_df.index.tolist()
x = np.arange(len(events))
width = 0.35


# === Generate one plot per metal ===
# --- Plot per metal: show both bars if Cancer exists; otherwise only Noncancer ---
plt.rcParams.update({'font.size': 14})
colors = {"Cancer": "red", "Noncancer": "orange"}

for metal in sorted(long_df["Metal"].unique()):
    # Pull rows for this metal
    mdf = long_df[long_df["Metal"] == metal].copy()

    # Build per-event series (preserve x-axis order from pivot_df.index)
    y_nc = (mdf[mdf["Risk Type"] == "Noncancer"]
              .set_index("Event")["Exceedance (%)"]
              .reindex(pivot_df.index))  # may contain NaN if no data for an event

    y_ca = (mdf[mdf["Risk Type"] == "Cancer"]
              .set_index("Event")["Exceedance (%)"]
              .reindex(pivot_df.index)
              if "Cancer" in mdf["Risk Type"].unique() else None)

    x = np.arange(len(pivot_df.index))
    fig, ax = plt.subplots(figsize=(7, 5))

    if y_ca is not None and not y_ca.dropna().empty:
        width = 0.38
        bars_ca = ax.bar(x - width/2, y_ca.values, width, label="Cancer", color=colors["Cancer"])
        bars_nc = ax.bar(x + width/2, y_nc.values, width, label="Noncancer", color=colors["Noncancer"])
        # labels
        for bar in list(bars_ca) + list(bars_nc):
            h = bar.get_height()
            if not np.isnan(h):
                ax.annotate(f'{h:.1f}%', (bar.get_x() + bar.get_width()/2, h),
                            xytext=(0, 3), textcoords="offset points",
                            ha='center', va='bottom', fontsize=11)
        ax.legend(loc='upper right', fontsize=12)
    else:
        # Noncancer only (e.g., Pb, Se)
        width = 0.6
        bars_nc = ax.bar(x, y_nc.values, width, label="Noncancer", color=colors["Noncancer"])
        for bar in bars_nc:
            h = bar.get_height()
            if not np.isnan(h):
                ax.annotate(f'{h:.1f}%', (bar.get_x() + bar.get_width()/2, h),
                            xytext=(0, 3), textcoords="offset points",
                            ha='center', va='bottom', fontsize=11)
        ax.legend(loc='upper right', fontsize=12)

    ax.set_title(f"{metal}", fontsize=16)
    ax.set_ylabel("% Exceedance")
    ax.set_ylim(0, 110)
    ax.set_xticks(x)
    ax.set_xticklabels(pivot_df.index, rotation=30, ha='right')

    plt.tight_layout()
    plt.show()
