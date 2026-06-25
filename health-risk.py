import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load your data
df = pd.read_csv("Xact_EST_May2023_May2025_combined.csv")
df['TIME'] = pd.to_datetime(df['TIME'], utc=True).dt.tz_convert('US/Eastern')
df.set_index('TIME', inplace=True)

# Exclude bad data periods
exclude_periods = [
    (pd.Timestamp('2024-01-09', tz='US/Eastern'), pd.Timestamp('2024-02-13', tz='US/Eastern')),
    (pd.Timestamp('2024-07-02', tz='US/Eastern'), pd.Timestamp('2024-08-08', tz='US/Eastern')),
]

global_valid_mask = pd.Series(True, index=df.index)
for start, end in exclude_periods:
    global_valid_mask &= ~((df.index >= start) & (df.index <= end))

plt.rcParams['font.size'] = 20

# NATTS thresholds in µg/m³
natts_thresholds = {
    "As": {"cancer": 0.00023, "noncancer": 0.003},
    "Cr": {"cancer": 0.00008, "noncancer": 0.01},
    "Ni": {"cancer": 0.0021, "noncancer": 0.009},
    "Pb": {"cancer": None, "noncancer": 0.015},
    "Mn": {"cancer": None, "noncancer": 0.03},
}

# Rolling averages
rolling_hours = [8, 24]
for metal_short in natts_thresholds.keys():
    conc_col = next((col for col in df.columns if col.startswith(metal_short + " ") and "(ng/m3)" in col and "uncert" not in col.lower()), None)
    uncert_col = next((col for col in df.columns if col.startswith(metal_short + " ") and "uncert" in col.lower()), None)
    if conc_col and uncert_col:
        for h in rolling_hours:
            df[f"{metal_short}_avg_{h}hr"] = df[conc_col].rolling(window=h, min_periods=int(h * 0.75)).mean()
            df[f"{metal_short}_uncert_avg_{h}hr"] = df[uncert_col].rolling(window=h, min_periods=int(h * 0.75)).mean()

# Build plot_data for 1hr, 8hr, 24hr
plot_data = {}
for metal_short, limits in natts_thresholds.items():
    conc_col = next((col for col in df.columns if col.startswith(metal_short + " ") and "(ng/m3)" in col and "uncert" not in col.lower()), None)
    uncert_col = next((col for col in df.columns if col.startswith(metal_short + " ") and "uncert" in col.lower()), None)
    if conc_col and uncert_col:
        # 1-hr
        valid_mask = (df[conc_col] > 3 * df[uncert_col]) & global_valid_mask
        valid_data = df.loc[valid_mask, [conc_col]].copy()
        valid_data.columns = [metal_short]
        plot_data[f"{metal_short}_1hr"] = (valid_data, limits)
        # 8-hr and 24-hr
        for h in rolling_hours:
            avg_col = f"{metal_short}_avg_{h}hr"
            uncert_avg_col = f"{metal_short}_uncert_avg_{h}hr"
            if avg_col in df.columns and uncert_avg_col in df.columns:
                valid_mask = (df[avg_col] > 3 * df[uncert_avg_col]) & global_valid_mask
                valid_data = df.loc[valid_mask, [avg_col]].copy()
                valid_data.columns = [metal_short]
                plot_data[f"{metal_short}_{h}hr"] = (valid_data, limits)

# Organize into grouped dictionaries
grouped_plot_data = {"1-hr": {}, "8-hr": {}, "24-hr": {}}
for key, val in plot_data.items():
    if key.endswith("1hr"):
        grouped_plot_data["1-hr"][key] = val
    elif key.endswith("8hr"):
        grouped_plot_data["8-hr"][key] = val
    elif key.endswith("24hr"):
        grouped_plot_data["24-hr"][key] = val

# Plotting
for label, subdata in grouped_plot_data.items():
    num = len(subdata)
    fig, axs = plt.subplots(num, 1, figsize=(10, 3 * num), sharex=True)
    if num == 1:
        axs = [axs]
    for ax, (metal_key, (df_metal, limits)) in zip(axs, subdata.items()):
        colname = df_metal.columns[0]
        ax.plot(df_metal.index, df_metal[colname], label=f"{colname} ({label})", color='black')
        ax.set_ylabel(colname, rotation=0, labelpad=30)
        if limits["cancer"]:
            ax.axhline(y=limits["cancer"] * 1000, color='red', linestyle='--', label="Cancer Threshold")
        if limits["noncancer"]:
            ax.axhline(y=limits["noncancer"] * 1000, color='orange', linestyle=':', label="Noncancer Threshold")
        ax.legend(loc='upper right')
    axs[-1].set_xlabel("Date")
    fig.suptitle(f"{label} Metal Concentrations with NATTS Risk Thresholds", fontsize=16)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

# --- Exceedance Percentages Summary Table ---
exceedance_percentages = []

for h in [1, 8, 24]:
    for metal_short, limits in natts_thresholds.items():
        if h == 1:
            conc_col = next((col for col in df.columns if col.startswith(metal_short + " ") and "(ng/m3)" in col and "uncert" not in col.lower()), None)
            uncert_col = next((col for col in df.columns if col.startswith(metal_short + " ") and "uncert" in col.lower()), None)
        else:
            conc_col = f"{metal_short}_avg_{h}hr"
            uncert_col = f"{metal_short}_uncert_avg_{h}hr"

        if conc_col not in df.columns or uncert_col not in df.columns:
            continue

        valid_mask = (df[conc_col] > 3 * df[uncert_col]) & global_valid_mask
        valid_total = valid_mask.sum()
        if valid_total == 0:
            continue

        cancer_exceed = ((df[conc_col] / 1000) > limits["cancer"]) & valid_mask if limits["cancer"] else None
        noncancer_exceed = ((df[conc_col] / 1000) > limits["noncancer"]) & valid_mask if limits["noncancer"] else None

        exceedance_percentages.append({
            "Metal": metal_short,
            "Averaging Window": f"{h}-hr",
            "% Time > Cancer Threshold": round(100 * cancer_exceed.sum() / valid_total, 2) if cancer_exceed is not None else "NA",
            "% Time > Noncancer Threshold": round(100 * noncancer_exceed.sum() / valid_total, 2) if noncancer_exceed is not None else "NA"
        })

# Display as DataFrame
exceedance_df = pd.DataFrame(exceedance_percentages)
print(exceedance_df)

import seaborn as sns

# Pivot data for plotting
df_melted = exceedance_df.melt(id_vars=["Metal", "Averaging Window"],
                                value_vars=["% Time > Cancer Threshold", "% Time > Noncancer Threshold"],
                                var_name="Threshold Type",
                                value_name="Exceedance (%)")

# Remove rows with 'NA'
df_melted = df_melted[df_melted["Exceedance (%)"] != "NA"]
df_melted["Exceedance (%)"] = pd.to_numeric(df_melted["Exceedance (%)"])

# Create the grouped bar plot
g = sns.catplot(
    data=df_melted,
    kind="bar",
    x="Metal",
    y="Exceedance (%)",
    hue="Averaging Window",
    col="Threshold Type",
    palette="Set2",
    height=6,
    aspect=1
)

g.set_titles("{col_name}")
g.set_axis_labels("Metal", "% Exceedance")
g.set(ylim=(0, 110))
for ax in g.axes.flatten():
    for c in ax.containers:
        ax.bar_label(c, fmt="%.1f", label_type="edge", fontsize=14)
plt.tight_layout()
plt.show()

# Optional: Save to CSV
# exceedance_df.to_csv("exceedance_summary.csv", index=False)
