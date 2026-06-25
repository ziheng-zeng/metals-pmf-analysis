import os
import pandas as pd
import matplotlib.pyplot as plt
import pytz
import seaborn as sns
from xact_header import headers  # Make sure this is correctly defined

# Read the CSV file
df = pd.read_csv('Xact_EST_May2023_Oct2025_combined.csv')
df['TIME'] = pd.to_datetime(df['TIME'], utc=True).dt.tz_convert('US/Eastern').dt.tz_localize(None)

# Apply time exclusion filters
exclude1_start = pd.Timestamp('2024-01-09')
exclude1_end = pd.Timestamp('2024-02-13')
exclude2_start = pd.Timestamp('2024-07-02')
exclude2_end = pd.Timestamp('2024-08-08')

df = df[(df['TIME'] < exclude1_start) | (df['TIME'] > exclude1_end)]
df = df[(df['TIME'] < exclude2_start) | (df['TIME'] > exclude2_end)]

# Set TIME as index
df.set_index('TIME', inplace=True)

# List of metals
extended_metals = ['Fe', 'K', 'Ca', 'Zn']

# Identify corresponding column names
metal_cols_ext = {
    m: [col for col in df.columns if f"{m} " in col and "(ng/m3)" in col and "uncert" not in col.lower()][0]
    for m in extended_metals
}

# Pair each concentration column with its corresponding uncertainty column
headers = df.columns.tolist()
concentration_cols = [col for col in headers if " (ng/m3)" in col and "uncert" not in col.lower()]
uncertainty_cols = [col for col in headers if "uncert" in col.lower()]

metal_uncert_pairs = {}
for conc_col in concentration_cols:
    element = conc_col.split()[0]
    uncert_col = next((u_col for u_col in uncertainty_cols if element in u_col), None)
    if uncert_col:
        metal_uncert_pairs[conc_col] = uncert_col

# Manually assign S uncertainty column
metal_uncert_pairs['S 16 (ng/m3)'] = 'S Uncert (ng/m3)'

# Calculate average concentrations for all metals after 3x uncertainty filtering
metal_averages = {}

print("Calculating average concentrations for all metals after 3x uncertainty filtering...\n")

for conc_col, uncert_col in metal_uncert_pairs.items():
    if conc_col in df.columns and uncert_col in df.columns:
        # Filter valid data: concentration > 3 × uncertainty
        valid_data = df[(df[conc_col] > 3 * df[uncert_col])][conc_col].dropna()

        if not valid_data.empty:
            avg_conc = valid_data.mean()
            metal_averages[conc_col] = avg_conc
            element = conc_col.split()[0]
            print(f"{element}: {avg_conc:.2f} ng/m³ (n={len(valid_data)} valid measurements)")
        else:
            element = conc_col.split()[0]
            print(f"{element}: No valid data after filtering")

# Rank metals from highest to lowest average concentration
print(f"\n{'=' * 60}")
print("RANKING: Average Concentrations (Highest to Lowest)")
print(f"{'=' * 60}")

sorted_metals = sorted(metal_averages.items(), key=lambda x: x[1], reverse=True)

for i, (conc_col, avg_conc) in enumerate(sorted_metals, 1):
    element = conc_col.split()[0]
    print(f"{i:2d}. {element:3s}: {avg_conc:8.2f} ng/m³")

print(f"\nTotal metals analyzed: {len(sorted_metals)}")
print(f"Highest: {sorted_metals[0][0].split()[0]} ({sorted_metals[0][1]:.2f} ng/m³)")
print(f"Lowest:  {sorted_metals[-1][0].split()[0]} ({sorted_metals[-1][1]:.2f} ng/m³)")
print(f"Range:   {sorted_metals[0][1] / sorted_metals[-1][1]:.1f}x difference")

# Define non-metals to exclude
non_metals = ['Al', 'S', 'Si', 'Cl', 'P', 'Br', 'I', 'As', 'Se', 'Te']  # Non-metals and metalloids

# Filter to get only true metals, excluding Al and non-metals
metals_only = []
for metal_data in sorted_metals:
    element = metal_data[0].split()[0]
    if element not in non_metals:
        metals_only.append(metal_data)

# Update metal selection for plotting - top 15 true metals
extended_metals = [metal[0].split()[0] for metal in metals_only[:15]]
print(f"\nFiltered out non-metals and Al. Top 15 true metals: {extended_metals}")

# Show what was excluded
excluded = [metal[0].split()[0] for metal in sorted_metals if metal[0].split()[0] in non_metals]
print(f"Excluded non-metals/metalloids: {excluded}")

# Update metal_cols_ext dictionary for the top metals
metal_cols_ext = {}
for metal in extended_metals:
    matching_cols = [col for col in df.columns if
                     f"{metal} " in col and "(ng/m3)" in col and "uncert" not in col.lower()]
    if matching_cols:
        metal_cols_ext[metal] = matching_cols[0]

# Generate time series plots with uncertainty shading
plt.rcParams['font.family'] = 'Times New Roman'

# Use the top metals from the ranking (this will be updated after the calculation above)
ts_metals = extended_metals  # This will use the top 4 metals

fig, axes = plt.subplots(len(ts_metals), 1, figsize=(12, 3 * len(ts_metals)))

# Handle case where there's only one subplot
if len(ts_metals) == 1:
    axes = [axes]

for i, metal in enumerate(ts_metals):
    col = metal_cols_ext[metal]
    uncert_col = metal_uncert_pairs.get(col)

    if not uncert_col:
        print(f"Skipping {metal} — no uncertainty column found.")
        continue

    # Filter valid data: concentration > 3 × uncertainty
    valid_data = df[(df[col] > 3 * df[uncert_col])][[col, uncert_col]].dropna()

    if valid_data.empty:
        print(f"Skipping {metal} — no valid data points.")
        continue

    ax = axes[i]

    # Plot concentration line
    line = ax.plot(valid_data.index, valid_data[col], color='black', linewidth=1, label='Concentration')[0]

    # Add uncertainty shading
    upper_bound = valid_data[col] + valid_data[uncert_col]
    lower_bound = valid_data[col] - valid_data[uncert_col]
    fill = ax.fill_between(valid_data.index, lower_bound, upper_bound,
                           alpha=0.3, color='grey', label='Uncertainty')

    # Formatting
    ax.set_ylabel(f'{metal} Concentration (ng/m³)', fontsize=12, fontfamily='Times New Roman')
    ax.set_title(f'{metal} Time Series', fontsize=14, fontfamily='Times New Roman')

    # Format ticks
    ax.tick_params(axis='both', which='major', labelsize=10)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontfamily('Times New Roman')

    # Rotate x-axis labels for better readability
    ax.tick_params(axis='x', rotation=45)

# Create shared legend below all plots
handles = [line, fill]
labels = ['Concentration', 'Uncertainty']
fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.02),
           ncol=2, fontsize=12, frameon=False)

# Set legend font to Times New Roman
legend = fig.legends[0]
for text in legend.get_texts():
    text.set_fontfamily('Times New Roman')

plt.suptitle('Metal Concentration Time Series with Uncertainty', fontsize=16, fontfamily='Times New Roman')
plt.tight_layout()
plt.subplots_adjust(bottom=0.08)  # Make room for legend below
plt.show()

# Separate plot for K only
plt.rcParams['font.family'] = 'Times New Roman'

metal = 'K'
if metal in metal_cols_ext:
    col = metal_cols_ext[metal]
    uncert_col = metal_uncert_pairs.get(col)

    if uncert_col:
        # Filter valid data: concentration > 3 × uncertainty
        valid_data = df[(df[col] > 3 * df[uncert_col])][[col, uncert_col]].dropna()

        if not valid_data.empty:
            plt.figure(figsize=(10, 3))

            # Plot concentration line
            plt.plot(valid_data.index, valid_data[col], color='black', linewidth=1, label='Concentration')

            # Add uncertainty shading
            upper_bound = valid_data[col] + valid_data[uncert_col]
            lower_bound = valid_data[col] - valid_data[uncert_col]
            plt.fill_between(valid_data.index, lower_bound, upper_bound,
                             alpha=0.3, color='grey', label='Uncertainty')

            # ====== NEW: set logarithmic y-axis ======
            ax = plt.gca()
            ax.set_yscale('log')

            # Optionally set y-limits (avoid negatives or zeros)
            ymin = max(valid_data[col].min() / 2, 0.1)  # choose a small >0 minimum
            ymax = valid_data[col].max() * 2
            ax.set_ylim(ymin, ymax)

            # Formatting
            plt.ylabel('K Concentration (ng/m³)', fontsize=14, fontfamily='Times New Roman')
            # plt.title('Potassium (K) Time Series', fontsize=16, fontfamily='Times New Roman')

            # Set x-axis to show every other month labels starting from 2023-05
            import matplotlib.dates as mdates

            ax = plt.gca()

            # Set specific start date for axis labels
            start_label = pd.Timestamp('2023-05-01')
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

            # Set x-axis limits to start from 2023-05
            ax.set_xlim(left=start_label)

            # Format ticks
            plt.tick_params(axis='both', which='major', labelsize=12)
            for label in ax.get_xticklabels() + ax.get_yticklabels():
                label.set_fontfamily('Times New Roman')

            # Rotate x-axis labels for better readability
            plt.xticks(rotation=45)

            # Add legend
            plt.legend(fontsize=12, frameon=False, loc='upper right')


            plt.tight_layout()
            plt.show()
        else:
            print("No valid K data found after filtering")
    else:
        print("No uncertainty column found for K")
else:
    print("K not found in metal_cols_ext dictionary")