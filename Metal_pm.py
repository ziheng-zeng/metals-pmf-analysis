import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

# === Load metal data ===
file_path = "Xact_EST_May2023_July2025_combined.csv"
df = pd.read_csv(file_path)

# Convert TIME column to datetime and localize to US Eastern time
df['TIME'] = pd.to_datetime(df['TIME'], utc=True).dt.tz_convert('US/Eastern')
df.set_index('TIME', inplace=True)

# === Load PM data ===
df_pm = pd.read_csv("D:/Documents/research-2024/LV PM25 + WDS 2023-05-01 to 2024-07-10.csv", skiprows=2)
df_pm['Date'] = pd.to_datetime(df_pm['Date'], format='%d-%b-%Y %H:%M', utc=True).dt.tz_convert('US/Eastern')
df_pm.set_index('Date', inplace=True)

# === Define metal and uncertainty columns automatically ===
# Identify corresponding column names
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
if 'S 16 (ng/m3)' in concentration_cols:
    metal_uncert_pairs['S 16 (ng/m3)'] = 'S Uncert (ng/m3)'

# Create metals_cols and uncert_cols dictionaries - only for successfully paired columns
metals_cols = {}
uncert_cols = {}
for conc_col, uncert_col in metal_uncert_pairs.items():
    element = conc_col.split()[0]
    # Verify both columns actually exist in the dataframe
    if conc_col in df.columns and uncert_col in df.columns:
        metals_cols[element] = conc_col
        uncert_cols[element] = uncert_col
    else:
        print(f"Warning: Missing columns for {element}: {conc_col} or {uncert_col}")

print(f"Successfully paired {len(metals_cols)} metals: {list(metals_cols.keys())}")

# Only include columns that were successfully paired
columns_needed = list(metals_cols.values()) + list(uncert_cols.values())


# === Function to process data for a given year ===
def process_year_data(year, start_date, end_date):
    df_period = df.loc[start_date:end_date, columns_needed].copy()

    # Filter values below 3x uncertainty - with proper error handling
    for metal in metals_cols:
        conc_col = metals_cols[metal]
        unc_col = uncert_cols[metal]

        # Check if both columns exist in the dataframe
        if conc_col in df_period.columns and unc_col in df_period.columns:
            try:
                # Ensure both series have the same index and handle NaN values
                conc_series = df_period[conc_col]
                unc_series = df_period[unc_col]

                # Apply 3x uncertainty filter only where both values are not NaN
                mask = (conc_series > 3 * unc_series) | conc_series.isna()
                df_period[conc_col] = conc_series.where(mask)

            except Exception as e:
                print(f"Warning: Could not process {metal} ({conc_col}): {e}")
                continue
        else:
            print(f"Warning: Missing columns for {metal}")

    # Calculate total metal concentration (sum of filtered metals) - only for valid columns
    valid_metal_cols = [col for col in metals_cols.values() if col in df_period.columns]
    df_period['Total_Metal'] = df_period[valid_metal_cols].sum(axis=1, skipna=True)

    # Get PM data for the same period
    df_pm_period = df_pm.loc[start_date:end_date].copy()

    # Merge PM data with metal data
    df_merged = df_period.merge(df_pm_period[['UG/M3']], left_index=True, right_index=True, how='left')

    # Calculate ratios - with error handling
    df_merged['Total_Metal_PM_Ratio'] = 0.001 * df_merged['Total_Metal'] / df_merged['UG/M3']

    # Calculate K/PM ratio if K exists
    if 'K' in metals_cols and metals_cols['K'] in df_merged.columns:
        df_merged['K_PM_Ratio'] = 0.001 * df_merged[metals_cols['K']] / df_merged['UG/M3']
    else:
        df_merged['K_PM_Ratio'] = np.nan

    return df_merged


# === Function to create plots ===
def create_plots(df_merged, year):
    plt.rcParams.update({'font.size': 14, 'font.family': 'Times New Roman'})

    # Create subplots with secondary y-axis for ratios
    fig, ax1 = plt.subplots(figsize=(12, 4))
    ax2 = ax1.twinx()  # Secondary y-axis for ratios

    # Plot ratios on secondary axis (only for 2023)
    if year == 2023 and df_merged is not None:
        # Plot Total Metal/PM ratio
        valid_total = df_merged['Total_Metal_PM_Ratio'].notna()
        if valid_total.any():
            ax2.plot(df_merged.index[valid_total], df_merged['Total_Metal_PM_Ratio'][valid_total],
                     'k-', linewidth=2, alpha=0.7, label='Total Metal/PM')

        # Plot K/PM ratio
        valid_k = df_merged['K_PM_Ratio'].notna()
        if valid_k.any():
            ax2.plot(df_merged.index[valid_k], df_merged['K_PM_Ratio'][valid_k],
                     'r--', linewidth=2, alpha=0.7, label='K/PM')

        # Plot PM2.5 on primary axis
        valid_pm = df_merged['UG/M3'].notna()
        if valid_pm.any():
            ax1.plot(df_merged.index[valid_pm], df_merged['UG/M3'][valid_pm],
                     'b-', linewidth=2, alpha=0.7, label='PM2.5')

    # # Set title and labels
    # if year == 2023:
    #     ax1.set_title('Metal/PM Ratios and PM2.5 Concentration (June 27 – July 10, 2023)')
    # else:
    #     ax1.set_title('Metal/PM Ratios and PM2.5 Concentration (June 27 – July 7, 2025)')

    ax1.set_ylabel('PM2.5 (μg/m³)', color='blue')
    ax1.legend(loc='upper left')
    ax1.tick_params(axis='y', labelcolor='blue')

    # Secondary axis labels (only for 2023)
    if year == 2023 and df_merged is not None:
        ax2.set_ylabel('Metal/PM Ratio', color='black')
        ax2.legend(loc='upper right')
        ax2.tick_params(axis='y', labelcolor='black')

    # Format x-axis to include year but no axis label
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %d, %Y'))
    ax1.xaxis.set_major_locator(mdates.DayLocator(interval=2))  # Show every 2 days

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45)

    # Remove x-axis label but keep the tick labels
    ax1.set_xlabel('')

    # Add background shading for 2023 plot
    if year == 2023:
        # Light dark green shading from June 28 to June 30 midday, 2023
        ax1.axvspan(pd.Timestamp('2023-06-28'), pd.Timestamp('2023-06-30 12:00:00'),
                    color='darkgreen', alpha=0.2, zorder=0)
        # Light purple shading from June 30 midday onwards, 2023
        ax1.axvspan(pd.Timestamp('2023-06-30 12:00:00'), pd.Timestamp('2023-07-09'),
                    color='purple', alpha=0.2, zorder=0)

        # Add text labels on the shaded regions
        ax1.text(pd.Timestamp('2023-06-29'), ax1.get_ylim()[1] * 0.9, 'Wildfire',
                 fontsize=12, ha='center', va='center',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        ax1.text(pd.Timestamp('2023-07-03'), ax1.get_ylim()[1] * 0.9, 'Fireworks',
                 fontsize=12, ha='center', va='center',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

        # Add subplot label (a)
        # ax1.text(0.02, 0.98, '(a)', transform=ax1.transAxes, fontsize=14,
        #          fontweight='bold', ha='left', va='top')

    if year == 2025:
        # Add subplot label (b)
        ax1.text(0.02, 0.98, '(b)', transform=ax1.transAxes, fontsize=14,
                 fontweight='bold', ha='left', va='top')

    # No grid
    ax1.grid(False)
    plt.tight_layout()
    plt.show()


# === Process and plot 2023 data ===
print("Processing 2023 data...")
df_2023_merged = process_year_data(2023, '2023-06-27', '2023-07-09')

# Plot 2023 with PM ratios
create_plots(df_2023_merged, 2023)

# # === Process and plot 2025 data ===
# print("Processing 2025 data...")
# df_2025_merged = process_year_data(2025, '2025-06-27', '2025-07-07')

# Plot 2025 (no PM ratios since no PM data for 2025)
# create_plots(df_2025_merged, 2025)