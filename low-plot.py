import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pytz
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec

# Load metal data
df = pd.read_csv('D:/Documents/research-2024/Xact python code/Xact_EasternTime_7_8_24.csv')
df['TIME'] = pd.to_datetime(df['TIME'], utc=True)
df['TIME'] = df['TIME'].dt.tz_convert('US/Eastern')
# df.dropna(subset=['TIME'] + [col for col in df.columns if 'uncert' not in col], inplace=True)

# Load pm data
df_pm = pd.read_csv('C:/Users/zengz/OneDrive/Desktop/LV PM25 + WDS 2023-05-01 to 2024-07-10.csv', skiprows=2)
df_pm['Date'] = pd.to_datetime(df_pm['Date'], utc=True)
df_pm['Date'] = df_pm['Date'].dt.tz_convert('US/Eastern')
# df_pm.dropna(subset=['Date', 'UG/M3'], inplace=True)

# Define the time range for filtering
start_time = pd.Timestamp('2023-07-03 00:00:00', tz='US/Eastern')
end_time = pd.Timestamp('2023-07-05 23:00:00', tz='US/Eastern')
df_filtered = df[(df['TIME'] >= start_time) & (df['TIME'] <= end_time)]
df_pm_filtered = df_pm[(df_pm['Date'] >= start_time) & (df_pm['Date'] <= end_time)]

# Species of interest
species_list = ['K', 'Ca', 'Ti', 'Cr', 'Mn', 'Fe', 'Cu', 'Zn', 'As', 'Se', 'Br', 'Sr', 'Zr', 'Mo', 'Ag', 'Cd', 'In', 'Ba', 'Pb', 'Bi']
for species in species_list:
    plt.figure(figsize=(12, 10))
    gs = GridSpec(2, 1, height_ratios=[3, 1])

    # Initialize correlation
    correlation = np.nan

    # Metal concentration plot
    ax1 = plt.subplot(gs[0])
    conc_col = [col for col in df.columns if species in col and 'uncert' not in col.lower()][0]
    uncert_col = [col for col in df.columns if species in col and 'uncert' in col.lower()][0]
    metal_values = df_filtered[conc_col].values
    pm_values = df_pm_filtered['UG/M3'].values
    ax1.errorbar(df_filtered['TIME'], metal_values, yerr=df_filtered[uncert_col], fmt='o', label=f'{species} concentration', capsize=3)
    ax1.set_ylabel(f'{species} Concentration (ng/m3)', fontsize=14)
    
    # PM concentration plot
    ax2 = ax1.twinx()
    ax2.plot(df_pm_filtered['Date'], pm_values, label='PM2.5 Concentration', color='r', linestyle='--')
    ax2.set_ylabel('PM2.5 Concentration (µg/m3)', fontsize=14)

    # Calculate and display the Pearson correlation coefficient
    # Ensure only common time indices are used
    df_filtered_sub = df_filtered.set_index('TIME').reindex(df_pm_filtered.set_index('Date').index, method='nearest')
    df_filtered_sub = df_filtered_sub.reset_index()
    if species in df_filtered_sub.columns:
        metal_values = df_filtered_sub[species].dropna().values
        pm_values = df_pm_filtered['UG/M3'].dropna().values

        # Only calculate correlation if arrays are not empty and of the same length
        if len(metal_values) == len(pm_values) and len(metal_values) > 1:
            correlation = np.corrcoef(metal_values, pm_values)[0, 1]
        else:
            correlation = np.nan
            print("Data mismatch or insufficient data for correlation.")
    else:
        print(f"{species} data not available for correlation.")
    ax1.set_title(f'{species} and PM2.5 Concentration (July 3-5, 2023) - Correlation: {correlation:.2f}', fontsize=16)

    # Ratio plot
    ax3 = plt.subplot(gs[1], sharex=ax1)
    ratio = metal_values / pm_values /1000
    ax3.plot(df_filtered['TIME'], ratio, label=f'{species}/PM2.5 Ratio', color='b')
    ax3.set_ylabel('Ratio', fontsize=14)
    ax3.set_xlabel('Time (US Eastern)', fontsize=14)
    ax3.legend(loc='upper left', fontsize=10)

    ax1.legend(loc='upper left', fontsize=10)
    ax2.legend(loc='upper right', fontsize=10)
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax3.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()

    plt.savefig(f'D:/Documents/research-2024/Xact data/New Folder/{species}_concentration_with_ratio_plot.png')
    plt.clf()