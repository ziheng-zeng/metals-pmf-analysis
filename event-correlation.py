import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

# === Load data ===
file_path = "Xact_EST_May2023_Oct2025_combined.csv"
df = pd.read_csv(file_path)

# Convert TIME column to datetime and localize to US Eastern time
df['TIME'] = pd.to_datetime(df['TIME'], utc=True).dt.tz_convert('US/Eastern')
df.set_index('TIME', inplace=True)

# === Define metal and uncertainty columns ===
metals_cols = {
    'K': 'K 19 (ng/m3)',
    'Cu': 'Cu 29 (ng/m3)',
    'Ba': 'Ba 56 (ng/m3)',
    'Sr': 'Sr 38 (ng/m3)'
}
uncert_cols = {
    'K': 'K Uncert (ng/m3)',
    'Cu': 'Cu Uncert (ng/m3)',
    'Ba': 'Ba Uncert (ng/m3)',
    'Sr': 'Sr Uncert (ng/m3)'
}

columns_needed = list(metals_cols.values()) + list(uncert_cols.values())


# === Function to process data for a given year ===
def process_year_data(year, start_date, end_date):
    df_period = df.loc[start_date:end_date, columns_needed].copy()

    # Filter values below 3x uncertainty
    for metal in metals_cols:
        conc_col = metals_cols[metal]
        unc_col = uncert_cols[metal]
        df_period[conc_col] = df_period[conc_col].where(df_period[conc_col] > 3 * df_period[unc_col])

    # Normalize (min-max)
    df_scaled = (df_period[list(metals_cols.values())] - df_period[list(metals_cols.values())].min()) / \
                (df_period[list(metals_cols.values())].max() - df_period[list(metals_cols.values())].min())

    # Normalize uncertainties to match scaled data
    df_uncert_scaled = pd.DataFrame(index=df_period.index)
    for metal in metals_cols:
        conc_col = metals_cols[metal]
        unc_col = uncert_cols[metal]
        df_uncert_scaled[metal] = df_period[unc_col] / (df_period[conc_col].max() - df_period[conc_col].min())

    # Clean data
    df_scaled_clean = df_scaled.astype('float64')
    df_uncert_scaled_clean = df_uncert_scaled.astype('float64')

    return df_scaled_clean, df_uncert_scaled_clean


# === Function to create plots ===
def create_plots(df_scaled_clean, df_uncert_scaled_clean, year, smoothed=False):
    # plt.rcParams.update({'font.size': 14, 'font.family': 'Times New Roman'})

    if smoothed:
        # Apply rolling mean smoothing (3-hour window)
        df_plot = df_scaled_clean.rolling(window=3, center=True).mean()
        df_uncert_plot = df_uncert_scaled_clean.rolling(window=3, center=True).mean()
        title_prefix = 'Smoothed Normalized'
    else:
        df_plot = df_scaled_clean
        df_uncert_plot = df_uncert_scaled_clean
        title_prefix = 'Normalized'

    fig, ax = plt.subplots(figsize=(6, 3))

    for metal in metals_cols:
        conc_col = metals_cols[metal]
        series = df_plot[conc_col]
        uncert = df_uncert_plot[metal]

        # Keep NaN values to show gaps in the plot
        time = df_plot.index.to_numpy()
        y = series.to_numpy(dtype=np.float64)
        yerr = uncert.to_numpy(dtype=np.float64)

        # Plot with NaN values - matplotlib will automatically break lines at NaN
        # Use errorbar instead of plot + fill_between for cleaner uncertainty display
        valid = ~(np.isnan(y) | np.isnan(yerr))
        if valid.any():
            ax.errorbar(time[valid], y[valid], yerr=yerr[valid],
                        label=metal, marker='o', markersize=3, capsize=3, capthick=1,
                        linestyle='', elinewidth=1)  # No connecting lines, just points and error bars


    ax.set_ylabel('Normalized Concentration')
    ax.legend()

    # Format x-axis to include year but no axis label
    tzinfo = df_plot.index.tz  # should be US/Eastern
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=1, tz=tzinfo))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d", tz=tzinfo))

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45)

    # Remove x-axis label but keep the tick labels
    ax.set_xlabel('Local Time (US Eastern)')

    # Add background shading for 2023 plot
    if year == 2023:
        # Light dark green shading from June 28 to June 30 midday, 2023
        ax.axvspan(pd.Timestamp('2023-06-27 16:00:00', tz='US/Eastern'), pd.Timestamp('2023-06-30 12:00:00', tz='US/Eastern'),
                   color='darkgreen', alpha=0.2, zorder=0)
        # Light purple shading from June 30 midday onwards, 2023
        ax.axvspan(pd.Timestamp('2023-06-30 12:00:00', tz='US/Eastern'), pd.Timestamp('2023-07-06 12:00:00', tz='US/Eastern'),
                   color='purple', alpha=0.2, zorder=0)

        # Add text labels on the shaded regions
        ax.text(pd.Timestamp('2023-06-29', tz='US/Eastern'), 0.9, 'Wildfire',
                fontsize=10, ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        ax.text(pd.Timestamp('2023-07-03', tz='US/Eastern'), 0.9, 'Fireworks',
                fontsize=10, ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

        # # Add subplot label (a)
        # ax.text(0.02, 0.98, '(a)', transform=ax.transAxes, fontsize=14,
        #         fontweight='bold', ha='left', va='top')

    if year == 2025:
        # Light purple shading
        ax.axvspan(pd.Timestamp('2025-06-28 20:00:00', tz='US/Eastern'), pd.Timestamp('2025-06-29 01:00:00', tz='US/Eastern'),
                   color='purple', alpha=0.2, zorder=0)
        # Light purple shading
        ax.axvspan(pd.Timestamp('2025-07-03 12:00:00', tz='US/Eastern'), pd.Timestamp('2025-07-07', tz='US/Eastern'),
                   color='purple', alpha=0.2, zorder=0)

        # Add text labels on the shaded regions
        # ax.text(pd.Timestamp('2025-06-29', tz='US/Eastern'), 0.9, 'Fireworks',
        #         fontsize=16, ha='center', va='center',
        #         bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        ax.text(pd.Timestamp('2025-07-02', tz='US/Eastern'), 0.9, 'Fireworks',
                fontsize=16, ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        # Add subplot label (b)
        ax.text(0.02, 0.98, '(b)', transform=ax.transAxes, fontsize=14,
                fontweight='bold', ha='left', va='top')

    # No grid
    ax.grid(False)
    plt.tight_layout()
    plt.show()


# === Process and plot 2023 data ===
print("Processing 2023 data...")
df_2023_scaled, df_2023_uncert_scaled = process_year_data(2023, '2023-06-27', '2023-07-07')

# Raw 2023 plot
create_plots(df_2023_scaled, df_2023_uncert_scaled, 2023, smoothed=False)

## Smoothed 2023 plot (commented out for now)
# create_plots(df_2023_scaled, df_2023_uncert_scaled, 2023, smoothed=True)

# === Process and plot 2025 data ===
print("Processing 2025 data...")
df_2025_scaled, df_2025_uncert_scaled = process_year_data(2025, '2025-06-27', '2025-07-07')

# Raw 2025 plot
create_plots(df_2025_scaled, df_2025_uncert_scaled, 2025, smoothed=False)

## Smoothed 2025 plot (commented out for now)
# create_plots(df_2025_scaled, df_2025_uncert_scaled, 2025, smoothed=True)