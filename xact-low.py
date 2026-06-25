import matplotlib.pyplot as plt
import pandas as pd
import pytz
from xact_header import headers

df = pd.read_csv('Xact_EasternTime_7_8_24.csv')
# Assuming 'Date' is the column containing the dates, convert it to datetime
df['TIME'] = pd.to_datetime(df['TIME'])

# Define the start and end dates with Eastern Time zone
eastern = pytz.timezone('US/Eastern')
start_date = pd.Timestamp('2024-01-09', tz='US/Eastern')
end_date = pd.Timestamp('2024-02-13', tz='US/Eastern')
# Filter out the tape-off date range
df = df[(df['TIME'] < start_date) | (df['TIME'] > end_date)]

df.fillna(0, inplace=True)

# Identifying concentration and uncertainty columns
concentration_cols = [col for col in headers if " (ng/m3)" in col and "uncert" not in col.lower()]
uncertainty_cols = [col for col in headers if "uncert" in col.lower()]

# Mapping concentration columns to their corresponding uncertainty columns
metal_uncert_pairs = {}
for conc_col in concentration_cols:
    element = conc_col.split()[0]  # Assuming the first word is the element symbol
    # Find the corresponding uncertainty column, accounting for case variations
    uncert_col = next((u_col for u_col in uncertainty_cols if element in u_col), None)
    if uncert_col:
        metal_uncert_pairs[conc_col] = uncert_col

metal_uncert_pairs['S 16 (ng/m3)'] = 'S Uncert (ng/m3)' # manually change mismatch

# Calculate the percentage of ratios above and below 3x uncertainty
ratios_above_uncertainty = {}
for metal_col, uncert_col in metal_uncert_pairs.items():
    if metal_col in df.columns and uncert_col in df.columns:
        above_uncertainty = df[metal_col] > 3 * df[uncert_col]
        ratios_above_uncertainty[metal_col.split(' ')[0]] = above_uncertainty.mean() * 100

# Filter poor species (less than 10% above uncertainty)
poor_species = [elem for elem, pct in ratios_above_uncertainty.items() if pct < 10]

# Analyze and plot spikes for poor species
for species in poor_species:
    conc_col = [col for col in df.columns if species in col and 'uncert' not in col.lower()][0]
    uncert_col = [col for col in df.columns if species in col and 'uncert' in col.lower()][0]

    # Ensure the selected columns contain only numeric data
    if pd.api.types.is_numeric_dtype(df[conc_col]) and pd.api.types.is_numeric_dtype(df[uncert_col]):
        df[species] = df[conc_col]
        df[f'{species}_uncert'] = df[uncert_col]
        
        # Define spike threshold as mean + 2 standard deviations
        spike_threshold = df[species].mean() + 2 * df[species].std()
        
        # Identify time periods with spikes
        spike_df = df[df[species] > spike_threshold]
        
        if not spike_df.empty:
            # Output time periods of spikes
            print(f"\nSpike time periods for {species}:")
            print(spike_df[['TIME', species, f'{species}_uncert']])

            # Scatter plot with error bars (no lines)
            plt.figure(figsize=(12, 6))
            plt.errorbar(df['TIME'], df[species], yerr=df[f'{species}_uncert'], fmt='o', color='black', 
                         ecolor='gray', elinewidth=1, capsize=3, label=f'{species} concentration')
            
            # Highlight spikes with error bars
            plt.errorbar(spike_df['TIME'], spike_df[species], yerr=spike_df[f'{species}_uncert'], fmt='o', 
                         color='red', ecolor='gray', elinewidth=1, capsize=3, label='Spikes', zorder=5)
            
            plt.axhline(y=spike_threshold, color='red', linestyle='--', label='Spike threshold')
            plt.title(f'Time Series of {species} Concentration and Spikes (Scatter with Error Bars)', fontsize=14)
            plt.xlabel('Time', fontsize=12)
            plt.ylabel(f'{species} (ng/m3)', fontsize=12)
            plt.xticks(rotation=45)
            plt.grid(True, linestyle='--', alpha=0.5)
            plt.legend()
            plt.tight_layout()
            
            # Save plot
            plt.savefig(f"D:/Documents/research-2024/Xact data/plots/low-spike-9-11/low-spike-{species}-scatter-9-11.png")
            plt.close()

            # Plot zoomed-in time series for the spike periods (scatter with error bars)
            plt.figure(figsize=(12, 6))
            plt.errorbar(spike_df['TIME'], spike_df[species], yerr=spike_df[f'{species}_uncert'], fmt='o', 
                         color='blue', ecolor='gray', elinewidth=1, capsize=3, label=f'{species} concentration during spikes')

            plt.title(f'Zoomed Time Series for {species} Spikes (Scatter with Error Bars)', fontsize=14)
            plt.xlabel('Time', fontsize=12)
            plt.ylabel(f'{species} (ng/m3)', fontsize=12)
            plt.xticks(rotation=45)
            plt.grid(True, linestyle='--', alpha=0.5)
            plt.legend()
            plt.tight_layout()
            
            # Save zoomed plot
            plt.savefig(f"D:/Documents/research-2024/Xact data/plots/low-spike-9-11/zoomed-low-spike-{species}-scatter-9-11.png")
            plt.close()