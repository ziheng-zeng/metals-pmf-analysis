import matplotlib.pyplot as plt
import pandas as pd
import pytz
from datetime import timedelta
from functools import reduce
from xact_header import headers

# Load data
df = pd.read_csv('Xact_EasternTime_7_8_24.csv')
df['TIME'] = pd.to_datetime(df['TIME'])

# Filter data to ignore tape-off date range
eastern = pytz.timezone('US/Eastern')
start_date = pd.Timestamp('2024-01-09', tz='US/Eastern')
end_date = pd.Timestamp('2024-02-13', tz='US/Eastern')
df = df[(df['TIME'] < start_date) | (df['TIME'] > end_date)]
df.fillna(0, inplace=True)

# Concentration and uncertainty columns
concentration_cols = [col for col in headers if " (ng/m3)" in col and "uncert" not in col.lower()]
uncertainty_cols = [col for col in headers if "uncert" in col.lower()]

metal_uncert_pairs = {}
for conc_col in concentration_cols:
    element = conc_col.split()[0]
    uncert_col = next((u_col for u_col in uncertainty_cols if element in u_col), None)
    if uncert_col:
        metal_uncert_pairs[conc_col] = uncert_col

metal_uncert_pairs['S 16 (ng/m3)'] = 'S Uncert (ng/m3)'  # Manual fix

# Calculate the percentage of ratios above and below 3x uncertainty
ratios_above_uncertainty = {}
for metal_col, uncert_col in metal_uncert_pairs.items():
    if metal_col in df.columns and uncert_col in df.columns:
        above_uncertainty = df[metal_col] > 3 * df[uncert_col]
        ratios_above_uncertainty[metal_col.split(' ')[0]] = above_uncertainty.mean() * 100

# Filter poor species (less than 10% above uncertainty)
poor_species = [elem for elem, pct in ratios_above_uncertainty.items() if pct < 10]

# Store spike time periods for each poor species
species_spike_periods = {}

# Analyze spikes for poor species and gather time periods
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
            # Store the spike time periods for each species
            species_spike_periods[species] = spike_df['TIME'].tolist()

# Set the time window for overlap (e.g., 1 hour)
time_window = timedelta(hours=1)

# Find overlapping time periods where at least 3 species have spikes within the time window
overlap_threshold = 3
overlapping_time_periods = []

# Dictionary to store which species overlap at each time period
overlap_species_at_time = {}

# Convert the list of spike times for each species to sets, then find common times
if len(species_spike_periods) >= overlap_threshold:
    all_spike_times = []
    
    # Collect all spike times in a unified list
    for species, times in species_spike_periods.items():
        all_spike_times.extend([(time, species) for time in times])  # Add species info to each time
    
    # Sort all spike times to make it easier to find overlaps
    all_spike_times.sort()  # Sort by time first, species second
    
    # Check for overlapping times across species
    for i, (time, species) in enumerate(all_spike_times):
        overlapping_species = {species}  # Use a set to automatically remove duplicates
        
        for j in range(i + 1, len(all_spike_times)):
            next_time, next_species = all_spike_times[j]
            if next_time - time <= time_window:
                overlapping_species.add(next_species)  # Add species to set
            else:
                break
        
        if len(overlapping_species) >= overlap_threshold:
            if time not in overlap_species_at_time:  # Ensure we only record each time once
                overlapping_time_periods.append(time)
                overlap_species_at_time[time] = list(overlapping_species)

# Output the overlapping time periods and species
with open('D:/Documents/research-2024/Xact data/overlapping_spikes_with_species.txt', 'w') as f:
    if overlapping_time_periods:
        f.write(f"Overlapping time periods where at least {overlap_threshold} species show spikes:\n")
        for time in overlapping_time_periods:
            species_list = ', '.join(overlap_species_at_time[time])
            f.write(f"{time} - Species: {species_list}\n")
    else:
        f.write(f"No overlapping time periods where at least {overlap_threshold} species show spikes within the {time_window} window.\n")
