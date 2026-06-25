import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load and prepare data
file_path = "Xact_EST_May2023_July2025_combined.csv"
df = pd.read_csv(file_path)
df['TIME'] = pd.to_datetime(df['TIME'], utc=True).dt.tz_convert('US/Eastern')
df.set_index('TIME', inplace=True)

# Auto-detect all metals and pair with uncertainties
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

# Create metals_cols and uncert_cols dictionaries
metals_cols = {}
uncert_cols = {}
for conc_col, uncert_col in metal_uncert_pairs.items():
    element = conc_col.split()[0]
    if conc_col in df.columns and uncert_col in df.columns:
        metals_cols[element] = conc_col
        uncert_cols[element] = uncert_col

print(f"Successfully paired {len(metals_cols)} metals")

# Apply 3× uncertainty filtering
df_filtered = df.copy()
for metal in metals_cols:
    conc_col = metals_cols[metal]
    unc_col = uncert_cols[metal]

    try:
        conc_series = df_filtered[conc_col]
        unc_series = df_filtered[unc_col]
        mask = (conc_series > 3 * unc_series) | conc_series.isna()
        df_filtered[conc_col] = conc_series.where(mask)
    except Exception as e:
        print(f"Warning: Could not filter {metal}: {e}")

# Exclude instrument down periods
exclude1_start = pd.Timestamp('2024-01-09', tz='US/Eastern')
exclude1_end = pd.Timestamp('2024-02-13', tz='US/Eastern')
exclude2_start = pd.Timestamp('2024-07-02', tz='US/Eastern')
exclude2_end = pd.Timestamp('2024-08-08', tz='US/Eastern')

exclude_mask1 = (df_filtered.index >= exclude1_start) & (df_filtered.index <= exclude1_end)
exclude_mask2 = (df_filtered.index >= exclude2_start) & (df_filtered.index <= exclude2_end)
background_mask = ~(exclude_mask1 | exclude_mask2)

df_background = df_filtered[background_mask].copy()

print(f"Excluded {(~background_mask).sum()} data points from instrument down periods")
print(f"Background dataset: {len(df_background)} data points")

# Add season column
df_background['season'] = df_background.index.month.map({
    12: 'Winter', 1: 'Winter', 2: 'Winter',
    3: 'Spring', 4: 'Spring', 5: 'Spring',
    6: 'Summer', 7: 'Summer', 8: 'Summer',
    9: 'Fall', 10: 'Fall', 11: 'Fall'
})

# Check data completeness for each metal in each season (75% threshold)
valid_metals_by_season = {}
seasons = ['Spring', 'Summer', 'Fall', 'Winter']

print("\nChecking data completeness (≥75% valid data required per season):")
print("-" * 80)

for season in seasons:
    season_data = df_background[df_background['season'] == season]
    total_hours_season = len(season_data)
    valid_metals_by_season[season] = []

    print(f"\n{season}: {total_hours_season} total hours")

    for metal in metals_cols:
        conc_col = metals_cols[metal]
        valid_data = season_data[conc_col].dropna()
        valid_hours = len(valid_data)

        if total_hours_season > 0:
            completeness = (valid_hours / total_hours_season) * 100

            if completeness >= 75.0:
                valid_metals_by_season[season].append(metal)
                print(f"  ✓ {metal}: {completeness:.1f}% ({valid_hours}/{total_hours_season})")
            else:
                print(f"  ✗ {metal}: {completeness:.1f}% ({valid_hours}/{total_hours_season})")

    print(f"  Valid metals in {season}: {len(valid_metals_by_season[season])}")

# Get metals that pass in at least one season
all_valid_metals = set()
for season in seasons:
    all_valid_metals.update(valid_metals_by_season[season])

all_valid_metals = list(all_valid_metals)
print(f"\nMetals with ≥75% data in at least one season: {len(all_valid_metals)}")
print(f"Valid metals: {all_valid_metals}")


# Calculate background statistics for valid metals
def calculate_background_stats(data):
    """Calculate background statistics for a data series"""
    clean_data = data.dropna()
    if len(clean_data) == 0:
        return {}

    return {
        'count': len(clean_data),
        '10th_percentile': np.percentile(clean_data, 10),
        '25th_percentile': np.percentile(clean_data, 25),
        'median': np.percentile(clean_data, 50),
        '75th_percentile': np.percentile(clean_data, 75),
        '90th_percentile': np.percentile(clean_data, 90),
        'mean': np.mean(clean_data),
        'std': np.std(clean_data)
    }


# Calculate background statistics
results = {}
for metal in all_valid_metals:
    conc_col = metals_cols[metal]
    results[metal] = {}

    for season in seasons:
        season_data = df_background[df_background['season'] == season][conc_col]

        # Only calculate stats if this metal passed the 75% test for this season
        if metal in valid_metals_by_season[season]:
            results[metal][season] = calculate_background_stats(season_data)
        else:
            results[metal][season] = {'insufficient_data': True}

# Create summary table
summary_data = []
for metal in all_valid_metals:
    for season in seasons:
        if 'insufficient_data' not in results[metal][season]:
            row = {
                'Element': metal,
                'Season': season,
                'Data_Points': results[metal][season]['count'],
                '25th_Percentile': results[metal][season]['25th_percentile'],
                'Median': results[metal][season]['median'],
                '75th_Percentile': results[metal][season]['75th_percentile'],
                'Mean': results[metal][season]['mean'],
                'Std': results[metal][season]['std']
            }
            summary_data.append(row)

summary_df = pd.DataFrame(summary_data)

# Display results
print(f"\nSeasonal Background Summary:")
print("=" * 100)
if len(summary_df) > 0:
    print(summary_df.round(2).to_string(index=False))
else:
    print("No metals met the 75% data completeness threshold")

# Show which metals are valid in which seasons
print(f"\nMetal validity by season:")
print("=" * 50)
for season in seasons:
    print(f"{season}: {len(valid_metals_by_season[season])} metals")
    if len(valid_metals_by_season[season]) > 0:
        print(f"  {valid_metals_by_season[season]}")

# Summary statistics
print(f"\nSummary:")
print(f"Total metals detected: {len(metals_cols)}")
print(f"Metals with ≥75% data in at least one season: {len(all_valid_metals)}")
print(f"Data filtered using 3× uncertainty threshold")
print(f"Excluded instrument down periods: 2024-01-09 to 2024-02-13, 2024-07-02 to 2024-08-08")