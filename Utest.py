import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu
import pytz

# Load Xact data and ensure 'TIME' is parsed as datetime with UTC handling
xact_df = pd.read_csv('Xact_EasternTime_7_8_24.csv')
xact_df['TIME'] = pd.to_datetime(xact_df['TIME'], utc=True).dt.tz_convert('US/Eastern')

# Load PM2.5 data and ensure 'TIME' is parsed as datetime with UTC handling
pm25_df = pd.read_csv('C:/Users/zengz/OneDrive/Desktop/LV PM25 + WDS 2023-05-01 to 2024-07-10.csv', skiprows=2)
pm25_df['TIME'] = pd.to_datetime(pm25_df['Date'], utc=True).dt.tz_convert('US/Eastern')

# Merge Xact and PM2.5 data on the TIME column
df = pd.merge(xact_df, pm25_df, on='TIME', how='inner')

# Unit conversion
df['UG/M3'] = df['UG/M3'] * 1000

# Define time periods for each event
fireworks_periods = [
    (pd.Timestamp('2023-07-01 17:00', tz='US/Eastern'), pd.Timestamp('2023-07-02 12:00', tz='US/Eastern')),
    (pd.Timestamp('2023-07-03 17:00', tz='US/Eastern'), pd.Timestamp('2023-07-04 16:00', tz='US/Eastern')),
    (pd.Timestamp('2023-07-04 17:00', tz='US/Eastern'), pd.Timestamp('2023-07-05 12:00', tz='US/Eastern')),
    (pd.Timestamp('2023-07-05 17:00', tz='US/Eastern'), pd.Timestamp('2023-07-06 12:00', tz='US/Eastern'))
]

# Filter data for fireworks and combine into one DataFrame
fireworks_df_list = [
    df[(df['TIME'] >= start) & (df['TIME'] <= end)]
    for start, end in fireworks_periods
]
fireworks_df = pd.concat(fireworks_df_list)

# Define wildfire and background periods
wildfire_start_time = pd.Timestamp('2023-06-28 00:00:00', tz='US/Eastern')
wildfire_end_time = pd.Timestamp('2023-06-30 12:00', tz='US/Eastern')
wildfire_df = df[(df['TIME'] >= wildfire_start_time) & (df['TIME'] <= wildfire_end_time)]

background_start_time = pd.Timestamp('2023-09-01 00:00:00', tz='US/Eastern')
background_end_time = pd.Timestamp('2023-09-07 23:59:59', tz='US/Eastern')
background_df = df[(df['TIME'] >= background_start_time) & (df['TIME'] <= background_end_time)]

# List of metals to analyze against potassium (K)
metals_of_interest = ['Pb', 'Zn', 'Fe', 'As', 'Ti', 'Sr', 'Ba', 'Bi', 'Cu', 'Ca', 'Mn', 'Cr', 'Se']

# Prepare lists to store the data for plotting
ratios_fireworks = []
ratios_wildfire = []
ratios_background = []

# Calculate Metal/K ratios for each period
for metal in metals_of_interest:
    conc_col = [col for col in df.columns if f"{metal} " in col and 'uncert' not in col.lower()][0]
    k_col = [col for col in df.columns if "K " in col and 'uncert' not in col.lower()][0]

    # Background Metal/K Ratio
    ratio_background = background_df[conc_col] / background_df[k_col]
    ratios_background.append(pd.DataFrame({
        'Ratio': ratio_background,
        'Metal': metal,
        'Event': 'Background'
    }))

    # Fireworks Metal/K Ratio
    ratio_fireworks = fireworks_df[conc_col] / fireworks_df[k_col]
    ratios_fireworks.append(pd.DataFrame({
        'Ratio': ratio_fireworks,
        'Metal': metal,
        'Event': 'Fireworks'
    }))

    # Wildfire Metal/K Ratio
    ratio_wildfire = wildfire_df[conc_col] / wildfire_df[k_col]
    ratios_wildfire.append(pd.DataFrame({
        'Ratio': ratio_wildfire,
        'Metal': metal,
        'Event': 'Wildfire'
    }))

# Combine all ratio data into a single DataFrame for plotting
all_ratios_df = pd.concat(ratios_background + ratios_fireworks + ratios_wildfire)

# Plotting the Metal/K Ratios as a Box Plot
plt.figure(figsize=(14, 7))
sns.boxplot(
    x='Metal', y='Ratio', hue='Event', data=all_ratios_df,
    palette={'Background': 'gray', 'Fireworks': 'blue', 'Wildfire': 'green'}, showfliers=False
)

# Customize the plot
plt.title('Distribution of Metal/K Ratios for Background, Fireworks, and Wildfire', fontsize=16)
plt.ylabel('Metal/K Ratio', fontsize=14)
plt.xlabel('Metal', fontsize=14)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Perform Mann-Whitney U Test and store results
results = []
for metal in metals_of_interest:
    # Filter data for each metal in the background, fireworks, and wildfire periods
    background_values = all_ratios_df[(all_ratios_df['Metal'] == metal) & (all_ratios_df['Event'] == 'Background')][
        'Ratio']
    fireworks_values = all_ratios_df[(all_ratios_df['Metal'] == metal) & (all_ratios_df['Event'] == 'Fireworks')][
        'Ratio']
    wildfire_values = all_ratios_df[(all_ratios_df['Metal'] == metal) & (all_ratios_df['Event'] == 'Wildfire')]['Ratio']

    # Mann-Whitney U test between each event and background
    u_stat_fireworks, p_value_fireworks = mannwhitneyu(fireworks_values, background_values, alternative='greater')
    u_stat_wildfire, p_value_wildfire = mannwhitneyu(wildfire_values, background_values, alternative='greater')

    # Store results
    results.append({
        'Metal': metal,
        'Fireworks vs Background p-value': p_value_fireworks,
        'Wildfire vs Background p-value': p_value_wildfire
    })

# Convert results to DataFrame and print
results_df = pd.DataFrame(results)
print("Mann-Whitney U Test Results (p-values):")
print(results_df)

# # Optional: Apply multiple comparison correction if needed
# from statsmodels.stats.multitest import multipletests
#
# # Adjust p-values for multiple comparisons
# p_values_fireworks = results_df['Fireworks vs Background p-value']
# p_values_wildfire = results_df['Wildfire vs Background p-value']
#
# _, p_values_fireworks_adj, _, _ = multipletests(p_values_fireworks, alpha=0.05, method='bonferroni')
# _, p_values_wildfire_adj, _, _ = multipletests(p_values_wildfire, alpha=0.05, method='bonferroni')
#
# # Add adjusted p-values to the results DataFrame
# results_df['Fireworks vs Background p-value (adj)'] = p_values_fireworks_adj
# results_df['Wildfire vs Background p-value (adj)'] = p_values_wildfire_adj
#
# print("Adjusted Mann-Whitney U Test Results (p-values):")
# print(results_df)
