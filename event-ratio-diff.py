import pandas as pd
import matplotlib.pyplot as plt
import pytz
import seaborn as sns

# Load data
df = pd.read_csv('D:/Documents/research-2024/Xact python code/Xact_EasternTime_7_8_24.csv')
eastern = pytz.timezone('US/Eastern')
df['TIME'] = pd.to_datetime(df['TIME'])

# Define multiple fireworks periods (start and end times)
fireworks_periods = [
    (pd.Timestamp('2023-07-01 17:00', tz='US/Eastern'), pd.Timestamp('2023-07-02 12:00', tz='US/Eastern')),
    (pd.Timestamp('2023-07-03 17:00', tz='US/Eastern'), pd.Timestamp('2023-07-04 16:00', tz='US/Eastern')),
    (pd.Timestamp('2023-07-04 17:00', tz='US/Eastern'), pd.Timestamp('2023-07-05 12:00', tz='US/Eastern')),
    (pd.Timestamp('2023-07-05 17:00', tz='US/Eastern'), pd.Timestamp('2023-07-06 12:00', tz='US/Eastern'))
]

# Filter data for all fireworks periods and combine them into a single DataFrame
fireworks_df_list = [
    df[(df['TIME'] >= start) & (df['TIME'] <= end)]
    for start, end in fireworks_periods
]
fireworks_df = pd.concat(fireworks_df_list)

# Define a wildfire period for comparison
wildfire_start_time = pd.Timestamp('2023-06-27 00:00:00', tz='US/Eastern')
wildfire_end_time = pd.Timestamp('2023-06-30 23:59:59', tz='US/Eastern')
wildfire_df = df[(df['TIME'] >= wildfire_start_time) & (df['TIME'] <= wildfire_end_time)]

# Define a background period in September (example: September 1-7)
background_start_time = pd.Timestamp('2023-09-01 00:00:00', tz='US/Eastern')
background_end_time = pd.Timestamp('2023-09-07 23:59:59', tz='US/Eastern')
background_df = df[(df['TIME'] >= background_start_time) & (df['TIME'] <= background_end_time)]

# Calculate the mean concentrations during the background period
background_means = background_df.select_dtypes(include=['number']).mean()

# List of metals to analyze against potassium (K)
metals_of_interest = ['Pb', 'Zn', 'Fe', 'As', 'Ti', 'Sr', 'Ba', 'Bi', 'Cu', 'Ca', 'Mn', 'Cr', 'Se']
# metals_of_interest = ['Pb', 'As', 'Ti', 'Sr', 'Ba', 'Bi', 'Cu', 'Mn', 'Cr', 'Se']
delta_ratios_fireworks = []
delta_ratios_wildfire = []

# Calculate delta ratios for each metal for both periods
for metal in metals_of_interest:
    # Get concentration columns
    conc_col = [col for col in df.columns if f"{metal} " in col and 'uncert' not in col.lower()][0]
    k_col = [col for col in df.columns if "K " in col and 'uncert' not in col.lower()][0]

    # Calculate ΔMetal and ΔK for fireworks
    delta_fireworks_metal = fireworks_df[conc_col] - background_means[conc_col]
    delta_fireworks_k = fireworks_df[k_col] - background_means[k_col]
    delta_fireworks_ratio = delta_fireworks_metal / delta_fireworks_k

    # Store ratios with labels
    delta_ratios_fireworks.append(pd.DataFrame({
        'Delta Ratio': delta_fireworks_ratio,
        'Metal': f'{metal}/K',
        'Event': 'Fireworks'
    }))

    # Calculate ΔMetal and ΔK for wildfires
    delta_wildfire_metal = wildfire_df[conc_col] - background_means[conc_col]
    delta_wildfire_k = wildfire_df[k_col] - background_means[k_col]
    delta_wildfire_ratio = delta_wildfire_metal / delta_wildfire_k

    # Store ratios with labels
    delta_ratios_wildfire.append(pd.DataFrame({
        'Delta Ratio': delta_wildfire_ratio,
        'Metal': f'{metal}/K',
        'Event': 'Wildfire'
    }))

# Combine all delta ratio data into a single DataFrame for plotting
all_delta_ratios_df = pd.concat(delta_ratios_fireworks + delta_ratios_wildfire)

# Create a box plot using seaborn
plt.figure(figsize=(12, 6))

# Use seaborn's boxplot, specifying the colors for each event
sns.boxplot(
    x='Metal', y='Delta Ratio', hue='Event', data=all_delta_ratios_df,
    palette={'Fireworks': 'blue', 'Wildfire': 'green'}, showfliers=False
)

plt.yscale('symlog', linthresh=0.01) # log scale

# Customize the plot
plt.title('Distribution of ΔMetal/ΔK Ratios for Fireworks vs. Wildfire', fontsize=16)
plt.ylabel('ΔMetal/ΔK Ratio', fontsize=14)
plt.xlabel('Metal', fontsize=14)
plt.xticks(rotation=45)
plt.tight_layout()

# Show the plot
plt.show()

