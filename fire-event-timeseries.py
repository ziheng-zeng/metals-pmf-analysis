import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pytz

# Load Xact data
df = pd.read_csv('D:/Documents/research-2024/Xact python code/Xact_EST_May2023_May2025_combined.csv')
df['TIME'] = pd.to_datetime(df['TIME'], utc=True).dt.tz_convert('US/Eastern')
df.set_index('TIME', inplace=True)

# Timezone
eastern = pytz.timezone('US/Eastern')

# Define wildfire event periods
wildfire_periods = [
    ("2023-06-04", "2023-06-09"),
    ("2023-06-27", "2023-07-01"),
    ("2023-07-16", "2023-07-20"),
    # ("2023-09-01", "2023-09-30"),
    ("2025-04-22", "2025-05-12")

    # (pd.Timestamp('2023-06-30 17:00', tz=eastern), pd.Timestamp('2023-07-06 23:59', tz=eastern)),
    # # (pd.Timestamp('2023-07-01 17:00', tz=eastern), pd.Timestamp('2023-07-02 12:00', tz=eastern)),
    # # (pd.Timestamp('2023-07-03 17:00', tz=eastern), pd.Timestamp('2023-07-04 16:00', tz=eastern)),
    # # (pd.Timestamp('2023-07-04 17:00', tz=eastern), pd.Timestamp('2023-07-05 12:00', tz=eastern)),
    # # (pd.Timestamp('2023-07-05 17:00', tz=eastern), pd.Timestamp('2023-07-06 12:00', tz=eastern)),
    # (pd.Timestamp('2024-06-29 00:00', tz=eastern), pd.Timestamp('2024-06-30 23:59', tz=eastern)),
    # (pd.Timestamp('2023-12-07 00:00', tz=eastern), pd.Timestamp('2023-12-14 23:59', tz=eastern)),
    # (pd.Timestamp('2024-12-01 00:00', tz=eastern), pd.Timestamp('2024-12-07 23:59', tz=eastern))

]

event_titles = [
    "June 5–8, 2023 – Code Orange: Canadian wildfire smoke",
    "June 28–30, 2023 – Code Red: Peak PM2.5 from wildfire",
    "July 17-18, 2023 – Lingering smoke, poor air quality",
    # "September 2023 – Intermittent wildfire impacts",
    "April 22–May 12, 2025 – Jones Road Fire (NJ), long-range smoke",

    # "2023 July 4th fireworks",
    # "Jun 29–30, 2024 - Lawrenceville fireworks",
    # "December 7-14, 2023 - Randomly Selected Background 1",
    # "December 1-7, 2024 - Randomly Selected Background 2"
]
# wildfire_periods = [(pd.Timestamp(start), pd.Timestamp(end)) for start, end in wildfire_periods]
wildfire_periods = [(pd.Timestamp(start, tz=eastern), pd.Timestamp(end, tz=eastern)) for start, end in wildfire_periods]

# Define exclusion period for mean calculation
exclude_start = pd.Timestamp('2024-01-09', tz=eastern)
exclude_end = pd.Timestamp('2024-02-13', tz=eastern)
df_included = df[~((df.index >= exclude_start) & (df.index <= exclude_end))]

# Calculate mean and propagated uncertainty
mean_k = df_included['K 19 (ng/m3)'].mean()
mean_uncert_k = (df_included['K Uncert (ng/m3)']**2).mean()**0.5  # Root mean square

# Create subplots
num_events = len(wildfire_periods)
fig, axes = plt.subplots(nrows=num_events, ncols=1, figsize=(6, 3 * num_events))

# Plot each wildfire period
for i, (start, end) in enumerate(wildfire_periods):
    ax = axes[i]
    df_period = df[(df.index >= start) & (df.index <= end)]

    # Plot error bars for K with uncertainty
    ax.errorbar(
        df_period.index, df_period['K 19 (ng/m3)'], yerr=df_period['K Uncert (ng/m3)'],
        fmt='o', color='blue', ecolor='lightblue', alpha=0.7, capsize=2, label='K Concentration'
    )

    # Plot average line and shaded uncertainty band
    ax.axhline(mean_k, color='red', linestyle='--', label='Mean K')
    ax.fill_between(df_period.index, mean_k - mean_uncert_k, mean_k + mean_uncert_k,
                    color='red', alpha=0.2, label='± Propagated Uncertainty' if i == 0 else None)

    ax.set_title(event_titles[i])
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    ax.tick_params(axis='x', rotation=45)
    if i == num_events - 1:
        ax.set_xlabel("Time (EST)")
    if i == 0:
        ax.legend(loc='upper right')
    ax.set_ylabel("K (ng/m³)")

plt.tight_layout()
plt.show()
