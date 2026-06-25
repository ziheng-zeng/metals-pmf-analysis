import pandas as pd
import matplotlib.pyplot as plt
from windrose import WindroseAxes

plt.rcParams.update({'font.size': 12})

# Load Xact data and ensure 'TIME' is parsed as datetime with UTC handling
xact_df = pd.read_csv('D:/Documents/research-2024/Xact python code/Xact_EasternTime_7_8_24.csv')
xact_df['TIME'] = pd.to_datetime(xact_df['TIME'], utc=True).dt.tz_convert('US/Eastern')

# Load PM2.5 data and ensure 'TIME' is parsed as datetime with UTC handling
pm25_df = pd.read_csv('C:/Users/zengz/OneDrive/Desktop/LV PM25 + WDS 2023-05-01 to 2024-07-10.csv', skiprows=2)
pm25_df['TIME'] = pd.to_datetime(pm25_df['Date'], utc=True).dt.tz_convert('US/Eastern')

# Merge Xact and PM2.5 data on the TIME column
df = pd.merge(xact_df, pm25_df, on='TIME', how='inner')

# Unit conversion
df['UG/M3'] = df['UG/M3'] * 1000

# Define the time periods for each event
time_periods = {
    "Fireworks 1": (
    pd.Timestamp('2023-07-01 17:00', tz='US/Eastern'), pd.Timestamp('2023-07-02 12:00', tz='US/Eastern')),
    "Fireworks 2": (
    pd.Timestamp('2023-07-03 17:00', tz='US/Eastern'), pd.Timestamp('2023-07-04 16:00', tz='US/Eastern')),
    "Fireworks 3": (
    pd.Timestamp('2023-07-04 17:00', tz='US/Eastern'), pd.Timestamp('2023-07-05 12:00', tz='US/Eastern')),
    "Fireworks 4": (
    pd.Timestamp('2023-07-05 17:00', tz='US/Eastern'), pd.Timestamp('2023-07-06 12:00', tz='US/Eastern')),
    "Wildfire": (pd.Timestamp('2023-06-28 00:00', tz='US/Eastern'), pd.Timestamp('2023-06-30 12:00', tz='US/Eastern'))
}

# Plot wind rose for each time period
for period_name, (start, end) in time_periods.items():
    # Filter data for the current period
    period_df = df[(df['TIME'] >= start) & (df['TIME'] <= end)]

    # Create wind rose plot
    ax = WindroseAxes.from_ax()
    ax.bar(period_df['DEG'], period_df['M/SEC'], normed=True, opening=0.8, edgecolor='white')

    # Customize the plot
    ax.set_legend()
    plt.title(f"Wind Rose for {period_name}\n({start.strftime('%Y-%m-%d %H:%M')} to {end.strftime('%Y-%m-%d %H:%M')})")
    plt.show()