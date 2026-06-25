import pandas as pd
import pytz

# Load the data
df = pd.read_csv("D:/Documents/research-2024/Xact python code/processed data results/Xact_EST_May2023_July2024_new_uncert.csv")

# Convert 'TIME' column to datetime
df['TIME'] = pd.to_datetime(df['TIME'])

# Define start and end dates with timezone
eastern = pytz.timezone('US/Eastern')
start_date = pd.Timestamp('2023-05-15 00:00', tz='US/Eastern')
end_date = pd.Timestamp('2023-05-29 23:59', tz='US/Eastern')

# Filter date range
df_filtered = df[(df['TIME'] >= start_date) & (df['TIME'] <= end_date)]

# Identify concentration and uncertainty columns
cols_to_keep = [col for col in df.columns if (" (ng/m3)" in col) or ("uncert" in col.lower())]

# Add 'TIME' to the list of columns to keep
cols_to_keep = ['TIME'] + cols_to_keep

# Filter the DataFrame
df_selected = df_filtered[cols_to_keep]

# Save to CSV
output_path = 'D:/Documents/research-2024/Xact python code/XACT_data_Lawrenceville_May15_May29_2023.csv'
df_selected.to_csv(output_path, index=False)



