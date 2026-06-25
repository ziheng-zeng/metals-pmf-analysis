from xact_header import headers, element_names, element_to_dl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.collections as mcoll
import pandas as pd
import os
import glob
import re
import numpy as np

path = "D:/Documents/research-2024/Xact data"

# Search CSV files in the subdirectories only
subfolders = ["july2025-oct2025"]
csv_files = []
for subfolder in subfolders:
    full_path = os.path.join(path, subfolder, "*.CSV")
    csv_files.extend(glob.glob(full_path))

df_list = []
for csv_file in csv_files:
    # get the actual number of columns in the file by reading only the header line
    num_columns = pd.read_csv(csv_file, index_col=1, nrows=0).shape[1]
    # construct the range of columns from 0 to num_columns
    range_columns = [i for i in range(0, num_columns + 1)]
    # read file only within the range of columns
    df = pd.read_csv(csv_file, usecols=range_columns, index_col=1, skiprows=1)
    df_list.append(df)
    # print(df[' K 19 (ng/m3)']) # this key name has format issue

# Concatenate all DataFrames in the list
df_combined = pd.concat(df_list, ignore_index=True)

# Convert 'TIME' to datetime to ease extraction of time
df_combined['TIME'] = pd.to_datetime(df_combined['TIME'])
# Replace rows where time is 00:30:00 with 00:00:00
df_combined.loc[df_combined['TIME'].dt.time == pd.to_datetime('00:30:00').time(), 'TIME'] = df_combined['TIME'].apply(
    lambda dt: dt.replace(hour=0, minute=0, second=0))
# Remove rows where time is 00:15:00
df_combined = df_combined[df_combined['TIME'].dt.time != pd.to_datetime('00:15:00').time()]

# Convert Timezone to Eastern Time Zone
df_combined['TIME'] = pd.to_datetime(df_combined['TIME'])
df_combined['TIME'] = df_combined['TIME'].dt.tz_localize('UTC')
df_combined['TIME'] = df_combined['TIME'].dt.tz_convert('US/Eastern')
# df_combined.to_csv('Xact_EasternTimeZone.csv', index=False)
df_combined['TIME'] = pd.to_datetime(df_combined['TIME'])
# df_combined.set_index('TIME', inplace=True)

# Export dataframe into csv
df_combined.to_csv('Xact_July_to_Oct2025.csv', index=False)


# new_data = pd.read_csv(file_path, usecols=['TIME'], low_memory=False)
#
# # Display the first few rows to inspect the format
# print("First few rows of the dataset:")
# print(new_data.head())
#
# # Convert the 'TIME' column to datetime format without specifying the format
# new_data['TIME'] = pd.to_datetime(new_data['TIME'], errors='coerce')
#
# # Display the first few rows after conversion to check for errors
# print("First few rows after date conversion:")
# print(new_data.head())
#
# # Drop any rows with NaT values resulting from conversion errors
# new_data = new_data.dropna(subset=['TIME'])
#
# # Analysis by hour
# new_data['Hour'] = new_data['TIME'].dt.floor('h')
# unique_hours = new_data['Hour'].nunique()
# total_hours_in_year = 8760
#
# # Analysis by day
# new_data['Day'] = new_data['TIME'].dt.date
# unique_days = new_data['Day'].nunique()
# total_days_in_year = 365
#
# # Output the results
# results = {
#     'unique_hours': unique_hours,
#     'total_hours_in_year': total_hours_in_year,
#     'unique_days': unique_days,
#     'total_days_in_year': total_days_in_year,
#     'hours_collected_percentage': (unique_hours / total_hours_in_year) * 100,
#     'days_collected_percentage': (unique_days / total_days_in_year) * 100
# }
#
# # Display the results
# print(f"Collected {results['unique_hours']} hours of data out of the possible {results['total_hours_in_year']} hours in the year.")
# print(f"Collected data on {results['unique_days']} days out of the possible {results['total_days_in_year']} days in the year.")
# print(f"Percentage of hours collected: {results['hours_collected_percentage']:.2f}%")
# print(f"Percentage of days collected: {results['days_collected_percentage']:.2f}%")


# # Identifying concentration columns
# concentration_cols = [col for col in headers if " (ng/m3)" in col and "uncert" not in col.lower()]
#
# # Detection limit
# detection_limits = {'Al 13 (ng/m3)': 100, 'Si 14 (ng/m3)': 17.8, 'P 15 (ng/m3)': 5.2, 'S 16 (ng/m3)': 3.16,
#                       'Cl 17 (ng/m3)': 1.73, ' K 19 (ng/m3)': 1.17, 'Ca 20 (ng/m3)': 0.3, 'Ti 22 (ng/m3)': 0.16,
#                       'V 23 (ng/m3)': 0.12, 'Cr 24 (ng/m3)': 0.12, 'Mn 25 (ng/m3)': 0.14, 'Fe 26 (ng/m3)': 0.17,
#                       'Co 27 (ng/m3)': 0.14, 'Ni 28 (ng/m3)': 0.1, 'Cu 29 (ng/m3)': 0.079, 'Zn 30 (ng/m3)': 0.067,
#                       'As 33 (ng/m3)': 0.063, 'Se 34 (ng/m3)': 0.081, 'Br 35 (ng/m3)': 0.1, 'Ag 47 (ng/m3)': 1.9,
#                       'Cd 48 (ng/m3)': 2.5, 'In 49 (ng/m3)': 3.1, 'Sn 50 (ng/m3)': 4.1, 'Sb 51 (ng/m3)': 5.2,
#                       'Ba 56 (ng/m3)': 0.39, 'Hg 80 (ng/m3)': 0.12, 'Tl 81 (ng/m3)': 0.12, 'Pb 82 (ng/m3)': 0.13,
#                       'Bi 83 (ng/m3)': 0.13}
#
# # Replace values below detection limits with NaN
# for col in concentration_cols:
#     if col in detection_limits:
#         df_combined[col] = df_combined[col].where(df_combined[col] >= detection_limits[col], np.nan)
#
# df_combined.to_csv('cleaned2.csv', index=False)
#
# # Sum concentrations for each day
# daily_sums = df_combined.groupby('TIME')[concentration_cols].sum()
#
# # Calculate the average of these daily sums
# average_daily_sum = daily_sums.sum(axis=1).mean()/1000  #ug/m^3
#
# print("Average of daily sums of all concentration columns:", average_daily_sum)


