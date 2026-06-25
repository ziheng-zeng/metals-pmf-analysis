from xact_header import headers, element_names, element_to_dl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.collections as mcoll
import pandas as pd
import pytz
import os
import glob
import re
import numpy as np

#path = "D:/Documents/research spring 24/Xact data/mar-apr"
save_dir = "D:/Documents/research spring 24/Xact data/plots/time-series-6-11"

df_combined = pd.read_csv('D:/Documents/research spring summer 24/Xact python code/jun23_may24_Xact_cleaned.csv')

# csv_files = glob.glob(os.path.join(path, "*.CSV"))
# # print(csv_files)
# df_list=[]
# for csv_file in csv_files:
#     # get the actual number of columns in the file by reading only the header line
#     num_columns = pd.read_csv(csv_file, index_col=1, nrows=0).shape[1]
#     # construct the range of columns from 0 to num_columns
#     range_columns = [i for i in range(0, num_columns+1)]
#     # read file only within the range of columns
#     df = pd.read_csv(csv_file, usecols=range_columns, index_col=1, skiprows=1)
#     df_list.append(df)
#     #print(df[' K 19 (ng/m3)']) # this key name has format issue
# # Concatenate all DataFrames in the list
# df_combined = pd.concat(df_list, ignore_index=True)
# df_combined.replace(0, np.nan, inplace=True)
# # df_combined.to_csv('filenew.csv', index=False)
# Convert 'TIME' to datetime to ease extraction of time
# df_combined['TIME'] = pd.to_datetime(df_combined['TIME'])
# # Replace rows where time is 00:30:00 with 00:00:00
# df_combined.loc[df_combined['TIME'].dt.time == pd.to_datetime('00:30:00').time(), 'TIME'] = df_combined['TIME'].apply(lambda dt: dt.replace(hour=0, minute=0, second=0))
# # Remove rows where time is 00:15:00
# df_combined = df_combined[df_combined['TIME'].dt.time != pd.to_datetime('00:15:00').time()]
#
# # Convert Timezone to Eastern Time Zone
# df_combined['TIME'] = pd.to_datetime(df_combined['TIME'])
# df_combined['TIME'] = df_combined['TIME'].dt.tz_localize('UTC')
# df_combined['TIME'] = df_combined['TIME'].dt.tz_convert('US/Eastern')
# # df_combined.to_csv('Xact_EasternTimeZone.csv', index=False)
# df_combined['TIME'] = pd.to_datetime(df_combined['TIME'])


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

# Detection limit
element_to_dl_full = {'Al 13 (ng/m3)': 100, 'Si 14 (ng/m3)': 17.8, 'P 15 (ng/m3)': 5.2,
                      'S 16 (ng/m3)': 3.16, 'Cl 17 (ng/m3)': 1.73, ' K 19 (ng/m3)': 1.17,
                      'Ca 20 (ng/m3)': 0.3, 'Ti 22 (ng/m3)': 0.16, 'V 23 (ng/m3)': 0.12,
                      'Cr 24 (ng/m3)': 0.12, 'Mn 25 (ng/m3)': 0.14, 'Fe 26 (ng/m3)': 0.17,
                      'Co 27 (ng/m3)': 0.14, 'Ni 28 (ng/m3)': 0.1, 'Cu 29 (ng/m3)': 0.079,
                      'Zn 30 (ng/m3)': 0.067, 'As 33 (ng/m3)': 0.063, 'Se 34 (ng/m3)': 0.081,
                      'Br 35 (ng/m3)': 0.1, 'Ag 47 (ng/m3)': 1.9, 'Cd 48 (ng/m3)': 2.5,
                      'In 49 (ng/m3)': 3.1, 'Sn 50 (ng/m3)': 4.1, 'Sb 51 (ng/m3)': 5.2,
                      'Ba 56 (ng/m3)': 0.39, 'Hg 80 (ng/m3)': 0.12, 'Tl 81 (ng/m3)': 0.12,
                      'Pb 82 (ng/m3)': 0.13, 'Bi 83 (ng/m3)': 0.13}


# # Ensure 'TIME' is set as the DataFrame index
# df_combined.set_index('TIME', inplace=True)
# # # Define a function to plot the time series for a given element, group by day of week
# def plot_element_time_series(element_name):
#     # Make sure the DataFrame is sorted by the index
#     df_combined.sort_index(inplace=True)
#     # Compute 'hour' column
#     df_combined['hour'] = df_combined.index.hour + df_combined.index.minute / 60
#     # Group by day of the week and hour, then calculate average
#     grouped = df_combined.groupby([df_combined.index.day_name(), df_combined['hour']])[element_name].mean().unstack(0)
#     fig, ax = plt.subplots(figsize=(12, 6))
#     # Dictionary to label days of the week
#     days_of_week = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
#
#     # Use the nipy_spectral colormap, retrieving 7 distinct colors
#     colors = plt.get_cmap('nipy_spectral', 7)
#
#     # Iterate over the days of the week
#     for i, day in enumerate(days_of_week):
#         if day in grouped.columns:
#             # The colormap index needs to be normalized between 0 and 1
#             color_index = i / 7  # normalize day index to 0-1 range
#             color = colors(color_index)
#             ax.plot(grouped.index, grouped[day], label=day, color=color)
#
#     if element_name in element_to_dl_full:
#         detection_limit = element_to_dl_full[element_name]
#         ax.axhline(y=detection_limit, color='r', linestyle='--', label='Detection Limit')
#     ax.set_xlim(0, 24)
#     ax.set_xticks(range(25))
#     ax.set_ylabel('Average Concentration (ng/m3)')
#     ax.legend(title="Day of Week", loc='upper right', fontsize='x-small')
#     ax.set_title(f'Weekly Average Time Series of {element_name}')
#     ax.set_xlabel('Hour of Day')
#
#     plt.tight_layout()

# Define a function to plot the time series for a given element, grouped by month
# def plot_element_time_series(element_name):
#     # Make sure the DataFrame is sorted by the index
#     df_combined.sort_index(inplace=True)
#
#     # Compute 'hour' column for the entire DataFrame
#     df_combined['hour'] = df_combined.index.hour + df_combined.index.minute / 60
#
#     # Group by month and calculate average for each hour
#     fig, ax = plt.subplots(figsize=(12, 6))
#     grouped = df_combined.groupby([df_combined.index.month, df_combined['hour']])[element_name].mean().unstack(0)
#
#     # Dictionary to label months
#     months = {
#         1: "January", 2: "February", 3: "March", 4: "April",
#         5: "May", 6: "June", 7: "July", 8: "August",
#         9: "September", 10: "October", 11: "November", 12: "December"
#     }
#     # Use a colormap
#     colors = plt.get_cmap('nipy_spectral', 12)
#
#     for month, name in months.items():
#         if month in grouped.columns:
#             # The colormap index needs to be normalized between 0 and 1
#             color_index = (month - 1) / 12  # normalize month to 0-1 range
#             color = colors(color_index)
#             ax.plot(grouped.index, grouped[month], label=name, color=color)
#
#     # Check for a detection limit and plot it if applicable
#     if element_name in element_to_dl_full:
#         detection_limit = element_to_dl_full[element_name]
#         ax.axhline(y=detection_limit, color='r', linestyle='--', label='Detection Limit')
#
#     # Set the x-axis to show 24 hours
#     ax.set_xlim(0, 24)
#     ax.set_xticks(np.arange(0, 25, 1))
#     # Set the y-axis label
#     ax.set_ylabel('Average Concentration (ng/m3)')
#     # Add legend
#     ax.legend(title='Month', loc='upper right', fontsize='x-small')
#
#     # Set title and labels
#     ax.set_title(f'Monthly Average Time Series of {element_name} Concentration')
#     ax.set_xlabel('Hour of Day')
#
#     plt.tight_layout()

    # # Save the plot
    # if not os.path.exists(save_dir):
    #     os.makedirs(save_dir)
    # sanitized_element_name = element_name.replace('/', '_').replace(' ', '_')
    # plot_filename = f'{sanitized_element_name}_daily_time_series.png'
    # save_path = os.path.join(save_dir, plot_filename)
    # plt.savefig(save_path)
    # plt.close()

# filtered_element_names = ['Al 13 (ng/m3)',
# 'Si 14 (ng/m3)','P 15 (ng/m3)','S 16 (ng/m3)',
# 'Cl 17 (ng/m3)', ' K 19 (ng/m3)','Ca 20 (ng/m3)',
# 'Sc 21 (ng/m3)','Ti 22 (ng/m3)','V 23 (ng/m3)',
# 'Cr 24 (ng/m3)','Mn 25 (ng/m3)','Fe 26 (ng/m3)','Co 27 (ng/m3)',
# 'Ni 28 (ng/m3)','Cu 29 (ng/m3)','Zn 30 (ng/m3)',
# 'Ga 31 (ng/m3)','Ge 32 (ng/m3)','As 33 (ng/m3)','Se 34 (ng/m3)',
# 'Br 35 (ng/m3)','Rb 37 (ng/m3)','Sr 38 (ng/m3)',
# 'Y 39 (ng/m3)','Zr 40 (ng/m3)','Nb 41(ng/m3)','Mo 42 (ng/m3)','Pd 46 (ng/m3)',
# 'Ag 47 (ng/m3)','Cd 48 (ng/m3)','In 49 (ng/m3)','Sn 50 (ng/m3)',
# 'Sb 51 (ng/m3)','Te 52 (ng/m3)','I 53 (ng/m3)',
# 'Cs 55 (ng/m3)','Ba 56 (ng/m3)','La 57 (ng/m3)','Ce 58 (ng/m3)',
# 'W 74 (ng/m3)','Pt 78 (ng/m3)','Au 79 (ng/m3)',
# 'Hg 80 (ng/m3)','Tl 81 (ng/m3)','Pb 82 (ng/m3)',
# 'Bi 83 (ng/m3)'
# ]
# # Iterate over all elements and plot
# for element in filtered_element_names:
#     plot_element_time_series(element)



# # color coding plots
# def make_segments(x, y):
#     points = np.array([x, y]).T.reshape(-1, 1, 2)
#     segments = np.concatenate([points[:-1], points[1:]], axis=1)
#     return segments
#
#
# def colorline_with_uncertainty(ax, dates, y, y_uncertainty, cmap='viridis'):
#     hours = dates.hour + dates.minute / 60  # Hour of the day for color coding
#     norm = plt.Normalize(0, 24)  # Normalize hour for the colormap
#
#     # Convert dates to matplotlib date format for plotting
#     x = mdates.date2num(dates.to_pydatetime())
#
#     # Create line segments and color them based on time of day
#     segments = make_segments(x, y)
#     lc = mcoll.LineCollection(segments, array=hours, cmap=cmap, norm=norm, linewidth=2, alpha=0.7)
#     ax.add_collection(lc)
#
#     # Plot uncertainty as shaded area
#     ax.fill_between(dates, y - y_uncertainty, y + y_uncertainty, color='grey', alpha=0.3)
#
#     # Plot the data points (dots)
#     #ax.scatter(dates, y, color='black', s=10, zorder=3)  # Zorder for drawing order
#
#     # Adjust axes limits to accommodate the line collection
#     ax.autoscale_view()
#
#     return lc
# #
# #
#
# def sanitize_filename(filename):
#     invalid_chars = ['\\', '/', ':', '*', '?', '"', '<', '>', '|']
#     for char in invalid_chars:
#         filename = filename.replace(char, '_')  # Replace invalid char with underscore
#     return filename
#
# # Create the directory for plots if it doesn't exist
# if not os.path.exists(save_dir):
#     os.makedirs(save_dir)
#
# # Detection limit
# element_to_dl_full = {'Al 13 (ng/m3)': 100, 'Si 14 (ng/m3)': 17.8, 'P 15 (ng/m3)': 5.2, 'S 16 (ng/m3)': 3.16, 'Cl 17 (ng/m3)': 1.73, ' K 19 (ng/m3)': 1.17, 'Ca 20 (ng/m3)': 0.3, 'Ti 22 (ng/m3)': 0.16, 'V 23 (ng/m3)': 0.12, 'Cr 24 (ng/m3)': 0.12, 'Mn 25 (ng/m3)': 0.14, 'Fe 26 (ng/m3)': 0.17, 'Co 27 (ng/m3)': 0.14, 'Ni 28 (ng/m3)': 0.1, 'Cu 29 (ng/m3)': 0.079, 'Zn 30 (ng/m3)': 0.067, 'As 33 (ng/m3)': 0.063, 'Se 34 (ng/m3)': 0.081, 'Br 35 (ng/m3)': 0.1, 'Ag 47 (ng/m3)': 1.9, 'Cd 48 (ng/m3)': 2.5, 'In 49 (ng/m3)': 3.1, 'Sn 50 (ng/m3)': 4.1, 'Sb 51 (ng/m3)': 5.2, 'Ba 56 (ng/m3)': 0.39, 'Hg 80 (ng/m3)': 0.12, 'Tl 81 (ng/m3)': 0.12, 'Pb 82 (ng/m3)': 0.13, 'Bi 83 (ng/m3)': 0.13}
#
# # filtered out no Xact data elements
# filtered_element_names = ['Al 13 (ng/m3)',
# 'Si 14 (ng/m3)','P 15 (ng/m3)','S 16 (ng/m3)',
# 'Cl 17 (ng/m3)', ' K 19 (ng/m3)','Ca 20 (ng/m3)',
# 'Sc 21 (ng/m3)','Ti 22 (ng/m3)','V 23 (ng/m3)',
# 'Cr 24 (ng/m3)','Mn 25 (ng/m3)','Fe 26 (ng/m3)','Co 27 (ng/m3)',
# 'Ni 28 (ng/m3)','Cu 29 (ng/m3)','Zn 30 (ng/m3)',
# 'Ga 31 (ng/m3)','Ge 32 (ng/m3)','As 33 (ng/m3)','Se 34 (ng/m3)',
# 'Br 35 (ng/m3)','Rb 37 (ng/m3)','Sr 38 (ng/m3)',
# 'Y 39 (ng/m3)','Zr 40 (ng/m3)','Nb 41(ng/m3)','Mo 42 (ng/m3)','Pd 46 (ng/m3)',
# 'Ag 47 (ng/m3)','Cd 48 (ng/m3)','In 49 (ng/m3)','Sn 50 (ng/m3)',
# 'Sb 51 (ng/m3)','Te 52 (ng/m3)','I 53 (ng/m3)',
# 'Cs 55 (ng/m3)','Ba 56 (ng/m3)','La 57 (ng/m3)','Ce 58 (ng/m3)',
# 'W 74 (ng/m3)','Pt 78 (ng/m3)','Au 79 (ng/m3)',
# 'Hg 80 (ng/m3)','Tl 81 (ng/m3)','Pb 82 (ng/m3)',
# 'Bi 83 (ng/m3)'
# ]
#
# for metal_conc, metal_uncert in metal_uncert_pairs.items():
#     if metal_conc in filtered_element_names:
#         fig, ax = plt.subplots(figsize=(10, 6))
#
#         dates = df_combined.index
#         y = df_combined[metal_conc].values
#         y_uncertainty = df_combined[metal_uncert].values  # Uncertainty values
#
#         # Generate the plot with uncertainties and dots
#         lc = colorline_with_uncertainty(ax, dates, y, y_uncertainty, cmap='viridis')
#         # Check if the element has a detection limit and plot it
#         element_name_with_units = f'{metal_conc}'  # Adjust this line to match how your metal_conc is named in the dictionary
#         if element_name_with_units in element_to_dl_full:
#             detection_limit = element_to_dl_full[element_name_with_units]
#             ax.axhline(y=detection_limit, color='r', linestyle='--', label='Detection Limit')
#
#         # Format the x-axis to display date format
#         ax.xaxis.set_major_locator(mdates.MonthLocator())
#         ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
#         plt.xticks(rotation=45)
#
#         ax.set_xlabel('Time (US/Eastern)')
#         ax.set_ylabel('Concentration (ng/m3)')
#         ax.set_title(f'Time Series of {metal_conc} Concentration with Time Color Coding')
#
#         # Add colorbar
#         cbar = fig.colorbar(lc, ax=ax)
#         cbar.set_label('Hour of Day')
#         cbar.set_ticks(range(0, 25, 3))
#
#         plt.tight_layout()
#         # Sanitize and save the plot to the specified directory
#         plot_filename = sanitize_filename(f'{metal_conc}_time_series_colored.png')
#         save_path = os.path.join(save_dir, plot_filename)
#         plt.savefig(save_path)
#         plt.close()


# for metal_conc, metal_uncert in metal_uncert_pairs.items():
#     if metal_conc in filtered_element_names:
#         fig, ax = plt.subplots(figsize=(10, 6))
#         # Plot the concentration and uncertainty as you already do
#         ax.plot(df_combined.index, df_combined[metal_conc], label=f'{metal_conc} Concentration', marker='o', linestyle='-')
#         ax.fill_between(df_combined.index, df_combined[metal_conc] - df_combined[metal_uncert],
#                         df_combined[metal_conc] + df_combined[metal_uncert], color='gray', alpha=0.2,
#                         label=f'{metal_conc} Uncertainty')
#
#         # Check if the element has a detection limit and plot it
#         element_name_with_units = f'{metal_conc}'  # Adjust this line to match how your metal_conc is named in the dictionary
#         if element_name_with_units in element_to_dl_full:
#             detection_limit = element_to_dl_full[element_name_with_units]
#             ax.axhline(y=detection_limit, color='r', linestyle='--', label='Detection Limit')
#
#         # Set major ticks to show at the start of every week
#         ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO))
#         # Set major formatter to show the date as 'Year-Month-Day'
#         ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
#         # (Optional) Set minor ticks to show every day
#         ax.xaxis.set_minor_locator(mdates.DayLocator())
#
#         # Optionally, for even less clutter, use mdates.YearLocator() to show only the year part at the beginning of each year
#
#         # Improve layout
#         fig.autofmt_xdate()  # Auto format date labels to avoid overlap
#         ax.set_xlabel('Time')
#         ax.set_ylabel('Concentration (ng/m3)')
#         ax.set_title(f'Time Series of {metal_conc} Concentration and Uncertainty')
#         ax.legend()
#
#         plt.xticks(rotation=45)  # Rotate the x-axis labels for better readability
#         plt.tight_layout()
#
#         # Sanitize and save the plot to the specified directory
#         plot_filename = sanitize_filename(f'{metal_conc}_time_series.png')
#         save_path = os.path.join(save_dir, plot_filename)
#         plt.savefig(save_path)
#         plt.close()
#

# # Go through each metal and its uncertainty and perform the operation
# for metal_col, uncert_col in metal_uncert_pairs.items():
#     if metal_col in df_combined.columns and uncert_col in df_combined.columns:
#         ratio = df_combined[metal_col] / df_combined[uncert_col]
#         mask = ratio < 5
#         df_combined.loc[mask, [metal_col, uncert_col]] = np.nan
#     else:
#         print(f"Column missing: {metal_col} or {uncert_col}")
#
# Save the modified DataFrame to a new CSV file
# df_combined.to_csv('valid.csv', index=False)

#
# # Plotting
# for metal in element_names:
#         df_combined['TIME'] = pd.to_datetime(df_combined['TIME'])
#         plt.figure(figsize=(10, 6))
#         plt.plot(df_combined['TIME'], df_combined[metal], marker='o', linestyle='-', label=metal)
#
#         # Format the x-axis to show a label for every few days (adjust as needed)
#         plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=5))  # Show a label every 5 days
#         plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
#
#         plt.xlabel('DateTime')
#         plt.ylabel(f'{metal} Concentration (ng/m3)')
#         plt.title(f'{metal} Concentration Over Time')
#         plt.grid(True)
#         plt.gcf().autofmt_xdate()  # Rotate dates for better spacing
#         plt.tight_layout()
#
#         # Replace problematic characters in metal names that might cause file saving issues
#         safe_metal_name = metal.replace('/', '_').replace('\\', '_')
#
#         # Save the figure to the specified directory with the custom filename
#         full_path = os.path.join(save_dir, f"{safe_metal_name}.png")
#         plt.savefig(full_path)
#         plt.close()  # Close the plot after saving to free up memory
#     # else:
#     #     print(f"Column '{metal}' not found in DataFrame or is an uncertainty column. Skipping...")

