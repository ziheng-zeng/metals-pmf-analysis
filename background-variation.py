import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore

# Load and parse data
xact_df = pd.read_csv('D:/Documents/research-2024/Xact python code/Xact_EasternTime_7_8_24.csv')
xact_df['TIME'] = pd.to_datetime(xact_df['TIME'], utc=True).dt.tz_convert('US/Eastern')

pm25_df = pd.read_csv('C:/Users/zengz/OneDrive/Desktop/LV PM25 + WDS 2023-05-01 to 2024-07-10.csv', skiprows=2)
pm25_df['TIME'] = pd.to_datetime(pm25_df['Date'], utc=True).dt.tz_convert('US/Eastern')

# Merge data
df = pd.merge(xact_df, pm25_df, on='TIME', how='inner')
df['UG/M3'] = df['UG/M3'] * 1000  # Unit conversion

# Set 'TIME' as the index
df.set_index('TIME', inplace=True)

# List of metals to analyze
metals_of_interest = ['Pb', 'K', 'Zn', 'Fe', 'As', 'Ti', 'Sr', 'Ba', 'Bi', 'Cu', 'Ca', 'Mn', 'Cr', 'Se']

# Iterate over each metal, calculate the Metal/PM2.5 ratio, plot control charts
for metal in metals_of_interest:
    conc_cols = [col for col in df.columns if f"{metal} " in col and 'uncert' not in col.lower()]
    if conc_cols:
        conc_col = conc_cols[0]  # Select the first matching column
        df[f'{metal}/PM2.5'] = df[conc_col] / df['UG/M3'].replace(0, pd.NA)

        # Calculate mean and control limits
        central_line = df[f'{metal}/PM2.5'].mean()
        control_limit_upper = central_line + 3*df[f'{metal}/PM2.5'].std()
        control_limit_lower = central_line - 3*df[f'{metal}/PM2.5'].std()

        # Plot control chart
        plt.figure(figsize=(10, 5))
        plt.plot(df.index, df[f'{metal}/PM2.5'], label=f'{metal}/PM2.5 Data', marker='o', linestyle='', alpha=0.5)
        plt.axhline(y=central_line, color='r', linestyle='-', label='Central Line')
        plt.axhline(y=control_limit_upper, color='g', linestyle='--', label='Upper Control Limit')
        plt.axhline(y=control_limit_lower, color='g', linestyle='--', label='Lower Control Limit')

        plt.title(f'Control Chart for {metal}/PM2.5 Ratio')
        plt.xlabel('Date')
        plt.ylabel(f'{metal}/PM2.5 Ratio')
        plt.legend()
        plt.grid(True)
        plt.show()
    else:
        print(f"Column for {metal} not found in DataFrame.")

#
# #boxlots
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
#
# # Load and parse data
# xact_df = pd.read_csv('D:/Documents/research-2024/Xact python code/Xact_EasternTime_7_8_24.csv')
# xact_df['TIME'] = pd.to_datetime(xact_df['TIME'], utc=True).dt.tz_convert('US/Eastern')
#
# pm25_df = pd.read_csv('C:/Users/zengz/OneDrive/Desktop/LV PM25 + WDS 2023-05-01 to 2024-07-10.csv', skiprows=2)
# pm25_df['TIME'] = pd.to_datetime(pm25_df['Date'], utc=True).dt.tz_convert('US/Eastern')
#
# # Merge data
# df = pd.merge(xact_df, pm25_df, on='TIME', how='inner')
# df['UG/M3'] = df['UG/M3'] * 1000  # Unit conversion
#
# # Set 'TIME' as the index
# df.set_index('TIME', inplace=True)
# df['Month'] = df.index.to_period('M')  # Grouping by month
#
# # List of metals to analyze
# metals_of_interest = ['Pb', 'K', 'Zn', 'Fe', 'As', 'Ti', 'Sr', 'Ba', 'Bi', 'Cu', 'Ca', 'Mn', 'Cr', 'Se']
#
# # Iterate over each metal, calculate the Metal/PM2.5 ratio, plot monthly box plots
# for metal in metals_of_interest:
#     conc_cols = [col for col in df.columns if f"{metal} " in col and 'uncert' not in col.lower()]
#     if conc_cols:
#         conc_col = conc_cols[0]  # Select the first matching column
#         df[f'{metal}/PM2.5'] = df[conc_col] / df['UG/M3'].replace(0, pd.NA)
#
#         # Plot monthly box plots
#         plt.figure(figsize=(12, 6))
#         sns.boxplot(x='Month', y=f'{metal}/PM2.5', data=df)
#         plt.title(f'Monthly Distribution of {metal}/PM2.5 Ratios')
#         plt.xlabel('Month')
#         plt.ylabel(f'{metal}/PM2.5 Ratio')
#         plt.xticks(rotation=45)  # Rotate the month labels for better readability
#         plt.grid(True)
#         plt.show()
#     else:
#         print(f"Column for {metal} not found in DataFrame.")
