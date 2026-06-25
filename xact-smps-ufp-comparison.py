import glob
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress
import numpy as np
import matplotlib.colors as mcolors
import os
from matplotlib.colors import ListedColormap
import matplotlib as mpl
import altair as alt

save_dir = "D:/Documents/research spring 24/Xact data/plots/comparison-mass-hourly-color-5-19"

# Load processed Xact data, output from the XAct_data.py file
xact_df = pd.read_csv('D:/Documents/research spring 24/Xact python code/may23_may24_Xact.csv')
xact_df['TIME'] = pd.to_datetime(xact_df['TIME'])
# manually selected interested metals list
filtered_metals = [' K 19 (ng/m3)','Ca 20 (ng/m3)', 'Ti 22 (ng/m3)',
'Cr 24 (ng/m3)','Mn 25 (ng/m3)','Fe 26 (ng/m3)',
'Ni 28 (ng/m3)','Cu 29 (ng/m3)','Zn 30 (ng/m3)','As 33 (ng/m3)']

for metal in filtered_metals:
    xact_df[metal] = xact_df[metal]/1000 # unit conversion from ng/m^3 to ug/m^3


# Load SMPS data
path = 'D:/Documents/Research spring 24/SMPS data/data-all-time' # change file path
csv_files = glob.glob(path + '/SMPS*.csv')
df_list = (pd.read_csv(files, skiprows=52) for files in csv_files)
smps_df = pd.concat(df_list, ignore_index=True)
# Get SMPS mass concentrations
tsdf = smps_df.set_index('DateTime Sample Start')
mid_D = np.array([float(x.replace('_', '')) for x in tsdf.columns[41:425]])
# calculate the average difference of the logarithmic midpoints
avg_diff = np.mean(np.diff(np.log10(mid_D)))
# calculate the low bound and higher bound of each bin
D_bound = np.full(mid_D.shape[0] + 1, np.nan)
for i in range(1, (len(D_bound) - 1)):
    D_bound[i] = 10 ** (0.5 * (np.log10(mid_D[i]) + np.log10(mid_D[i - 1])))
D_bound[0] = 10 ** (np.log10(mid_D[0]) - 0.5 * avg_diff)
D_bound[-1] = 10 ** (np.log10(mid_D[-1]) + 0.5 * avg_diff)
D_low = D_bound[0:-1]
D_high = D_bound[1:]
dlogDp = np.log10(D_high) - np.log10(D_low)

# Create a mask for mid_D values that are less than or equal to 100nm
mask = mid_D <= 100
# Apply this mask to filter dNdlogDp for mid_D <= 100
filtered_mid_D = mid_D[mask]
filtered_dNdlogDp = tsdf.iloc[:, 41:425].loc[:, mask]
# Calculate the dN for the filtered data
filtered_dN = filtered_dNdlogDp.values * dlogDp[mask]
# Sum across columns to get total number concentration N
filtered_N = np.nansum(filtered_dN, axis=1)
# Calculate mass distribution and total mass using filtered mid_D and dNdlogDp
density = 1.4  # g/cm3 assumed particle density
filtered_dMdlogDp = (density / 1e9) * (np.pi / 6.) * filtered_mid_D ** 3 * filtered_dNdlogDp  # ug/cm3
filtered_dM = filtered_dMdlogDp * dlogDp[mask]
filtered_M = np.nansum(filtered_dM, axis=1)  # ug/m3

# Create 2 new columns in the original dataframe for the UFP concentration
smps_df['UFP_Mass_Conc_SMPS'] = pd.to_numeric(filtered_M, errors='coerce') # mass
smps_df['UFP_Num_Conc_SMPS'] = pd.to_numeric(filtered_N, errors='coerce') # number


# Ensure datetime formats are consistent and set as index
smps_df['DateTime Sample Start'] = pd.to_datetime(smps_df['DateTime Sample Start'], format='%d/%m/%Y %H:%M:%S')
smps_df.rename(columns={'DateTime Sample Start': 'TIME'}, inplace=True)

# Select numeric columns only for resampling
numeric_cols_smps = smps_df.select_dtypes(include=[np.number]).columns.tolist()
# Resample data to hourly means, using only numeric columns
smps_df_numeric = smps_df.set_index('TIME')[numeric_cols_smps].resample('1h').mean().reset_index()

# Merge the datasets on the 't_base' timestamp column
merged_df = pd.merge(xact_df, smps_df_numeric, on='TIME', suffixes=('_XACT', '_SMPS'))
merged_df['TIME'] = pd.to_datetime(merged_df['TIME'])
merged_df['TIME'] = merged_df['TIME'].dt.tz_localize('UTC')
merged_df['TIME'] = merged_df['TIME'].dt.tz_convert('US/Eastern')

# print(merged_df['TIME'].head())  # check to ensure time is correct

# hourly color coding
merged_df['hour'] = merged_df['TIME'].dt.hour
# Define a colormap - one color for each hour
cmap = plt.get_cmap('viridis')  # Good for continuous data
norm = mcolors.Normalize(vmin=0, vmax=23)

# merged_df['day_of_week'] = merged_df['TIME'].dt.dayofweek  # Monday=0, Sunday=6
# # Define a colormap - one color for each day of the week
# # Define a custom colormap with 7 distinct colors
# day_colors = ['#e6194B', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#46f0f0']
# days_of_week = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
# cmap = ListedColormap(day_colors)

# Create a dictionary to map original names to cleaned names
name_map = {name: name.split()[0] for name in filtered_metals}

# Ploting
# Loop through each metal for comparison and plotting
for metal in filtered_metals:
    y = merged_df[metal]  # XACT metal concentration
    x = merged_df['UFP_Mass_Conc_SMPS']  # SMPS total mass concentration

    # Create a mask that will be True for indices where neither x nor y is NaN
    valid_mask = ~np.isnan(x) & ~np.isnan(y)

    # Perform linear regression on the cleaned data
    slope, intercept, r_value, p_value, std_err = linregress(x[valid_mask], y[valid_mask])

    # Plot the data and the fitted regression line
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(x[valid_mask], y[valid_mask], c=merged_df['hour'][valid_mask], cmap=cmap, norm=norm, alpha=0.5)
    plt.colorbar(scatter, label='Hour of the Day')
    # # Create colorbar with custom labels
    # cbar = plt.colorbar(scatter, ticks=range(7))
    # cbar.set_label('Day of the Week')
    # cbar.ax.set_yticklabels(days_of_week)  # Set text labels
    cleaned_label = name_map[metal]
    plt.plot(x[valid_mask], intercept + slope * x[valid_mask], 'r', label='fitted line')
    plt.annotate(f'r$^2$ = {r_value ** 2:.2f}\nSlope = {slope:.2f}', xy=(0.1, 0.7), xycoords='axes fraction',
                 fontsize=15, fontname="Times New Roman", bbox=dict(facecolor='white', alpha=0.5))
    plt.xlabel('UFP Mass Concentration (ng/m$^{3}$)',
                 fontsize=15, fontname="Times New Roman")
    plt.ylabel(f'{cleaned_label} Mass Concentration (µg/m$^{3}$)',
                 fontsize=15, fontname="Times New Roman")
    plt.xticks(fontsize=12, fontname="Times New Roman")
    plt.yticks(fontsize=12, fontname="Times New Roman")
    plt.legend()
    plt.title(f'Comparison of XACT {cleaned_label} and SMPS(UFP) Data')


    sanitized_element_name = cleaned_label.replace('/', '_').replace(' ', '_')
    plot_filename = f'{sanitized_element_name}_UFP_mass_comparison.png'
    save_path = os.path.join(save_dir, plot_filename)
    plt.savefig(save_path)
    plt.close()

# # Altair interactive plotting
# # Ensure Altair can handle large data by disabling max rows
# alt.data_transformers.disable_max_rows()
#
# # Loop through each metal for comparison and plotting
# for metal in filtered_metals:
#     # Create a Chart object
#     chart = alt.Chart(merged_df).mark_point().encode(
#         x=alt.X('UFP_Mass_Conc_SMPS', title='UFP Mass Concentration'),
#         y=alt.Y(f'{metal}', title=f'{metal} Mass Concentration (µg/m^3)'),
#         tooltip=[
#             alt.Tooltip('TIME', title='Timestamp', format='%Y-%m-%d %H:%M:%S'),  # Custom datetime format
#             alt.Tooltip('UFP_Mass_Conc_SMPS', title='UFP Mass'),
#             alt.Tooltip(f'{metal}', title=f'{metal} Concentration')
#         ]
#     ).properties(
#         title=f'Comparison of XACT {metal} and SMPS(UFP) Data',
#         width=800,
#         height=600
#     )
#
#     # Add regression line
#     regression = chart.transform_regression(
#         'UFP_Mass_Conc_SMPS', metal, method="linear"
#     ).mark_line(color='red')
#
#     # Combine the points and the regression line
#     final_chart = (chart + regression).interactive()
#
#     # Save chart to HTML file
#     sanitized_element_name = metal.replace('/', '_').replace(' ', '_')
#     plot_filename = f'{sanitized_element_name}_UFP_mass_comparison.html'
#     save_path = os.path.join(save_dir, plot_filename)
#     final_chart.save(save_path)
