import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress
import numpy as np
import matplotlib.dates as mdates
import pytz
import matplotlib as mpl
from xact_header import headers

df = pd.read_csv('D:/Documents/research-2024/Xact python code/Xact_EST_May2023_May2025_combined.csv')
df['TIME'] = pd.to_datetime(df['TIME'])
eastern = pytz.timezone('US/Eastern')
# Define the periods
# periods1 = [
#     (pd.Timestamp('2023-07-01 17:00', tz='US/Eastern'), pd.Timestamp('2023-07-02 09:00', tz='US/Eastern')),
#     (pd.Timestamp('2023-07-03 17:00', tz='US/Eastern'), pd.Timestamp('2023-07-05 12:00', tz='US/Eastern')),
#     #  (pd.Timestamp('2024-06-29 20:00', tz='US/Eastern'), pd.Timestamp('2024-06-30 12:00', tz='US/Eastern'))
# ]
# periods2 = [
#     (pd.Timestamp('2024-06-29 17:00', tz='US/Eastern'), pd.Timestamp('2024-06-30 18:00', tz='US/Eastern')),
#     (pd.Timestamp('2024-06-30 21:00', tz='US/Eastern'), pd.Timestamp('2024-07-01 01:00', tz='US/Eastern'))
# ]
#
# periods3 = [
#     (pd.Timestamp('2023-06-27', tz='US/Eastern'), pd.Timestamp('2023-06-30', tz='US/Eastern'))
# ]
#
# # Create masks for each period
# mask1 = pd.Series(False, index=df.index)
# mask2 = pd.Series(False, index=df.index)
# mask3 = pd.Series(False, index=df.index)
#
# for start, end in periods1:
#     mask1 |= (df['TIME'] >= start) & (df['TIME'] <= end)
#
# for start, end in periods2:
#     mask2 |= (df['TIME'] >= start) & (df['TIME'] <= end)
#
# for start, end in periods3:
#     mask3 |= (df['TIME'] >= start) & (df['TIME'] <= end)
#
# # Filter data for each period
# df1 = df[mask1].copy()
# df2 = df[mask2].copy()
# df3 = df[mask3].copy()  # New dataframe for the third period
#
# # Strip spaces from column names
# df1.columns = df1.columns.str.strip()
# df2.columns = df2.columns.str.strip()
# df3.columns = df3.columns.str.strip()
#
# # Filter for necessary elements
# elements = ['K 19 (ng/m3)', 'Ti 22 (ng/m3)', 'Cu 29 (ng/m3)', 'Ba 56 (ng/m3)']
# df1 = df1[elements]
# df2 = df2[elements]
# df3 = df3[elements]
#
# # Prepare for plotting
# pairs = [
#     ('K 19 (ng/m3)', 'Ti 22 (ng/m3)'),
#     ('K 19 (ng/m3)', 'Cu 29 (ng/m3)'),
#     ('K 19 (ng/m3)', 'Ba 56 (ng/m3)'),
#     ('Ti 22 (ng/m3)', 'Cu 29 (ng/m3)'),
#     ('Ti 22 (ng/m3)', 'Ba 56 (ng/m3)'),
#     ('Cu 29 (ng/m3)', 'Ba 56 (ng/m3)')
# ]
#
# # Set the global font to Times New Roman
# mpl.rcParams['font.family'] = 'Times New Roman'
# mpl.rcParams['font.size'] = 28
#
# fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(20, 14))
# axes = axes.flatten()
#
# for i, (elem1, elem2) in enumerate(pairs):
#     ax = axes[i]
#
#     # Plot Period 1
#     regression1 = linregress(df1[elem1], df1[elem2])
#     r_squared1 = regression1.rvalue ** 2
#     x1 = np.linspace(df1[elem1].min(), df1[elem1].max(), 100)
#     y1 = regression1.intercept + regression1.slope * x1
#     ax.scatter(df1[elem1], df1[elem2], alpha=0.7, color='black', s=100, label='2023 Fourth of July fireworks events')  # small black points
#     ax.plot(x1, y1, color='blue')
#
#     # Plot Period 2
#     ax.scatter(df2[elem1], df2[elem2], alpha=0.7, edgecolor='red', facecolor='none', s=400, marker='s',
#                label='2024 fireworks events')  # large empty boxes
#
#     # Plot Period 3 - Canadian wildfires
#     ax.scatter(df3[elem1], df3[elem2], alpha=0.7, color='green', facecolor='none' , s=280, marker='^',
#                label='2023 Canadian wildfires')  # triangles
#
#     # Formatting
#     ax.set_xlabel(f"{elem1.split(' ')[0]} Concentration (ng/m³)")
#     ax.set_ylabel(f"{elem2.split(' ')[0]} Concentration (ng/m³)")
#
#     # Display slope and R² value
#     ax.text(0.65, 0.2, f'Slope: {regression1.slope:.2f}\n$R^2$: {r_squared1:.2f}',
#             transform=ax.transAxes, fontsize=25, verticalalignment='top', color='black')
#
# # Update the legend to include all periods
# handles = [plt.Line2D([], [], marker='o', color='black', linestyle='None', markersize=10, label='2023 Fourth of July fireworks events'),
#            plt.Line2D([], [], marker='s', color='red', linestyle='None', markersize=10, markerfacecolor='none', label='2024 fireworks events'),
#            plt.Line2D([], [], marker='^', color='green', linestyle='None', markersize=10, markerfacecolor='none', label='2023 Canadian wildfires')]
#
# # Create custom legend for the entire figure
# fig.legend(handles=handles,
#            loc='lower center', ncol=3, bbox_to_anchor=(0.5, 0), markerscale=2,
#            fancybox=True, shadow=False, prop={'size': 28}, handletextpad=0.5, handlelength=2)
#
# plt.tight_layout()
# plt.subplots_adjust(bottom=0.16)  # Adjust plot to make room for the legend
# plt.show()


periods = [
    (pd.Timestamp('2024-06-29', tz='US/Eastern'), pd.Timestamp('2024-07-02', tz='US/Eastern')),

]
mask = pd.Series(False, index=df.index)
for start, end in periods:
    mask |= (df['TIME'] >= start) & (df['TIME'] <= end)

df = df[mask]
df.set_index('TIME', inplace=True)
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

plt.rcParams['font.family'] = 'Times New Roman'
mpl.rcParams['font.size'] = 26
# Plotting
fig, ax1 = plt.subplots(figsize=(10, 6))

marker_style_k = {'marker': 'o', 'markersize': 8, 'markerfacecolor': 'blue', 'markeredgecolor': 'blue'}
marker_style_ba = {'marker': 's', 'markersize': 8, 'markerfacecolor': 'none', 'markeredgecolor': 'darkgoldenrod'}

ax1.set_xlabel('Time (EST)')
ax1.set_ylabel('K Concentration (ng/m$^3$)', color='blue')
ax1.errorbar(df.index, df['K 19 (ng/m3)'], yerr=df['K Uncert (ng/m3)'], label='K Concentration', fmt='o', color='blue', capsize=3, **marker_style_k)
ax1.tick_params(axis='y', labelcolor='blue')
ax1.tick_params(axis='x')
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
ax1.xaxis.set_major_locator(mdates.DayLocator(interval=1))
ax1.xaxis.set_minor_locator(mdates.DayLocator(interval=1))
ax1.tick_params(axis='x', which='major', length=8)
ax1.tick_params(axis='x', which='minor', length=3)
# ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
# ax2.set_ylabel('Ba Concentration (ng/m3)', color='red',fontsize = 20)  # we already handled the x-label with ax1
# ax2.errorbar(df.index, df['Ba 56 (ng/m3)'], yerr=df['Ba Uncert (ng/m3)'], label='Ba Concentration', fmt='s', color='black', capsize=3, **marker_style_ba)
# ax2.tick_params(axis='y', labelcolor='red',labelsize = 20)
#
# fig.legend(loc='upper left', bbox_to_anchor=(0.15,0.9), fontsize=15)
# fig.tight_layout()  # otherwise the right y-label is slightly clipped
# plt.show()
ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
ax2.set_ylabel('Cu Concentration (ng/m$^3$)', color='darkgoldenrod')  # we already handled the x-label with ax1
ax2.errorbar(df.index, df['Cu 29 (ng/m3)'], yerr=df['Cu Uncert (ng/m3)'], label='Cu Concentration', fmt='s', color='black', capsize=3, **marker_style_ba)
ax2.tick_params(axis='y', labelcolor='darkgoldenrod')

fig.legend(loc='upper left', bbox_to_anchor=(0.15,0.96))
fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()




