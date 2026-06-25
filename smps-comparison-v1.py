import glob, os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib import dates as mdates

### CONFIGURATION ###
smps_folder = "D:/Documents/2025/SMPS Comparison/Data/Lawrenceville/April-2025"
spider_folder = "C:/Users/zengz/Box/Jen Lab Data Archive/Bigelow 2025/Spider Data/inverted"
smps_pattern = os.path.join(smps_folder, "SMPS*.csv")
spider_pattern = os.path.join(spider_folder, "SpiderMAGIC_SN289_N_*.txt")


### HELPER FUNCTION ###
def get_bounds_and_dlogDp(mid_D):
    log_mid = np.log10(mid_D)
    avg_diff = np.mean(np.diff(log_mid))
    D_bound = np.empty(len(mid_D) + 1)
    D_bound[1:-1] = 10**((log_mid[1:] + log_mid[:-1]) / 2)
    D_bound[0] = 10**(log_mid[0] - 0.5 * avg_diff)
    D_bound[-1] = 10**(log_mid[-1] + 0.5 * avg_diff)
    dlogDp = np.log10(D_bound[1:]) - np.log10(D_bound[:-1])
    return D_bound, dlogDp

### LOAD SMPS DATA ###
smps_files = glob.glob(smps_pattern)
smps_df = pd.concat([pd.read_csv(f, skiprows=52) for f in smps_files])

# Strip column names to remove any accidental leading/trailing spaces
smps_df.columns = smps_df.columns.str.strip()

# Convert datetime column from UTC to US/Eastern
smps_df['DateTime Sample Start'] = pd.to_datetime(
    smps_df['DateTime Sample Start'], format='%d/%m/%Y %H:%M:%S', utc=True)
smps_df['DateTime Sample Start'] = smps_df['DateTime Sample Start'].dt.tz_convert("US/Eastern")

# Filter for LAWRENCEVILLE date range
start_date_smps = pd.Timestamp("2025-04-15", tz="US/Eastern")
end_date_smps = pd.Timestamp("2025-04-25", tz="US/Eastern")
smps_df = smps_df[(smps_df['DateTime Sample Start'] >= start_date_smps) &
                  (smps_df['DateTime Sample Start'] <= end_date_smps)]

# Set datetime as index AFTER filtering
smps_df.set_index('DateTime Sample Start', inplace=True)

# --- STP Correction Factor ---
# P in kPa, T in °C
P = smps_df['Sheath Pressure (kPa)'].values
T = smps_df['Sheath Temp (C)'].values

STP_factor = (101.35 / P) * ((273.15 + T) / 273.15)

mid_D_smps = np.array([float(c) for c in smps_df.columns[41:425]])
# print(mid_D_smps)
D_bound_smps, dlogDp_smps = get_bounds_and_dlogDp(mid_D_smps)
dNdlogDp_smps = smps_df.iloc[:, 41:425].values
# Apply STP correction per scan (broadcast across bins)
dNdlogDp_smps = dNdlogDp_smps * STP_factor[:, None]

time_smps = smps_df.index
# # Confirm shapes
# print("dNdlogDp shape:", dNdlogDp_smps.shape)
# print("mid_D shape:", mid_D_smps.shape)

# Make sure mid_D is 1D and aligned correctly
mid_D_smps = mid_D_smps.flatten()
assert dNdlogDp_smps.shape[1] == mid_D_smps.shape[0], "Mismatch in bin count"

# Total number concentration
dlogDp_smps = np.log10(D_bound_smps[1:]) - np.log10(D_bound_smps[:-1])
# Number in each bin
dN = dNdlogDp_smps * dlogDp_smps[np.newaxis, :]  # shape (3427, 384)
# Total number per scan
N_smps = np.nansum(dN, axis=1)
# Compute number-weighted mean diameter properly
Dmean_smps = np.nansum(dN * mid_D_smps[np.newaxis, :], axis=1) / N_smps


# Sanity check
# print("Dmean range (nm):", np.nanmin(Dmean_smps), "to", np.nanmax(Dmean_smps))

### LOAD SPIDER DATA ###
spider_files = sorted(glob.glob(spider_pattern), key=os.path.getmtime)[:-1]
spider_df = pd.concat([pd.read_csv(f) for f in spider_files])
spider_df['Start datetime (PC)'] = pd.to_datetime(spider_df['Start datetime (PC)'])

# Ensure datetime column is parsed and localized
spider_df['Start datetime (PC)'] = pd.to_datetime(spider_df['Start datetime (PC)'])
spider_df.set_index('Start datetime (PC)', inplace=True)
spider_df.index = spider_df.index.tz_localize('US/Eastern')  # Only do this if not already localized

# Filter to only positive polarity scans (negative ions)
spider_df = spider_df[spider_df['V1 (V)'] > 0]

# Filter for BIGELOW date range
start_date_spider = pd.Timestamp("2025-04-12", tz="US/Eastern")  # Your desired range
end_date_spider = pd.Timestamp("2025-04-17", tz="US/Eastern")    # Your desired range
spider_df = spider_df.loc[start_date_spider:end_date_spider]


dp_cols_spider = [c for c in spider_df.columns if c.replace('.', '', 1).isdigit()]
mid_D_spider = np.array([float(c) for c in dp_cols_spider])
D_bound_spider, dlogDp_spider = get_bounds_and_dlogDp(mid_D_spider)
dNdlogDp_spider = spider_df[dp_cols_spider].values
time_spider = spider_df.index

# Calculate D_mean_spider
# Calculate Dp bin edges and dlogDp
log_mid = np.log10(mid_D_spider)
avg_diff = np.mean(np.diff(log_mid))
D_bound_spider = np.empty(len(mid_D_spider) + 1)
D_bound_spider[1:-1] = 10 ** ((log_mid[1:] + log_mid[:-1]) / 2)
D_bound_spider[0] = 10 ** (log_mid[0] - 0.5 * avg_diff)
D_bound_spider[-1] = 10 ** (log_mid[-1] + 0.5 * avg_diff)
dlogDp_spider = np.log10(D_bound_spider[1:]) - np.log10(D_bound_spider[:-1])
# Number per bin
dN_spider = dNdlogDp_spider * dlogDp_spider[np.newaxis, :]
# Total number per scan
N_spider = np.nansum(dN_spider, axis=1)
# Number-weighted mean diameter
Dmean_spider = np.nansum(dN_spider * mid_D_spider[np.newaxis, :], axis=1) / N_spider

### Load Kigali Data ###
file_path_kigali = "D:/Documents/2025/SMPS Comparison/Data/Kigali/Kigali_SMPS_2024_01_30.1_no_raw_data.txt"
df_kigali = pd.read_csv(file_path_kigali, skiprows=17, encoding='latin-1')

# --- Process Time ---
df_kigali['Date'] = df_kigali['Date'].astype(str).str.strip()
df_kigali['Start Time'] = df_kigali['Start Time'].astype(str).str.strip()
df_kigali['Time'] = pd.to_datetime(df_kigali['Date'] + ' ' + df_kigali['Start Time'], format='%m/%d/%y %H:%M:%S', errors='coerce')
df_kigali.dropna(subset=['Time'], inplace=True)

# --- Reorder columns to put time first ---
columns = list(df_kigali.columns)
df_kigali = df_kigali[[columns[-1]] + columns[:-1]]
df_kigali.fillna(0, inplace=True)
tsdf_kigali = df_kigali.set_index('Time')

# Filter Kigali data to February 10–17, 2024
start_kigali = pd.Timestamp("2024-02-10")
end_kigali = pd.Timestamp("2024-02-17")
tsdf_kigali = tsdf_kigali.loc[start_kigali:end_kigali]

# Ensure index is datetime
tsdf_kigali.index = pd.to_datetime(tsdf_kigali.index)

# # Exclude specific dates
# tsdf_kigali = tsdf_kigali[~tsdf_kigali.index.normalize().isin([
#     pd.Timestamp("2024-02-08"),
#     pd.Timestamp("2024-02-17")
# ])]
#
# # Then re-extract the time series and dNdlogDp from this filtered DataFrame
# time_kigali = tsdf_kigali.index

# --- Extract midpoints and dNdlogDp ---
mid_D_kigali = np.array([float(x) for x in tsdf_kigali.columns[8:200]])
dNdlogDp_kigali = tsdf_kigali.iloc[:, 8:200].to_numpy()

# --- Compute dlogDp ---
log_mid = np.log10(mid_D_kigali)
avg_diff = np.mean(np.diff(log_mid))
D_bound_kigali = np.empty(len(mid_D_kigali) + 1)
D_bound_kigali[1:-1] = 10 ** ((log_mid[1:] + log_mid[:-1]) / 2)
D_bound_kigali[0] = 10 ** (log_mid[0] - 0.5 * avg_diff)
D_bound_kigali[-1] = 10 ** (log_mid[-1] + 0.5 * avg_diff)
dlogDp_kigali = np.log10(D_bound_kigali[1:]) - np.log10(D_bound_kigali[:-1])

# --- Compute dN, N, and Dmean ---
dN_kigali = dNdlogDp_kigali * dlogDp_kigali[np.newaxis, :]
N_kigali = np.nansum(dN_kigali, axis=1)
Dmean_kigali = np.nansum(dN_kigali * mid_D_kigali[np.newaxis, :], axis=1) / N_kigali
time_kigali = tsdf_kigali.index


### 1. DIURNAL AVERAGE BANANA PLOTS ###
def prepare_diurnal_average(time, dp, dNdlogDp):
    """
    Calculate diurnal average (average by hour of day across all days)
    """
    # Create DataFrame with hour of day
    df = pd.DataFrame(dNdlogDp, index=time, columns=dp)
    df['hour'] = df.index.hour

    # Group by hour and take mean
    diurnal_avg = df.groupby('hour').mean()

    return diurnal_avg.index.values, dp, diurnal_avg.values


# Prepare diurnal averages for each site
hours_smps, dp_smps, diurnal_smps = prepare_diurnal_average(time_smps, mid_D_smps, dNdlogDp_smps)
hours_spider, dp_spider, diurnal_spider = prepare_diurnal_average(time_spider, mid_D_spider, dNdlogDp_spider)
hours_kigali, dp_kigali, diurnal_kigali = prepare_diurnal_average(time_kigali, mid_D_kigali, dNdlogDp_kigali)

# Create figure with 3 subplots sharing x-axis
fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

# Plot Pittsburgh (Lawrenceville)
H, D = np.meshgrid(hours_smps, dp_smps)
Z = diurnal_smps.T
Z[Z <= 0] = np.nan
pcm1 = axes[0].pcolormesh(H, D, Z, shading='auto', norm=LogNorm(vmin=1e3, vmax=1e5), cmap='turbo')
axes[0].set_yscale('log')
axes[0].set_ylim(10, 1000)
axes[0].set_ylabel("Dp (nm)")
axes[0].set_title("Pittsburgh")
plt.colorbar(pcm1, ax=axes[0], label='dN/dlogDp (cm⁻³)')

# Plot Maine (Bigelow)
H, D = np.meshgrid(hours_spider, dp_spider)
Z = diurnal_spider.T
Z[Z <= 0] = np.nan
pcm2 = axes[1].pcolormesh(H, D, Z, shading='auto', norm=LogNorm(vmin=1e3, vmax=1e5), cmap='turbo')
axes[1].set_yscale('log')
axes[1].set_ylim(10, 1000)
axes[1].set_ylabel("Dp (nm)")
axes[1].set_title("Maine")
plt.colorbar(pcm2, ax=axes[1], label='dN/dlogDp (cm⁻³)')

# Plot Kigali
H, D = np.meshgrid(hours_kigali, dp_kigali)
Z = diurnal_kigali.T
Z[Z <= 0] = np.nan
pcm3 = axes[2].pcolormesh(H, D, Z, shading='auto', norm=LogNorm(vmin=1e3, vmax=1e5), cmap='turbo')
axes[2].set_yscale('log')
axes[2].set_ylim(10, 1000)
axes[2].set_ylabel("Dp (nm)")
axes[2].set_xlabel("Hour of Day")
axes[2].set_title("Kigali, Rwanda")
plt.colorbar(pcm3, ax=axes[2], label='dN/dlogDp (cm⁻³)')

# Set x-axis to show 0-24 hours
for ax in axes:
    ax.set_xlim(0, 23)
    ax.set_xticks(range(0, 24, 3))

plt.tight_layout()
plt.show()

### 1. BANANA PLOTS ###
def plot_banana(time, dp, dNdlogDp, title):
    T, D = np.meshgrid(mdates.date2num(time), dp)
    Z = dNdlogDp.T
    Z[Z <= 0] = np.nan
    fig, ax = plt.subplots(figsize=(10, 5))
    pcm = ax.pcolormesh(T, D, Z, shading='auto', norm=LogNorm(vmin=1e3, vmax=1e5), cmap='turbo')
    ax.set_yscale('log')
    ax.set_ylim(10, 1000)
    ax.set_ylabel("Dp [nm]")
    ax.set_xlabel("Time")
    ax.set_title(title)
    # plt.colorbar(pcm, ax=ax, label='dN/dlogDp [cm⁻³]')
    # Set major ticks every 6 hours
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=12))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%y-%m-%d %H:%M'))

    plt.tight_layout()
    plt.show()

plot_banana(time_smps, mid_D_smps, dNdlogDp_smps, "Lawrenceville")
# plot_banana(time_spider, mid_D_spider, dNdlogDp_spider, "Bigelow")
# plot_banana(time_kigali, mid_D_kigali, dNdlogDp_kigali, "Kigali")

# ### 2. TOTAL NUMBER CONCENTRATION ###
# N_smps = np.nansum(dNdlogDp_smps * dlogDp_smps, axis=1)
# N_spider = np.nansum(dNdlogDp_spider * dlogDp_spider, axis=1)
#
# plt.figure(figsize=(14,5))
# plt.plot(time_smps, N_smps, label='SMPS', alpha=0.7)
# plt.plot(time_spider, N_spider, label='Spider', alpha=0.7)
# plt.ylabel("Total N (cm⁻³)")
# plt.xlabel("Time")
# plt.legend()
# plt.title("Total Number Concentration")
# plt.tight_layout()
# # plt.show()

# ### 3. MEAN DIAMETER ###
# plt.figure(figsize=(14,5))
# plt.plot(time_smps, Dmean_smps, label='SMPS', alpha=0.7)
# plt.plot(time_spider, Dmean_spider, label='Spider', alpha=0.7)
# plt.plot(time_kigali, Dmean_kigali, label='Kigali', alpha=0.7)
# plt.ylabel("Mean Diameter (nm)")
# plt.xlabel("Time")
# plt.legend()
# plt.title("Mean Particle Diameter Over Time")
# plt.tight_layout()
# # plt.show()

## 4. AVERAGE SIZE DISTRIBUTIONS WITH GMD AND GSD ###
avg_dNdlogDp_smps = np.nanmean(dNdlogDp_smps, axis=0)
avg_dNdlogDp_spider = np.nanmean(dNdlogDp_spider, axis=0)
avg_dNdlogDp_kigali = np.nanmean(dNdlogDp_kigali, axis=0)

# Convert zeros to NaN for cleaner plotting
avg_dNdlogDp_smps[avg_dNdlogDp_smps == 0] = np.nan
avg_dNdlogDp_spider[avg_dNdlogDp_spider == 0] = np.nan
avg_dNdlogDp_kigali[avg_dNdlogDp_kigali == 0] = np.nan


def calculate_gmd_gsd(mid_D, dNdlogDp, dlogDp):
    """
    Calculate Geometric Mean Diameter (GMD) and Geometric Standard Deviation (GSD)
    from size distribution data.

    GMD = exp(sum(ln(Dp) * N * dlogDp) / sum(N * dlogDp))
    GSD = exp(sqrt(sum((ln(Dp) - ln(GMD))^2 * N * dlogDp) / sum(N * dlogDp)))
    """
    # Remove NaN values
    mask = ~np.isnan(dNdlogDp)
    mid_D_clean = mid_D[mask]
    dNdlogDp_clean = dNdlogDp[mask]
    dlogDp_clean = dlogDp[mask]

    # Calculate dN (number in each bin)
    dN = dNdlogDp_clean * dlogDp_clean

    # Total number
    N_total = np.sum(dN)

    # GMD (count-weighted geometric mean)
    ln_Dp = np.log(mid_D_clean)
    GMD = np.exp(np.sum(ln_Dp * dN) / N_total)

    # GSD (geometric standard deviation)
    variance = np.sum(((ln_Dp - np.log(GMD)) ** 2) * dN) / N_total
    GSD = np.exp(np.sqrt(variance))

    return GMD, GSD


# Calculate GMD and GSD for each site
GMD_smps, GSD_smps = calculate_gmd_gsd(mid_D_smps, avg_dNdlogDp_smps, dlogDp_smps)
GMD_spider, GSD_spider = calculate_gmd_gsd(mid_D_spider, avg_dNdlogDp_spider, dlogDp_spider)
GMD_kigali, GSD_kigali = calculate_gmd_gsd(mid_D_kigali, avg_dNdlogDp_kigali, dlogDp_kigali)

# Calculate total number concentrations
# Calculate total number concentrations (use nansum to handle NaN values)
N_total_smps = np.nansum(avg_dNdlogDp_smps * dlogDp_smps)
N_total_spider = np.nansum(avg_dNdlogDp_spider * dlogDp_spider)
N_total_kigali = np.nansum(avg_dNdlogDp_kigali * dlogDp_kigali)

# Print results
print(f"\nPittsburgh:")
print(f"  GMD = {GMD_smps:.1f} nm")
print(f"  GSD = {GSD_smps:.2f}")
print(f"  Total N = {N_total_smps:.1f} cm⁻³")

print(f"\nMaine:")
print(f"  GMD = {GMD_spider:.1f} nm")
print(f"  GSD = {GSD_spider:.2f}")
print(f"  Total N = {N_total_spider:.1f} cm⁻³")

print(f"\nKigali:")
print(f"  GMD = {GMD_kigali:.1f} nm")
print(f"  GSD = {GSD_kigali:.2f}")
print(f"  Total N = {N_total_kigali:.1f} cm⁻³")

# Plot with statistics in legend
plt.figure(figsize=(5, 3))
plt.plot(mid_D_smps, avg_dNdlogDp_smps,
         label=f'Pittsburgh (GMD={GMD_smps:.0f}nm, GSD={GSD_smps:.2f})')
plt.plot(mid_D_spider, avg_dNdlogDp_spider,
         label=f'Maine (GMD={GMD_spider:.0f}nm, GSD={GSD_spider:.2f})')
plt.plot(mid_D_kigali, avg_dNdlogDp_kigali,
         label=f'Kigali (GMD={GMD_kigali:.0f}nm, GSD={GSD_kigali:.2f})')
plt.xscale('log')
plt.xlim(10, 300)
plt.xlabel("Dp (nm)")
plt.ylabel("Avg dN/dlogDp (cm⁻³)")
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.35), ncol=1, fontsize=9)
plt.tight_layout()
plt.show()