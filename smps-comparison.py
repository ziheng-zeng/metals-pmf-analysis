import glob, os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib import dates as mdates

### CONFIGURATION ###
# Lawrenceville folders
smps_folder_spring = "D:/Documents/2025/SMPS Comparison/Data/Lawrenceville/April-2025"
smps_folder_fall = "D:/Documents/research-2024/SMPS data/data-all-time"

# Spider folders
spider_folder_spring = "C:/Users/zengz/Box/Jen Lab Data Archive/Bigelow 2025/Spider Data/inverted"
spider_folder_fall = "C:/Users/zengz/Box/Jen Lab Data Archive/Bigelow 2025/Spider Data/inverted"

# Kigali files
kigali_file_dry = "D:/Documents/2025/SMPS Comparison/Data/Kigali/Kigali_SMPS_2024_01_30.1_no_raw_data.txt"
kigali_file_wet = "D:/Documents/2025/SMPS Comparison/Data/Kigali/Kigali_SMPS_2024_03_25.txt"



### HELPER FUNCTION ###
def get_bounds_and_dlogDp(mid_D):
    log_mid = np.log10(mid_D)
    avg_diff = np.mean(np.diff(log_mid))
    D_bound = np.empty(len(mid_D) + 1)
    D_bound[1:-1] = 10 ** ((log_mid[1:] + log_mid[:-1]) / 2)
    D_bound[0] = 10 ** (log_mid[0] - 0.5 * avg_diff)
    D_bound[-1] = 10 ** (log_mid[-1] + 0.5 * avg_diff)
    dlogDp = np.log10(D_bound[1:]) - np.log10(D_bound[:-1])
    return D_bound, dlogDp


def load_smps_data(folder_pattern, start_date, end_date):
    """Load and process SMPS data for a given date range"""
    smps_files = glob.glob(folder_pattern)
    if len(smps_files) == 0:
        print(f"Warning: No files found matching {folder_pattern}")
        return None, None, None, None

    smps_df = pd.concat([pd.read_csv(f, skiprows=52) for f in smps_files])

    # Strip column names
    smps_df.columns = smps_df.columns.str.strip()

    # Convert datetime column from UTC to US/Eastern
    smps_df['DateTime Sample Start'] = pd.to_datetime(
        smps_df['DateTime Sample Start'], format='%d/%m/%Y %H:%M:%S', utc=True)
    smps_df['DateTime Sample Start'] = smps_df['DateTime Sample Start'].dt.tz_convert("US/Eastern")

    # Filter for date range
    smps_df = smps_df[(smps_df['DateTime Sample Start'] >= start_date) &
                      (smps_df['DateTime Sample Start'] <= end_date)]

    if len(smps_df) == 0:
        print(f"Warning: No data in date range {start_date} to {end_date}")
        return None, None, None, None

    # Set datetime as index
    smps_df.set_index('DateTime Sample Start', inplace=True)

    # Extract size distribution data
    mid_D = np.array([float(c) for c in smps_df.columns[41:425]])
    D_bound, dlogDp = get_bounds_and_dlogDp(mid_D)
    dNdlogDp = smps_df.iloc[:, 41:425].values
    time = smps_df.index

    return mid_D, dlogDp, dNdlogDp, time


def load_spider_data(folder_pattern, start_date, end_date):
    """Load and process Spider data for a given date range"""
    spider_files = sorted(glob.glob(folder_pattern), key=os.path.getmtime)
    if len(spider_files) == 0:
        print(f"Warning: No files found matching {folder_pattern}")
        return None, None, None, None

    # Remove last file if needed (your original code did this)
    spider_files = spider_files[:-1] if len(spider_files) > 1 else spider_files

    spider_df = pd.concat([pd.read_csv(f) for f in spider_files])
    spider_df['Start datetime (PC)'] = pd.to_datetime(spider_df['Start datetime (PC)'])

    # Set index and localize
    spider_df.set_index('Start datetime (PC)', inplace=True)
    try:
        spider_df.index = spider_df.index.tz_localize('US/Eastern', ambiguous='infer')
    except:
        spider_df.index = spider_df.index.tz_localize('US/Eastern', ambiguous='NaT', nonexistent='shift_forward')
        spider_df = spider_df[spider_df.index.notna()]

    # Filter to only positive polarity scans (negative ions)
    spider_df = spider_df[spider_df['V1 (V)'] > 0]

    # Filter date range
    spider_df = spider_df.loc[start_date:end_date]

    if len(spider_df) == 0:
        print(f"Warning: No Spider data in date range {start_date} to {end_date}")
        return None, None, None, None

    # Extract size distribution data
    dp_cols = [c for c in spider_df.columns if c.replace('.', '', 1).isdigit()]
    mid_D = np.array([float(c) for c in dp_cols])
    D_bound, dlogDp = get_bounds_and_dlogDp(mid_D)
    dNdlogDp = spider_df[dp_cols].values
    time = spider_df.index

    return mid_D, dlogDp, dNdlogDp, time


def load_kigali_data(file_path, start_date, end_date):
    """Load and process Kigali data for a given date range"""
    df = pd.read_csv(file_path, skiprows=17, encoding='latin-1',on_bad_lines="warn")

    # Process Time
    df['Date'] = df['Date'].astype(str).str.strip()
    df['Start Time'] = df['Start Time'].astype(str).str.strip()
    df['Time'] = pd.to_datetime(df['Date'] + ' ' + df['Start Time'],
                                format='%m/%d/%y %H:%M:%S', errors='coerce')
    df.dropna(subset=['Time'], inplace=True)

    # Reorder columns
    columns = list(df.columns)
    df = df[[columns[-1]] + columns[:-1]]
    df.fillna(0, inplace=True)
    tsdf = df.set_index('Time')

    # Filter date range
    tsdf = tsdf.loc[start_date:end_date]

    if len(tsdf) == 0:
        print(f"Warning: No Kigali data in date range {start_date} to {end_date}")
        return None, None, None, None

    mid_D = np.array([float(x) for x in tsdf.columns[8:200]])
    dNdlogDp = tsdf.iloc[:, 8:200].to_numpy()

    # Compute dlogDp
    D_bound, dlogDp = get_bounds_and_dlogDp(mid_D)
    time = tsdf.index

    return mid_D, dlogDp, dNdlogDp, time


### DEFINE DATE RANGES ###
# Lawrenceville Spring (April 2025)
smps_spring_start = pd.Timestamp("2025-04-01", tz="US/Eastern")
smps_spring_end = pd.Timestamp("2025-04-30", tz="US/Eastern")

# Lawrenceville Fall (October 2024)
smps_fall_start = pd.Timestamp("2024-09-01", tz="US/Eastern")
smps_fall_end = pd.Timestamp("2024-09-30", tz="US/Eastern")

# Spider Spring (April 2025)
spider_spring_start = pd.Timestamp("2025-04-12", tz="US/Eastern")
spider_spring_end = pd.Timestamp("2025-04-17", tz="US/Eastern")

# Spider Fall (September 2025)
spider_fall_start = pd.Timestamp("2025-09-17", tz="US/Eastern")
spider_fall_end = pd.Timestamp("2025-09-22", tz="US/Eastern")

# Kigali Dry Season (January-February 2024)
kigali_dry_start = pd.Timestamp("2024-02-10")
kigali_dry_end = pd.Timestamp("2024-02-20")

# Kigali Wet Season (March-May 2024)
kigali_wet_start = pd.Timestamp("2024-03-10")
kigali_wet_end = pd.Timestamp("2024-03-20")

### LOAD ALL DATASETS ###
print("Loading Lawrenceville Spring...")
smps_spring_pattern = os.path.join(smps_folder_spring, "SMPS*.csv")
mid_D_smps_spring, dlogDp_smps_spring, dNdlogDp_smps_spring, time_smps_spring = \
    load_smps_data(smps_spring_pattern, smps_spring_start, smps_spring_end)

print("Loading Lawrenceville Fall...")
smps_fall_pattern = os.path.join(smps_folder_fall, "SMPS*.csv")
mid_D_smps_fall, dlogDp_smps_fall, dNdlogDp_smps_fall, time_smps_fall = \
    load_smps_data(smps_fall_pattern, smps_fall_start, smps_fall_end)

print("Loading Spider Spring...")
spider_spring_pattern = os.path.join(spider_folder_spring, "SpiderMAGIC_SN289_N_*.txt")
mid_D_spider_spring, dlogDp_spider_spring, dNdlogDp_spider_spring, time_spider_spring = \
    load_spider_data(spider_spring_pattern, spider_spring_start, spider_spring_end)

print("Loading Spider Fall...")
spider_fall_pattern = os.path.join(spider_folder_fall, "SpiderMAGIC_SN666_N_*.txt")
mid_D_spider_fall, dlogDp_spider_fall, dNdlogDp_spider_fall, time_spider_fall = \
    load_spider_data(spider_fall_pattern, spider_fall_start, spider_fall_end)

print("Loading Kigali Dry Season...")
mid_D_kigali_dry, dlogDp_kigali_dry, dNdlogDp_kigali_dry, time_kigali_dry = \
    load_kigali_data(kigali_file_dry, kigali_dry_start, kigali_dry_end)

print("Loading Kigali Wet Season...")
mid_D_kigali_wet, dlogDp_kigali_wet, dNdlogDp_kigali_wet, time_kigali_wet = \
    load_kigali_data(kigali_file_wet, kigali_wet_start, kigali_wet_end)

### COMPUTE AVERAGE SIZE DISTRIBUTIONS ###
datasets = {
    'Lawrenceville Spring': (mid_D_smps_spring, dNdlogDp_smps_spring),
    'Lawrenceville Fall': (mid_D_smps_fall, dNdlogDp_smps_fall),
    'Maine Spring': (mid_D_spider_spring, dNdlogDp_spider_spring),
    'Maine Fall': (mid_D_spider_fall, dNdlogDp_spider_fall),
    'Kigali Dry': (mid_D_kigali_dry, dNdlogDp_kigali_dry),
    'Kigali Wet': (mid_D_kigali_wet, dNdlogDp_kigali_wet)
}

### PLOT AVERAGE SIZE DISTRIBUTIONS ###
plt.figure(figsize=(8, 4))

colors = ['blue', 'cyan', 'orange', 'red', 'green', 'lime']
linestyles = ['-', '--', '-', '--', '-', '--']

for i, (name, (mid_D, dNdlogDp)) in enumerate(datasets.items()):
    if mid_D is not None and dNdlogDp is not None:
        avg_dNdlogDp = np.nanmean(dNdlogDp, axis=0)

        # 👉 For Lawrenceville only: don’t plot bins whose avg is exactly 0
        if 'Lawrenceville' in name:
            avg_dNdlogDp[avg_dNdlogDp == 0] = np.nan

        # Mask artificial values for Kigali data
        if 'Kigali' in name:
            mask = mid_D < 0
            avg_dNdlogDp[mask] = np.nan

        plt.plot(mid_D, avg_dNdlogDp,
                 label=name,
                 color=colors[i],
                 linestyle=linestyles[i],
                 linewidth=2,
                 alpha=0.8)
        print(f"{name}: {len(mid_D)} bins, avg concentration: {np.nanmean(avg_dNdlogDp):.1f} cm⁻³")
    else:
        print(f"{name}: No data available")

plt.xscale('log')
plt.xlim(10, 400)
# plt.yscale('log')  # Optional: use log scale for y-axis too
plt.xlabel("Dp [nm]", fontsize=12)
plt.ylabel("Avg dN/dlogDp (cm⁻³)", fontsize=12)
plt.title("Average Size Distribution - Seasonal Comparison", fontsize=14)
plt.legend(loc='best', fontsize=10)
plt.grid(True, alpha=0.3, which='both')
plt.tight_layout()
plt.show()


### OPTIONAL: BANANA PLOTS FOR EACH DATASET ###
def plot_banana(time, dp, dNdlogDp, title):
    if time is None or len(time) == 0:
        print(f"Skipping {title} - no data available")
        return

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
    plt.colorbar(pcm, ax=ax, label='dN/dlogDp [cm⁻³]')
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    plt.tight_layout()
    plt.show()

# Uncomment to generate banana plots
# plot_banana(time_smps_spring, mid_D_smps_spring, dNdlogDp_smps_spring, "Lawrenceville Spring")
# plot_banana(time_smps_fall, mid_D_smps_fall, dNdlogDp_smps_fall, "Lawrenceville Fall")
# plot_banana(time_spider_spring, mid_D_spider_spring, dNdlogDp_spider_spring, "Spider Spring")
# plot_banana(time_spider_fall, mid_D_spider_fall, dNdlogDp_spider_fall, "Spider Fall")
# plot_banana(time_kigali_dry, mid_D_kigali_dry, dNdlogDp_kigali_dry, "Kigali Dry")
# plot_banana(time_kigali_wet, mid_D_kigali_wet, dNdlogDp_kigali_wet, "Kigali Wet")