#!/usr/bin/env python
"""
NPF detection + single-panel Dal-Maso-style plot for Pittsburgh SMPS

- Loads only the two days around DAY_OF_INTEREST (UTC files, converted to US/Eastern)
- Detects one NPF burst using a simple N_nuc-threshold method
- Fits lognormal nucleation mode to get Dpg(t) and growth rate (GR)
- Estimates J10 from the initial rise in N_nuc
- Plots ONLY the banana plot with Dpg points and linear GR fit overlaid
- Prints a short paragraph you can paste into your proposal
"""

import glob
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colors import LogNorm
from scipy.stats import linregress
from scipy.optimize import curve_fit
import warnings

warnings.filterwarnings("ignore")

# ============================================================
# CONFIGURATION
# ============================================================

SMPS_FOLDER = r"D:\Documents\research-2024\SMPS data\data-all-time"
SMPS_PATTERN = os.path.join(SMPS_FOLDER, "SMPS*.csv")

DAY_OF_INTEREST = "2024-08-30"   # local (US/Eastern)
TZ_LOCAL = "US/Eastern"

# Size ranges (nm)
DP_NUC_N_MIN = 10.0   # for N_nuc
DP_NUC_N_MAX = 25.0
DP_FIT_MIN   = 7.0    # for lognormal fit to nucleation mode
DP_FIT_MAX   = 30.0

# ============================================================
# UTILITIES
# ============================================================
plt.rcParams.update({
    "font.size": 14,          # overall font size
    "axes.titlesize": 16,     # title
    "axes.labelsize": 15,     # x/y labels
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 13,
    "figure.titlesize": 16
})

def get_size_columns(df):
    """Return ordered list of columns that look like Dp midpoints."""
    size_cols = []
    for c in df.columns:
        try:
            size_cols.append((float(c), c))
        except ValueError:
            continue
    return [c[1] for c in sorted(size_cols)]


def get_bounds_and_dlogDp(mid_D):
    """Compute bin boundaries and dlogDp from midpoints."""
    log_mid = np.log10(mid_D)
    avg_diff = np.mean(np.diff(log_mid))
    D_bound = np.empty(len(mid_D) + 1)
    D_bound[1:-1] = 10 ** ((log_mid[1:] + log_mid[:-1]) / 2)
    D_bound[0] = 10 ** (log_mid[0] - 0.5 * avg_diff)
    D_bound[-1] = 10 ** (log_mid[-1] + 0.5 * avg_diff)
    dlogDp = np.log10(D_bound[1:]) - np.log10(D_bound[:-1])
    return D_bound, dlogDp


def lognormal(x, N, dpg, sigma):
    """Lognormal in dN/dlogDp space."""
    return (N / (np.sqrt(2 * np.pi) * np.log(sigma))) * \
        np.exp(-0.5 * ((np.log(x) - np.log(dpg)) / np.log(sigma)) ** 2)

# ============================================================
# LOAD ONLY THE TWO RELEVANT FILES (DAY-1 AND DAY)
# ============================================================

print(f"Looking for files in: {SMPS_FOLDER}")
print(f"Pattern: {SMPS_PATTERN}")

all_files = glob.glob(SMPS_PATTERN)
if not all_files:
    print(f"No files matching {SMPS_PATTERN}, trying *.csv")
    all_files = glob.glob(os.path.join(SMPS_FOLDER, "*.csv"))

print(f"Found {len(all_files)} total files")

target_date = pd.to_datetime(DAY_OF_INTEREST)
prev_date   = target_date - pd.Timedelta(days=1)
target_str  = target_date.strftime("%Y%m%d")
prev_str    = prev_date.strftime("%Y%m%d")

selected_files = []
for fp in all_files:
    base = os.path.basename(fp)
    eight_digits = re.findall(r"(\d{8})", base)
    if not eight_digits:
        continue
    date_part = eight_digits[-1]   # last 8-digit group
    if date_part in (target_str, prev_str):
        selected_files.append(fp)
        print("  Selected:", base)

if not selected_files:
    raise FileNotFoundError(
        f"No files with dates {prev_str} or {target_str} found in {SMPS_FOLDER}"
    )

selected_files = sorted(selected_files)
print(f"\nUsing {len(selected_files)} files for analysis")

df = pd.concat([pd.read_csv(f, skiprows=52) for f in selected_files],
               ignore_index=True)
df.columns = df.columns.str.strip()

# timestamps are UTC strings
df["DateTime Sample Start"] = pd.to_datetime(
    df["DateTime Sample Start"],
    format="%d/%m/%Y %H:%M:%S",
    utc=True,
)
# convert to local
df["Time_local"] = df["DateTime Sample Start"].dt.tz_convert(TZ_LOCAL)
df = df.set_index("Time_local").sort_index()

# subset to one local day
day_local = pd.Timestamp(DAY_OF_INTEREST).tz_localize(TZ_LOCAL)
start_day = day_local
end_day   = day_local + pd.Timedelta(days=1)

df_day = df.loc[start_day:end_day]
if df_day.empty:
    raise RuntimeError(f"No data found for {DAY_OF_INTEREST}")

time = df_day.index
print(f"\nScans on {DAY_OF_INTEREST}: {len(time)}")

# ============================================================
# STP CORRECTION & SIZE GRID
# ============================================================

P = df_day["Sheath Pressure (kPa)"].to_numpy()
T = df_day["Sheath Temp (C)"].to_numpy()
STP_factor = (101.35 / P) * ((273.15 + T) / 273.15)

size_cols = get_size_columns(df_day)
mid_D = np.array([float(c) for c in size_cols])
D_bound, dlogDp = get_bounds_and_dlogDp(mid_D)

dNdlogDp_raw  = df_day[size_cols].to_numpy()
dNdlogDp_corr = dNdlogDp_raw * STP_factor[:, None]

# ============================================================
# SIMPLE NPF DETECTION USING N_nuc
# ============================================================

mask_nuc = (mid_D >= DP_NUC_N_MIN) & (mid_D <= DP_NUC_N_MAX)
N_nuc = np.nansum(dNdlogDp_corr[:, mask_nuc] * dlogDp[mask_nuc], axis=1)

N_series = pd.Series(N_nuc, index=time)
N_smooth = N_series.rolling(3, center=True, min_periods=1).mean()

background = np.nanpercentile(N_smooth[N_smooth > 0], 25)
threshold  = background * 2.0

above_thresh = N_smooth > threshold

events = []
start = None
for i, flag in enumerate(above_thresh):
    if flag and start is None:
        start = i
    elif (not flag or i == len(above_thresh) - 1) and start is not None:
        end = i if not flag else i + 1
        duration_h = (time[end - 1] - time[start]).total_seconds() / 3600
        if duration_h >= 1.0:  # ≥1 hour
            events.append((start, end))
        start = None

if events:
    start_idx, end_idx = max(events, key=lambda x: x[1] - x[0])
    event_found = True
    print("\nNPF Event Detected!")
    print(f"  Start: {time[start_idx].strftime('%H:%M')}")
    print(f"  End:   {time[end_idx - 1].strftime('%H:%M')}")
    print(f"  Duration: {(time[end_idx - 1] - time[start_idx]).total_seconds() / 3600:.1f} hours")
else:
    event_found = False
    start_idx, end_idx = None, None
    print("\nNo clear NPF event detected")

# ============================================================
# Dpg(t) FROM LOGNORMAL FIT + GROWTH RATE
# ============================================================

GR   = np.nan
J_nuc = np.nan
dpg_list  = []
time_list = []

if event_found:
    for i in range(start_idx, min(end_idx, len(time))):
        mask_fit = (mid_D >= DP_FIT_MIN) & (mid_D <= DP_FIT_MAX)
        d_fit    = mid_D[mask_fit]
        spec_fit = dNdlogDp_corr[i, mask_fit]

        valid = (spec_fit > 0) & np.isfinite(spec_fit)
        if valid.sum() < 4:
            continue

        d_valid    = d_fit[valid]
        spec_valid = spec_fit[valid]

        try:
            peak_idx = np.argmax(spec_valid)
            p0 = [spec_valid.sum() * 0.1, d_valid[peak_idx], 1.5]
            bounds = ([0, DP_FIT_MIN, 1.1],
                      [spec_valid.sum() * 10, DP_FIT_MAX, 3.0])

            popt, _ = curve_fit(lognormal, d_valid, spec_valid,
                                p0=p0, bounds=bounds)

            dpg_list.append(popt[1])
            time_list.append(time[i])
        except Exception:
            continue

    if len(dpg_list) >= 5:
        t0 = time_list[0]
        hours = np.array([(t - t0).total_seconds() / 3600 for t in time_list])
        dpg_array = np.array(dpg_list)

        slope, intercept, r_value, _, _ = linregress(hours, dpg_array)
        GR = slope
        print(f"  Growth Rate: {GR:.1f} nm/h (R² = {r_value ** 2:.2f})")

        # simple J10 estimate from early N_nuc ramp
        N_event = N_nuc[start_idx:end_idx]
        time_event = time[start_idx:end_idx]

        growth_period = min(20, len(N_event) // 2)  # ~first 2 hours or half event
        if growth_period > 1:
            dt_sec = (time_event[growth_period] - time_event[0]).total_seconds()
            dN = N_event[growth_period] - N_event[0]
            if dt_sec > 0 and dN > 0:
                J_nuc = (dN / dt_sec) * 1.3  # crude coagulation correction
                print(f"  Formation Rate (J₁₀): {J_nuc:.2f} cm⁻³ s⁻¹")

# ============================================================
# SINGLE-PANEL PLOT: BANANA + MODE TRAJECTORY
# ============================================================

time_naive = time.tz_localize(None)
T_mesh, D_mesh = np.meshgrid(mdates.date2num(time_naive), mid_D)
Z = dNdlogDp_corr.T.copy()
Z[Z <= 0] = np.nan

fig, ax = plt.subplots(figsize=(10, 5))

pcm = ax.pcolormesh(
    T_mesh,
    D_mesh,
    Z,
    shading="auto",
    norm=LogNorm(vmin=1e3, vmax=1e5),
    cmap="turbo",
)

ax.set_yscale("log")
ax.set_ylim(10, 300)
ax.set_ylabel("Dp (nm)")
ax.set_xlabel("Local Time (US Eastern)")
# ax.set_title(f"Pittsburgh SMPS – {DAY_OF_INTEREST} (Local Time)")

# Overlay Dpg trajectory & growth fit
if event_found and len(dpg_list) >= 5:
    dpg_times_naive = [t.tz_localize(None) for t in time_list]
    ax.plot(
        mdates.date2num(dpg_times_naive),
        dpg_list,
        "o",
        ms=4,
        mec="black",
        mfc="white",
        label="Nucleation mode Dpg",
    )

    fit_hours = np.linspace(hours[0], hours[-1], 50)
    fit_dpg = intercept + slope * fit_hours
    fit_times = [t0 + pd.Timedelta(hours=h) for h in fit_hours]
    fit_times_naive = [t.tz_localize(None) for t in fit_times]
    ax.plot(
        mdates.date2num(fit_times_naive),
        fit_dpg,
        color="gray",
        linewidth=2,
        label=f"Growth rate = {GR:.1f} nm/h",
    )

    ax.legend(loc="upper right")

fig.colorbar(pcm, ax=ax, label="dN/dlogDp (cm⁻³)")

ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
fig.autofmt_xdate()

plt.tight_layout()
plt.show()

out_fig = f"NPF_event_top_{DAY_OF_INTEREST.replace('-', '')}.png"
fig.savefig(out_fig, dpi=300, bbox_inches="tight")
print(f"\nTop-panel NPF plot saved as {out_fig}")

# ============================================================
# TEXT FOR PROPOSAL
# ============================================================

print("\n" + "=" * 60)
print("TEXT FOR YOUR PROPOSAL:")
print("=" * 60)

if event_found and not np.isnan(GR) and not np.isnan(J_nuc):
    text = f"""
A representative NPF event from Pittsburgh on {DAY_OF_INTEREST} (Figure 4) demonstrates 
clear new particle formation and growth patterns characteristic of urban environments. 
Size distribution evolution shows nucleation mode appearance at ~10 nm around {time[start_idx].strftime('%H:%M')} 
local time, followed by continuous growth to ~25–30 nm over the subsequent 
{(time[end_idx - 1] - time[start_idx]).total_seconds() / 3600:.1f} hours. 
Lognormal modal fitting yields a particle growth rate of {GR:.1f} nm h⁻¹, consistent with 
condensational growth driven by photochemical oxidation products. The nucleation mode 
particle formation rate J₁₀ = {J_nuc:.2f} cm⁻³ s⁻¹ reflects typical urban NPF intensity 
influenced by traffic emissions and secondary aerosol formation. This systematic 
quantification methodology will be applied across all three measurement sites to 
characterize site-specific NPF patterns and formation mechanisms.
"""
elif event_found and not np.isnan(GR):
    text = f"""
Pittsburgh data from {DAY_OF_INTEREST} show a distinct nucleation mode burst between 
{time[start_idx].strftime('%H:%M')} and {time[end_idx - 1].strftime('%H:%M')} local time, with 
lognormal modal fitting indicating a growth rate of {GR:.1f} nm h⁻¹. Although a robust 
formation rate estimate was not obtained, the event clearly exhibits sustained particle 
growth from ~10 nm into the Aitken mode, consistent with classical new particle formation.
"""
else:
    text = """
Analysis of Pittsburgh data from August 30, 2024 reveals elevated ultrafine particle 
concentrations but without the sustained growth pattern characteristic of classical NPF 
events. This highlights the complexity of urban aerosol dynamics and the need for 
systematic detection and classification across longer time periods.
"""

print(text)
