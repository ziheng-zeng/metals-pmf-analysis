import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pytz

# Load the Xact data
df = pd.read_csv('Xact_EST_May2023_July2025_combined.csv')
df['TIME'] = pd.to_datetime(df['TIME'], utc=True).dt.tz_convert('US/Eastern')
df.set_index('TIME', inplace=True)

# Define Eastern timezone
eastern = pytz.timezone('US/Eastern')

# Define event periods with simplified categorization
fireworks_periods = [
    ("Fireworks 1", "Fireworks", pd.Timestamp('2023-06-30 17:00', tz=eastern),
     pd.Timestamp('2023-07-01 12:00', tz=eastern)),
    ("Fireworks 2", "Fireworks", pd.Timestamp('2023-07-01 17:00', tz=eastern),
     pd.Timestamp('2023-07-02 12:00', tz=eastern)),
    ("Fireworks 3", "Fireworks", pd.Timestamp('2023-07-03 17:00', tz=eastern),
     pd.Timestamp('2023-07-04 16:00', tz=eastern)),
    ("Fireworks 4", "Fireworks", pd.Timestamp('2023-07-04 17:00', tz=eastern),
     pd.Timestamp('2023-07-05 12:00', tz=eastern)),
    ("Fireworks 5", "Fireworks", pd.Timestamp('2023-07-05 17:00', tz=eastern),
     pd.Timestamp('2023-07-06 12:00', tz=eastern)),
    ("Fireworks 6", "Fireworks", pd.Timestamp('2024-06-29 20:00', tz=eastern),
     pd.Timestamp('2024-06-29 23:59', tz=eastern)),
    ("Fireworks 7", "Fireworks", pd.Timestamp('2025-06-28 20:00', tz=eastern),
     pd.Timestamp('2025-06-29 3:00', tz=eastern)),
    ("Fireworks 8", "Fireworks", pd.Timestamp('2025-07-03 17:00', tz=eastern),
     pd.Timestamp('2025-07-04 12:00', tz=eastern)),
    ("Fireworks 9", "Fireworks", pd.Timestamp('2025-07-04 17:00', tz=eastern),
     pd.Timestamp('2025-07-05 16:00', tz=eastern)),
    ("Fireworks 10", "Fireworks", pd.Timestamp('2025-07-05 17:00', tz=eastern),
     pd.Timestamp('2025-07-06 12:00', tz=eastern))
]

wildfire_periods = [
    ("Wildfire 1", "Wildfire", pd.Timestamp('2023-06-05 12:00', tz=eastern),
     pd.Timestamp('2023-06-08', tz=eastern)),
    ("Wildfire 2", "Wildfire", pd.Timestamp('2023-06-28', tz=eastern),
     pd.Timestamp('2023-06-30', tz=eastern)),
    ("Wildfire 3", "Wildfire", pd.Timestamp('2023-07-16', tz=eastern),
     pd.Timestamp('2023-07-18', tz=eastern))
]

# --- New: Coke Plume periods (Eastern time) ---
coke_plume_periods = []
_coke_ranges = [
    ("2024-10-22 23:00", "2024-10-23 09:00"),
    ("2025-03-18 23:00", "2025-03-19 11:00"),
    ("2025-03-11 03:00", "2025-03-11 09:00"),
    ("2024-09-08 21:00", "2024-09-09 09:00"),
    # ("2024-10-04 16:00", "2024-10-05 09:00"),
    # ("2023-09-06 03:00", "2023-09-06 10:00"),
]
for i, (s, e) in enumerate(_coke_ranges, start=1):
    coke_plume_periods.append((
        f"Coke Plume {i}", "Coke Plume",
        pd.Timestamp(s, tz=eastern),
        pd.Timestamp(e, tz=eastern)
    ))

# Seasonal background periods
background_periods = [
    ("Winter", "Winter", pd.Timestamp('2023-12-15 00:00', tz=eastern),
     pd.Timestamp('2023-12-22 23:59', tz=eastern)),
    ("Spring", "Spring", pd.Timestamp('2024-04-15 00:00', tz=eastern),
     pd.Timestamp('2024-04-22 23:59', tz=eastern)),
    ("Summer", "Summer", pd.Timestamp('2024-07-15 00:00', tz=eastern),
     pd.Timestamp('2024-07-22 23:59', tz=eastern)),
    ("Fall", "Fall", pd.Timestamp('2024-09-13 00:00', tz=eastern),
     pd.Timestamp('2024-09-20 23:59', tz=eastern))
]

# Identify concentration/uncertainty column pairs
headers = df.columns.tolist()
concentration_cols = [col for col in headers if " (ng/m3)" in col and "uncert" not in col.lower()]
uncertainty_cols = [col for col in headers if "uncert" in col.lower()]

metal_uncert_pairs = {}
for conc_col in concentration_cols:
    element = conc_col.split()[0]
    uncert_col = next((u_col for u_col in uncertainty_cols if element in u_col), None)
    if uncert_col:
        metal_uncert_pairs[conc_col] = uncert_col

print(f"Found {len(metal_uncert_pairs)} metals with uncertainty pairs")

# Calculate background concentrations from all background periods
all_background_data = []
for label, category, start, end in background_periods:
    df_bg = df[(df.index >= start) & (df.index <= end)]
    all_background_data.append(df_bg)

df_all_background = pd.concat(all_background_data)

background_concentrations = {}
for conc_col, uncert_col in metal_uncert_pairs.items():
    if conc_col in df_all_background.columns and uncert_col in df_all_background.columns:
        valid = df_all_background[conc_col] > 3 * df_all_background[uncert_col]
        mean_bg = df_all_background.loc[valid, conc_col].mean()
        if pd.notna(mean_bg):
            metal_name = conc_col.split(" (ng/m3)")[0].strip()
            background_concentrations[metal_name] = mean_bg

print(f"Calculated background concentrations for {len(background_concentrations)} metals")


def get_event_profile(df_event, top_n=10, use_raw=False):
    """Get metal profile for an event period with top N metals"""
    event_metals = {}

    for conc_col, uncert_col in metal_uncert_pairs.items():
        if conc_col in df_event.columns and uncert_col in df_event.columns:
            valid = df_event[conc_col] > 3 * df_event[uncert_col]

            if valid.any():
                mean_val = df_event.loc[valid, conc_col].mean()

                if pd.notna(mean_val) and mean_val > 0:
                    metal_name = conc_col.split(" (ng/m3)")[0].strip()
                    background_val = background_concentrations.get(metal_name, 0)

                    event_metals[metal_name] = {
                        'raw_concentration': mean_val,
                        'background_concentration': background_val,
                        'corrected_concentration': max(0, mean_val - background_val)
                    }

    # Sort by raw or corrected concentration and take top N
    sort_key = 'raw_concentration' if use_raw else 'corrected_concentration'
    sorted_metals = sorted(event_metals.items(),
                           key=lambda x: x[1][sort_key],
                           reverse=True)[:top_n]

    return dict(sorted_metals)


# Process all events
all_events = fireworks_periods + wildfire_periods + coke_plume_periods + background_periods
event_data = []

print("\nProcessing events...")
for label, category, start, end in all_events:
    print(f"Processing: {label}")

    df_event = df[(df.index >= start) & (df.index <= end)]

    if len(df_event) == 0:
        print(f"  Warning: No data found for {label}")
        continue

    # Use raw concentrations for background periods, corrected for others
    use_raw = category in ['Winter', 'Spring', 'Summer', 'Fall']
    profile = get_event_profile(df_event, top_n=10, use_raw=use_raw)

    # Calculate totals
    if use_raw:
        total_conc = sum([data['raw_concentration'] for data in profile.values()])
    else:
        total_conc = sum([data['corrected_concentration'] for data in profile.values()])

    # Create event record
    event_record = {
        'Event': label,
        'Category': category,
        'Start': start,
        'End': end,
        'Total_concentration': total_conc
    }

    # Add individual metal data
    for metal, data in profile.items():
        if use_raw:
            event_record[f'{metal}_conc'] = data['raw_concentration']
            event_record[f'{metal}_pct'] = (
                    data['raw_concentration'] / total_conc * 100) if total_conc > 0 else 0
        else:
            event_record[f'{metal}_conc'] = data['corrected_concentration']
            event_record[f'{metal}_pct'] = (
                    data['corrected_concentration'] / total_conc * 100) if total_conc > 0 else 0

    event_data.append(event_record)

# Create DataFrame
df_profiles = pd.DataFrame(event_data)
print(f"\nCreated profiles for {len(df_profiles)} events")

# Paul Tol's colorblind-friendly palette
COLORBLIND_COLORS = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
    '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5'
]

import matplotlib.cm as cm
import matplotlib.colors as mcolors
import seaborn as sns

def make_color_map(metals, scheme="husl"):
    """
    Return {metal -> color} for an arbitrary-length metal list.
    Good schemes: 'husl' (seaborn), 'tab20'/'tab20b' (matplotlib), 'Spectral', 'viridis'.
    """
    metals = list(metals)
    n = len(metals)

    if scheme.lower() in {"husl", "hls"}:
        palette = sns.color_palette("husl", n)         # many distinct hues
        colors = [mcolors.to_hex(c) for c in palette]
    else:
        cmap = cm.get_cmap(scheme, n)                   # matplotlib colormap
        colors = [mcolors.to_hex(cmap(i)) for i in range(n)]

    return {m: colors[i] for i, m in enumerate(metals)}

def _stacked_bars_per_event_all(ax, df_cat, title,
                                palette=COLORBLIND_COLORS,
                                drop_below_pct=None,
                                metal_order=None):
    """
    Build stacked bars for ALL metals per event (no top-10).
    - Re-normalizes each event to 100%.
    - If metal_order is provided, uses it; otherwise orders metals by total across events (DESC),
      which makes the largest components draw first (sit at the bottom).
    - drop_below_pct: hide metals whose max contribution across events is below this % (e.g., 0.05).
    """
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')

    if df_cat.empty:
        return

    df_cat = df_cat.set_index('Event')
    all_pct_cols = [c for c in df_cat.columns if c.endswith('_pct')]
    for col in all_pct_cols:
        df_cat[col] = pd.to_numeric(df_cat[col], errors='coerce')

    plot_data = df_cat[all_pct_cols].copy().fillna(0.0)

    # Normalize each event to 100%
    row_sums = plot_data.sum(axis=1).replace(0, np.nan)
    plot_data = plot_data.div(row_sums, axis=0).multiply(100.0).fillna(0.0)

    # Optionally drop globally tiny metals
    if drop_below_pct is not None:
        keep = (plot_data.max(axis=0) >= drop_below_pct)
        plot_data = plot_data.loc[:, keep]

    # Decide order: provided order or by total across events (DESC -> largest first -> bottom)
    if metal_order is None:
        order = plot_data.sum(axis=0).sort_values(ascending=False).index.tolist()
    else:
        # keep only metals present; preserve provided order
        order = [m for m in metal_order if m in plot_data.columns]

    bottom = np.zeros(len(plot_data))
    for i, col in enumerate(order):
        metal_name = col.replace('_pct', '')
        vals = plot_data[col].values
        ax.bar(range(len(plot_data)), vals, bottom=bottom,
               label=metal_name, color=palette[i % len(palette)],
               alpha=0.85, edgecolor='white', linewidth=0.8, width=0.8)
        bottom += vals

    ax.set_xticks(range(len(plot_data)))
    ax.set_xticklabels(plot_data.index, rotation=45, ha='right', fontsize=10)
    ax.set_ylabel('Percentage of Total Corrected Metal (%)', fontsize=12, fontweight='bold')



def _mean_all(df_cat):
    """
    Return a Series of mean %-profile over ALL metals for a category.
    """
    if df_cat.empty:
        return None
    df_cat = df_cat.copy()
    all_pct_cols = [c for c in df_cat.columns if c.endswith('_pct')]
    for col in all_pct_cols:
        df_cat[col] = pd.to_numeric(df_cat[col], errors='coerce')

    # Mean over events, then normalize to 100%
    m = df_cat[all_pct_cols].mean(skipna=True)
    if m.isna().all():
        return None
    m = m.fillna(0.0)
    s = m.sum()
    if s > 0:
        m = 100.0 * m / s
    return m

def create_separate_panel_plots(df_profiles, save_path=None):
    """
    Four panels: Fireworks | Wildfire | Coke Plumes | Mean comparison (ALL metals).
    - Event panels: each bar is one event, 100% stack across all metals (largest components at bottom).
    - Mean panel: 3 stacked bars (Fireworks/Wildfire/Coke) using a common union of metals and the same order.
    """
    fw_df = df_profiles[df_profiles['Category'] == 'Fireworks'].copy()
    wf_df = df_profiles[df_profiles['Category'] == 'Wildfire' ].copy()
    ck_df = df_profiles[df_profiles['Category'] == 'Coke Plume'].copy()

    fig, axes = plt.subplots(1, 4, figsize=(32, 8))
    ax1, ax2, ax3, ax4 = axes

    # ---------- Build a global metal order from all three categories ----------
    def _collect_pct_cols(d):
        cols = [c for c in d.columns if c.endswith('_pct')]
        d = d.copy()
        for c in cols:
            d[c] = pd.to_numeric(d[c], errors='coerce')
        return d[cols].fillna(0.0)

    parts = []
    for d in (fw_df, wf_df, ck_df):
        if not d.empty:
            parts.append(_collect_pct_cols(d))
    union_df = pd.concat(parts, axis=0) if parts else None

    if union_df is not None and not union_df.empty:
        # Normalize each event row to 100% to get fair totals, then sum by metal
        norm_union = union_df.div(union_df.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
        # DESC so the largest components draw first -> bottom
        global_metal_order = norm_union.sum(axis=0).sort_values(ascending=False).index.tolist()
    else:
        global_metal_order = None
    # ------------------------------------------------------------------------

    # Panels 1–3: per-event profiles (ALL metals, common order)
    _stacked_bars_per_event_all(ax1, fw_df, 'Fireworks Events',
                                metal_order=global_metal_order, drop_below_pct=None)
    _stacked_bars_per_event_all(ax2, wf_df, 'Wildfire Events',
                                metal_order=global_metal_order, drop_below_pct=None)
    _stacked_bars_per_event_all(ax3, ck_df, 'Coke Plume Events',
                                metal_order=global_metal_order, drop_below_pct=None)

    # Panel 4: Mean comparison (ALL metals; same union/ordering)
    def _mean_all(df_cat):
        if df_cat.empty: return None
        pct_cols = [c for c in df_cat.columns if c.endswith('_pct')]
        d = df_cat[pct_cols].copy()
        for c in pct_cols:
            d[c] = pd.to_numeric(d[c], errors='coerce')
        m = d.mean(skipna=True).fillna(0.0)
        s = m.sum()
        return (100.0 * m / s) if s > 0 else m

    fw_mean = _mean_all(fw_df); wf_mean = _mean_all(wf_df); ck_mean = _mean_all(ck_df)

    mean_dict = {}
    if fw_mean is not None: mean_dict['Fireworks Mean'] = fw_mean
    if wf_mean is not None: mean_dict['Wildfire Mean']  = wf_mean
    if ck_mean is not None: mean_dict['Coke Plume Mean'] = ck_mean

    ax = ax4
    if mean_dict:
        # 1) Build union frame and row-normalize to 100%
        metals = sorted(set().union(*[s.index for s in mean_dict.values()]))
        plot_data = pd.DataFrame(index=mean_dict.keys(), columns=metals, dtype=float)
        for k, s in mean_dict.items():
            plot_data.loc[k, s.index] = s.values
        plot_data = plot_data.fillna(0.0)
        plot_data = plot_data.div(plot_data.sum(axis=1), axis=0).multiply(100.0)

        # 2) Stable color map (consistent colors even though order differs per bar)
        all_metals = list(plot_data.columns)
        color_map = {m: COLORBLIND_COLORS[i % len(COLORBLIND_COLORS)] for i, m in enumerate(all_metals)}

        # 3) Draw each bar with its own descending order so largest is at bottom
        x = np.arange(len(plot_data))
        width = 0.6
        for j, (row_name, row) in enumerate(plot_data.iterrows()):
            order = row.sort_values(ascending=False).index.tolist()
            bottom = 0.0
            for m in order:
                val = row[m]
                if val <= 0:
                    continue
                ax.bar(x[j], val, width=width, bottom=bottom,
                       color=color_map[m], edgecolor='white', linewidth=0.8)
                bottom += val

        ax.set_xticks(x)
        ax.set_xticklabels(plot_data.index, rotation=0, ha='center',
                           fontsize=11, fontweight='bold')
        ax.set_ylabel('Percentage of Total Corrected Metal (%)', fontsize=12, fontweight='bold')
        ax.set_title('Mean Fingerprint Comparison (All Metals)', fontsize=14, fontweight='bold', pad=15)
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.3, linestyle='--', axis='y')

        # Legend (order by overall contribution just for readability)
        legend_order = plot_data.sum(axis=0).sort_values(ascending=False).index.tolist()
        handles = [plt.Rectangle((0, 0), 1, 1, color=color_map[m]) for m in legend_order]
        ax.legend(handles, [m.replace('_pct', '') for m in legend_order],
                  bbox_to_anchor=(1.02, 1), loc='upper left', frameon=True, fontsize=9)

    else:
        ax.set_title('Mean Fingerprint Comparison (All Metals)', fontsize=14, fontweight='bold', pad=15)
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.3, linestyle='--', axis='y')

    plt.suptitle('Metal Composition Profiles (Background-Corrected)', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')

    return fig, axes



def create_seasonal_background_plot(df_profiles, save_path=None):
    """
    Create plot showing 4 seasonal background fingerprints (using raw concentrations)
    """

    # Filter to background events
    bg_events = df_profiles[df_profiles['Category'].isin(['Winter', 'Spring', 'Summer', 'Fall'])].copy()

    if bg_events.empty:
        print("No background events found")
        return None, None

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))

    # Get all _pct columns (raw percentages for background)
    pct_cols = [col for col in bg_events.columns if col.endswith('_pct')]

    # Get top 10 metals across all background periods
    metal_sums = bg_events[pct_cols].sum().sort_values(ascending=False).head(10)
    top_pct_cols = metal_sums.index.tolist()

    # Order by seasons
    season_order = ['Winter', 'Spring', 'Summer', 'Fall']
    bg_events['Category'] = pd.Categorical(bg_events['Category'], categories=season_order, ordered=True)
    bg_events = bg_events.sort_values('Category')
    bg_events = bg_events.set_index('Event')

    # Prepare data
    plot_data = bg_events[top_pct_cols].fillna(0)

    # Renormalize each row to 100%
    for idx in plot_data.index:
        row_sum = plot_data.loc[idx].sum()
        if row_sum > 0:
            plot_data.loc[idx] = (plot_data.loc[idx] / row_sum) * 100

    # Create stacked bars
    bottom = np.zeros(len(plot_data))
    bar_width = 0.7

    for i, col in enumerate(top_pct_cols):
        metal_name = col.replace('_pct', '')
        values = plot_data[col].values

        bars = ax.bar(range(len(plot_data)), values, bottom=bottom,
                      label=metal_name, color=COLORBLIND_COLORS[i % len(COLORBLIND_COLORS)],
                      alpha=0.85, edgecolor='white', linewidth=0.8,
                      width=bar_width)
        bottom += values

    # Customize axes
    ax.set_xticks(range(len(plot_data)))
    ax.set_xticklabels(plot_data.index, rotation=0, ha='center', fontsize=12, fontweight='bold')
    ax.set_ylabel('Percentage of Total Metal Concentration (%)', fontsize=13, fontweight='bold')
    ax.set_title('Seasonal Background Metal Composition (Raw Concentrations)',
                 fontsize=15, fontweight='bold', pad=20)

    # Add legend
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left',
              frameon=True, fontsize=11)

    # Set limits and grid
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')

    return fig, ax


# Create the plots
print("\nCreating enhanced metal composition plots...")

# # Plot 1: Fireworks and Wildfire panels with means
# fig1, axes1 = create_separate_panel_plots(df_profiles,
#                                           'metal_composition_fireworks_wildfire.png')
# plt.show()

# Plot 2: Seasonal backgrounds
# fig2, ax2 = create_seasonal_background_plot(df_profiles,
#                                             'metal_composition_seasonal_background.png')
# plt.show()
#
# print("\nPlots created successfully!")
# print("- Fireworks and Wildfire panels (with mean fingerprints)")
# print("- Seasonal background fingerprints")

def plot_event_panels_all_metals(df_profiles, save_path=None, drop_below_pct=None):
    fw_df = df_profiles[df_profiles['Category'] == 'Fireworks'].copy()
    wf_df = df_profiles[df_profiles['Category'] == 'Wildfire' ].copy()
    ck_df = df_profiles[df_profiles['Category'] == 'Coke Plume'].copy()

    def stacked_events(ax, df_cat, title, drop_below_pct):
        ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
        ax.set_ylim(0, 100); ax.grid(True, alpha=0.3, linestyle='--', axis='y')
        if df_cat.empty: return

        df_cat = df_cat.set_index('Event')
        pct_cols = [c for c in df_cat.columns if c.endswith('_pct')]
        for c in pct_cols: df_cat[c] = pd.to_numeric(df_cat[c], errors='coerce')
        plot = df_cat[pct_cols].fillna(0.0)
        plot = plot.div(plot.sum(axis=1).replace(0, np.nan), axis=0).multiply(100.0).fillna(0.0)

        if drop_below_pct is not None:
            keep = (plot.max(axis=0) >= drop_below_pct)
            plot = plot.loc[:, keep]

        # stable colors across bars
        metals = list(plot.columns)
        color_map = make_color_map(metals, scheme="husl")  # try "tab20b" or "Spectral" if you prefer

        x = np.arange(len(plot)); width = 0.8
        for j, (ev, row) in enumerate(plot.iterrows()):
            order = row.sort_values(ascending=False).index.tolist()  # largest -> bottom
            bottom = 0.0
            for m in order:
                val = row[m]
                if val <= 0: continue
                ax.bar(x[j], val, width=width, bottom=bottom,
                       color=color_map[m], edgecolor='white', linewidth=0.8, alpha=0.85)
                bottom += val
        ax.set_xticks(x); ax.set_xticklabels(plot.index, rotation=45, ha='right', fontsize=10)
        ax.set_ylabel('Percentage of Total Corrected Metal (%)', fontsize=12, fontweight='bold')

    fig, axes = plt.subplots(1, 3, figsize=(26, 8))
    stacked_events(axes[0], fw_df, 'Fireworks Events', drop_below_pct)
    stacked_events(axes[1], wf_df, 'Wildfire Events',  drop_below_pct)
    stacked_events(axes[2], ck_df, 'Coke Plume Events', drop_below_pct)

    plt.suptitle('Metal Composition Profiles (Background-Corrected, All Metals)', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    return fig, axes

import matplotlib.patheffects as pe

def plot_mean_fingerprint_panel(df_profiles,
                                save_path=None,
                                title='Mean Fingerprint Comparison (All Metals)',
                                label_threshold_pct=1.0,   # only label slices ≥ this height (to avoid clutter)
                                zoom_ylim=(98, 100),       # zoomed-in y-limits
                                zoom_save_path='mean_fingerprint_zoom_98_100.png',
                                color_scheme='husl'):
    """
    Mean fingerprint stacked bars + labels (metal symbols) and a 90–100% zoom figure.

    Parameters
    ----------
    label_threshold_pct : float
        Minimum slice height (%) to draw a label (keeps tiny labels from overlapping).
    zoom_ylim : tuple(float, float)
        Y-range for the zoomed panels (e.g., (90, 100)).
    color_scheme : str
        Palette name passed to make_color_map(...).
    """

    cats = ['Fireworks', 'Wildfire', 'Coke Plume']
    mean_dict = {}

    def mean_profile(df_cat):
        if df_cat.empty: return None
        pct_cols = [c for c in df_cat.columns if c.endswith('_pct')]
        d = df_cat[pct_cols].apply(pd.to_numeric, errors='coerce').fillna(0.0)
        m = d.mean(skipna=True)
        s = m.sum()
        return (100.0 * m / s) if s > 0 else None

    # Build mean profiles
    for cat in cats:
        s = mean_profile(df_profiles[df_profiles['Category'] == cat])
        if s is not None:
            mean_dict[f'{cat} Mean'] = s

    # Nothing to plot
    if not mean_dict:
        fig, ax = plt.subplots(figsize=(9, 8))
        ax.set_title(title)
        return fig, ax

    # Union of metals and row-normalize
    metals = sorted(set().union(*[s.index for s in mean_dict.values()]))
    plot = pd.DataFrame(index=mean_dict.keys(), columns=metals, dtype=float)
    for k, s in mean_dict.items():
        plot.loc[k, s.index] = s.values
    plot = plot.fillna(0.0)
    plot = plot.div(plot.sum(axis=1), axis=0).multiply(100.0)

    # Colors: distinct hues for many categories
    color_map = make_color_map(plot.columns, scheme=color_scheme)

    # ---------- MAIN MEAN FIGURE (with labels) ----------
    fig, ax = plt.subplots(figsize=(9, 8))
    x = np.arange(len(plot))
    width = 0.6

    # Draw each mean bar with its own descending order (largest at bottom)
    for j, (row_name, row) in enumerate(plot.iterrows()):
        order = row.sort_values(ascending=False).index.tolist()
        bottom = 0.0
        for m in order:
            val = float(row[m])
            if val <= 0:
                continue
            rect = ax.bar(x[j], val, width=width, bottom=bottom,
                          color=color_map[m], edgecolor='white', linewidth=0.8)
            # Label if sufficiently tall
            if val >= label_threshold_pct:
                # take only the first token (chemical symbol) before any space
                symbol = m.replace('_pct', '').split()[0]
                ax.text(x[j], bottom + val/2.0, symbol,
                        ha='center', va='center', fontsize=9, color='black',
                        path_effects=[pe.withStroke(linewidth=2, foreground='white')])
            bottom += val

    ax.set_xticks(x)
    ax.set_xticklabels(plot.index, fontsize=11, fontweight='bold')
    ax.set_ylabel('Percentage of Total Corrected Metal (%)', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')

    # Legend ordered by overall contribution (optional; you can comment this out)
    legend_order = plot.sum(axis=0).sort_values(ascending=False).index.tolist()
    handles = [plt.Rectangle((0,0),1,1, color=color_map[m]) for m in legend_order]
    ax.legend(handles, [m.replace('_pct','') for m in legend_order],
              bbox_to_anchor=(1.02, 1), loc='upper left', frameon=True, fontsize=9)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')

    # ---------- ZOOMED 90–100% FIGURE ----------
    zfig, zaxes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    zfig.suptitle(f'{title} — Zoomed {zoom_ylim[0]}–{zoom_ylim[1]}%', fontsize=14, fontweight='bold', y=1.05)

    for j, (row_name, row) in enumerate(plot.iterrows()):
        axz = zaxes[j]
        order = row.sort_values(ascending=False).index.tolist()
        bottom = 0.0
        for m in order:
            val = float(row[m])
            if val <= 0:
                continue
            axz.bar(0, val, width=0.6, bottom=bottom,
                    color=color_map[m], edgecolor='white', linewidth=0.8)
            # label in zoom too (use smaller font)
            if val >= label_threshold_pct:
                # take only the first token (chemical symbol) before any space
                symbol = m.replace('_pct', '').split()[0]
                axz.text(0, bottom + val/2.0, symbol,
                         ha='center', va='center', fontsize=10, color='black',
                         path_effects=[pe.withStroke(linewidth=2, foreground='white')])
            bottom += val

        axz.set_xlim(-0.8, 0.8)
        axz.set_ylim(*zoom_ylim)
        axz.set_title(row_name, fontsize=12, fontweight='bold')
        axz.set_xticks([])  # hide x ticks for single-bar panels
        axz.grid(True, alpha=0.3, linestyle='--', axis='y')

        if j == 0:
            axz.set_ylabel('Percentage of Total Corrected Metal (%)', fontsize=11, fontweight='bold')

    zfig.tight_layout()
    if zoom_save_path:
        zfig.savefig(zoom_save_path, dpi=300, bbox_inches='tight', facecolor='white')

    return fig, ax, zfig, zaxes


# # A) event panels only
# plot_event_panels_all_metals(df_profiles, 'events_fireworks_wildfire_coke_allmetals.png', drop_below_pct=None)
# plt.show()

# B) standalone mean panel
# Standalone mean panel + zoomed panels (labels included)
plot_mean_fingerprint_panel(
    df_profiles,
    save_path='mean_fingerprint_allmetals_labeled.png',
    zoom_save_path='mean_fingerprint_zoom_90_100.png',
    label_threshold_pct=1,     # tweak if you want more/less labels
    color_scheme='husl'          # try 'tab20b' or 'Spectral' if preferred
)
plt.show()


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_mean_fingerprint_pies(
    df_profiles,
    categories=('Fireworks', 'Wildfire', 'Coke Plume'),
    threshold_pct=98.0,          # show a second row of pies for the 98–100% tail
    label_min_slice_pct=1.0,      # only label slices ≥ this % in the FULL pies
    figsize=(12, 8),
    save_path_full='mean_fingerprint_pies.png',
    save_path_zoom='mean_fingerprint_pies_zoom_98_100.png'
):
    """
    Build a 2x3 panel of pies:
      Row 1: one FULL pie per category's mean fingerprint (sums to 100%).
      Row 2: a ZOOM pie showing the tiny 'tail' that occupies threshold–100% (e.g., 98–100%) of the FULL pie.
              The zoom pie is renormalized to 100% of that tail so you can read the composition of the 'last 2%'.

    Notes
    -----
    • Labels are element symbols (first token of the column name before any space).
    • Slices < label_min_slice_pct in FULL pies are unlabeled to avoid clutter.
    • If a category is missing, its pies are skipped.
    """

    def mean_profile(df_cat):
        if df_cat.empty:
            return None
        pct_cols = [c for c in df_cat.columns if c.endswith('_pct')]
        d = df_cat[pct_cols].apply(pd.to_numeric, errors='coerce').fillna(0.0)
        m = d.mean(skipna=True)
        s = m.sum()
        return (100.0 * m / s) if s > 0 else None

    # -------- compute mean profiles --------
    mean_series = {}
    for cat in categories:
        s = mean_profile(df_profiles[df_profiles['Category'] == cat])
        if s is not None:
            # sort descending so the “tail” really is the tiny components
            s = s.sort_values(ascending=False)
            mean_series[cat] = s

    if not mean_series:
        print("No categories had data to plot.")
        return None, None

    # union of metals for consistent color assignment
    metals_union = sorted(set().union(*[s.index for s in mean_series.values()]))

    # make a simple, stable color map (don’t hardcode colors → let mpl pick)
    # but keep color per metal consistent across pies by mapping order to colormap indices
    cmap = plt.get_cmap('tab20') if len(metals_union) <= 20 else plt.get_cmap('tab20b')
    color_map = {m: cmap(i % cmap.N) for i, m in enumerate(metals_union)}

    # ---------- FULL pies ----------
    n = len(mean_series)
    fig_full, axes_full = plt.subplots(1, n, figsize=(figsize[0], figsize[1]/2))
    if n == 1:
        axes_full = [axes_full]

    for ax, (cat, s) in zip(axes_full, mean_series.items()):
        vals = s.values.astype(float)
        labs_all = [idx.split()[0] for idx in s.index]  # element symbol = first token
        # hide labels for tiny wedges to reduce clutter
        labels = [lab if v >= label_min_slice_pct else '' for lab, v in zip(labs_all, vals)]

        wedges, _ = ax.pie(
            vals,
            labels=labels,
            startangle=90,
            counterclock=False,
            wedgeprops=dict(edgecolor='white', linewidth=0.8)
        )
        ax.set_title(f'{cat} Mean (100%)', fontweight='bold')
        # enforce consistent colors by remapping wedges in the order of s.index
        for w, m in zip(wedges, s.index):
            w.set_facecolor(color_map[m])

    fig_full.suptitle('Mean Fingerprint — Full Pies (All Metals)', y=0.98, fontweight='bold')
    fig_full.tight_layout()
    if save_path_full:
        fig_full.savefig(save_path_full, dpi=300, bbox_inches='tight', facecolor='white')

    # ---------- ZOOM pies (threshold–100%) ----------
    fig_zoom, axes_zoom = plt.subplots(1, n, figsize=(figsize[0], figsize[1]/2))
    if n == 1:
        axes_zoom = [axes_zoom]

    for ax, (cat, s) in zip(axes_zoom, mean_series.items()):
        # take the *tail* that occupies (threshold_pct → 100%)
        cumulative = s.cumsum()
        tail_mask = cumulative > threshold_pct - 1e-9  # numeric robustness
        s_tail = s[tail_mask]
        if s_tail.sum() <= 0 or len(s_tail) == 0:
            ax.axis('off')
            ax.set_title(f'{cat} Mean — No slices in {threshold_pct}–100%', fontweight='bold')
            continue

        # renormalize tail to 100% so you can read its composition
        vals = (100.0 * s_tail / s_tail.sum()).values.astype(float)
        labels = [idx.split()[0] for idx in s_tail.index]

        wedges, _ = ax.pie(
            vals,
            labels=labels,
            startangle=90,
            counterclock=False,
            wedgeprops=dict(edgecolor='white', linewidth=0.8)
        )
        for w, m in zip(wedges, s_tail.index):
            w.set_facecolor(color_map[m])
        ax.set_title(f'{cat} Mean — Zoom {threshold_pct:.0f}–100%', fontweight='bold')

    fig_zoom.suptitle(f'Mean Fingerprint — Zoomed Tail ({threshold_pct:.0f}–100%)', y=0.98, fontweight='bold')
    fig_zoom.tight_layout()
    if save_path_zoom:
        fig_zoom.savefig(save_path_zoom, dpi=300, bbox_inches='tight', facecolor='white')

    return fig_full, axes_full, fig_zoom, axes_zoom

plot_mean_fingerprint_pies(
    df_profiles,
    threshold_pct=98.0,             # ← this gives you the “98–100%” zoom row
    label_min_slice_pct=1.0,        # labels only on slices ≥1% in the FULL pies
    save_path_full='mean_fingerprint_pies.png',
    save_path_zoom='mean_fingerprint_pies_zoom_98_100.png'
)
plt.show()
