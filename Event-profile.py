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

# Define event periods with enhanced categorization
fireworks_periods = [
    ("Small Fireworks 1", "Small", pd.Timestamp('2023-06-30 17:00', tz=eastern),
     pd.Timestamp('2023-07-01 12:00', tz=eastern)),
    ("Large Fireworks 1", "Large", pd.Timestamp('2023-07-01 17:00', tz=eastern),
     pd.Timestamp('2023-07-02 12:00', tz=eastern)),
    ("Large Fireworks 2", "Large", pd.Timestamp('2023-07-03 17:00', tz=eastern),
     pd.Timestamp('2023-07-04 16:00', tz=eastern)),
    ("Large Fireworks 3", "Large", pd.Timestamp('2023-07-04 17:00', tz=eastern),
     pd.Timestamp('2023-07-05 12:00', tz=eastern)),
    ("Small Fireworks 2", "Small", pd.Timestamp('2023-07-05 17:00', tz=eastern),
     pd.Timestamp('2023-07-06 12:00', tz=eastern)),
    ("Fireworks 2024", "Medium", pd.Timestamp('2024-06-29 20:00', tz=eastern),
     pd.Timestamp('2024-06-29 23:59', tz=eastern)),
    ("Small Fireworks 3", "Small", pd.Timestamp('2025-06-28 20:00', tz=eastern),
     pd.Timestamp('2025-06-29 3:00', tz=eastern)),
    ("Large Fireworks 4", "Large", pd.Timestamp('2025-07-03 17:00', tz=eastern),
     pd.Timestamp('2025-07-04 12:00', tz=eastern)),
    ("Large Fireworks 5", "Large", pd.Timestamp('2025-07-04 17:00', tz=eastern),
     pd.Timestamp('2025-07-05 16:00', tz=eastern)),
    ("Small Fireworks 4", "Small", pd.Timestamp('2025-07-05 17:00', tz=eastern),
     pd.Timestamp('2025-07-06 12:00', tz=eastern))
]

wildfire_periods = [
    ("Wildfire 1 (Minor)", "Minor", pd.Timestamp('2023-06-05 12:00', tz=eastern),
     pd.Timestamp('2023-06-08', tz=eastern)),
    ("Wildfire 2 (Major)", "Major", pd.Timestamp('2023-06-28', tz=eastern), pd.Timestamp('2023-06-30', tz=eastern)),
    ("Wildfire 3 (Minor)", "Minor", pd.Timestamp('2023-07-16', tz=eastern), pd.Timestamp('2023-07-18', tz=eastern))
]

# Enhanced background selection - multiple representative periods
background_periods = [
    ("Winter Background", "Background", pd.Timestamp('2023-12-15 00:00', tz=eastern),
     pd.Timestamp('2023-12-22 23:59', tz=eastern)),
    ("Spring Background", "Background", pd.Timestamp('2024-04-15 00:00', tz=eastern),
     pd.Timestamp('2024-04-22 23:59', tz=eastern)),
    ("Fall Background", "Background", pd.Timestamp('2024-09-15 00:00', tz=eastern),
     pd.Timestamp('2024-09-22 23:59', tz=eastern))
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

# Calculate average background concentrations from multiple periods
background_concentrations = {}
all_background_data = []

for label, category, start, end in background_periods:
    df_bg = df[(df.index >= start) & (df.index <= end)]
    all_background_data.append(df_bg)

df_all_background = pd.concat(all_background_data)

for conc_col, uncert_col in metal_uncert_pairs.items():
    if conc_col in df_all_background.columns and uncert_col in df_all_background.columns:
        valid = df_all_background[conc_col] > 3 * df_all_background[uncert_col]
        mean_bg = df_all_background.loc[valid, conc_col].mean()
        if pd.notna(mean_bg):
            metal_name = conc_col.split(" (ng/m3)")[0].strip()
            background_concentrations[metal_name] = mean_bg

print(f"\nCalculated background concentrations for {len(background_concentrations)} metals")


# Function to get metal profiles for events
def get_event_profile(df_event, top_n=15):
    """Get metal profile for an event period"""
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
                        'corrected_concentration': max(0, mean_val - background_val),
                        'enhancement_factor': mean_val / background_val if background_val > 0 else np.inf,
                        'valid_measurements': valid.sum()
                    }

    # Sort by raw concentration and take top N
    sorted_metals = sorted(event_metals.items(),
                           key=lambda x: x[1]['raw_concentration'],
                           reverse=True)[:top_n]

    return dict(sorted_metals)


# Process all events and collect data
all_events = fireworks_periods + wildfire_periods + background_periods
event_data = []

print("\nProcessing events...")
for label, category, start, end in all_events:
    print(f"Processing: {label}")

    df_event = df[(df.index >= start) & (df.index <= end)]

    if len(df_event) == 0:
        print(f"  Warning: No data found for {label}")
        continue

    profile = get_event_profile(df_event)

    # Calculate totals
    total_raw = sum([data['raw_concentration'] for data in profile.values()])
    total_corrected = sum([data['corrected_concentration'] for data in profile.values()])

    # Create event record
    event_record = {
        'Event': label,
        'Category': category,
        'Start': start,
        'End': end,
        'Duration_hours': (end - start).total_seconds() / 3600,
        'Data_points': len(df_event),
        'Total_raw_concentration': total_raw,
        'Total_corrected_concentration': total_corrected
    }

    # Add individual metal data
    for metal, data in profile.items():
        event_record[f'{metal}_raw'] = data['raw_concentration']
        event_record[f'{metal}_corrected'] = data['corrected_concentration']
        event_record[f'{metal}_enhancement'] = data['enhancement_factor']
        event_record[f'{metal}_raw_pct'] = (data['raw_concentration'] / total_raw * 100) if total_raw > 0 else 0
        event_record[f'{metal}_corrected_pct'] = (
                    data['corrected_concentration'] / total_corrected * 100) if total_corrected > 0 else 0

    event_data.append(event_record)

# Create DataFrame
df_profiles = pd.DataFrame(event_data)

print(f"\nCreated profiles for {len(df_profiles)} events")

# Display event boundaries table
print("\nEvent Time Boundaries:")
print("=" * 90)
print(f"{'Event':<25} {'Category':<12} {'Start':<20} {'End':<20} {'Duration (h)':<12}")
print("=" * 90)
for _, row in df_profiles.iterrows():
    print(f"{row['Event']:<25} {row['Category']:<12} {row['Start'].strftime('%Y-%m-%d %H:%M'):<20} "
          f"{row['End'].strftime('%Y-%m-%d %H:%M'):<20} {row['Duration_hours']:<12.1f}")


# Identify fireworks-enhanced metals
def identify_fireworks_metals(enhancement_threshold=2.0, consistency_threshold=0.6):
    """Identify metals consistently enhanced during fireworks events"""

    fireworks_events = df_profiles[df_profiles['Category'].isin(['Small', 'Large', 'Medium'])]

    if fireworks_events.empty:
        return {}

    enhanced_metals = {}
    enhancement_cols = [col for col in df_profiles.columns if '_enhancement' in col]

    for col in enhancement_cols:
        metal = col.replace('_enhancement', '')

        if col in fireworks_events.columns:
            enhancements = fireworks_events[col].dropna()

            if len(enhancements) > 0:
                enhanced_count = (enhancements >= enhancement_threshold).sum()
                consistency = enhanced_count / len(enhancements)

                if consistency >= consistency_threshold:
                    enhanced_metals[metal] = {
                        'mean_enhancement': enhancements.mean(),
                        'max_enhancement': enhancements.max(),
                        'consistency': consistency,
                        'events_enhanced': enhanced_count,
                        'total_events': len(enhancements)
                    }

    return enhanced_metals


# Compare metal diversity between fireworks and wildfires
def compare_metal_diversity():
    """Compare metal diversity between fireworks and wildfires"""

    fireworks_events = df_profiles[df_profiles['Category'].isin(['Small', 'Large', 'Medium'])]
    wildfire_events = df_profiles[df_profiles['Category'].isin(['Minor', 'Major'])]

    enhancement_cols = [col for col in df_profiles.columns if '_enhancement' in col]

    fireworks_diversity = {}
    wildfire_diversity = {}

    for col in enhancement_cols:
        metal = col.replace('_enhancement', '')

        # Fireworks frequency
        if not fireworks_events.empty:
            fw_enhanced = (fireworks_events[col] >= 2.0).sum()
            fw_total = len(fireworks_events)
            fireworks_diversity[metal] = fw_enhanced / fw_total

        # Wildfire frequency
        if not wildfire_events.empty:
            wf_enhanced = (wildfire_events[col] >= 2.0).sum()
            wf_total = len(wildfire_events)
            wildfire_diversity[metal] = wf_enhanced / wf_total

    return fireworks_diversity, wildfire_diversity


# Run the analysis
fireworks_metals = identify_fireworks_metals()
fw_diversity, wf_diversity = compare_metal_diversity()

print("\nMetals Consistently Enhanced During Fireworks (>2× background in >60% of events):")
print("=" * 80)
print(f"{'Metal':<8} {'Mean Enh.':<12} {'Max Enh.':<12} {'Consistency':<12} {'Events Enhanced':<15}")
print("=" * 80)

for metal, stats in sorted(fireworks_metals.items(), key=lambda x: x[1]['mean_enhancement'], reverse=True):
    print(f"{metal:<8} {stats['mean_enhancement']:<12.1f}× {stats['max_enhancement']:<12.1f}× "
          f"{stats['consistency']:<12.1%} {stats['events_enhanced']}/{stats['total_events']}")

print("\nMetal Enhancement Frequency Comparison (Fireworks vs Wildfires):")
print("=" * 60)
print(f"{'Metal':<8} {'Fireworks':<12} {'Wildfires':<12} {'Difference':<12}")
print("=" * 60)

all_compared_metals = set(fw_diversity.keys()) | set(wf_diversity.keys())
fireworks_specific_metals = []

for metal in sorted(all_compared_metals):
    fw_freq = fw_diversity.get(metal, 0)
    wf_freq = wf_diversity.get(metal, 0)
    difference = fw_freq - wf_freq

    print(f"{metal:<8} {fw_freq:<12.1%} {wf_freq:<12.1%} {difference:+<12.1%}")

    # Identify fireworks-specific metals
    if fw_freq >= 0.5 and difference >= 0.3:
        fireworks_specific_metals.append(metal)

print(f"\nFireworks-Specific Metals (≥50% fireworks frequency, ≥30% selectivity):")
print(", ".join(fireworks_specific_metals))

# Set plotting parameters
plt.rcParams['font.size'] = 12
plt.rcParams['font.family'] = 'Times New Roman'


# Function to create grouped stacked bar plot
def create_stacked_plot(plot_type='raw_pct'):
    """Create normalized stacked bar plot"""

    # Get top metals for plotting
    if plot_type == 'raw_pct':
        metal_cols = [col for col in df_profiles.columns if col.endswith('_raw')]
        pct_suffix = '_raw_pct'
        title_suffix = "(Raw Concentrations)"
    else:
        metal_cols = [col for col in df_profiles.columns if col.endswith('_corrected')]
        pct_suffix = '_corrected_pct'
        title_suffix = "(Background-Corrected)"

    # Find top 10 metals by total concentration across all events
    metal_totals = df_profiles[metal_cols].sum().sort_values(ascending=False)
    top_metals = [col.replace('_raw', '').replace('_corrected', '') for col in metal_totals.head(10).index]

    # Prepare plotting data - group by category
    plot_order = []
    category_order = ['Small', 'Large', 'Medium', 'Minor', 'Major', 'Background']

    for category in category_order:
        cat_events = df_profiles[df_profiles['Category'] == category]
        for _, row in cat_events.iterrows():
            plot_order.append(row['Event'])

    plot_df = df_profiles[df_profiles['Event'].isin(plot_order)].set_index('Event').loc[plot_order]

    # Create the plot
    fig, ax = plt.subplots(figsize=(16, 8))

    # Get percentage data for top metals
    pct_cols = [f'{metal}{pct_suffix}' for metal in top_metals if f'{metal}{pct_suffix}' in plot_df.columns]
    plot_data = plot_df[pct_cols].fillna(0)

    # Colors for metals
    colors = plt.cm.Set3(np.linspace(0, 1, len(top_metals)))

    # Create stacked bars
    bottom = np.zeros(len(plot_data))

    for i, col in enumerate(pct_cols):
        metal_name = col.replace(pct_suffix, '')
        values = plot_data[col].values
        ax.bar(range(len(plot_data)), values, bottom=bottom,
               label=metal_name, color=colors[i], alpha=0.8)
        bottom += values

    # Customize plot
    ax.set_xticks(range(len(plot_data)))
    ax.set_xticklabels(plot_data.index, rotation=45, ha='right')
    ax.set_ylabel('Percentage of Total Metal Concentration (%)')
    ax.set_title(f'Metal Composition Profiles {title_suffix}')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.set_ylim(0, 100)

    # Add category separators
    current_cat = None
    for i, event in enumerate(plot_data.index):
        event_cat = df_profiles[df_profiles['Event'] == event]['Category'].iloc[0]
        if event_cat != current_cat:
            if current_cat is not None:
                ax.axvline(x=i - 0.5, color='black', linestyle='--', alpha=0.5)
            current_cat = event_cat

    plt.tight_layout()
    return fig, ax


# Create both plots
print("\nCreating metal composition plots...")

# Raw percentages
fig1, ax1 = create_stacked_plot('raw_pct')
plt.show()

# Background-corrected percentages
fig2, ax2 = create_stacked_plot('corrected_pct')
plt.show()

# Summary statistics
print("\nSummary Statistics by Event Category:")
print("=" * 60)

# Check what columns exist
print("Available columns with 'enhancement':")
enhancement_cols = [col for col in df_profiles.columns if 'enhancement' in col]
for col in enhancement_cols[:10]:  # Show first 10
    print(f"  {col}")

# Check if K_enhancement exists, if not use available columns
if 'K_enhancement' in df_profiles.columns:
    k_col = 'K_enhancement'
else:
    k_col = next((col for col in enhancement_cols if col.startswith('K_')), None)

# Use available columns for summary
summary_cols = ['Total_raw_concentration', 'Duration_hours']
if k_col:
    summary_cols.append(k_col)

print(f"\nUsing columns for summary: {summary_cols}")

try:
    summary_stats = df_profiles.groupby('Category')[summary_cols].agg(['mean', 'std', 'count']).round(2)
    print(summary_stats)
except Exception as e:
    print(f"Error creating summary: {e}")
    print("\nAlternative summary:")
    for category in df_profiles['Category'].unique():
        cat_data = df_profiles[df_profiles['Category'] == category]
        print(f"\n{category}:")
        print(f"  Events: {len(cat_data)}")
        print(f"  Avg Total Concentration: {cat_data['Total_raw_concentration'].mean():.1f} ng/m³")
        if k_col and k_col in cat_data.columns:
            print(f"  Avg K Enhancement: {cat_data[k_col].mean():.1f}×")

print(f"\nTotal fireworks-enhanced metals identified: {len(fireworks_metals)}")
print(f"Fireworks-specific metals: {len(fireworks_specific_metals)}")
print(f"Analysis complete!")

# Enhanced Metal Analysis Plotting Code with Colorblind-Friendly Palette
# For Section 3.3 of Metal Paper

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pytz

# Set high-quality plotting parameters for publication
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'Arial',
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'figure.figsize': (12, 8),
    'axes.linewidth': 1.2,
    'grid.alpha': 0.3,
    'grid.linestyle': '--'
})

# Paul Tol's colorblind-friendly palette
COLORBLIND_COLORS = [
    '#1f77b4',  # Blue
    '#ff7f0e',  # Orange
    '#2ca02c',  # Green
    '#d62728',  # Red
    '#9467bd',  # Purple
    '#8c564b',  # Brown
    '#e377c2',  # Pink
    '#7f7f7f',  # Gray
    '#bcbd22',  # Olive
    '#17becf',  # Cyan
    '#aec7e8',  # Light blue
    '#ffbb78',  # Light orange
    '#98df8a',  # Light green
    '#ff9896',  # Light red
    '#c5b0d5'  # Light purple
]

# Category colors for consistent theming
CATEGORY_COLORS = {
    'Small': '#1f77b4',  # Blue
    'Large': '#ff7f0e',  # Orange
    'Medium': '#2ca02c',  # Green
    'Minor': '#d62728',  # Red
    'Major': '#9467bd',  # Purple
    'Background': '#8c564b'  # Brown
}


def create_enhanced_stacked_plot(df_profiles, plot_type='raw_pct', save_path=None):
    """
    Create publication-quality stacked bar plot showing metal composition profiles

    Parameters:
    -----------
    df_profiles : pd.DataFrame
        Event profiles dataframe
    plot_type : str
        'raw_pct' for raw concentrations or 'corrected_pct' for background-corrected
    save_path : str, optional
        Path to save the figure
    """

    # Configure plot parameters based on type
    if plot_type == 'raw_pct':
        metal_cols = [col for col in df_profiles.columns if col.endswith('_raw')]
        pct_suffix = '_raw_pct'
        title_suffix = "(Raw Concentrations)"
        ylabel = "Percentage of Total Raw Metal Concentration (%)"
    else:
        metal_cols = [col for col in df_profiles.columns if col.endswith('_corrected')]
        pct_suffix = '_corrected_pct'
        title_suffix = "(Background-Corrected)"
        ylabel = "Percentage of Total Corrected Metal Concentration (%)"

    # Find top 10 metals by total concentration
    metal_totals = df_profiles[metal_cols].sum().sort_values(ascending=False)
    top_metals = [col.replace('_raw', '').replace('_corrected', '') for col in metal_totals.head(10).index]

    # Order events by category for logical grouping
    category_order = ['Small', 'Large', 'Medium', 'Minor', 'Major', 'Background']
    plot_order = []

    for category in category_order:
        cat_events = df_profiles[df_profiles['Category'] == category]
        for _, row in cat_events.iterrows():
            plot_order.append(row['Event'])

    plot_df = df_profiles[df_profiles['Event'].isin(plot_order)].set_index('Event').loc[plot_order]

    # Create figure with optimal dimensions
    fig, ax = plt.subplots(figsize=(18, 10))

    # Prepare percentage data
    pct_cols = [f'{metal}{pct_suffix}' for metal in top_metals if f'{metal}{pct_suffix}' in plot_df.columns]
    plot_data = plot_df[pct_cols].fillna(0)

    # Create stacked bars with colorblind-friendly colors
    bottom = np.zeros(len(plot_data))
    bar_width = 0.8

    for i, col in enumerate(pct_cols):
        metal_name = col.replace(pct_suffix, '')
        values = plot_data[col].values

        bars = ax.bar(range(len(plot_data)), values, bottom=bottom,
                      label=metal_name, color=COLORBLIND_COLORS[i],
                      alpha=0.85, edgecolor='white', linewidth=0.8,
                      width=bar_width)
        bottom += values

    # Customize axes and labels
    ax.set_xticks(range(len(plot_data)))
    ax.set_xticklabels(plot_data.index, rotation=45, ha='right', fontsize=11)
    ax.set_ylabel(ylabel, fontsize=14, fontweight='bold')
    ax.set_title(f'Metal Composition Profiles {title_suffix}',
                 fontsize=16, fontweight='bold', pad=25)

    # Enhanced legend
    legend = ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left',
                       frameon=True, fancybox=True, shadow=True,
                       ncol=1, fontsize=11)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_alpha(0.9)

    # Set limits and grid
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')

    # Add category separators and labels
    current_cat = None
    category_positions = {}
    separator_positions = []

    for i, event in enumerate(plot_data.index):
        event_cat = df_profiles[df_profiles['Event'] == event]['Category'].iloc[0]
        if event_cat != current_cat:
            if current_cat is not None:
                separator_positions.append(i - 0.5)
            current_cat = event_cat
            if event_cat not in category_positions:
                category_positions[event_cat] = []
            category_positions[event_cat].append(i)

    # Draw category separators
    for pos in separator_positions:
        ax.axvline(x=pos, color='black', linestyle='-', alpha=0.8, linewidth=2)

    # Add category labels at bottom
    for cat, positions in category_positions.items():
        if positions:
            center_pos = np.mean(positions)
            ax.text(center_pos, -8, cat, ha='center', va='top',
                    fontsize=12, fontweight='bold',
                    color=CATEGORY_COLORS.get(cat, 'black'),
                    bbox=dict(boxstyle="round,pad=0.4",
                              facecolor=CATEGORY_COLORS.get(cat, 'lightgray'),
                              alpha=0.2, edgecolor='none'))

    # Adjust layout
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')

    return fig, ax


def create_enhancement_heatmap(df_profiles, key_metals=None, save_path=None):
    """
    Create heatmap showing enhancement factors for key metals across events
    """

    if key_metals is None:
        key_metals = ['K 19', 'Sr 38', 'Ba 56', 'Cu 29', 'Bi 83', 'S 16', 'Al 13']

    # Filter to fireworks and wildfire events only
    plot_events = df_profiles[~df_profiles['Category'].isin(['Background'])]

    # Prepare enhancement data
    enhancement_cols = [f'{metal}_enhancement' for metal in key_metals
                        if f'{metal}_enhancement' in df_profiles.columns]

    if not enhancement_cols:
        print("Warning: No enhancement columns found for heatmap")
        return None, None

    # Create data matrix
    heatmap_data = plot_events[['Event', 'Category'] + enhancement_cols].set_index('Event')
    enhancement_matrix = heatmap_data[enhancement_cols]

    # Handle infinite values for visualization
    enhancement_matrix = enhancement_matrix.replace([np.inf, -np.inf], 1000)
    enhancement_matrix = enhancement_matrix.fillna(0)

    # Apply log transformation for better visualization
    log_matrix = np.log10(enhancement_matrix + 1)

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))

    # Create heatmap
    heatmap = sns.heatmap(log_matrix.T,
                          annot=True,
                          fmt='.1f',
                          cmap='plasma',  # Colorblind-friendly sequential colormap
                          cbar_kws={'label': 'log₁₀(Enhancement Factor + 1)',
                                    'shrink': 0.8},
                          ax=ax,
                          linewidths=0.5,
                          linecolor='white')

    # Customize labels
    ax.set_title('Metal Enhancement Factors During Events',
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Events', fontsize=14, fontweight='bold')
    ax.set_ylabel('Metals', fontsize=14, fontweight='bold')

    # Clean up y-axis labels (remove _enhancement suffix)
    yticklabels = [label.get_text().replace('_enhancement', '') for label in ax.get_yticklabels()]
    ax.set_yticklabels(yticklabels, rotation=0, fontsize=11)

    # Rotate x-axis labels for better readability
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=10)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')

    return fig, ax


def create_time_series_plot(df, fireworks_periods, key_metals=None, save_path=None):
    """
    Create time series plot showing metal concentrations during events
    """

    if key_metals is None:
        key_metals = ['K 19', 'Sr 38', 'Ba 56', 'Cu 29']

    # Define time range around first few events for clarity
    start_date = pd.Timestamp('2023-06-25', tz='US/Eastern')
    end_date = pd.Timestamp('2023-07-10', tz='US/Eastern')

    plot_df = df[(df.index >= start_date) & (df.index <= end_date)]

    # Create subplots for each metal
    fig, axes = plt.subplots(len(key_metals), 1, figsize=(16, 12), sharex=True)
    if len(key_metals) == 1:
        axes = [axes]

    for i, metal in enumerate(key_metals):
        conc_col = f'{metal} (ng/m3)'
        uncert_col = f'{metal} uncert (ng/m3)'

        if conc_col in plot_df.columns and uncert_col in plot_df.columns:
            # Filter valid data (concentration > 3 * uncertainty)
            valid_mask = plot_df[conc_col] > 3 * plot_df[uncert_col]
            valid_data = plot_df.loc[valid_mask]

            # Plot time series
            axes[i].plot(valid_data.index, valid_data[conc_col],
                         color=COLORBLIND_COLORS[i], linewidth=2.5,
                         marker='o', markersize=3, alpha=0.8,
                         label=f'{metal} Concentration')

            # Add fireworks event shading
            event_colors = {'Small': 'orange', 'Large': 'red', 'Medium': 'yellow'}
            legend_added = {}

            for label, category, fw_start, fw_end in fireworks_periods[:5]:  # First 5 events
                if fw_start >= start_date and fw_end <= end_date:
                    color = event_colors.get(category, 'gray')
                    alpha = 0.3

                    axes[i].axvspan(fw_start, fw_end, alpha=alpha, color=color,
                                    label=f'{category} Fireworks' if category not in legend_added else "")
                    legend_added[category] = True

            # Customize subplot
            axes[i].set_ylabel(f'{metal}\n(ng/m³)', fontsize=12, fontweight='bold')
            axes[i].grid(True, alpha=0.3)
            axes[i].set_yscale('log')
            axes[i].tick_params(axis='y', labelsize=10)

            # Add legend to first subplot only
            if i == 0:
                axes[i].legend(loc='upper right', fontsize=10, framealpha=0.9)

    # Final formatting
    axes[-1].set_xlabel('Date', fontsize=14, fontweight='bold')
    axes[-1].tick_params(axis='x', labelsize=11, rotation=45)

    plt.suptitle('Time Series of Key Fireworks Metals During July 2023 Events',
                 fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')

    return fig, axes


def create_enhancement_comparison_plot(fireworks_metals, save_path=None):
    """
    Create bar plot comparing enhancement factors for fireworks metals
    """

    # Prepare data
    metals = list(fireworks_metals.keys())
    mean_enhancements = []
    consistencies = []

    for metal in metals:
        enh = fireworks_metals[metal]['mean_enhancement']
        mean_enhancements.append(1000 if enh == np.inf else enh)  # Cap infinite values
        consistencies.append(fireworks_metals[metal]['consistency'])

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # Plot 1: Enhancement factors
    bars1 = ax1.bar(metals, mean_enhancements,
                    color=COLORBLIND_COLORS[:len(metals)],
                    alpha=0.8, edgecolor='black', linewidth=1.2)

    ax1.set_ylabel('Mean Enhancement Factor', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Metal', fontsize=14, fontweight='bold')
    ax1.set_title('Mean Enhancement Factors', fontsize=14, fontweight='bold')
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.tick_params(axis='x', rotation=45, labelsize=11)

    # Add value labels on bars
    for bar, metal in zip(bars1, metals):
        height = bar.get_height()
        original_val = fireworks_metals[metal]['mean_enhancement']
        label = '∞' if original_val == np.inf else f'{height:.1f}×'
        ax1.text(bar.get_x() + bar.get_width() / 2., height * 1.1,
                 label, ha='center', va='bottom', fontweight='bold', fontsize=10)

    # Plot 2: Consistency percentages
    bars2 = ax2.bar(metals, [c * 100 for c in consistencies],
                    color=COLORBLIND_COLORS[:len(metals)],
                    alpha=0.8, edgecolor='black', linewidth=1.2)

    ax2.set_ylabel('Consistency (%)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Metal', fontsize=14, fontweight='bold')
    ax2.set_title('Enhancement Consistency', fontsize=14, fontweight='bold')
    ax2.set_ylim(0, 105)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.tick_params(axis='x', rotation=45, labelsize=11)

    # Add percentage labels
    for bar, consistency in zip(bars2, consistencies):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2., height + 2,
                 f'{height:.0f}%', ha='center', va='bottom', fontweight='bold', fontsize=10)

    plt.suptitle('Fireworks-Associated Metal Enhancement Analysis',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')

    return fig, (ax1, ax2)


def create_summary_tables():
    """
    Create formatted summary tables for publication
    """

    print("\n" + "=" * 100)
    print("TABLE 1: Fireworks-Enhanced Metals Summary (Section 3.3)")
    print("=" * 100)
    print(f"{'Metal':<12} {'Element':<15} {'Mean Enh.':<12} {'Max Enh.':<12} {'Frequency':<12} {'Typical Source':<20}")
    print("-" * 100)

    # Metal source mapping based on fireworks chemistry
    metal_sources = {
        'K 19': 'Gunpowder/Fuel',
        'Sr 38': 'Red colorant',
        'Ba 56': 'Green colorant',
        'Cu 29': 'Blue colorant',
        'Bi 83': 'Crackling agent',
        'S 16': 'Oxidizer',
        'In 49': 'Specialty effect'
    }

    for metal, stats in sorted(fireworks_metals.items(),
                               key=lambda x: x[1]['mean_enhancement'] if x[1]['mean_enhancement'] != np.inf else 10000,
                               reverse=True):
        element = metal.split()[0]
        mean_enh = '∞' if stats['mean_enhancement'] == np.inf else f"{stats['mean_enhancement']:.1f}×"
        max_enh = '∞' if stats['max_enhancement'] == np.inf else f"{stats['max_enhancement']:.1f}×"
        frequency = f"{stats['consistency']:.0%}"
        source = metal_sources.get(metal, 'Unknown')

        print(f"{metal:<12} {element:<15} {mean_enh:<12} {max_enh:<12} {frequency:<12} {source:<20}")

    print("=" * 100)
    print("Enhancement Factor = Event Concentration ÷ Background Concentration")
    print("Frequency = Percentage of fireworks events where metal enhanced >2× background")
    print("∞ = Metal not detected in background periods")

    print("\n" + "=" * 80)
    print("TABLE 2: Event Category Statistics")
    print("=" * 80)
    print(f"{'Category':<12} {'Events':<8} {'Avg Duration (h)':<16} {'Avg Total Conc (ng/m³)':<22}")
    print("-" * 80)

    summary_stats = df_profiles.groupby('Category').agg({
        'Duration_hours': 'mean',
        'Total_raw_concentration': 'mean'
    }).round(1)

    event_counts = df_profiles['Category'].value_counts()

    for category in ['Small', 'Large', 'Medium', 'Minor', 'Major', 'Background']:
        if category in summary_stats.index:
            duration = summary_stats.loc[category, 'Duration_hours']
            conc = summary_stats.loc[category, 'Total_raw_concentration']
            count = event_counts[category]
            print(f"{category:<12} {count:<8} {duration:<16.1f} {conc:<22.1f}")

    print("=" * 80)


# Main execution block
def main():
    """
    Execute all plotting functions with the analyzed data
    """

    print("Creating enhanced publication-quality plots...")
    print("Using colorblind-friendly Paul Tol palette")

    # Create all plots
    plots_created = []

    # 1. Stacked composition plots
    try:
        fig1, ax1 = create_enhanced_stacked_plot(df_profiles, 'raw_pct',
                                                 'metal_composition_raw_enhanced.png')
        plt.show()
        plots_created.append("Raw concentration stacked plot")

        fig2, ax2 = create_enhanced_stacked_plot(df_profiles, 'corrected_pct',
                                                 'metal_composition_corrected_enhanced.png')
        plt.show()
        plots_created.append("Background-corrected stacked plot")
    except Exception as e:
        print(f"Error creating stacked plots: {e}")

    # 2. Enhancement heatmap
    try:
        fig3, ax3 = create_enhancement_heatmap(df_profiles,
                                               save_path='enhancement_heatmap_enhanced.png')
        if fig3:
            plt.show()
            plots_created.append("Enhancement heatmap")
    except Exception as e:
        print(f"Error creating heatmap: {e}")

    # 3. Time series plot
    try:
        fig4, ax4 = create_time_series_plot(df, fireworks_periods,
                                            save_path='time_series_enhanced.png')
        plt.show()
        plots_created.append("Time series plot")
    except Exception as e:
        print(f"Error creating time series: {e}")

    # 4. Enhancement comparison plot
    try:
        fig5, axes5 = create_enhancement_comparison_plot(fireworks_metals,
                                                         'enhancement_comparison_enhanced.png')
        plt.show()
        plots_created.append("Enhancement comparison plot")
    except Exception as e:
        print(f"Error creating comparison plot: {e}")

    # 5. Summary tables
    try:
        create_summary_tables()
        plots_created.append("Summary tables")
    except Exception as e:
        print(f"Error creating tables: {e}")

    print(f"\nSuccessfully created: {', '.join(plots_created)}")
    print("All plots saved with 300 DPI for publication quality")


# Execute main function when script is run
if __name__ == "__main__":
    main()