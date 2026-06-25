
import os
import pandas as pd
import matplotlib.pyplot as plt
import pytz
import seaborn as sns
import numpy as np

# Read the CSV file
df = pd.read_csv('Xact_EST_May2023_Oct2025_combined.csv')
df['TIME'] = pd.to_datetime(df['TIME'], utc=True).dt.tz_convert('US/Eastern').dt.tz_localize(None)

# Apply time exclusion filters
exclude1_start = pd.Timestamp('2024-01-09')
exclude1_end = pd.Timestamp('2024-02-13')
exclude2_start = pd.Timestamp('2024-07-02')
exclude2_end = pd.Timestamp('2024-08-08')

df = df[(df['TIME'] < exclude1_start) | (df['TIME'] > exclude1_end)]
df = df[(df['TIME'] < exclude2_start) | (df['TIME'] > exclude2_end)]
df.set_index('TIME', inplace=True)


# Assign season
def get_season(date):
    month = date.month
    if 3 <= month <= 5:
        return 'SP'
    elif 6 <= month <= 8:
        return 'SU'
    elif 9 <= month <= 11:
        return 'F'
    else:
        return 'W'


df['Season'] = df.index.map(get_season)
df['Hour'] = df.index.hour

# Set parameters
season_order = ['SP', 'SU', 'F', 'W']
season_names = ['Spring', 'Summer', 'Fall', 'Winter']
season_colors = {'SP': '#2ecc71', 'SU': '#e74c3c', 'F': '#f39c12', 'W': '#3498db'}
extended_metals = ['Fe', 'K', 'Ca', 'Zn']
# extended_metals = ['As', 'Se', 'Pb', 'Zn']
# Identify metal columns
metal_cols_ext = {
    m: [col for col in df.columns if f"{m} " in col and "(ng/m3)" in col and "uncert" not in col.lower()][0]
    for m in extended_metals
}

# Pair with uncertainty columns
headers = df.columns.tolist()
concentration_cols = [col for col in headers if " (ng/m3)" in col and "uncert" not in col.lower()]
uncertainty_cols = [col for col in headers if "uncert" in col.lower()]

metal_uncert_pairs = {}
for conc_col in concentration_cols:
    element = conc_col.split()[0]
    uncert_col = next((u_col for u_col in uncertainty_cols if element in u_col), None)
    if uncert_col:
        metal_uncert_pairs[conc_col] = uncert_col

# Create figure with 4 subplots (4 rows x 1 column)
fig, axes = plt.subplots(4, 1, figsize=(10, 12), sharex=True)

# Store handles and labels for legend
legend_handles = []
legend_labels = []

for idx, metal in enumerate(extended_metals):
    ax = axes[idx]
    col = metal_cols_ext[metal]
    uncert_col = metal_uncert_pairs.get(col)

    if not uncert_col:
        ax.text(0.5, 0.5, f'No uncertainty data for {metal}',
                ha='center', va='center', transform=ax.transAxes)
        continue

    # Filter valid data: concentration > 3 × uncertainty
    valid_mask = df[col] > 3 * df[uncert_col]
    valid_df = df[valid_mask][['Season', 'Hour', col]].dropna()

    if valid_df.empty:
        ax.text(0.5, 0.5, f'No valid data for {metal}',
                ha='center', va='center', transform=ax.transAxes)
        continue

    # Calculate percentiles for each season and hour
    for season in season_order:
        season_data = valid_df[valid_df['Season'] == season]

        if len(season_data) == 0:
            continue

        # Group by hour and calculate percentiles
        hourly_stats = season_data.groupby('Hour')[col].agg([
            ('median', 'median'),
            ('p10', lambda x: np.percentile(x, 10)),
            ('p90', lambda x: np.percentile(x, 90)),
            ('count', 'count')
        ]).reset_index()

        # Only plot if we have data for at least a few hours
        if len(hourly_stats) < 3:
            continue

        hours = hourly_stats['Hour']
        median = hourly_stats['median']
        p10 = hourly_stats['p10']
        p90 = hourly_stats['p90']

        color = season_colors[season]
        season_name = season_names[season_order.index(season)]

        # Plot median as solid line
        line = ax.plot(hours, median, color=color, linewidth=2.5,
                       label=season_name, alpha=0.9)

        # Store handles and labels from first subplot only
        if idx == 0:
            legend_handles.append(line[0])
            legend_labels.append(season_name)

        # Plot 10th and 90th percentiles as dashed lines
        ax.plot(hours, p10, color=color, linewidth=1.5,
                linestyle='--', alpha=0.6)
        ax.plot(hours, p90, color=color, linewidth=1.5,
                linestyle='--', alpha=0.6)

        # Shade between 10th and 90th percentiles
        ax.fill_between(hours, p10, p90, color=color, alpha=0.15)

    # Formatting
    # Only add x-label to bottom subplot
    if idx == len(extended_metals) - 1:
        ax.set_xlabel('Hour of Day (Local Time)', fontsize=14, fontweight='bold')

    ax.set_ylabel(f'{metal} Concentration (ng/m³)', fontsize=14, fontweight='bold')

    # Add panel label (a, b, c, d)
    panel_labels = ['a', 'b', 'c', 'd']
    ax.text(0.02, 0.95, f'({panel_labels[idx]}) {metal}', transform=ax.transAxes,
            fontsize=16, fontweight='bold', verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax.set_xlim(0, 23)
    ax.set_xticks(range(0, 24, 3))
    ax.tick_params(labelsize=12)
    ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)

    # Add subtle background
    ax.set_facecolor('#f8f9fa')

# Add single legend to the top right outside all panels
fig.legend(legend_handles, legend_labels, loc='upper right',
           framealpha=0.9, fontsize=13, bbox_to_anchor=(0.98, 0.98))

plt.tight_layout(rect=[0, 0, 1, 0.99])  # Adjust layout
plt.savefig('seasonal_diurnal_patterns_metals.png', dpi=300, bbox_inches='tight')
plt.show()

print("Plot saved as 'seasonal_diurnal_patterns_metals.png'")
print("\nPlot Details:")
print("- Solid lines: Median concentrations")
print("- Dashed lines: 10th and 90th percentiles")
print("- Shaded areas: Range between 10th and 90th percentiles")
print("- Each panel shows one metal with all four seasons overlaid")

## ---------- FINAL seasonal boxplots by year (no outliers + mean dots; clean (a) Fe labels) ----------
AGGREGATE_DAILY = False   # True → daily medians before boxplot
LOG_SCALE = False         # True → log-scale y-axis

df['Year'] = df.index.year
df['Date'] = df.index.date

def valid_metal_df(metal_symbol: str) -> pd.DataFrame:
    """Return filtered dataframe for one metal with >3×uncertainty rule applied."""
    col = metal_cols_ext[metal_symbol]
    uncert_col = metal_uncert_pairs.get(col)
    if uncert_col is None:
        return pd.DataFrame(columns=['Season', 'Year', 'Date', col])
    vmask = df[col] > 3 * df[uncert_col]
    out = df.loc[vmask, ['Season', 'Year', 'Date', col]].dropna()
    if AGGREGATE_DAILY:
        out = (
            out.groupby(['Season', 'Year', 'Date'], as_index=False)[col]
            .median()
        )
    return out

# Assemble long dataframe
long_frames = []
for m in extended_metals:
    if m not in metal_cols_ext:
        continue
    tmp = valid_metal_df(m).copy()
    if tmp.empty:
        continue
    tmp['Metal'] = m
    long_frames.append(tmp)

if len(long_frames) == 0:
    print("No valid data to draw seasonal boxplots.")
else:
    long_df = pd.concat(long_frames, ignore_index=True)

    season_label_map = {'SP': 'Spring', 'SU': 'Summer', 'F': 'Fall', 'W': 'Winter'}
    long_df['Season'] = pd.Categorical(long_df['Season'], categories=season_order, ordered=True)
    long_df['SeasonLabel'] = long_df['Season'].map(season_label_map)

    years_sorted = sorted(long_df['Year'].unique())
    year_palette = sns.color_palette('Set2', n_colors=len(years_sorted))
    year_colors = {y: c for y, c in zip(years_sorted, year_palette)}

    # --- Create 4 stacked subplots (one per metal) ---
    nrows = len(extended_metals)
    fig2, axes2 = plt.subplots(nrows, 1, figsize=(3.5, 2.8 * nrows), sharex=False)
    if nrows == 1:
        axes2 = [axes2]

    panel_labels = ['(a)', '(b)', '(c)', '(d)']

    for idx, (ax, m) in enumerate(zip(axes2, extended_metals)):
        sub = long_df[long_df['Metal'] == m]
        if sub.empty:
            ax.text(0.5, 0.5, f'No valid data for {m}', ha='center', va='center', transform=ax.transAxes)
            continue

        YCOL = metal_cols_ext[m]  # exact column name

        # --- Boxplot (no outliers) with mean dots ---
        sns.boxplot(
            data=sub,
            x='SeasonLabel',
            y=YCOL,
            hue='Year',
            order=['Spring', 'Summer', 'Fall', 'Winter'],
            hue_order=years_sorted,
            palette=year_colors,
            ax=ax,
            showcaps=True,
            linewidth=1.0,
            showfliers=False,  # hide outliers
            showmeans=True,    # draw mean marker
            meanprops=dict(
                marker='o',
                markerfacecolor='black',
                markeredgecolor='black',
                markersize=5,
                zorder=5,
            ),
        )

        # Axes formatting
        ax.set_ylabel(f'{m} (ng/m³)', fontsize=12, fontweight='bold')
        ax.set_xlabel('' if ax is not axes2[-1] else 'Season', fontsize=12, fontweight='bold')
        ax.grid(True, axis='y', alpha=0.25, linestyle=':', linewidth=0.6)
        if LOG_SCALE:
            ax.set_yscale('log')

        # --- Combined clean label (no box background) ---
        if idx < len(panel_labels):
            ax.text(
                0.02, 0.95, f'{panel_labels[idx]} {m}',
                transform=ax.transAxes,
                fontsize=14, fontweight='bold',
                va='top', ha='left',
                color='black'
            )

        # Remove per-axes legend; global legend later
        leg = ax.get_legend()
        if leg is not None:
            leg.remove()

    # --- Single legend at bottom of the figure ---
    handles, labels = axes2[0].get_legend_handles_labels()
    fig2.legend(
        handles, labels,
        title='Year',
        loc='lower center',
        ncol=min(len(years_sorted), 6),
        frameon=True,
        bbox_to_anchor=(0.5, 0.0)
    )

    plt.tight_layout(rect=[0, 0.06, 1, 0.98])  # leave space at bottom for legend
    outname = 'seasonal_boxplots_by_year_meandots_stacked_cleanlabels.png'
    plt.savefig(outname, dpi=300, bbox_inches='tight')
    plt.show()

    print(f"Saved seasonal-by-year boxplots with mean dots (stacked; labeled cleanly (a) Fe etc.): {outname}")
# ---------- END FINAL ----------

# ============================================================
# Seasonal & Diurnal Statistical Analysis for Metals (Hourly)
# Requirements: pandas, numpy, scipy, statsmodels
# Assumes your df already has: 'Season' (W/SP/SU/F), 'Year', 'Hour', 'Date'
# and your metal columns exist (e.g., the ones in metal_cols_ext).
# ============================================================

import numpy as np
import pandas as pd
from itertools import combinations
from scipy.stats import kruskal, mannwhitneyu
from statsmodels.stats.multitest import multipletests
import statsmodels.api as sm
import statsmodels.formula.api as smf

# -----------------------------
# User inputs expected to exist:
# -----------------------------
# df : your filtered hourly dataframe (index TIME), with columns:
#   'Season' in codes like ['SP','SU','F','W'] or similar, plus 'Hour','Year','Date'
# extended_metals : list like ['Fe','K','Ca','Zn']
# metal_cols_ext : dict mapping symbol -> exact column name in df (as you already built)
# season_order   : e.g., ['SP','SU','F','W']  (we’ll map to labels for tables)
season_label_map = {'SP':'Spring', 'SU':'Summer', 'F':'Fall', 'W':'Winter'}

# Ensure canonical categories/order
df = df.copy()
if 'Season' in df:
    df['Season'] = pd.Categorical(df['Season'], categories=season_order, ordered=True)
if 'Hour' in df:
    # treat as categorical for factorial tests
    df['HourCat'] = pd.Categorical(df['Hour'], categories=list(range(24)), ordered=True)

# Helper: safe log1p column creation
def add_log1p(df, colname, outname):
    df[outname] = np.log1p(df[colname])
    return df

# ============================================================
# PART 1: "Are seasons significantly different?" (by year)
# ============================================================

def kruskal_by_year(df, metal_col, season_order):
    """Kruskal–Wallis across 4 seasons within each year."""
    out = []
    for year in sorted(df['Year'].dropna().unique()):
        groups = []
        sizes = []
        for s in season_order:
            vals = df.loc[(df['Year']==year) & (df['Season']==s), metal_col].dropna()
            groups.append(vals.values)
            sizes.append(len(vals))
        if all(len(g) >= 5 for g in groups):  # minimal data check
            H, p = kruskal(*groups)
        else:
            H, p = np.nan, np.nan
        out.append({'Year': int(year), 'H': H, 'pval': p,
                    'n_W': sizes[0] if len(sizes)>0 else np.nan,
                    'n_SP': sizes[1] if len(sizes)>1 else np.nan,
                    'n_SU': sizes[2] if len(sizes)>2 else np.nan,
                    'n_F': sizes[3] if len(sizes)>3 else np.nan})
    return pd.DataFrame(out)

def pairwise_mwu_by_year(df, metal_col, season_order, padjust='holm'):
    """Pairwise Mann–Whitney U tests between seasons within each year + p-adjust."""
    rows = []
    season_pairs = list(combinations(season_order, 2))
    for year in sorted(df['Year'].dropna().unique()):
        raw_tests = []
        for (s1, s2) in season_pairs:
            x = df.loc[(df['Year']==year) & (df['Season']==s1), metal_col].dropna()
            y = df.loc[(df['Year']==year) & (df['Season']==s2), metal_col].dropna()
            if len(x) >= 5 and len(y) >= 5:
                stat, p = mannwhitneyu(x, y, alternative='two-sided')
            else:
                stat, p = np.nan, np.nan
            raw_tests.append({'Year': int(year), 'Season1': s1, 'Season2': s2,
                              'U': stat, 'p_raw': p, 'n1': len(x), 'n2': len(y)})
        # adjust within this year
        pvals = [r['p_raw'] for r in raw_tests]
        if np.isfinite(pvals).sum() > 0:
            mask = np.isfinite(pvals)
            padj = np.full_like(pvals, np.nan, dtype=float)
            _, p_adj, _, _ = multipletests(np.array(pvals)[mask], method=padjust)
            padj[mask] = p_adj
        else:
            padj = [np.nan]*len(pvals)
        for r, pa in zip(raw_tests, padj):
            r['p_adj_' + padjust] = pa
            rows.append(r)
    out = pd.DataFrame(rows)
    if not out.empty:
        out['Season1Label'] = out['Season1'].map(season_label_map)
        out['Season2Label'] = out['Season2'].map(season_label_map)
    return out

def two_way_anova_log(df, metal_col):
    """
    Two-way ANOVA on log1p(metal) with fixed factors Season and Year, incl. interaction.
    Uses OLS (hourly). Report Type II ANOVA table (season, year, season:year).
    """
    tmp = df[['Season','Year',metal_col]].dropna().copy()
    if tmp.empty:
        return pd.DataFrame()
    add_log1p(tmp, metal_col, 'y')
    model = smf.ols('y ~ C(Season) * C(Year)', data=tmp).fit()
    aov = sm.stats.anova_lm(model, typ=2)  # Type II
    aov = aov.reset_index().rename(columns={'index':'Effect'})
    return aov

# Run PART 1 for all metals
seasonal_kw_all = []
seasonal_posthoc_all = []
two_way_all = []

for m in extended_metals:
    mcol = metal_cols_ext[m]
    kw_df = kruskal_by_year(df, mcol, season_order)
    kw_df.insert(0, 'Metal', m)
    seasonal_kw_all.append(kw_df)

    posthoc_df = pairwise_mwu_by_year(df, mcol, season_order, padjust='holm')
    if not posthoc_df.empty:
        posthoc_df.insert(0, 'Metal', m)
        seasonal_posthoc_all.append(posthoc_df)

    aov_df = two_way_anova_log(df, mcol)
    if not aov_df.empty:
        aov_df.insert(0, 'Metal', m)
        two_way_all.append(aov_df)

seasonal_kw_all = pd.concat(seasonal_kw_all, ignore_index=True) if seasonal_kw_all else pd.DataFrame()
seasonal_posthoc_all = pd.concat(seasonal_posthoc_all, ignore_index=True) if seasonal_posthoc_all else pd.DataFrame()
two_way_all = pd.concat(two_way_all, ignore_index=True) if two_way_all else pd.DataFrame()

# Pretty labels
if not seasonal_kw_all.empty:
    seasonal_kw_all = seasonal_kw_all.sort_values(['Metal','Year'])
if not seasonal_posthoc_all.empty:
    seasonal_posthoc_all = seasonal_posthoc_all.sort_values(['Metal','Year','Season1','Season2'])
if not two_way_all.empty:
    # rename columns for clarity
    two_way_all = two_way_all.rename(columns={'sum_sq':'SS', 'df':'DF', 'F':'F', 'PR(>F)':'pval'})

# Save results
seasonal_kw_all.to_csv('seasonal_kruskal_by_year.csv', index=False)
seasonal_posthoc_all.to_csv('seasonal_pairwise_MWU_by_year_holm.csv', index=False)
two_way_all.to_csv('two_way_anova_log_season_year.csv', index=False)

print("\n=== Seasonal Kruskal–Wallis (by year) ===")
print(seasonal_kw_all.head(20))
print("\n=== Seasonal Pairwise MWU (Holm-adjusted) ===")
print(seasonal_posthoc_all.head(20))
print("\n=== Two-way ANOVA on log1p (Season, Year, Interaction) ===")
print(two_way_all.head(20))


# ============================================================
# PART 2: Diurnal analysis (seasonal diurnal differences)
# Mixed-effects: log1p(metal) ~ C(Season) * C(Hour) + (1 | Date)
# Report LRT p-values for Season, Hour, and Interaction effects.
# ============================================================

from scipy.stats import chi2

def lrt(full_fit, reduced_fit):
    """
    Likelihood-ratio test for nested models.
    Uses chi-square with df = (#params_full - #params_reduced).
    """
    # log-likelihoods
    llf_full = float(getattr(full_fit, "llf", np.nan))
    llf_red  = float(getattr(reduced_fit, "llf", np.nan))

    # robust dof difference:
    # prefer number of free params from results; fall back to len(params)
    def nparams(res):
        k = getattr(res, "df_modelwc", None)
        if k is None:
            k = len(getattr(res, "params", []))
        return int(k)

    k_full = nparams(full_fit)
    k_red  = nparams(reduced_fit)
    df_diff = max(1, k_full - k_red)

    # test statistic (guard tiny negatives from rounding)
    LR = max(0.0, 2.0 * (llf_full - llf_red))
    p = chi2.sf(LR, df_diff)
    return LR, df_diff, p


def mixedlm_diurnal_tests(df, metal_col):
    """
    MixedLM on hourly data with random intercept by Date:
        log1p(metal) ~ C(Season) * C(HourCat)  +  (1 | Date)

    Returns:
        (results_table, full_fit) if success, else (None)
    Notes:
        - Fit with ML (reml=False) so LRTs are valid.
        - Includes try/except fallbacks on optimizer.
    """
    tmp = df[['Season', 'HourCat', 'Date', metal_col]].dropna().copy()
    if tmp.empty or tmp['Season'].isna().all():
        return None

    # random-effects grouping as string to avoid categorical dtype quirks
    tmp['Date'] = pd.to_datetime(tmp['Date'])
    tmp['DateGroup'] = tmp['Date'].astype(str)

    # response transform
    tmp['y'] = np.log1p(tmp[metal_col])

    # helpers to fit robustly
    def _fit(formula):
        # Try lbfgs ML; fallback to nm if needed
        for meth in ('lbfgs', 'nm'):
            try:
                model = smf.mixedlm(formula, data=tmp, groups=tmp['DateGroup'])
                res = model.fit(method=meth, reml=False, disp=False)
                if res.converged:
                    return res
            except Exception:
                pass
        # last resort: return whatever lbfgs gave even if not flagged converged
        model = smf.mixedlm(formula, data=tmp, groups=tmp['DateGroup'])
        return model.fit(method='lbfgs', reml=False, disp=False)

    # Full and reduced models
    full_fit    = _fit('y ~ C(Season) * C(HourCat)')
    no_inter    = _fit('y ~ C(Season) + C(HourCat)')
    no_season   = _fit('y ~ C(HourCat)')
    no_hour     = _fit('y ~ C(Season)')

    # LRTs
    LR_int, df_int, p_int = lrt(full_fit, no_inter)    # interaction
    LR_sea, df_sea, p_sea = lrt(no_inter, no_hour)     # Season main effect
    LR_hou, df_hou, p_hou = lrt(no_inter, no_season)   # Hour main effect

    out = pd.DataFrame([
        {'Effect': 'Season',      'LR': LR_sea, 'df': df_sea, 'pval': p_sea},
        {'Effect': 'Hour',        'LR': LR_hou, 'df': df_hou, 'pval': p_hou},
        {'Effect': 'Season:Hour', 'LR': LR_int, 'df': df_int, 'pval': p_int},
    ])
    return out, full_fit


def per_hour_kruskal(df, metal_col, season_order):
    """At each hour 0–23, Kruskal–Wallis across seasons (hourly resolution)."""
    rows = []
    for h in range(24):
        groups = []
        sizes = []
        for s in season_order:
            vals = df.loc[(df['Hour']==h) & (df['Season']==s), metal_col].dropna()
            groups.append(vals.values)
            sizes.append(len(vals))
        if all(len(g)>=5 for g in groups):
            H, p = kruskal(*groups)
        else:
            H, p = np.nan, np.nan
        rows.append({'Hour':h, 'H':H, 'pval':p,
                     'n_W':sizes[0] if len(sizes)>0 else np.nan,
                     'n_SP':sizes[1] if len(sizes)>1 else np.nan,
                     'n_SU':sizes[2] if len(sizes)>2 else np.nan,
                     'n_F':sizes[3] if len(sizes)>3 else np.nan})
    return pd.DataFrame(rows)

# Run PART 2 for all metals
diurnal_mixed_all = []
diurnal_perhour_all = []

for m in extended_metals:
    mcol = metal_cols_ext[m]

    res = mixedlm_diurnal_tests(df, mcol)
    if res is not None:
        out_table, _full_fit = res
        out_table.insert(0, 'Metal', m)
        diurnal_mixed_all.append(out_table)

    pht = per_hour_kruskal(df, mcol, season_order)
    if pht is not None and not pht.empty:
        pht.insert(0, 'Metal', m)
        diurnal_perhour_all.append(pht)

diurnal_mixed_all = pd.concat(diurnal_mixed_all, ignore_index=True) if diurnal_mixed_all else pd.DataFrame()
diurnal_perhour_all = pd.concat(diurnal_perhour_all, ignore_index=True) if diurnal_perhour_all else pd.DataFrame()

# Save results
diurnal_mixed_all.to_csv('diurnal_mixedlm_LRT_season_hour_interaction.csv', index=False)
diurnal_perhour_all.to_csv('diurnal_perhour_kruskal.csv', index=False)

print("\n=== MixedLM Diurnal Effects (LRT) ===")
print(diurnal_mixed_all)
print("\n=== Per-hour Kruskal–Wallis across seasons ===")
print(diurnal_perhour_all.head(30))


# ============================================================
# HOW TO TALK ABOUT THE RESULTS (cheat-sheet for the paper)
# ============================================================
# - Seasonal differences (by year): report Kruskal–Wallis H and p per metal & year.
#   If significant, cite pairwise MWU (Holm-adjusted) to say which seasons differ.
# - Global Season/Year effects: cite two-way ANOVA (log1p) p-values for Season, Year, and Season×Year.
# - Diurnal: report MixedLM LRT p-values for Season, Hour, and Season×Hour.
#   If Season×Hour significant, you can say "the diurnal pattern differs by season".
# - Optionally plot the diurnal per-hour Kruskal p-values (Hour vs p) to show *when* seasons differ most.

# ============================================================
# Diurnal plotting options: main-text + SI versions
# Assumes you already created:
# df, extended_metals, metal_cols_ext, metal_uncert_pairs,
# season_order, season_names, season_colors
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------- shared helper ----------
season_label_map = dict(zip(season_order, season_names))

def get_valid_diurnal_stats(df, metal, agg="median"):
    """
    Returns hourly stats by season for one metal after 3x uncertainty filter.
    agg = "median" or "mean"
    """
    col = metal_cols_ext[metal]
    uncert_col = metal_uncert_pairs.get(col)

    if uncert_col is None:
        return pd.DataFrame()

    valid = df.loc[df[col] > 3 * df[uncert_col], ['Season', 'Hour', col]].dropna()

    if valid.empty:
        return pd.DataFrame()

    stats = (
        valid.groupby(['Season', 'Hour'])[col]
        .agg(
            median='median',
            mean='mean',
            p10=lambda x: np.percentile(x, 10),
            p90=lambda x: np.percentile(x, 90),
            count='count'
        )
        .reset_index()
    )

    stats['center'] = stats[agg]
    return stats


# ============================================================
# OPTION 1: Main text — median-only seasonal diurnal lines
# ============================================================

def plot_option1_median_only():
    fig, axes = plt.subplots(
        len(extended_metals), 1,
        figsize=(7.2, 8.5),
        sharex=True
    )

    if len(extended_metals) == 1:
        axes = [axes]

    panel_labels = ['a', 'b', 'c', 'd']

    for idx, metal in enumerate(extended_metals):
        ax = axes[idx]
        stats = get_valid_diurnal_stats(df, metal, agg="median")

        for season in season_order:
            sub = stats[stats['Season'] == season]
            if sub.empty:
                continue

            ax.plot(
                sub['Hour'], sub['median'],
                color=season_colors[season],
                linewidth=2.4,
                label=season_label_map[season]
            )

        ax.set_ylabel(f'{metal}\n(ng/m³)', fontsize=11)
        ax.text(
            0.02, 0.90, f'({panel_labels[idx]}) {metal}',
            transform=ax.transAxes,
            fontsize=13, fontweight='bold'
        )
        ax.set_xlim(0, 23)
        ax.set_xticks(range(0, 24, 3))
        ax.grid(True, alpha=0.25, linestyle=':', linewidth=0.6)

    axes[-1].set_xlabel('Hour of Day (Local Time)', fontsize=12)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc='lower center',
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, -0.01)
    )

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.savefig('OPTION1_main_median_only_seasonal_diurnal.png', dpi=300, bbox_inches='tight')
    plt.show()


# ============================================================
# OPTION 2: Main text — overall diurnal pattern only
# One black line per metal, seasons removed
# ============================================================

def plot_option2_overall_diurnal(agg="median"):
    fig, axes = plt.subplots(
        len(extended_metals), 1,
        figsize=(6.5, 8.5),
        sharex=True
    )

    if len(extended_metals) == 1:
        axes = [axes]

    panel_labels = ['a', 'b', 'c', 'd']

    for idx, metal in enumerate(extended_metals):
        ax = axes[idx]
        col = metal_cols_ext[metal]
        uncert_col = metal_uncert_pairs.get(col)

        valid = df.loc[df[col] > 3 * df[uncert_col], ['Hour', col]].dropna()

        if agg == "mean":
            hourly = valid.groupby('Hour')[col].mean()
            label = 'Mean'
        else:
            hourly = valid.groupby('Hour')[col].median()
            label = 'Median'

        ax.plot(
            hourly.index, hourly.values,
            color='black',
            linewidth=2.6
        )

        ax.set_ylabel(f'{metal}\n(ng/m³)', fontsize=11)
        ax.text(
            0.02, 0.90, f'({panel_labels[idx]}) {metal}',
            transform=ax.transAxes,
            fontsize=13, fontweight='bold'
        )
        ax.grid(True, alpha=0.25, linestyle=':', linewidth=0.6)
        ax.set_xlim(0, 23)
        ax.set_xticks(range(0, 24, 3))

    axes[-1].set_xlabel('Hour of Day (Local Time)', fontsize=12)
    fig.suptitle(f'Overall {label} Diurnal Patterns', fontsize=14, fontweight='bold')

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(f'OPTION2_main_overall_{agg}_diurnal.png', dpi=300, bbox_inches='tight')
    plt.show()


# ============================================================
# OPTION 3: Main text — summer-only diurnal pattern
# Shows summer patterns for Fe, K, Ca, Zn
# ============================================================

def plot_option3_summer_only():
    fig, axes = plt.subplots(
        2, 2,
        figsize=(8, 5.8),
        sharex=True
    )

    axes = axes.flatten()
    panel_labels = ['a', 'b', 'c', 'd']

    for idx, metal in enumerate(extended_metals):
        ax = axes[idx]
        stats = get_valid_diurnal_stats(df, metal, agg="median")
        sub = stats[stats['Season'] == 'SU']

        ax.plot(
            sub['Hour'], sub['median'],
            color=season_colors['SU'],
            linewidth=2.6
        )

        ax.set_title(f'({panel_labels[idx]}) {metal}', fontsize=13, fontweight='bold')
        ax.set_ylabel('ng/m³')
        ax.set_xlim(0, 23)
        ax.set_xticks(range(0, 24, 3))
        ax.grid(True, alpha=0.25, linestyle=':', linewidth=0.6)

    for ax in axes[-2:]:
        ax.set_xlabel('Hour of Day (Local Time)')

    plt.tight_layout()
    plt.savefig('OPTION3_main_summer_only_diurnal.png', dpi=300, bbox_inches='tight')
    plt.show()


# ============================================================
# SI FIGURE: detailed seasonal diurnal with median + 10/90 range
# Cleaner version of your current figure
# ============================================================

def plot_SI_detailed_percentile():
    fig, axes = plt.subplots(
        len(extended_metals), 1,
        figsize=(8.2, 9.5),
        sharex=True
    )

    if len(extended_metals) == 1:
        axes = [axes]

    panel_labels = ['a', 'b', 'c', 'd']

    for idx, metal in enumerate(extended_metals):
        ax = axes[idx]
        stats = get_valid_diurnal_stats(df, metal, agg="median")

        for season in season_order:
            sub = stats[stats['Season'] == season]
            if sub.empty:
                continue

            ax.plot(
                sub['Hour'], sub['median'],
                color=season_colors[season],
                linewidth=2.2,
                label=season_label_map[season]
            )

            ax.fill_between(
                sub['Hour'],
                sub['p10'],
                sub['p90'],
                color=season_colors[season],
                alpha=0.13,
                linewidth=0
            )

        ax.set_ylabel(f'{metal}\n(ng/m³)', fontsize=11)
        ax.text(
            0.02, 0.90, f'({panel_labels[idx]}) {metal}',
            transform=ax.transAxes,
            fontsize=13, fontweight='bold'
        )
        ax.set_xlim(0, 23)
        ax.set_xticks(range(0, 24, 3))
        ax.grid(True, alpha=0.25, linestyle=':', linewidth=0.6)

    axes[-1].set_xlabel('Hour of Day (Local Time)', fontsize=12)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc='lower center',
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, -0.01)
    )

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.savefig('SI_detailed_median_p10_p90_seasonal_diurnal.png', dpi=300, bbox_inches='tight')
    plt.show()


# ============================================================
# Run all figures
# ============================================================

plot_option1_median_only()
plot_option2_overall_diurnal(agg="median")   # use "mean" only if you really want average
plot_option3_summer_only()
plot_SI_detailed_percentile()