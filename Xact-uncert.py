import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import re

# ================= USER SETTINGS =================
INPUT_FILE = Path(r"Xact_EST_May2023_Oct2025_combined.csv")
OUTPUT_DIR = Path(r"Xact_uncertainty_QA_prePMF")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TIME_COL = "TIME"

EXCLUDE_PERIODS = [
    ("instrument_down_2024_winter", "2024-01-09 00:00", "2024-02-13 23:59"),
    ("instrument_down_2024_summer", "2024-07-02 00:00", "2024-08-08 23:59"),
]

EXCLUDE_SINGLE_TIMES = [
    # "2023-07-10 10:00",
]

MISSING_CODES = [-999, -999.0, -9999, -9999.0]

# Use the 46 species retained in your study.
# Set ELEMENTS_TO_INCLUDE = None if you want to plot every concentration/uncertainty pair found.
ELEMENTS_TO_INCLUDE = [
    "Al", "Si", "P", "S", "Cl", "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn",
    "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Rb",
    "Sr", "Y", "Zr", "Mo", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te",
    "I", "Cs", "Ba", "La", "Ce", "W", "Pt", "Au", "Hg", "Tl", "Pb", "Bi"
]

# Bigger figure fonts
FONT_SIZE = 16
TICK_SIZE = 14
LEGEND_SIZE = 14

# Plot settings
ADD_75_LINE = True
LEGEND_BBOX = (0.98, 1.03)   # lower second number to move legend down
# ==================================================


def clean_colname(c):
    """Normalize spacing in column names."""
    return re.sub(r"\s+", " ", str(c).strip())


def extract_element_from_col(col):
    """
    Extract element symbol from columns like:
    'Fe 26 (ng/m3)', 'Fe Uncert (ng/m3)', 'Fe'
    """
    col = clean_colname(col)
    m = re.match(r"^([A-Z][a-z]?)\b", col)
    return m.group(1) if m else None


def is_concentration_col(col):
    """Identify concentration columns, not uncertainty columns."""
    col_clean = clean_colname(col)
    col_lower = col_clean.lower()

    if "uncert" in col_lower or "unc" in col_lower:
        return False

    return "(ng/m3)" in col_lower or "ng/m3" in col_lower


def is_uncertainty_col(col):
    """Identify uncertainty columns."""
    col_clean = clean_colname(col)
    col_lower = col_clean.lower()

    return "uncert" in col_lower or "unc" in col_lower


def find_conc_uncert_pairs(df, elements_to_include=None):
    """
    Pair concentration columns with uncertainty columns by element symbol.
    Handles columns like:
      Al 13 (ng/m3)      and Al Uncert (ng/m3)
      Mg 12 (ng/m3)      and Mg uncert (ng/m3)
      Nb 41(ng/m3)       and Nb Uncert (ng/m3)
    """
    conc_cols = {}
    uncert_cols = {}

    for col in df.columns:
        col_clean = clean_colname(col)
        elem = extract_element_from_col(col_clean)

        if elem is None:
            continue

        if elements_to_include is not None and elem not in elements_to_include:
            continue

        if is_uncertainty_col(col_clean):
            uncert_cols[elem] = col_clean
        elif is_concentration_col(col_clean):
            conc_cols[elem] = col_clean

    pairs = {}
    for elem in sorted(conc_cols.keys()):
        if elem in uncert_cols:
            pairs[elem] = {
                "conc_col": conc_cols[elem],
                "uncert_col": uncert_cols[elem],
            }

    missing_uncert = sorted([e for e in conc_cols if e not in uncert_cols])
    missing_conc = sorted([e for e in uncert_cols if e not in conc_cols])

    return pairs, missing_uncert, missing_conc


def load_and_filter_data():
    df = pd.read_csv(INPUT_FILE)
    df.columns = [clean_colname(c) for c in df.columns]

    if TIME_COL not in df.columns:
        raise ValueError(
            f"Cannot find TIME_COL = {TIME_COL}. "
            f"Available columns include: {df.columns[:10].tolist()}"
        )

    # Parse timezone-aware timestamps safely, e.g. 2023-05-02 12:00:00-04:00
    df[TIME_COL] = pd.to_datetime(df[TIME_COL], errors="coerce", utc=True)

    # Convert to Eastern local time and drop timezone
    df[TIME_COL] = df[TIME_COL].dt.tz_convert("America/New_York").dt.tz_localize(None)

    # Drop rows with invalid/unparseable time
    df = df.loc[df[TIME_COL].notna()].copy()

    print(f"Rows before exclusions: {len(df)}")

    keep = pd.Series(True, index=df.index)

    for name, start, end in EXCLUDE_PERIODS:
        start = pd.to_datetime(start)
        end = pd.to_datetime(end)
        in_period = (df[TIME_COL] >= start) & (df[TIME_COL] <= end)
        print(f"Excluding {name}: {in_period.sum()} rows")
        keep &= ~in_period

    for t in EXCLUDE_SINGLE_TIMES:
        t = pd.to_datetime(t)
        in_time = df[TIME_COL] == t
        print(f"Excluding single time {t}: {in_time.sum()} rows")
        keep &= ~in_time

    df = df.loc[keep].copy()
    print(f"Rows after exclusions: {len(df)}")

    return df


def calc_uncertainty_summary(df, pairs, multiplier):
    """
    Calculate above/below/missing percentages for 1x or 3x paired uncertainty.

    Denominator = all hourly rows after downtime/outlier exclusions.
    Above = concentration >= multiplier * uncertainty.
    Below = concentration < multiplier * uncertainty.
    Missing/invalid = concentration missing OR uncertainty missing OR uncertainty <= 0.
    """
    rows = []
    n_total_hours = len(df)

    for elem, cols in pairs.items():
        conc_col = cols["conc_col"]
        uncert_col = cols["uncert_col"]

        conc = pd.to_numeric(df[conc_col], errors="coerce").replace(MISSING_CODES, np.nan)
        uncert = pd.to_numeric(df[uncert_col], errors="coerce").replace(MISSING_CODES, np.nan)

        # Treat uncertainty <= 0 as invalid for uncertainty-based thresholding
        valid_pair = conc.notna() & uncert.notna() & (uncert > 0)

        threshold = multiplier * uncert

        n_valid_pair = valid_pair.sum()
        n_missing_or_invalid = n_total_hours - n_valid_pair

        n_above = ((conc >= threshold) & valid_pair).sum()
        n_below = ((conc < threshold) & valid_pair).sum()

        pct_above = 100 * n_above / n_total_hours if n_total_hours > 0 else np.nan
        pct_below = 100 * n_below / n_total_hours if n_total_hours > 0 else np.nan
        pct_missing_or_invalid = 100 * n_missing_or_invalid / n_total_hours if n_total_hours > 0 else np.nan

        rows.append({
            "Element": elem,
            "Concentration_column": conc_col,
            "Uncertainty_column": uncert_col,
            "Threshold_multiplier": multiplier,
            "n_total_hours_after_exclusions": n_total_hours,
            "n_valid_conc_uncert_pairs": n_valid_pair,
            "n_missing_or_invalid_pairs": n_missing_or_invalid,
            "n_above_threshold": n_above,
            "n_below_threshold": n_below,
            "pct_above_threshold_all_hours": pct_above,
            "pct_below_threshold_all_hours": pct_below,
            "pct_missing_or_invalid_all_hours": pct_missing_or_invalid,
            "median_conc_nonmissing_ng_m3": conc.median(skipna=True),
            "mean_conc_nonmissing_ng_m3": conc.mean(skipna=True),
            "max_conc_nonmissing_ng_m3": conc.max(skipna=True),
            "median_uncert_valid_ng_m3": uncert.where(uncert > 0).median(skipna=True),
            "mean_uncert_valid_ng_m3": uncert.where(uncert > 0).mean(skipna=True),
        })

    summary = pd.DataFrame(rows)
    summary = summary.sort_values(
        "pct_above_threshold_all_hours",
        ascending=False
    ).reset_index(drop=True)

    return summary


def plot_above_below_uncertainty(summary, multiplier, output_dir):
    label = f"{multiplier}× reported uncertainty"

    elements = summary["Element"].to_numpy()
    above = summary["pct_above_threshold_all_hours"].to_numpy()
    below = -summary["pct_below_threshold_all_hours"].to_numpy()

    fig, ax = plt.subplots(figsize=(17, 7))

    ax.bar(elements, above, color="black", label=f"Above {label}")
    ax.bar(elements, below, color="gray", label=f"Below {label}")

    ax.axhline(0, color="black", linewidth=1.0)

    if ADD_75_LINE:
        ax.axhline(75, color="black", linestyle="--", linewidth=1.0)
        ax.text(
            len(elements) - 1.2,
            71,
            "75%",
            va="top",
            ha="right",
            fontsize=TICK_SIZE
        )

    ax.set_ylabel("Hourly sample periods (%)", fontsize=FONT_SIZE, labelpad=12)
    ax.set_xlabel("Element", fontsize=FONT_SIZE, labelpad=8)

    ax.set_xticks(range(len(elements)))
    ax.set_xticklabels(elements, rotation=60, ha="right", fontsize=TICK_SIZE)
    ax.tick_params(axis="y", labelsize=TICK_SIZE)

    ax.set_ylim(-105, 105)

    ax.legend(
        frameon=False,
        fontsize=LEGEND_SIZE,
        loc="upper right",
        bbox_to_anchor=LEGEND_BBOX
    )

    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.25, top=0.86)

    safe_label = f"{multiplier}x_uncertainty"

    png_out = output_dir / f"prePMF_species_above_below_{safe_label}_all_hours_denominator.png"
    pdf_out = output_dir / f"prePMF_species_above_below_{safe_label}_all_hours_denominator.pdf"

    fig.savefig(png_out, dpi=300, bbox_inches="tight", pad_inches=0.15)
    fig.savefig(pdf_out, bbox_inches="tight", pad_inches=0.15)
    plt.show()

    print(f"Saved PNG: {png_out}")
    print(f"Saved PDF: {pdf_out}")


# ==================================================
# Main workflow
# ==================================================

df = load_and_filter_data()

pairs, missing_uncert, missing_conc = find_conc_uncert_pairs(
    df,
    elements_to_include=ELEMENTS_TO_INCLUDE
)

print(f"\nNumber of paired species found: {len(pairs)}")

if missing_uncert:
    print("\nWARNING: Concentration columns found without uncertainty columns:")
    print(missing_uncert)

if missing_conc:
    print("\nWARNING: Uncertainty columns found without concentration columns:")
    print(missing_conc)

print("\nMatched concentration/uncertainty pairs:")
for elem, cols in pairs.items():
    print(f"{elem}: {cols['conc_col']}  |  {cols['uncert_col']}")

# Calculate 1x and 3x uncertainty summaries
summary_1x = calc_uncertainty_summary(df, pairs, multiplier=1)
summary_3x = calc_uncertainty_summary(df, pairs, multiplier=3)

# Save summaries
summary_1x_csv = OUTPUT_DIR / "prePMF_species_above_below_1x_uncertainty_summary_all_hours_denominator.csv"
summary_3x_csv = OUTPUT_DIR / "prePMF_species_above_below_3x_uncertainty_summary_all_hours_denominator.csv"

summary_1x.to_csv(summary_1x_csv, index=False)
summary_3x.to_csv(summary_3x_csv, index=False)

print(f"\nSaved 1x uncertainty summary: {summary_1x_csv}")
print(f"Saved 3x uncertainty summary: {summary_3x_csv}")

combined_summary = pd.concat(
    [
        summary_1x.assign(Threshold_type="1x_reported_uncertainty"),
        summary_3x.assign(Threshold_type="3x_reported_uncertainty"),
    ],
    ignore_index=True
)

combined_csv = OUTPUT_DIR / "prePMF_species_above_below_1x_and_3x_uncertainty_combined_summary.csv"
combined_summary.to_csv(combined_csv, index=False)

print(f"Saved combined summary: {combined_csv}")

# Make plots
plot_above_below_uncertainty(summary_1x, multiplier=1, output_dir=OUTPUT_DIR)
plot_above_below_uncertainty(summary_3x, multiplier=3, output_dir=OUTPUT_DIR)