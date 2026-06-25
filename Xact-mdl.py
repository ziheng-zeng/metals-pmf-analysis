import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import re

# ================= USER SETTINGS =================
INPUT_FILE = Path(r"Xact_EST_May2023_Oct2025_combined.csv")
OUTPUT_DIR = Path(r"Xact_MDL_QA_prePMF")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TIME_COL = "TIME"

EXCLUDE_PERIODS = [
    ("instrument_down_2024_winter", "2024-01-09 00:00", "2024-02-13 23:59"),
    ("instrument_down_2024_summer", "2024-07-02 00:00", "2024-08-08 23:59"),
]

EXCLUDE_SINGLE_TIMES = [
    # "2023-07-10 10:00",
]

# Existing missing codes, if present in the original file
MISSING_CODES = [-999, -999.0, -9999, -9999.0]

# Bigger figure fonts
FONT_SIZE = 16
TITLE_SIZE = 18
TICK_SIZE = 14
LEGEND_SIZE = 14
# ==================================================


# Published MDLs for 29 elemental species, ng/m3
MDL = {
    "Al": 100,
    "Si": 17.8,
    "P": 5.2,
    "S": 3.16,
    "Cl": 1.73,
    "K": 1.17,
    "Ca": 0.3,
    "Ti": 0.16,
    "V": 0.12,
    "Cr": 0.12,
    "Mn": 0.14,
    "Fe": 0.17,
    "Co": 0.14,
    "Ni": 0.10,
    "Cu": 0.079,
    "Zn": 0.067,
    "As": 0.063,
    "Se": 0.081,
    "Br": 0.10,
    "Ag": 1.9,
    "Cd": 2.5,
    "In": 3.1,
    "Sn": 4.1,
    "Sb": 5.2,
    "Ba": 0.39,
    "Hg": 0.12,
    "Tl": 0.12,
    "Pb": 0.13,
    "Bi": 0.13,
}


def clean_colname(c):
    """Normalize spacing in column names."""
    return re.sub(r"\s+", " ", str(c).strip())


def extract_element_from_col(col):
    """
    Extract element symbol from columns like:
    'Fe 26 (ng/m3)', 'Fe (ng/m3)', or 'Fe'
    """
    col = clean_colname(col)
    m = re.match(r"^([A-Z][a-z]?)\b", col)
    return m.group(1) if m else None


def find_concentration_columns(df):
    """
    Build mapping from element symbol to concentration column.
    Avoid uncertainty columns.
    """
    element_to_col = {}

    for col in df.columns:
        col_clean = clean_colname(col)
        col_lower = col_clean.lower()

        # Skip uncertainty columns
        if "uncert" in col_lower or "unc" in col_lower:
            continue

        # Keep concentration columns like "Fe 26 (ng/m3)"
        if "(ng/m3)" not in col_lower and "ng/m3" not in col_lower:
            # Also allow simple columns like "Fe"
            if col_clean not in MDL:
                continue

        elem = extract_element_from_col(col_clean)

        if elem in MDL:
            element_to_col[elem] = col_clean

    return element_to_col


def calc_threshold_summary(df, element_to_col, multiplier):
    """
    Calculate above/below/missing percentages for either 1x MDL or 3x MDL.

    Denominator = all hourly rows after downtime/outlier exclusions.
    Zeros count as valid measurements below threshold.
    Blanks/NA/-999 count as missing, but remain in denominator.
    """
    rows = []
    n_total_hours = len(df)

    for elem, mdl_value in MDL.items():
        if elem not in element_to_col:
            continue

        col = element_to_col[elem]
        threshold = multiplier * mdl_value

        s = pd.to_numeric(df[col], errors="coerce")
        s = s.replace(MISSING_CODES, np.nan)

        n_missing = s.isna().sum()
        n_nonmissing = s.notna().sum()

        n_above = (s >= threshold).sum()
        n_below = ((s < threshold) & s.notna()).sum()

        pct_above = 100 * n_above / n_total_hours if n_total_hours > 0 else np.nan
        pct_below = 100 * n_below / n_total_hours if n_total_hours > 0 else np.nan
        pct_missing = 100 * n_missing / n_total_hours if n_total_hours > 0 else np.nan

        rows.append({
            "Element": elem,
            "Column": col,
            "Published_MDL_ng_m3": mdl_value,
            "Threshold_multiplier": multiplier,
            "Threshold_ng_m3": threshold,
            "n_total_hours_after_exclusions": n_total_hours,
            "n_nonmissing": n_nonmissing,
            "n_missing": n_missing,
            "n_above_threshold": n_above,
            "n_below_threshold": n_below,
            "pct_above_threshold_all_hours": pct_above,
            "pct_below_threshold_all_hours": pct_below,
            "pct_missing_all_hours": pct_missing,
            "median_nonmissing_ng_m3": s.median(skipna=True),
            "mean_nonmissing_ng_m3": s.mean(skipna=True),
            "max_nonmissing_ng_m3": s.max(skipna=True),
        })

    summary = pd.DataFrame(rows)
    summary = summary.sort_values(
        "pct_above_threshold_all_hours",
        ascending=False
    ).reset_index(drop=True)

    return summary


def plot_above_below(summary, multiplier, output_dir):
    label = f"{multiplier}× published MDL" if multiplier != 1 else "published MDL"

    elements = summary["Element"].to_numpy()
    above = summary["pct_above_threshold_all_hours"].to_numpy()
    below = -summary["pct_below_threshold_all_hours"].to_numpy()

    fig, ax = plt.subplots(figsize=(16, 7))

    ax.bar(elements, above, color="black", label=f"Above {label}")
    ax.bar(elements, below, color="gray", label=f"Below {label}")

    ax.axhline(0, color="black", linewidth=1.0)

    # 75% reference line
    ax.axhline(75, color="black", linestyle="--", linewidth=1.0)

    # 75% text, shifted away from legend
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

    # Legend stays top right but slightly above plotting area
    ax.legend(
        frameon=False,
        fontsize=LEGEND_SIZE,
        loc="upper right",
        bbox_to_anchor=(0.98, 1.03)
    )

    # Push plotting area slightly downward to make room for legend
    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.24, top=0.82)

    safe_label = "1xMDL" if multiplier == 1 else f"{multiplier}xMDL"

    png_out = output_dir / f"prePMF_29_species_above_below_{safe_label}_all_hours_denominator.png"
    pdf_out = output_dir / f"prePMF_29_species_above_below_{safe_label}_all_hours_denominator.pdf"

    fig.savefig(png_out, dpi=300, bbox_inches="tight", pad_inches=0.15)
    fig.savefig(pdf_out, bbox_inches="tight", pad_inches=0.15)
    plt.show()

    print(f"Saved PNG: {png_out}")
    print(f"Saved PDF: {pdf_out}")


# ==================================================
# Main workflow
# ==================================================

# ---------- Load original/pre-PMF data ----------
df = pd.read_csv(INPUT_FILE)
df.columns = [clean_colname(c) for c in df.columns]

if TIME_COL not in df.columns:
    raise ValueError(
        f"Cannot find TIME_COL = {TIME_COL}. "
        f"Available columns include: {df.columns[:10].tolist()}"
    )

# Parse TIME safely with timezone offsets like 2023-05-02 12:00:00-04:00
df[TIME_COL] = pd.to_datetime(df[TIME_COL], errors="coerce", utc=True)

# Convert to Eastern local time and drop timezone
df[TIME_COL] = df[TIME_COL].dt.tz_convert("America/New_York").dt.tz_localize(None)

# Drop rows with invalid/unparseable time
df = df.loc[df[TIME_COL].notna()].copy()

print(f"Rows before exclusions: {len(df)}")

# ---------- Exclude downtime periods ----------
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

# ---------- Find concentration columns ----------
element_to_col = find_concentration_columns(df)

missing_mdl_cols = [e for e in MDL if e not in element_to_col]
if missing_mdl_cols:
    print("\nWARNING: These MDL elements were not found as concentration columns:")
    print(missing_mdl_cols)

print("\nMatched concentration columns:")
for e, c in element_to_col.items():
    print(f"{e}: {c}")

# ---------- Calculate summaries ----------
summary_1x = calc_threshold_summary(df, element_to_col, multiplier=1)
summary_3x = calc_threshold_summary(df, element_to_col, multiplier=3)

# Save individual summaries
summary_1x_csv = OUTPUT_DIR / "prePMF_29_species_above_below_1xMDL_summary_all_hours_denominator.csv"
summary_3x_csv = OUTPUT_DIR / "prePMF_29_species_above_below_3xMDL_summary_all_hours_denominator.csv"

summary_1x.to_csv(summary_1x_csv, index=False)
summary_3x.to_csv(summary_3x_csv, index=False)

print(f"\nSaved 1x MDL summary: {summary_1x_csv}")
print(f"Saved 3x MDL summary: {summary_3x_csv}")

# Save combined summary
combined_summary = pd.concat(
    [
        summary_1x.assign(Threshold_type="1x_MDL"),
        summary_3x.assign(Threshold_type="3x_MDL"),
    ],
    ignore_index=True
)

combined_csv = OUTPUT_DIR / "prePMF_29_species_above_below_1x_and_3xMDL_combined_summary.csv"
combined_summary.to_csv(combined_csv, index=False)

print(f"Saved combined summary: {combined_csv}")

# ---------- Plot both ----------
plot_above_below(summary_1x, multiplier=1, output_dir=OUTPUT_DIR)
plot_above_below(summary_3x, multiplier=3, output_dir=OUTPUT_DIR)