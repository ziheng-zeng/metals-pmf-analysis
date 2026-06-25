import pandas as pd
import re
from pathlib import Path

# ============================================================
# USER SETTINGS
# ============================================================
input_file = Path("Xact_EST_May2023_Oct2025_combined.csv")
output_dir = Path("pmf_ready")
output_prefix = "Xact_PMF"

time_col = "TIME"
local_tz = "America/New_York"

output_dir.mkdir(exist_ok=True)

MISSING_VALUE = -999.0

# Use simple SampleID instead of timestamp in EPA PMF input files.
# Timestamps are saved separately in *_sample_key.csv.
USE_SAMPLE_ID_IN_PMF = True
ID_COL = "SampleID"

EXPORT_MASTER_TRACKING_FILE = True

# Detection limits, ng/m3
element_to_dl_full = {
    'Al 13 (ng/m3)': 100,
    'Si 14 (ng/m3)': 17.8,
    'P 15 (ng/m3)': 5.2,
    'S 16 (ng/m3)': 3.16,
    'Cl 17 (ng/m3)': 1.73,
    ' K 19 (ng/m3)': 1.17,
    'Ca 20 (ng/m3)': 0.3,
    'Ti 22 (ng/m3)': 0.16,
    'V 23 (ng/m3)': 0.12,
    'Cr 24 (ng/m3)': 0.12,
    'Mn 25 (ng/m3)': 0.14,
    'Fe 26 (ng/m3)': 0.17,
    'Co 27 (ng/m3)': 0.14,
    'Ni 28 (ng/m3)': 0.1,
    'Cu 29 (ng/m3)': 0.079,
    'Zn 30 (ng/m3)': 0.067,
    'As 33 (ng/m3)': 0.063,
    'Se 34 (ng/m3)': 0.081,
    'Br 35 (ng/m3)': 0.1,
    'Ag 47 (ng/m3)': 1.9,
    'Cd 48 (ng/m3)': 2.5,
    'In 49 (ng/m3)': 3.1,
    'Sn 50 (ng/m3)': 4.1,
    'Sb 51 (ng/m3)': 5.2,
    'Ba 56 (ng/m3)': 0.39,
    'Hg 80 (ng/m3)': 0.12,
    'Tl 81 (ng/m3)': 0.12,
    'Pb 82 (ng/m3)': 0.13,
    'Bi 83 (ng/m3)': 0.13,
}

# Full-data exclusions
EXCLUDE_PERIODS = [
    ("instrument_down_2024_winter",    "2024-01-09 00:00", "2024-02-13 23:59"),
    ("instrument_down_2024_summer",    "2024-07-02 00:00", "2024-08-08 23:59"),
    ("single_outlier_2023_07_10_10am", "2023-07-10 10:00", "2023-07-10 10:00"),
]

# Separate PMF event windows
PMF_WINDOWS = {
    "mixed_june_july_2023":   ("2023-06-25 00:00", "2023-07-10 23:59"),
    "fireworks_2023":         ("2023-07-01 00:00", "2023-07-06 23:59"),
    "wildfire_2023_major":    ("2023-06-28 00:00", "2023-06-30 23:59"),
    "background_summer_2023": ("2023-08-10 00:00", "2023-08-17 23:59"),
}

# Combined coke-event file
COKE_EVENTS = {
    "coke_2024_10_22": ("2024-10-22 23:00", "2024-10-23 09:00"),
    "coke_2025_03_11": ("2025-03-11 03:00", "2025-03-11 09:00"),
    "coke_2025_03_18": ("2025-03-18 23:00", "2025-03-19 11:00"),
    "coke_2025_03_10": ("2025-03-10 16:00", "2025-03-11 15:00"),
    "coke_2024_09_08": ("2024-09-08 16:00", "2024-09-09 15:00"),
    "coke_2024_10_04": ("2024-10-04 16:00", "2024-10-05 15:00"),
    "coke_2023_09_05": ("2023-09-05 16:00", "2023-09-06 15:00"),
}

# Species to completely remove from PMF
EXCLUDED_SPECIES = [
    "Nb",
]


# ============================================================
# HELPER FUNCTIONS
# ============================================================
def parse_local_time(series, local_tz):
    """
    Handles mixed UTC offsets:
      2023-05-02 12:00:00-04:00
      2023-12-02 12:00:00-05:00
    """
    t = pd.to_datetime(series, errors="coerce", utc=True)
    t = t.dt.tz_convert(local_tz)
    return t


def clean_metal_name(col):
    """
    Examples:
      'Mg 12 (ng/m3)' -> 'Mg'
      'Nb 41(ng/m3)'  -> 'Nb'
      'Al 13 (ng/m3)' -> 'Al'
    """
    col = str(col).strip()
    col = re.sub(r"\s*\(ng/m3\)", "", col, flags=re.IGNORECASE)
    col = re.sub(r"\s+\d+$", "", col)
    col = re.sub(r"\d+$", "", col)
    return col.strip()


MDL_BY_METAL = {
    clean_metal_name(col): dl
    for col, dl in element_to_dl_full.items()
}


def is_conc_col(col):
    col_lower = str(col).lower()
    return (
        "(ng/m3)" in col_lower
        and "uncert" not in col_lower
        and col != time_col
    )


def find_uncert_col(conc_col, all_cols):
    metal = clean_metal_name(conc_col).lower()

    for c in all_cols:
        c_lower = str(c).strip().lower()
        if "uncert" in c_lower and c_lower.startswith(metal + " "):
            return c

    return None


def get_conc_uncert_pairs(df):
    all_cols = list(df.columns)
    conc_cols = [c for c in all_cols if is_conc_col(c)]

    pairs = []
    missing_uncert = []

    for conc_col in conc_cols:
        uncert_col = find_uncert_col(conc_col, all_cols)
        metal = clean_metal_name(conc_col)

        if metal in EXCLUDED_SPECIES:
            print(f"  Skipping excluded species: {metal}")
            continue

        if uncert_col is None:
            missing_uncert.append(conc_col)
        else:
            pairs.append((metal, conc_col, uncert_col))

    if missing_uncert:
        print("\n  WARNING: These concentration columns had no matching uncertainty column:")
        for c in missing_uncert:
            print("    ", c)

    print(f"  Found {len(pairs)} concentration/uncertainty species pairs.")
    return pairs


def make_sample_key(df, label, sample_ids):
    sample_key = pd.DataFrame({
        ID_COL: sample_ids,
        "Dataset_Label": label,
        "Master_Row": df["Master_Row"].to_numpy() if "Master_Row" in df.columns else "",
        "Date_Local": df[time_col].dt.strftime("%Y-%m-%d %H:%M:%S").to_numpy(),
        "Date_UTC": df[time_col].dt.tz_convert("UTC").dt.strftime("%Y-%m-%d %H:%M:%S").to_numpy(),
    })

    if "event_name" in df.columns:
        sample_key["Event_Name"] = df["event_name"].to_numpy()

    return sample_key


def make_pmf_files(df, label):
    """
    Produces:
      {label}_conc.csv
      {label}_unc.csv
      {label}_sample_key.csv
      {label}_species_mapping.csv

    PMF files use SampleID instead of timestamp.
    Timestamps are preserved in sample_key.
    """
    if df.empty:
        print(f"\n  WARNING: {label} is empty — no files exported.")
        return

    # CRITICAL: reset index so pandas does not align sliced Series by old row index.
    df = df.copy().reset_index(drop=True)

    pairs = get_conc_uncert_pairs(df)

    if not pairs:
        print(f"\n  WARNING: {label} has no valid species pairs — no files exported.")
        return

    # Use stable master-row IDs so sliced files keep their original master-file identity.
    if "Master_Row" in df.columns:
        sample_ids = df["Master_Row"].astype(int).to_list()
    else:
        sample_ids = list(range(1, len(df) + 1))

    if USE_SAMPLE_ID_IN_PMF:
        first_col_name = ID_COL
        first_col_values = sample_ids
    else:
        first_col_name = "Date"
        first_col_values = df[time_col].dt.strftime("%m/%d/%Y %H:%M").to_list()

    conc_out = pd.DataFrame({first_col_name: first_col_values})
    unc_out = pd.DataFrame({first_col_name: first_col_values})

    kept_pairs = []
    mdl_sub_counts = {}

    for metal, conc_col, uncert_col in pairs:
        conc = pd.to_numeric(df[conc_col], errors="coerce")
        unc = pd.to_numeric(df[uncert_col], errors="coerce")

        true_missing = conc.isna() | unc.isna()

        # Xact real non-detect / very low result
        zero_zero = (conc == 0) & (unc == 0)
        zero_zero_count = int(zero_zero.sum())

        # PMF cannot use zero/negative uncertainty
        bad_uncert = (conc > 0) & (unc <= 0)

        if metal in MDL_BY_METAL:
            mdl = MDL_BY_METAL[metal]
            mdl_sub_counts[metal] = zero_zero_count

            # true missing stays missing
            conc = conc.mask(true_missing, MISSING_VALUE)
            unc = unc.mask(true_missing, MISSING_VALUE)

            # 0/0 non-detect becomes EPA-style BDL substitution
            conc = conc.mask(zero_zero, mdl / 2)
            unc = unc.mask(zero_zero, (5 / 6) * mdl)

            # positive conc with bad uncertainty gets uncertainty floor
            unc = unc.mask(bad_uncert, (5 / 6) * mdl)

        else:
            # no MDL available, so zero/negative uncertainty becomes missing
            bad = true_missing | (unc <= 0)
            conc = conc.mask(bad, MISSING_VALUE)
            unc = unc.mask(bad, MISSING_VALUE)

        # CRITICAL: use .to_numpy() to avoid index alignment bugs
        conc_out[metal] = conc.to_numpy()
        unc_out[metal] = unc.to_numpy()
        kept_pairs.append((metal, conc_col, uncert_col))

    print("\n  MDL substitution check:")
    any_subs = False
    for metal, n in mdl_sub_counts.items():
        if n > 0:
            any_subs = True
            mdl = MDL_BY_METAL[metal]
            print(
                f"    {metal}: {n} rows changed from 0/0 "
                f"to conc={mdl / 2:.4f}, unc={(5 / 6) * mdl:.4f}"
            )

    if not any_subs:
        print("    No 0/0 rows found for species with MDLs.")

    conc_out = conc_out.fillna(MISSING_VALUE)
    unc_out = unc_out.fillna(MISSING_VALUE)

    # --------------------------------------------------------
    # Missing summary + remove fully missing species
    # --------------------------------------------------------
    drop_species = []

    print("\n  Species missing summary before dropping:")
    for col in conc_out.columns:
        if col == first_col_name:
            continue

        valid_count = int((conc_out[col] != MISSING_VALUE).sum())
        missing_frac = float((conc_out[col] == MISSING_VALUE).mean())

        print(f"    {col}: valid={valid_count}, missing={missing_frac:.1%}")

        if missing_frac == 1.0:
            drop_species.append(col)

    if drop_species:
        print(f"\n  Removing {len(drop_species)} species that are entirely missing:")
        print("   ", drop_species)

        conc_out = conc_out.drop(columns=drop_species)
        unc_out = unc_out.drop(columns=drop_species)

        kept_pairs = [
            pair for pair in kept_pairs
            if pair[0] not in drop_species
        ]

    if len(conc_out.columns) <= 1:
        raise ValueError(
            f"{label}: all species were dropped, leaving only {first_col_name}. "
            "This means all species became -999 or the conc/uncert pairing failed."
        )

    print(f"\n  Final PMF columns ({len(conc_out.columns)}):")
    print("   ", list(conc_out.columns))

    # --------------------------------------------------------
    # Consistency checks
    # --------------------------------------------------------
    assert len(conc_out) == len(unc_out), (
        f"Row count mismatch: {len(conc_out)} vs {len(unc_out)}"
    )

    assert list(conc_out.columns) == list(unc_out.columns), (
        "Column lists differ between concentration and uncertainty files"
    )

    assert (
        conc_out[first_col_name].astype(str).to_numpy()
        == unc_out[first_col_name].astype(str).to_numpy()
    ).all(), f"{first_col_name} mismatch before writing"

    for col in conc_out.columns:
        if col == first_col_name:
            continue

        conc_missing = conc_out[col].eq(MISSING_VALUE)
        unc_missing = unc_out[col].eq(MISSING_VALUE)

        if not conc_missing.equals(unc_missing):
            mismatch_count = int((conc_missing != unc_missing).sum())
            raise ValueError(
                f"Missing-value mismatch in {label}, species {col}: "
                f"{mismatch_count} rows"
            )

    unc_species = unc_out.drop(columns=[first_col_name])
    bad_unc_values = ((unc_species <= 0) & (unc_species != MISSING_VALUE)).sum().sum()

    if bad_unc_values > 0:
        raise ValueError(
            f"{label}: uncertainty file still has {bad_unc_values} "
            f"zero/negative non-missing uncertainty values."
        )

    # --------------------------------------------------------
    # Write files
    # --------------------------------------------------------
    conc_path = output_dir / f"{label}_conc.csv"
    unc_path = output_dir / f"{label}_unc.csv"
    key_path = output_dir / f"{label}_sample_key.csv"
    map_path = output_dir / f"{label}_species_mapping.csv"

    conc_out.to_csv(conc_path, index=False, encoding="utf-8", float_format="%.4f")
    unc_out.to_csv(unc_path, index=False, encoding="utf-8", float_format="%.4f")

    sample_key = make_sample_key(df, label, sample_ids)
    sample_key.to_csv(key_path, index=False, encoding="utf-8")

    pd.DataFrame(
        kept_pairs,
        columns=["PMF_species_name", "concentration_column", "uncertainty_column"]
    ).to_csv(map_path, index=False, encoding="utf-8")

    # --------------------------------------------------------
    # Post-write check
    # --------------------------------------------------------
    conc_check = pd.read_csv(conc_path, usecols=[first_col_name])
    unc_check = pd.read_csv(unc_path, usecols=[first_col_name])

    assert len(conc_check) == len(unc_check), (
        f"Post-write row count mismatch for {label}: "
        f"{len(conc_check)} conc vs {len(unc_check)} unc"
    )

    assert (
        conc_check[first_col_name].astype(str).to_numpy()
        == unc_check[first_col_name].astype(str).to_numpy()
    ).all(), f"Post-write {first_col_name} mismatch for {label}"

    print(f"\n  Exported: {label}")
    print(f"    Rows:          {len(df)}")
    print(f"    Species kept:  {len(kept_pairs)}")
    print(f"    First column:  {first_col_name}")
    print(f"    Conc file:     {conc_path}")
    print(f"    Uncert file:   {unc_path}")
    print(f"    Sample key:    {key_path}")
    print(f"    Mapping file:  {map_path}")


def exclude_periods(df, exclude_periods):
    df_out = df.copy()

    for name, start, end in exclude_periods:
        start_ts = pd.Timestamp(start, tz=local_tz)
        end_ts = pd.Timestamp(end, tz=local_tz)

        mask = (df_out[time_col] >= start_ts) & (df_out[time_col] <= end_ts)
        print(f"  Excluding {name}: {mask.sum()} rows")

        df_out = df_out.loc[~mask].copy()

    return df_out


def slice_window(df, start, end):
    start_ts = pd.Timestamp(start, tz=local_tz)
    end_ts = pd.Timestamp(end, tz=local_tz)

    return df[
        (df[time_col] >= start_ts) &
        (df[time_col] <= end_ts)
    ].copy()


def combine_event_windows(df, event_windows):
    pieces = []

    for event_name, (start, end) in event_windows.items():
        this_event = slice_window(df, start, end)

        if this_event.empty:
            print(f"  WARNING: {event_name} has no rows.")
            continue

        this_event = this_event.copy()
        this_event.insert(1, "event_name", event_name)
        pieces.append(this_event)

        print(f"  Adding {event_name}: {len(this_event)} rows")

    if not pieces:
        return pd.DataFrame()

    return pd.concat(pieces, ignore_index=True)


def export_master_tracking_file(df):
    master_key = pd.DataFrame({
        "Master_Row": df["Master_Row"].to_numpy(),
        "Date_Local": df[time_col].dt.strftime("%Y-%m-%d %H:%M:%S").to_numpy(),
        "Date_UTC": df[time_col].dt.tz_convert("UTC").dt.strftime("%Y-%m-%d %H:%M:%S").to_numpy(),
    })

    master_path = output_dir / f"{output_prefix}_MASTER_time_key_all_rows.csv"

    master_key.to_csv(
        master_path,
        index=False,
        encoding="utf-8"
    )

    print(f"\nMaster tracking file exported:")
    print(f"  {master_path}")


# ============================================================
# AUTOMATIC TIME-SLICE EXPORT HELPERS
# ============================================================
MONTH_NAME = {
    1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr",
    5: "May", 6: "Jun", 7: "Jul", 8: "Aug",
    9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec",
}


def label_date(ts):
    """For obvious filenames."""
    return ts.strftime("%Y%m%d")


def make_monthly_windows(df):
    """
    One PMF file set per calendar month.
    Example:
      Xact_PMF_month_2024_10_Oct
    """
    windows = []

    month_periods = (
        df[time_col]
        .dt.tz_localize(None)
        .dt.to_period("M")
        .drop_duplicates()
        .sort_values()
    )

    for p in month_periods:
        start = pd.Timestamp(p.start_time, tz=local_tz)
        end = pd.Timestamp(p.end_time, tz=local_tz)

        label = f"{output_prefix}_month_{p.year}_{p.month:02d}_{MONTH_NAME[p.month]}"
        windows.append((label, start, end))

    return windows


def make_yearly_windows(df):
    """
    One PMF file set per calendar year.
    Example:
      Xact_PMF_year_2024
    """
    windows = []

    years = sorted(df[time_col].dt.year.dropna().unique())

    for y in years:
        start = pd.Timestamp(f"{y}-01-01 00:00", tz=local_tz)
        end = pd.Timestamp(f"{y}-12-31 23:59:59", tz=local_tz)

        label = f"{output_prefix}_year_{y}"
        windows.append((label, start, end))

    return windows


def make_6month_windows(df):
    """
    Two PMF file sets per year:
      Jan-Jun
      Jul-Dec

    Examples:
      Xact_PMF_6month_2024_H1_Jan_Jun
      Xact_PMF_6month_2024_H2_Jul_Dec
    """
    windows = []

    years = sorted(df[time_col].dt.year.dropna().unique())

    for y in years:
        windows.append((
            f"{output_prefix}_6month_{y}_H1_Jan_Jun",
            pd.Timestamp(f"{y}-01-01 00:00", tz=local_tz),
            pd.Timestamp(f"{y}-06-30 23:59:59", tz=local_tz),
        ))

        windows.append((
            f"{output_prefix}_6month_{y}_H2_Jul_Dec",
            pd.Timestamp(f"{y}-07-01 00:00", tz=local_tz),
            pd.Timestamp(f"{y}-12-31 23:59:59", tz=local_tz),
        ))

    return windows


def make_season_year_windows(df):
    """
    Meteorological seasons:
      Spring = Mar-Apr-May
      Summer = Jun-Jul-Aug
      Fall   = Sep-Oct-Nov
      Winter = Dec-Jan-Feb

    Winter is labeled by the Jan/Feb year.
    Example:
      Xact_PMF_season_2024_Winter_DJF_20231201_to_20240229
      means Dec 2023 + Jan/Feb 2024.
    """
    windows = []

    min_time = df[time_col].min()
    max_time = df[time_col].max()

    min_year = int(min_time.year) - 1
    max_year = int(max_time.year) + 1

    for y in range(min_year, max_year + 1):

        spring_start = pd.Timestamp(f"{y}-03-01 00:00", tz=local_tz)
        spring_end = pd.Timestamp(f"{y}-05-31 23:59:59", tz=local_tz)

        summer_start = pd.Timestamp(f"{y}-06-01 00:00", tz=local_tz)
        summer_end = pd.Timestamp(f"{y}-08-31 23:59:59", tz=local_tz)

        fall_start = pd.Timestamp(f"{y}-09-01 00:00", tz=local_tz)
        fall_end = pd.Timestamp(f"{y}-11-30 23:59:59", tz=local_tz)

        winter_start = pd.Timestamp(f"{y - 1}-12-01 00:00", tz=local_tz)
        feb_end = pd.Timestamp(f"{y}-02-01 00:00", tz=local_tz) + pd.offsets.MonthEnd(0)
        winter_end = pd.Timestamp(f"{feb_end.strftime('%Y-%m-%d')} 23:59:59", tz=local_tz)

        season_defs = [
            (f"{output_prefix}_season_{y}_Spring_MAM", spring_start, spring_end),
            (f"{output_prefix}_season_{y}_Summer_JJA", summer_start, summer_end),
            (f"{output_prefix}_season_{y}_Fall_SON", fall_start, fall_end),
            (f"{output_prefix}_season_{y}_Winter_DJF", winter_start, winter_end),
        ]

        for base_label, start, end in season_defs:
            label = f"{base_label}_{label_date(start)}_to_{label_date(end)}"
            windows.append((label, start, end))

    return windows


def make_combined_season_windows(df):
    """
    Combined meteorological seasons across all available years:
      all Springs = Mar-Apr-May from every year
      all Summers = Jun-Jul-Aug from every year
      all Falls   = Sep-Oct-Nov from every year
      all Winters = Dec-Jan-Feb from every year

    These are non-continuous grouped windows, so this returns masks.
    """
    return [
        (
            f"{output_prefix}_season_combined_Spring_MAM_all_years",
            df[time_col].dt.month.isin([3, 4, 5])
        ),
        (
            f"{output_prefix}_season_combined_Summer_JJA_all_years",
            df[time_col].dt.month.isin([6, 7, 8])
        ),
        (
            f"{output_prefix}_season_combined_Fall_SON_all_years",
            df[time_col].dt.month.isin([9, 10, 11])
        ),
        (
            f"{output_prefix}_season_combined_Winter_DJF_all_years",
            df[time_col].dt.month.isin([12, 1, 2])
        ),
    ]


def export_window_list(df_source, window_list, group_name):
    """
    Exports continuous time windows, e.g. each month or each year.
    """
    print(f"\n--- Automatic exports: {group_name} ---")

    for label, start, end in window_list:
        df_window = df_source[
            (df_source[time_col] >= start) &
            (df_source[time_col] <= end)
        ].copy()

        if df_window.empty:
            print(f"\n  Skipping empty window: {label}")
            continue

        print(f"\n--- {group_name}: {label} ---")
        print(f"  Window: {start} to {end}")
        print(f"  Rows:   {len(df_window)}")

        make_pmf_files(df_window, label)


def export_masked_window_list(df_source, masked_window_list, group_name):
    """
    Exports non-continuous grouped windows, e.g. all summers across years.
    """
    print(f"\n--- Automatic exports: {group_name} ---")

    for label, mask in masked_window_list:
        df_window = df_source.loc[mask].copy()

        if df_window.empty:
            print(f"\n  Skipping empty grouped window: {label}")
            continue

        print(f"\n--- {group_name}: {label} ---")
        print(f"  Rows:   {len(df_window)}")
        print(f"  Start:  {df_window[time_col].min()}")
        print(f"  End:    {df_window[time_col].max()}")

        make_pmf_files(df_window, label)


# ============================================================
# MAIN
# ============================================================
print("=" * 60)
print("PMF Input File Generator")
print("=" * 60)

df = pd.read_csv(input_file)

df[time_col] = parse_local_time(df[time_col], local_tz)
df = df.sort_values(time_col).reset_index(drop=True)

# Stable row ID for tracking back to the sorted master file
df["Master_Row"] = range(1, len(df) + 1)

print(f"\nLoaded: {input_file}")
print(f"  Rows:  {len(df)}")
print(f"  Start: {df[time_col].min()}")
print(f"  End:   {df[time_col].max()}")

if EXPORT_MASTER_TRACKING_FILE:
    export_master_tracking_file(df)

# --------------------------------------------------------
# 1. Full clean dataset excluding downtime periods and outlier
# --------------------------------------------------------
print("\n--- Full dataset excluding downtime + outlier ---")
df_full_clean = exclude_periods(df, EXCLUDE_PERIODS)

make_pmf_files(
    df_full_clean,
    output_prefix + "_full_excluding_downtime_outlier"
)

# --------------------------------------------------------
# 2. Automatic routine PMF windows
#    These use df_full_clean, so downtime/outlier rows are removed.
# --------------------------------------------------------
monthly_windows = make_monthly_windows(df_full_clean)
season_year_windows = make_season_year_windows(df_full_clean)
combined_season_windows = make_combined_season_windows(df_full_clean)
sixmonth_windows = make_6month_windows(df_full_clean)
yearly_windows = make_yearly_windows(df_full_clean)

export_window_list(
    df_full_clean,
    monthly_windows,
    "monthly"
)

export_window_list(
    df_full_clean,
    season_year_windows,
    "season_year"
)

export_masked_window_list(
    df_full_clean,
    combined_season_windows,
    "combined_season_all_years"
)

export_window_list(
    df_full_clean,
    sixmonth_windows,
    "6month"
)

export_window_list(
    df_full_clean,
    yearly_windows,
    "yearly"
)

# --------------------------------------------------------
# 3. Current individual event PMF windows
#    Kept from your original code.
#    These use raw df, like before.
# --------------------------------------------------------
for window_name, (start, end) in PMF_WINDOWS.items():
    print(f"\n--- Event/window: {window_name} ---")

    df_window = slice_window(df, start, end)

    make_pmf_files(
        df_window,
        output_prefix + "_event_" + window_name
    )

# --------------------------------------------------------
# 4. Combined coke-event file
#    Kept from your original code.
# --------------------------------------------------------
print("\n--- Combined coke events ---")
df_coke = combine_event_windows(df, COKE_EVENTS)

make_pmf_files(
    df_coke,
    output_prefix + "_event_combined_coke_events"
)

print("\n" + "=" * 60)
print("Done.")
print("=" * 60)