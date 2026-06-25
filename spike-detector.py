# -*- coding: utf-8 -*-
"""
All-metal exploratory peak and co-occurrence screening.

Goal:
- Detect small/recurrent metal peaks across all available Xact metals.
- Summarize which metals spike together.
- Identify possible source groups beyond known fireworks/wildfire/coke plumes.

Outputs:
1. allmetal_hourly_exceedance_flags.csv
2. allmetal_event_summary.csv
3. allmetal_metal_frequency_summary.csv
4. allmetal_cooccurrence_matrix.csv
5. allmetal_top_pairs.csv
6. allmetal_event_by_metal_matrix.csv
"""

import re
import itertools
import numpy as np
import pandas as pd
from pathlib import Path

# ================= USER SETTINGS =================

INPUT_FILE = Path(r"Xact_EST_May2023_Oct2025_combined.csv")

OUTPUT_FOLDER = Path(r"plume_screening_all_metals")
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

TIME_COL = "TIME"
LOCAL_TZ = "America/New_York"

EXCLUDE_RANGES = [
    ("2024-01-09 00:00", "2024-02-13 23:59"),
    ("2024-07-02 00:00", "2024-08-08 23:59"),
]

# Peak-screening settings
BACKGROUND_WINDOW_HOURS = 24
BACKGROUND_QUANTILE = 0.25
MIN_BACKGROUND = 0.01

# Lower threshold to catch smaller peaks
ER_THRESHOLD = 2.0
UNC_MULT = 1.0

# Event grouping settings
MIN_METALS_EXCEEDING_FOR_EVENT = 2
MIN_EVENT_HOURS = 1

# Exclude metals that almost never have valid signal
MIN_VALID_SIGNAL_FRACTION = 0.00

EXCLUDE_METALS = [
    "Nb",
]

RETAINED_SPECIES = [
    "Al", "Si", "P", "S", "Cl", "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn",
    "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Rb",
    "Sr", "Y", "Zr", "Mo", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te",
    "I", "Cs", "Ba", "La", "Ce", "W", "Pt", "Au", "Hg", "Tl", "Pb", "Bi"
]
# ==================================================


def clean_metal_name(col):
    s = str(col).strip()
    s = re.sub(r"\(.*?\)", "", s)
    s = re.sub(r"\bng\s*/?\s*m3\b", "", s, flags=re.I)
    s = re.sub(r"\bng\s*/?\s*m\^?3\b", "", s, flags=re.I)
    s = s.replace("Uncert", "").replace("uncert", "")
    s = re.sub(r"\d+", "", s)
    s = re.sub(r"[^A-Za-z]", "", s)
    return s


def find_xact_pairs(df):
    cols = list(df.columns)

    conc_cols = []
    uncert_cols = []

    for c in cols:
        cl = str(c).lower()
        if c == TIME_COL:
            continue
        if "uncert" in cl:
            uncert_cols.append(c)
        elif "ng/m3" in cl or "ng/m³" in cl or "(ng" in cl:
            conc_cols.append(c)

    conc_map = {}
    uncert_map = {}

    for c in conc_cols:
        m = clean_metal_name(c)
        if m:
            conc_map[m] = c

    for c in uncert_cols:
        m = clean_metal_name(c)
        if m:
            uncert_map[m] = c

    pairs = {}
    for m in sorted(set(conc_map) & set(uncert_map)):
        if m not in EXCLUDE_METALS:
            pairs[m] = {
                "conc": conc_map[m],
                "unc": uncert_map[m],
            }

    return pairs


def parse_time_column(df):
    """
    Parse TIME column robustly.

    Handles:
    - timezone-aware strings like 2023-05-02 12:00:00-04:00
    - mixed daylight saving offsets (-04:00 and -05:00)
    - timezone-naive strings
    """
    df = df.copy()

    # First parse everything as UTC.
    # This avoids pandas mixed-timezone problems from daylight saving time.
    t_utc = pd.to_datetime(df[TIME_COL], errors="coerce", utc=True)

    # Convert UTC to local Eastern time, then remove timezone info
    # so downstream code uses simple local timestamps.
    t_local = t_utc.dt.tz_convert(LOCAL_TZ).dt.tz_localize(None)

    df["time_local"] = t_local

    n_bad = df["time_local"].isna().sum()
    if n_bad > 0:
        print(f"WARNING: {n_bad} rows had unparseable times and were dropped.")

    df = df.dropna(subset=["time_local"])
    df = df.sort_values("time_local").reset_index(drop=True)

    return df


def apply_exclusions(df):
    df = df.copy()
    keep = pd.Series(True, index=df.index)

    for start, end in EXCLUDE_RANGES:
        s = pd.to_datetime(start)
        e = pd.to_datetime(end)
        keep &= ~df["time_local"].between(s, e)

    return df.loc[keep].reset_index(drop=True)


def get_season(ts):
    m = ts.month
    if m in [3, 4, 5]:
        return "Spring"
    elif m in [6, 7, 8]:
        return "Summer"
    elif m in [9, 10, 11]:
        return "Fall"
    else:
        return "Winter"


def prepare_all_metals(df, pairs):
    out = pd.DataFrame({"time_local": df["time_local"]})

    for m, p in pairs.items():
        conc = pd.to_numeric(df[p["conc"]], errors="coerce")
        unc = pd.to_numeric(df[p["unc"]], errors="coerce")

        conc = conc.mask(conc < 0)
        unc = unc.mask(unc < 0)

        out[f"{m}_conc"] = conc
        out[f"{m}_unc"] = unc

    return out


def calculate_allmetal_flags(metal_df, metals):
    out = pd.DataFrame({"time_local": metal_df["time_local"]})

    valid_signal_matrix = {}
    exceed_matrix = {}
    er_matrix = {}

    keep_metals = []

    for m in metals:
        conc_col = f"{m}_conc"
        unc_col = f"{m}_unc"

        conc = metal_df[conc_col]
        unc = metal_df[unc_col]

        valid_signal = (
            conc.notna()
            & unc.notna()
            & (unc > 0)
            & (conc > UNC_MULT * unc)
        )

        valid_frac = valid_signal.mean()

        if valid_frac < MIN_VALID_SIGNAL_FRACTION:
            continue

        bg = (
            conc.shift(1)
            .rolling(BACKGROUND_WINDOW_HOURS, min_periods=max(6, BACKGROUND_WINDOW_HOURS // 4))
            .quantile(BACKGROUND_QUANTILE)
            .clip(lower=MIN_BACKGROUND)
        )

        er = conc / bg
        exceed = valid_signal & (er >= ER_THRESHOLD)

        keep_metals.append(m)
        valid_signal_matrix[m] = valid_signal
        exceed_matrix[m] = exceed
        er_matrix[m] = er

        out[f"{m}_conc"] = conc
        out[f"{m}_ER"] = er
        out[f"{m}_exceed"] = exceed

    exceed_df = pd.DataFrame(exceed_matrix)
    er_df = pd.DataFrame(er_matrix)

    out["n_metals_exceeding"] = exceed_df.sum(axis=1)

    out["metals_exceeding"] = exceed_df.apply(
        lambda row: ",".join(row.index[row.values].tolist()),
        axis=1,
    )

    return out, exceed_df, er_df, keep_metals


def group_allmetal_events(hourly, exceed_df, er_df):
    hourly = hourly.copy()

    hourly["is_event_hour"] = hourly["n_metals_exceeding"] >= MIN_METALS_EXCEEDING_FOR_EVENT

    event_start = hourly["is_event_hour"] & (~hourly["is_event_hour"].shift(1, fill_value=False))
    hourly["event_id_raw"] = event_start.cumsum()
    hourly.loc[~hourly["is_event_hour"], "event_id_raw"] = np.nan

    events = []
    event_by_metal_rows = []

    for event_id, g in hourly.dropna(subset=["event_id_raw"]).groupby("event_id_raw"):
        idx = g.index
        duration_h = len(g)

        if duration_h < MIN_EVENT_HOURS:
            continue

        sub_exceed = exceed_df.loc[idx]
        sub_er = er_df.loc[idx]

        metals_in_event = sub_exceed.columns[sub_exceed.any(axis=0)].tolist()

        start = g["time_local"].min()
        end = g["time_local"].max()

        row = {
            "event_id": int(event_id),
            "start_time": start,
            "end_time": end,
            "duration_hours": duration_h,
            "season": get_season(start),
            "start_month": start.month,
            "n_unique_metals": len(metals_in_event),
            "max_metals_in_one_hour": int(g["n_metals_exceeding"].max()),
            "mean_metals_per_hour": g["n_metals_exceeding"].mean(),
            "metals_in_event": ",".join(metals_in_event),
        }

        # Top metals ranked by number of exceeding hours, then max ER
        metal_stats = []
        for m in metals_in_event:
            metal_stats.append({
                "metal": m,
                "hours_exceeding": int(sub_exceed[m].sum()),
                "max_ER": sub_er[m].max(),
                "mean_ER_when_exceeding": sub_er.loc[sub_exceed[m], m].mean(),
            })

        metal_stats_df = pd.DataFrame(metal_stats)
        if not metal_stats_df.empty:
            metal_stats_df = metal_stats_df.sort_values(
                ["hours_exceeding", "max_ER"],
                ascending=[False, False],
            )
            row["top_metals_by_hours"] = ",".join(metal_stats_df["metal"].head(10))
            row["top_metals_by_max_ER"] = ",".join(
                metal_stats_df.sort_values("max_ER", ascending=False)["metal"].head(10)
            )
        else:
            row["top_metals_by_hours"] = ""
            row["top_metals_by_max_ER"] = ""

        events.append(row)

        event_binary = {"event_id": int(event_id)}
        for m in exceed_df.columns:
            event_binary[m] = bool(sub_exceed[m].any())
        event_by_metal_rows.append(event_binary)

    event_summary = pd.DataFrame(events)
    event_by_metal = pd.DataFrame(event_by_metal_rows)

    if not event_summary.empty:
        old_to_new = {
            old: new for new, old in enumerate(event_summary["event_id"].tolist(), start=1)
        }

        hourly["event_id"] = hourly["event_id_raw"].map(old_to_new)
        event_summary["event_id_old"] = event_summary["event_id"]
        event_summary["event_id"] = range(1, len(event_summary) + 1)

        if not event_by_metal.empty:
            event_by_metal["event_id_old"] = event_by_metal["event_id"]
            event_by_metal["event_id"] = event_by_metal["event_id"].map(old_to_new)
            cols = ["event_id", "event_id_old"] + [
                c for c in event_by_metal.columns if c not in ["event_id", "event_id_old"]
            ]
            event_by_metal = event_by_metal[cols]
    else:
        hourly["event_id"] = np.nan

    hourly = hourly.drop(columns=["event_id_raw"])

    return hourly, event_summary, event_by_metal


def make_frequency_summary(exceed_df, event_by_metal):
    rows = []

    for m in exceed_df.columns:
        row = {
            "metal": m,
            "hours_exceeding": int(exceed_df[m].sum()),
            "fraction_hours_exceeding": exceed_df[m].mean(),
        }

        if not event_by_metal.empty and m in event_by_metal.columns:
            row["events_present"] = int(event_by_metal[m].sum())
            row["fraction_events_present"] = event_by_metal[m].mean()
        else:
            row["events_present"] = 0
            row["fraction_events_present"] = np.nan

        rows.append(row)

    return pd.DataFrame(rows).sort_values(
        ["events_present", "hours_exceeding"],
        ascending=[False, False],
    )


def make_cooccurrence_outputs(event_by_metal):
    if event_by_metal.empty:
        return pd.DataFrame(), pd.DataFrame()

    metal_cols = [
        c for c in event_by_metal.columns
        if c not in ["event_id", "event_id_old"]
    ]

    binary = event_by_metal[metal_cols].astype(bool)

    # Co-occurrence count matrix: number of events where both metals appear
    cooc = binary.T.dot(binary).astype(int)

    pair_rows = []

    for a, b in itertools.combinations(metal_cols, 2):
        both = int((binary[a] & binary[b]).sum())
        a_count = int(binary[a].sum())
        b_count = int(binary[b].sum())
        either = int((binary[a] | binary[b]).sum())

        if either == 0:
            jaccard = np.nan
        else:
            jaccard = both / either

        if min(a_count, b_count) == 0:
            conditional_min = np.nan
        else:
            conditional_min = both / min(a_count, b_count)

        pair_rows.append({
            "metal_a": a,
            "metal_b": b,
            "events_both": both,
            "events_a": a_count,
            "events_b": b_count,
            "jaccard": jaccard,
            "conditional_on_smaller_count": conditional_min,
        })

    pairs = pd.DataFrame(pair_rows)
    pairs = pairs.sort_values(
        ["events_both", "jaccard"],
        ascending=[False, False],
    )

    return cooc, pairs


def main():
    print("Loading data...")
    df = pd.read_csv(INPUT_FILE, low_memory=False)

    print("Parsing time and applying exclusions...")
    df = parse_time_column(df)
    df = apply_exclusions(df)

    print("Finding paired metals...")
    pairs = find_xact_pairs(df)

    # Keep only the 46 species retained after removing completely missing species
    pairs = {m: p for m, p in pairs.items() if m in RETAINED_SPECIES}

    metals = sorted(pairs.keys())
    print(f"Found {len(metals)} retained paired metals for plume screening:")
    print(", ".join(metals))

    print("Preparing metal data...")
    metal_df = prepare_all_metals(df, pairs)

    print("Calculating all-metal enhancement flags...")
    hourly, exceed_df, er_df, keep_metals = calculate_allmetal_flags(metal_df, metals)

    print(f"Kept {len(keep_metals)} retained metals:")
    print(", ".join(keep_metals))

    print("Grouping all-metal events...")
    hourly, event_summary, event_by_metal = group_allmetal_events(hourly, exceed_df, er_df)

    print("Making frequency and co-occurrence summaries...")
    freq_summary = make_frequency_summary(exceed_df, event_by_metal)
    cooc_matrix, top_pairs = make_cooccurrence_outputs(event_by_metal)

    hourly_out = OUTPUT_FOLDER / "allmetal_hourly_exceedance_flags.csv"
    event_out = OUTPUT_FOLDER / "allmetal_event_summary.csv"
    freq_out = OUTPUT_FOLDER / "allmetal_metal_frequency_summary.csv"
    cooc_out = OUTPUT_FOLDER / "allmetal_cooccurrence_matrix.csv"
    pairs_out = OUTPUT_FOLDER / "allmetal_top_pairs.csv"
    event_matrix_out = OUTPUT_FOLDER / "allmetal_event_by_metal_matrix.csv"

    hourly.to_csv(hourly_out, index=False)
    event_summary.to_csv(event_out, index=False)
    freq_summary.to_csv(freq_out, index=False)
    cooc_matrix.to_csv(cooc_out)
    top_pairs.to_csv(pairs_out, index=False)
    event_by_metal.to_csv(event_matrix_out, index=False)

    print("\nDone.")
    print(f"Hourly flags: {hourly_out}")
    print(f"Event summary: {event_out}")
    print(f"Metal frequency summary: {freq_out}")
    print(f"Co-occurrence matrix: {cooc_out}")
    print(f"Top pairs: {pairs_out}")
    print(f"Event-by-metal matrix: {event_matrix_out}")

    print("\n========== QUICK SUMMARY ==========")
    print(f"Total hours analyzed: {len(hourly)}")
    print(f"Event hours: {hourly['is_event_hour'].sum()}")
    print(f"Number of events: {len(event_summary)}")

    if not event_summary.empty:
        print("\nEvents by season:")
        print(event_summary["season"].value_counts().to_string())

        print("\nTop 20 metals by event frequency:")
        print(freq_summary.head(20).to_string(index=False))

        print("\nTop 20 co-occurring metal pairs:")
        print(
            top_pairs[
                ["metal_a", "metal_b", "events_both", "events_a", "events_b", "jaccard"]
            ]
            .head(20)
            .to_string(index=False)
        )


if __name__ == "__main__":
    main()