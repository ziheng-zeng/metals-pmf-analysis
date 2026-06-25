"""
Xact auto-QC concat + NOT-too-strict timestamp bug screening + final override file

What this does:
1) Load all monthly L1b CSVs from a folder and concatenate them
2) Screen for *likely* software timestamp bug hours (duplicates / non-monotonic / pump-vs-sample mismatch)
   - intentionally NOT too strict, but avoids the "flags everything" trap by:
     * NOT using missing-hours detection
     * using a reasonable duplicate key
     * requiring anomalies to cluster into periods (min markers per period)
3) Produce two override sets:
   A) Tube replacement/calibration invalidation (2024-07-04 to 2024-08-08) -> flag 659
   B) Software timestamp bug periods -> flag 659
4) Output ONE final override CSV with ONLY:
   sample_datetime_UTC_start, sample_datetime_UTC_end, flag, comment
   - granular hourly rows (so you don't accidentally invalidate extra good data)
"""

import os
import glob
import re
import numpy as np
import pandas as pd

# ----------------------------
# USER SETTINGS
# ----------------------------
IN_DIR = r"D:/Documents/PhD-Research/ASCENT-intercomparison/Xact2024L1b"
FILE_GLOB = "ASCENT_Xact_Lawrenceville_*_L1b.csv"  # adjust if needed

OUT_DIR = r"D:/Documents/PhD-Research/ASCENT-intercomparison/Xact2024L2"
COMBINED_CSV = os.path.join(OUT_DIR, "ASCENT_Xact_Lawrenceville_2024_L1b_all_months.csv")

FINAL_OVERRIDE = os.path.join(OUT_DIR, "Lawrenceville_Xact_ManualQC_20240101_20241231.csv")

# Column names
DT_COL = "sample_datetime_UTC"
PUMP_COL = "pump_start_time_UTC"
SITE_COL = "site_code"
ELEM_COL = "element"
SAMPLE_TYPE_COL = "sample_type"
ALARM_COL = "alarm"
FLAG_COL = "flag"
QC_COL = "qc_outcome"
COMMENT_COL = "comment"
DUR_COL = "sample_time_min"

# Filter tape misalignment invalidation window (UTC)  [ADD THIS]
TAPE_START = pd.Timestamp("2024-01-09 00:00:00", tz="UTC")
TAPE_END_EXCL = pd.Timestamp("2024-02-14 00:00:00", tz="UTC")  # end-exclusive (through Feb 13)

TAPE_COMMENT = "Manually invalidated: filter tape misalignment period (2024-01-09 to 2024-02-13)."

EXPAND_TAPE_TO_HOURLY = True

# Tube replacement / calibration invalidation window (UTC)
TUBE_START = pd.Timestamp("2024-07-04 00:00:00", tz="UTC")
TUBE_END_EXCL = pd.Timestamp("2024-08-09 00:00:00", tz="UTC")  # end-exclusive

INVALID_FLAG = "659"
TUBE_COMMENT = "Manually invalidated: tube replacement and calibration period (2024-07-04 to 2024-08-08)."
BUG_COMMENT = "Manually invalidated because of software bug (incorrect timestamps / duplicated data points)."

# Timestamp bug detection knobs (NOT too strict)
# - No missing-hours check (avoids flagging downtime)
# - Duplicate key is moderately specific (edit if needed)
DUP_KEYS = [SITE_COL, ELEM_COL, SAMPLE_TYPE_COL, DT_COL]  # add 'sample_analysis_id' if duplicates are still huge

# "badness" signals
PUMP_MISMATCH_MIN = 60   # minutes; mismatch beyond this is suspicious (1 hour)
USE_PUMP_MISMATCH = True  # set False if pump_start_time_UTC isn't reliable in your files
USE_NONMONO = True        # non-monotonic timestamps within a series (rare, but strong)

# Period building
GAP_TOL_HOURS = 2         # merge markers within 2 hours into same bug period
MIN_MARKERS_PER_PERIOD = 2  # require at least 2 anomaly markers in a period to accept it (reduces false positives)

# Output style
EXPAND_TUBE_TO_HOURLY = True
EXPAND_BUG_PERIODS_TO_HOURLY = True  # recommended for hourly data

# ----------------------------
# UTIL
# ----------------------------
def clean_text(s: pd.Series) -> pd.Series:
    return (
        s.astype("string")
         .fillna("")
         .str.strip()
         .replace({"nan": "", "NaN": "", "None": "", "NULL": "", "null": ""})
    )

def expand_to_hourly(start_utc, end_utc_excl, flag, comment):
    """Expand [start,end) into hourly [h, h+1h) rows, trimmed to overlap."""
    start_utc = pd.to_datetime(start_utc, utc=True)
    end_utc_excl = pd.to_datetime(end_utc_excl, utc=True)

    if pd.isna(start_utc) or pd.isna(end_utc_excl) or end_utc_excl <= start_utc:
        return pd.DataFrame(columns=["sample_datetime_UTC_start", "sample_datetime_UTC_end", "flag", "comment"])

    h0 = start_utc.floor("H")
    h1 = end_utc_excl.ceil("H")
    hours = pd.date_range(h0, h1, freq="H", tz="UTC", inclusive="left")

    out = pd.DataFrame({
        "sample_datetime_UTC_start": hours,
        "sample_datetime_UTC_end": hours + pd.Timedelta(hours=1),
        "flag": str(flag),
        "comment": str(comment),
    })
    out = out[(out["sample_datetime_UTC_end"] > start_utc) & (out["sample_datetime_UTC_start"] < end_utc_excl)].copy()
    return out

def build_periods_from_times(times, gap_tol_hours=2):
    """Given sorted unique timestamps, group into contiguous periods based on gap tolerance."""
    if len(times) == 0:
        return []
    times = sorted(pd.to_datetime(pd.Series(times), utc=True).dropna().unique())
    if len(times) == 0:
        return []
    periods = []
    start = times[0]
    end = times[0]
    gap = pd.Timedelta(hours=gap_tol_hours)

    for t in times[1:]:
        if (t - end) <= gap:
            end = t
        else:
            periods.append((start, end))
            start, end = t, t
    periods.append((start, end))
    return periods

# ----------------------------
# 1) CONCAT MONTHLY FILES
# ----------------------------
os.makedirs(OUT_DIR, exist_ok=True)

paths = sorted(glob.glob(os.path.join(IN_DIR, FILE_GLOB)))
if not paths:
    raise FileNotFoundError(f"No files matched: {os.path.join(IN_DIR, FILE_GLOB)}")

dfs = []
for p in paths:
    d = pd.read_csv(p)
    d.columns = d.columns.str.strip()
    d["source_file"] = os.path.basename(p)
    dfs.append(d)

df = pd.concat(dfs, ignore_index=True, sort=False)
print(f"Files found: {len(paths)}")
print(f"Combined rows: {len(df):,}")

# Ensure expected columns exist
for c in [SITE_COL, ELEM_COL, SAMPLE_TYPE_COL, DT_COL, FLAG_COL, QC_COL, COMMENT_COL, ALARM_COL]:
    if c not in df.columns:
        df[c] = ""

df[FLAG_COL] = clean_text(df[FLAG_COL])
df[QC_COL] = clean_text(df[QC_COL])
df[COMMENT_COL] = clean_text(df[COMMENT_COL])
df[ALARM_COL] = clean_text(df[ALARM_COL])

df[DT_COL] = pd.to_datetime(df[DT_COL], errors="coerce", utc=True)
if PUMP_COL in df.columns:
    df[PUMP_COL] = pd.to_datetime(df[PUMP_COL], errors="coerce", utc=True)

# Save combined for reuse
df.to_csv(COMBINED_CSV, index=False)
print("Saved combined CSV:", COMBINED_CSV)

# ----------------------------
# 2) TIMESTAMP BUG SCREEN (NOT too strict)
# ----------------------------
# Only consider rows with a valid timestamp
df2 = df[df[DT_COL].notna()].copy()
df2 = df2.sort_values([SITE_COL, ELEM_COL, SAMPLE_TYPE_COL, DT_COL])

bad_period_rows = []

for (site, elem, stype), g in df2.groupby([SITE_COL, ELEM_COL, SAMPLE_TYPE_COL], sort=False):
    g = g.sort_values(DT_COL).copy()
    if len(g) < 5:
        continue

    # A) Duplicates on moderate key (often a strong bug signal)
    dup = g.duplicated(subset=DUP_KEYS, keep=False)

    # B) Non-monotonic timestamps BEFORE sorting isn't available; after sorting it won't show.
    # Instead, check if raw-order has backward steps (requires preserving input order).
    # We approximate with pump-based ordering (below) and/or duplicated points.
    nonmono = pd.Series(False, index=g.index)
    if USE_NONMONO:
        # "Nonmono" proxy: if within a single hour, there are multiple different pump start times far apart,
        # or the same pump start maps to inconsistent sample times.
        # We'll keep it simple: repeated sample_datetime with different pump_start_time can be suspicious.
        if PUMP_COL in g.columns and g[PUMP_COL].notna().any():
            nonmono = dup.copy()  # keep nonmono mild; dup already catches a lot

    # C) Pump mismatch (optional)
    pump_bad = pd.Series(False, index=g.index)
    if USE_PUMP_MISMATCH and (PUMP_COL in g.columns) and g[PUMP_COL].notna().any():
        delta_min = (g[PUMP_COL] - g[DT_COL]).dt.total_seconds() / 60.0
        pump_bad = delta_min.abs() > PUMP_MISMATCH_MIN

    # Combine anomaly markers
    bad_idx = g.index[dup | nonmono | pump_bad].tolist()
    if not bad_idx:
        continue

    bad_times = g.loc[bad_idx, DT_COL].tolist()

    # Group anomaly markers into periods (avoid flagging isolated 1-offs)
    periods = build_periods_from_times(bad_times, gap_tol_hours=GAP_TOL_HOURS)

    # Keep only periods with enough markers
    for (s, e) in periods:
        # count markers within this period
        n_markers = sum((pd.Series(bad_times) >= s) & (pd.Series(bad_times) <= e))
        if n_markers < MIN_MARKERS_PER_PERIOD:
            continue

        # Expand to cover full hourly blocks: make end exclusive by +1h
        start = pd.Timestamp(s).tz_convert("UTC")
        end_excl = pd.Timestamp(e).tz_convert("UTC") + pd.Timedelta(hours=1)

        bad_period_rows.append({
            "sample_datetime_UTC_start": start,
            "sample_datetime_UTC_end": end_excl,
            "flag": INVALID_FLAG,
            "comment": BUG_COMMENT,
            "site_code": site,
            "element": elem,
            "sample_type": stype,
            "n_markers": int(n_markers),
        })

bug_periods = pd.DataFrame(bad_period_rows)

print(f"Detected bug periods (pre-expand): {len(bug_periods):,}")

# ----------------------------
# 3) BUILD OVERRIDE ROWS (4 columns only)
# ----------------------------
need4 = ["sample_datetime_UTC_start", "sample_datetime_UTC_end", "flag", "comment"]

# Tube override
if EXPAND_TUBE_TO_HOURLY:
    tube_override = expand_to_hourly(TUBE_START, TUBE_END_EXCL, INVALID_FLAG, TUBE_COMMENT)
else:
    tube_override = pd.DataFrame([{
        "sample_datetime_UTC_start": TUBE_START,
        "sample_datetime_UTC_end": TUBE_END_EXCL,
        "flag": INVALID_FLAG,
        "comment": TUBE_COMMENT,
    }])

# Tape misalignment override  [ADD THIS]
if EXPAND_TAPE_TO_HOURLY:
    tape_override = expand_to_hourly(TAPE_START, TAPE_END_EXCL, INVALID_FLAG, TAPE_COMMENT)
else:
    tape_override = pd.DataFrame([{
        "sample_datetime_UTC_start": TAPE_START,
        "sample_datetime_UTC_end": TAPE_END_EXCL,
        "flag": INVALID_FLAG,
        "comment": TAPE_COMMENT,
    }])

print(f"Tape override rows: {len(tape_override):,}")

# Bug override
if bug_periods.empty:
    bug_override = pd.DataFrame(columns=need4)
else:
    if EXPAND_BUG_PERIODS_TO_HOURLY:
        chunks = []
        for r in bug_periods.itertuples(index=False):
            chunks.append(expand_to_hourly(r.sample_datetime_UTC_start, r.sample_datetime_UTC_end, r.flag, r.comment))
        bug_override = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame(columns=need4)
    else:
        bug_override = bug_periods[need4].copy()

print(f"Bug override rows (post-expand): {len(bug_override):,}")
print(f"Tube override rows: {len(tube_override):,}")

# Combine + dedupe exact duplicates
final = pd.concat([tape_override, tube_override, bug_override], ignore_index=True)
final = final.drop_duplicates(subset=need4).copy()
final = final.sort_values(["sample_datetime_UTC_start", "sample_datetime_UTC_end"]).reset_index(drop=True)

# Write final file
final.to_csv(FINAL_OVERRIDE, index=False)
print("Wrote FINAL override CSV:", FINAL_OVERRIDE)
print("Final override rows:", len(final))

# ----------------------------
# 4) OPTIONAL: Quick diagnostics (helps you see if it's over-triggering)
# ----------------------------
# How many hours flagged per month?
final["month"] = pd.to_datetime(final["sample_datetime_UTC_start"], utc=True).dt.to_period("M").astype(str)
print("\nOverride hours by month (top 12):")
print(final["month"].value_counts().sort_index().head(12))

# If you want to inspect detected bug periods with metadata (site/element/sample_type),
# write a debug file (not required for final override application):
DEBUG_BUG_PERIODS = os.path.join(OUT_DIR, "DEBUG_detected_bug_periods_with_metadata.csv")
if not bug_periods.empty:
    bug_periods.sort_values(["sample_datetime_UTC_start"]).to_csv(DEBUG_BUG_PERIODS, index=False)
    print("Wrote debug bug periods:", DEBUG_BUG_PERIODS)

# import os
# import pandas as pd
#
# OVERRIDE_PATH = r"D:/Documents/PhD-Research/ASCENT-intercomparison/Xact2024L2/Lawrenceville_Xact_ManualQC_20240101_20241231.csv"
#
# t_bad = pd.Timestamp("2024-07-01 01:00:00", tz="UTC")
#
# INVALID_FLAG = "659"
# COMMENT = "Manually invalidated: extreme Xact sulfur outlier (ACSM sulfate near zero)."
#
# new_row = pd.DataFrame([{
#     "sample_datetime_UTC_start": t_bad,
#     "sample_datetime_UTC_end": t_bad + pd.Timedelta("1H"),
#     "flag": INVALID_FLAG,
#     "comment": COMMENT,
# }])
#
#
# # Load existing override file (if exists)
# if os.path.exists(OVERRIDE_PATH):
#     ov = pd.read_csv(OVERRIDE_PATH)
#     ov.columns = ov.columns.str.strip()
#
#     ov["sample_datetime_UTC_start"] = pd.to_datetime(
#         ov["sample_datetime_UTC_start"], utc=True, errors="coerce"
#     )
#     ov["sample_datetime_UTC_end"] = pd.to_datetime(
#         ov["sample_datetime_UTC_end"], utc=True, errors="coerce"
#     )
# else:
#     ov = pd.DataFrame(columns=new_row.columns)
#
#
# # Append + deduplicate
# ov2 = pd.concat([ov, new_row], ignore_index=True)
#
# ov2 = ov2.drop_duplicates(
#     subset=["sample_datetime_UTC_start", "sample_datetime_UTC_end", "flag", "comment"]
# )
#
# ov2 = ov2.sort_values(
#     ["sample_datetime_UTC_start", "sample_datetime_UTC_end"]
# ).reset_index(drop=True)
#
#
# # Save
# ov2.to_csv(OVERRIDE_PATH, index=False)
#
# print("Added manual flag for:", t_bad)
# print("Saved override file:", OVERRIDE_PATH)
# print("Total rows now:", len(ov2))
