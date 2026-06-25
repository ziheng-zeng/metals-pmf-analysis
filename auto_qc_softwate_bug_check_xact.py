import os
import numpy as np
import pandas as pd

# ----------------------------
# SETTINGS
# ----------------------------
COMBINED_CSV = r"D:/Documents/PhD-Research/ASCENT-intercomparison/Xact2024L2/ASCENT_Xact_Lawrenceville_2024_L1b_all_months.csv"
OUT_DIR = r"D:/Documents/PhD-Research/ASCENT-intercomparison/Xact2024L2"
OUT_OVERRIDE = os.path.join(OUT_DIR, "ManualInvalidate_SoftwareBug_TimestampIssue.csv")

DT_COL = "sample_datetime_UTC"
SITE_COL = "site_code"
ELEM_COL = "element"
PUMP_COL = "pump_start_time_UTC"

FLAG_OUT = "659"
COMMENT_OUT = "Manually invalidated because of software bug (incorrect timestamps/duplicated data points)."

# How aggressively to merge nearby bad points into one period
GAP_TOL_HOURS = 2   # treat bad segments separated by <=2 hours as one continuous bad period

# For missing-hour detection (optional): expected hourly cadence
CHECK_MISSING_HOURS = True

os.makedirs(OUT_DIR, exist_ok=True)

# ----------------------------
# LOAD
# ----------------------------
df = pd.read_csv(COMBINED_CSV)
df.columns = df.columns.str.strip()

for c in [SITE_COL, ELEM_COL, DT_COL]:
    if c not in df.columns:
        raise ValueError(f"Missing required column: {c}")

df[DT_COL] = pd.to_datetime(df[DT_COL], errors="coerce", utc=True)
if PUMP_COL in df.columns:
    df[PUMP_COL] = pd.to_datetime(df[PUMP_COL], errors="coerce", utc=True)

df = df[df[DT_COL].notna()].copy()
df = df.sort_values([SITE_COL, ELEM_COL, DT_COL])

# ----------------------------
# DETECT BAD HOURS PER SERIES
# ----------------------------
bad_records = []

for (site, elem), g in df.groupby([SITE_COL, ELEM_COL], sort=False):
    g = g.sort_values(DT_COL).copy()

    # 1) duplicate timestamps
    dup = g[DT_COL].duplicated(keep=False)

    # 2) non-monotonic (shouldn't happen after sorting unless there are identical timestamps)
    # still useful if source data isn't sorted or has weird timestamps
    dt_prev = g[DT_COL].shift(1)
    nonmono = (dt_prev.notna()) & (g[DT_COL] < dt_prev)

    # 3) pump start mismatch (optional): pump start far from sample time
    pump_bad = pd.Series(False, index=g.index)
    if PUMP_COL in g.columns:
        delta_min = (g[PUMP_COL] - g[DT_COL]).dt.total_seconds() / 60.0
        # flag if pump start is weirdly far away (tune as needed)
        pump_bad = delta_min.abs() > 180  # >3 hours mismatch is suspicious

    # 4) missing hours detection (optional): find gaps in expected hourly series
    missing_bad_times = []
    if CHECK_MISSING_HOURS and len(g) >= 24:
        # round to hour (since should be hourly)
        hours = g[DT_COL].dt.floor("H").dropna().unique()
        if len(hours) > 0:
            hmin, hmax = hours.min(), hours.max()
            full = pd.date_range(hmin, hmax, freq="H", tz="UTC")
            missing = sorted(set(full) - set(hours))
            # treat missing hours as "bad markers" (we'll invalidate around them)
            missing_bad_times = missing

    # union of bad points
    bad_idx = g.index[dup | nonmono | pump_bad].tolist()
    bad_times = list(g.loc[bad_idx, DT_COL].values)

    # also include missing-hour markers
    bad_times.extend(missing_bad_times)

    if not bad_times:
        continue

    # convert to sorted unique timestamps
    bad_times = sorted(pd.to_datetime(pd.Series(bad_times)).dropna().unique())

    # ----------------------------
    # GROUP INTO PERIODS
    # ----------------------------
    periods = []
    start = bad_times[0]
    end = bad_times[0]

    for t in bad_times[1:]:
        if (t - end) <= pd.Timedelta(hours=GAP_TOL_HOURS):
            end = t
        else:
            periods.append((start, end))
            start, end = t, t
    periods.append((start, end))

    # expand to cover full hour blocks (optional): invalidate whole hours
    # end + 1 hour so the "bad hour" is fully included
    for (s, e) in periods:
        bad_records.append({
            "site_code": site,
            "element": elem,
            "sample_datetime_UTC_start": pd.Timestamp(s).tz_convert("UTC"),
            "sample_datetime_UTC_end": (pd.Timestamp(e).tz_convert("UTC") + pd.Timedelta(hours=1)),
            "flag": FLAG_OUT,
            "comment": COMMENT_OUT,
            "n_bad_markers": len(bad_times),
        })

bad_df = pd.DataFrame(bad_records)

if bad_df.empty:
    print("No timestamp-bug-like periods detected.")
else:
    # Merge overlapping periods across element if you want one combined window per site:
    # For now keep site+element windows (more conservative).
    bad_df = bad_df.sort_values(["site_code", "element", "sample_datetime_UTC_start"])
    bad_df.to_csv(OUT_OVERRIDE, index=False)
    print(f"Detected {len(bad_df)} bad periods. Wrote override file:\n{OUT_OVERRIDE}")
