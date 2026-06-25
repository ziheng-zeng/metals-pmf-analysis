import pandas as pd
import numpy as np
import re
from pathlib import Path


# ================= USER SETTINGS =================

PMF_OUTPUTS_ROOT = Path(
    r"D:\Documents\PhD-Research\Xact python code\pmf_outputs"
)

SAMPLE_KEY_FOLDER = Path(
    r"D:\Documents\PhD-Research\Xact python code\pmf_ready"
)

OUTPUT_ROOT = Path(
    r"D:\Documents\PhD-Research\Xact python code\pmf_outputs_merged"
)

OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

# Keep raw PMF contribution values.
# Do not silently clip negatives unless you decide to later.
CLIP_NEGATIVE_CONTRIBUTIONS_FOR_SUMMARY = False

DAY_START_HOUR = 7
NIGHT_START_HOUR = 19

TOP_QUANTILES = [0.90, 0.95]  # top 10% and top 5%

# ==================================================


def infer_year_month_from_text(text):
    """
    Finds year/month from folder or file names.

    Works for:
      2023_06_Jun_3f
      Xact_PMF_month_2023_06_Jun_sample_key.csv
      month202306_contributions.txt
    """
    text = str(text)

    m = re.search(r"(20\d{2})[_-](\d{2})(?:[_-]([A-Za-z]{3}))?", text)
    if m:
        year = m.group(1)
        month = m.group(2)
        month_name = m.group(3)
        return year, month, month_name

    m = re.search(r"month(20\d{2})(\d{2})", text, flags=re.IGNORECASE)
    if m:
        return m.group(1), m.group(2), None

    return None, None, None


def infer_factor_count_from_folder(folder_name):
    """
    Finds factor count from folders like:
      2023_06_Jun_3f
      2023_06_Jun_6f
    """
    m = re.search(r"_(\d+)f$", folder_name, flags=re.IGNORECASE)
    if m:
        return int(m.group(1))
    return None


def build_sample_key_index(sample_key_folder):
    """
    Index sample keys by (year, month).
    """
    index = {}

    for path in sample_key_folder.glob("*sample_key*.csv"):
        year, month, month_name = infer_year_month_from_text(path.name)

        if year is None or month is None:
            continue

        key = (year, month)

        if key in index:
            print(f"WARNING: duplicate sample key for {key}:")
            print(f"  existing: {index[key]}")
            print(f"  new:      {path}")
            print("  Keeping existing one.")

        else:
            index[key] = path

    return index


def read_pmf_contribution_file(path):
    """
    Reads EPA PMF contribution txt file.

    Handles:
      Case A: PMF IDs are ID1, ID2, ID3...
      Case B: PMF IDs are original SampleIDs, e.g. ID702, ID703...
      Case C: PMF txt has extra sections; only reads the first Factor Contributions block.

    Returns:
      PMF_ID_Number, PMF_ID, Base_Run, Factor_1, Factor_2, ...
    """
    path = Path(path)
    lines = path.read_text(errors="replace").splitlines()

    # Find the contribution section
    start_idx = None
    for i, line in enumerate(lines):
        if "Factor Contributions" in line:
            start_idx = i
            break

    if start_idx is None:
        raise ValueError("This does not look like a Factor Contributions file.")

    # Read only the first contiguous data block after the Factor Contributions header
    data_lines = []
    started_data = False

    for line in lines[start_idx + 1:]:
        is_data = bool(re.match(r"^\s*\d+\s+ID\d+\s+", line))

        if is_data:
            data_lines.append(line)
            started_data = True

        elif started_data:
            # Once data started, stop at the first non-data line.
            # This avoids accidentally reading another later section.
            break

    if len(data_lines) == 0:
        raise ValueError("No PMF data rows found. Expected rows like: 7 ID1 ...")

    rows = [re.split(r"\s+", line.strip()) for line in data_lines]

    n_factors = len(rows[0]) - 2

    df = pd.DataFrame(
        rows,
        columns=["Base_Run", "PMF_ID"] + [f"Factor_{i}" for i in range(1, n_factors + 1)]
    )

    df["Base_Run"] = pd.to_numeric(df["Base_Run"])
    df["PMF_ID_Number"] = df["PMF_ID"].str.extract(r"ID(\d+)").astype(int)

    for col in [f"Factor_{i}" for i in range(1, n_factors + 1)]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df[["PMF_ID_Number", "PMF_ID", "Base_Run"] + [f"Factor_{i}" for i in range(1, n_factors + 1)]]

    return df


def find_local_time_column(sample_key):
    """
    Finds the local timestamp column in the sample key.
    """
    candidates = [
        "Date_Local",
        "timestamp",
        "Timestamp",
        "datetime",
        "DateTime",
        "date_time",
        "Time",
        "time"
    ]

    for col in candidates:
        if col in sample_key.columns:
            return col

    raise ValueError(
        f"Could not find local timestamp column. Columns are: {list(sample_key.columns)}"
    )


def add_time_variables(df, local_time_col):
    """
    Adds timing variables used for PMF factor diagnostics.
    """
    df = df.copy()

    df[local_time_col] = pd.to_datetime(df[local_time_col])

    df["hour"] = df[local_time_col].dt.hour
    df["date"] = df[local_time_col].dt.date
    df["month"] = df[local_time_col].dt.month
    df["year"] = df[local_time_col].dt.year

    df["weekday_num"] = df[local_time_col].dt.dayofweek  # Monday=0
    df["weekday_name"] = df[local_time_col].dt.day_name()
    df["weekday_weekend"] = np.where(
        df["weekday_num"].isin([5, 6]),
        "weekend",
        "weekday"
    )

    df["day_night_fixed"] = np.where(
        (df["hour"] >= DAY_START_HOUR) & (df["hour"] < NIGHT_START_HOUR),
        "day",
        "night"
    )

    return df


def summarize_factor_timing(merged, factor_cols, local_time_col, run_folder, year, month, folder_factor_count):
    """
    Creates one timing summary table per PMF contribution file.
    """
    rows = []

    for fac in factor_cols:
        if CLIP_NEGATIVE_CONTRIBUTIONS_FOR_SUMMARY:
            x = merged[fac].clip(lower=0)
        else:
            x = merged[fac]

        day_mean = x[merged["day_night_fixed"] == "day"].mean()
        night_mean = x[merged["day_night_fixed"] == "night"].mean()

        weekday_mean = x[merged["weekday_weekend"] == "weekday"].mean()
        weekend_mean = x[merged["weekday_weekend"] == "weekend"].mean()

        hourly_mean = merged.assign(_x=x).groupby("hour")["_x"].mean()
        peak_hour = int(hourly_mean.idxmax())

        base = {
            "run_folder": run_folder,
            "year": year,
            "month": month,
            "folder_factor_count": folder_factor_count,
            "factor": fac,
            "n_hours": x.notna().sum(),
            "mean": x.mean(),
            "median": x.median(),
            "min": x.min(),
            "max": x.max(),
            "peak_hour_by_mean": peak_hour,
            "day_mean_fixed": day_mean,
            "night_mean_fixed": night_mean,
            "night_day_ratio_fixed": night_mean / day_mean if day_mean not in [0, np.nan] else np.nan,
            "weekday_mean": weekday_mean,
            "weekend_mean": weekend_mean,
            "weekend_weekday_ratio": weekend_mean / weekday_mean if weekday_mean not in [0, np.nan] else np.nan,
        }

        for q in TOP_QUANTILES:
            threshold = x.quantile(q)
            top = merged[x >= threshold]

            suffix = f"top{int((1 - q) * 100)}pct"

            base[f"{suffix}_threshold"] = threshold
            base[f"{suffix}_n_hours"] = len(top)
            base[f"{suffix}_night_fraction"] = (top["day_night_fixed"] == "night").mean()
            base[f"{suffix}_weekend_fraction"] = (top["weekday_weekend"] == "weekend").mean()
            base[f"{suffix}_first_time"] = top[local_time_col].min()
            base[f"{suffix}_last_time"] = top[local_time_col].max()

        rows.append(base)

    return pd.DataFrame(rows)


def make_hour_of_day_table(merged, factor_cols, run_folder, year, month, folder_factor_count):
    hourly = (
        merged.groupby("hour")[factor_cols]
        .mean()
        .reset_index()
    )

    hourly.insert(0, "folder_factor_count", folder_factor_count)
    hourly.insert(0, "month", month)
    hourly.insert(0, "year", year)
    hourly.insert(0, "run_folder", run_folder)

    return hourly


def make_top_hours_table(merged, factor_cols, local_time_col, run_folder, year, month, folder_factor_count):
    rows = []

    id_cols = []
    for col in ["SampleID", "PMF_Row", "PMF_ID", local_time_col, "Date_UTC"]:
        if col in merged.columns:
            id_cols.append(col)

    for fac in factor_cols:
        x = merged[fac].clip(lower=0) if CLIP_NEGATIVE_CONTRIBUTIONS_FOR_SUMMARY else merged[fac]

        for q in TOP_QUANTILES:
            threshold = x.quantile(q)
            top = merged[x >= threshold].copy()

            for _, r in top.iterrows():
                row = {
                    "run_folder": run_folder,
                    "year": year,
                    "month": month,
                    "folder_factor_count": folder_factor_count,
                    "factor": fac,
                    "top_group": f"top{int((1 - q) * 100)}pct",
                    "threshold": threshold,
                    "contribution": r[fac],
                    "hour": r["hour"],
                    "weekday_name": r["weekday_name"],
                    "weekday_weekend": r["weekday_weekend"],
                    "day_night_fixed": r["day_night_fixed"],
                }

                for col in id_cols:
                    row[col] = r[col]

                rows.append(row)

    return pd.DataFrame(rows)


def process_one_contribution_file(contrib_path, sample_key_index):
    contrib_path = Path(contrib_path)

    run_folder = contrib_path.parent.name
    folder_factor_count = infer_factor_count_from_folder(run_folder)

    # Infer year/month from parent folder first, then file name
    year, month, month_name = infer_year_month_from_text(run_folder)

    if year is None or month is None:
        year, month, month_name = infer_year_month_from_text(contrib_path.name)

    if year is None or month is None:
        raise ValueError(f"Could not infer year/month from {contrib_path}")

    key = (year, month)

    if key not in sample_key_index:
        raise FileNotFoundError(
            f"No matching sample key found for {year}_{month}. "
            f"Expected something like Xact_PMF_month_{year}_{month}_*_sample_key.csv"
        )

    sample_key_path = sample_key_index[key]

    sample_key = pd.read_csv(sample_key_path)
    contrib = read_pmf_contribution_file(contrib_path)

    local_time_col = find_local_time_column(sample_key)

    # Decide merge method
    merge_method = None

    # Case 1: PMF IDs are original SampleID values, e.g. ID702, ID703...
    if "SampleID" in sample_key.columns:
        sample_ids = set(pd.to_numeric(sample_key["SampleID"], errors="coerce").dropna().astype(int))
        pmf_ids = set(contrib["PMF_ID_Number"].dropna().astype(int))

        if pmf_ids.issubset(sample_ids):
            merge_method = "SampleID"

    # Case 2: PMF IDs are row numbers, e.g. ID1, ID2, ID3...
    expected_row_ids = list(range(1, len(contrib) + 1))
    pmf_id_numbers = contrib["PMF_ID_Number"].tolist()

    if merge_method is None and pmf_id_numbers == expected_row_ids:
        merge_method = "row_order_ID1_to_IDn"

    # Case 3: IDs are not useful, but row counts match
    # This is acceptable if EPA PMF preserved the input order.
    if merge_method is None and len(sample_key) == len(contrib):
        merge_method = "row_order_fallback"

    # Stop only if neither ID merge nor row-order merge is possible
    if merge_method is None:
        raise ValueError(
            f"Could not safely merge {run_folder}: "
            f"sample key has {len(sample_key)} rows, "
            f"contribution file has {len(contrib)} rows, "
            f"PMF IDs range from {contrib['PMF_ID'].iloc[0]} to {contrib['PMF_ID'].iloc[-1]}."
        )

    if merge_method == "SampleID":
        contrib_for_merge = contrib.rename(columns={"PMF_ID_Number": "SampleID"})
        merged = sample_key.merge(contrib_for_merge, on="SampleID", how="left")

        if merged[[col for col in contrib.columns if col.startswith("Factor_")]].isna().all(axis=None):
            raise ValueError(f"SampleID merge failed for {run_folder}.")

    else:
        if len(sample_key) != len(contrib):
            raise ValueError(
                f"Row mismatch for {run_folder}: "
                f"sample key has {len(sample_key)} rows, "
                f"contribution file has {len(contrib)} rows."
            )

        merged = pd.concat(
            [
                sample_key.reset_index(drop=True),
                contrib.reset_index(drop=True)
            ],
            axis=1
        )

    merged["merge_method"] = merge_method

    merged = add_time_variables(merged, local_time_col)

    factor_cols = [col for col in contrib.columns if col.startswith("Factor_")]

    # Add run metadata
    merged.insert(0, "run_folder", run_folder)
    merged.insert(1, "source_contribution_file", str(contrib_path))
    merged.insert(2, "source_sample_key_file", str(sample_key_path))
    merged.insert(3, "folder_factor_count", folder_factor_count)

    timing_summary = summarize_factor_timing(
        merged=merged,
        factor_cols=factor_cols,
        local_time_col=local_time_col,
        run_folder=run_folder,
        year=year,
        month=month,
        folder_factor_count=folder_factor_count
    )

    hourly = make_hour_of_day_table(
        merged=merged,
        factor_cols=factor_cols,
        run_folder=run_folder,
        year=year,
        month=month,
        folder_factor_count=folder_factor_count
    )

    top_hours = make_top_hours_table(
        merged=merged,
        factor_cols=factor_cols,
        local_time_col=local_time_col,
        run_folder=run_folder,
        year=year,
        month=month,
        folder_factor_count=folder_factor_count
    )

    # Output folder mirrors PMF run folder
    out_dir = OUTPUT_ROOT / run_folder
    out_dir.mkdir(parents=True, exist_ok=True)

    merged_out = out_dir / f"{run_folder}_contributions_merged_with_time.csv"
    clean_contrib_out = out_dir / f"{run_folder}_contributions_clean_merge_ready.csv"
    summary_out = out_dir / f"{run_folder}_factor_timing_summary.csv"
    hourly_out = out_dir / f"{run_folder}_factor_hour_of_day_means.csv"
    top_hours_out = out_dir / f"{run_folder}_factor_top_hours.csv"

    merged.to_csv(merged_out, index=False)
    contrib.to_csv(clean_contrib_out, index=False)
    timing_summary.to_csv(summary_out, index=False)
    hourly.to_csv(hourly_out, index=False)
    top_hours.to_csv(top_hours_out, index=False)

    log_row = {
        "status": "success",
        "run_folder": run_folder,
        "contribution_file": str(contrib_path),
        "sample_key_file": str(sample_key_path),
        "merge_method": merge_method,
        "n_rows": len(merged),
        "n_factors_in_file": len(factor_cols),
        "factor_count_from_folder": folder_factor_count,
        "merged_out": str(merged_out),
        "summary_out": str(summary_out),
        "hourly_out": str(hourly_out),
        "top_hours_out": str(top_hours_out),
        "error": ""
    }

    return log_row, timing_summary, hourly, top_hours


def main():
    sample_key_index = build_sample_key_index(SAMPLE_KEY_FOLDER)

    print(f"Found {len(sample_key_index)} sample key files.")
    print("Sample key index:")
    for key, path in sorted(sample_key_index.items()):
        print(f"  {key}: {path.name}")

    contribution_files = sorted(PMF_OUTPUTS_ROOT.rglob("*contribution*.txt"))

    # Keep only actual Factor Contributions files
    real_contribution_files = []
    for path in contribution_files:
        try:
            head = path.read_text(errors="replace")[:500]
            if "Factor Contributions" in head:
                real_contribution_files.append(path)
        except Exception:
            pass

    print()
    print(f"Found {len(real_contribution_files)} PMF contribution files.")

    log_rows = []
    all_summaries = []
    all_hourly = []
    all_top_hours = []

    for path in real_contribution_files:
        print()
        print(f"Processing: {path}")

        try:
            log_row, timing_summary, hourly, top_hours = process_one_contribution_file(
                path,
                sample_key_index
            )

            log_rows.append(log_row)
            all_summaries.append(timing_summary)
            all_hourly.append(hourly)
            all_top_hours.append(top_hours)

            print(f"  OK: {log_row['n_rows']} rows, {log_row['n_factors_in_file']} factors")

        except Exception as e:
            print(f"  ERROR: {e}")

            log_rows.append({
                "status": "error",
                "run_folder": path.parent.name,
                "contribution_file": str(path),
                "merge_method": "",
                "sample_key_file": "",
                "n_rows": np.nan,
                "n_factors_in_file": np.nan,
                "factor_count_from_folder": infer_factor_count_from_folder(path.parent.name),
                "merged_out": "",
                "summary_out": "",
                "hourly_out": "",
                "top_hours_out": "",
                "error": str(e)
            })

    log = pd.DataFrame(log_rows)
    log_out = OUTPUT_ROOT / "_batch_merge_log.csv"
    log.to_csv(log_out, index=False)

    if len(all_summaries) > 0:
        all_summary = pd.concat(all_summaries, ignore_index=True)
        all_summary_out = OUTPUT_ROOT / "_ALL_factor_timing_summaries.csv"
        all_summary.to_csv(all_summary_out, index=False)
    else:
        all_summary_out = None

    if len(all_hourly) > 0:
        all_hourly_df = pd.concat(all_hourly, ignore_index=True)
        all_hourly_out = OUTPUT_ROOT / "_ALL_factor_hour_of_day_means.csv"
        all_hourly_df.to_csv(all_hourly_out, index=False)
    else:
        all_hourly_out = None

    if len(all_top_hours) > 0:
        all_top_hours_df = pd.concat(all_top_hours, ignore_index=True)
        all_top_hours_out = OUTPUT_ROOT / "_ALL_factor_top_hours.csv"
        all_top_hours_df.to_csv(all_top_hours_out, index=False)
    else:
        all_top_hours_out = None

    print()
    print("====================================")
    print("DONE")
    print("====================================")
    print(f"Batch log: {log_out}")

    if all_summary_out:
        print(f"All timing summaries: {all_summary_out}")

    if all_hourly_out:
        print(f"All hour-of-day means: {all_hourly_out}")

    if all_top_hours_out:
        print(f"All top hours: {all_top_hours_out}")

    print()
    print("Check _batch_merge_log.csv for skipped/error files.")


if __name__ == "__main__":
    main()