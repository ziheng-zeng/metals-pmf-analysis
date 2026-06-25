"""
SMPS 2024: concatenate all monthly files + QC/flag frequency report

What it does:
1) Finds all SMPS CSVs in a folder whose filename contains "2024"
2) Concatenates them (keeps a source_file column)
3) Writes a combined CSV
4) Generates summary reports:
   - qc_outcome counts
   - flag token counts (splits multi-flag cells)
   - flag combo counts (exact strings)
   - comment exact + bucketed counts
   - optional status/error fields counts (detector_status, classifier_errors, etc.)
"""

import os
import glob
import re
import pandas as pd

# ----------------------------
# USER SETTINGS
# ----------------------------
IN_DIR = r"D:/Documents/PhD-Research/ASCENT-intercomparison/SMPS2024L1b"  # <-- change to your SMPS folder
FILE_GLOB = "*2024*.csv"  # filenames include 2024 (adjust if needed)

OUT_DIR = r"D:/Documents/PhD-Research/ASCENT-intercomparison/SMPS2024L2"  # <-- output folder
COMBINED_CSV = os.path.join(OUT_DIR, "ASCENT_SMPS_2024_all_files_concat.csv")

# Key columns
DT_COL = "sample_datetime_utc"   # note: your header uses _utc (lowercase)
QC_COL = "qc_outcome"
FLAG_COL = "flag"
COMMENT_COL = "comment"

# Optional “status/error” columns to summarize if they exist
STATUS_COLS = [
    "detector_status",
    "classifier_errors",
    "communication_status",
    "neutralizer_status",
    "test_name",
    "scan_direction",
    "hv_polarity",
    "wide_range_scan_mode",
]

os.makedirs(OUT_DIR, exist_ok=True)

# ----------------------------
# HELPERS
# ----------------------------
def clean_text(s: pd.Series) -> pd.Series:
    return (
        s.astype("string")
         .fillna("")
         .str.strip()
         .replace({"nan": "", "NaN": "", "None": "", "NULL": "", "null": ""})
    )

def split_flags(x: str):
    x = str(x).strip()
    if x == "":
        return []
    # split on common separators: comma, semicolon, pipe, whitespace
    parts = re.split(r"[;,|]\s*|\s+", x)
    return [p.strip() for p in parts if p.strip()]

def bucket_comment(s: str) -> str:
    s = str(s)
    s = re.sub(r"\d{4}-\d{2}-\d{2}.*", "", s)  # remove dates and after
    s = re.sub(r"\d+(\.\d+)?", "#", s)         # replace numbers with #
    s = re.sub(r"\s+", " ", s).strip()
    return s if s else "(blank)"

def save_counts(series: pd.Series, out_path: str, name_col: str, n_col: str = "n_rows"):
    tbl = (
        series.replace("", "(blank)")
              .value_counts(dropna=False)
              .rename_axis(name_col)
              .reset_index(name=n_col)
    )
    tbl.to_csv(out_path, index=False)
    return tbl

# ----------------------------
# 1) FIND + CONCAT FILES
# ----------------------------
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
df.columns = df.columns.str.strip()

print(f"Files found: {len(paths)}")
print(f"Combined rows: {len(df):,}")

# Save combined CSV for reuse
df.to_csv(COMBINED_CSV, index=False)
print("Saved combined CSV:", COMBINED_CSV)

# ----------------------------
# 2) CLEAN TARGET COLUMNS
# ----------------------------
for c in [QC_COL, FLAG_COL, COMMENT_COL]:
    if c not in df.columns:
        df[c] = ""

df[QC_COL] = clean_text(df[QC_COL])
df[FLAG_COL] = clean_text(df[FLAG_COL])
df[COMMENT_COL] = clean_text(df[COMMENT_COL])

# Parse datetime (optional but helpful)
if DT_COL in df.columns:
    df[DT_COL] = pd.to_datetime(df[DT_COL], errors="coerce", utc=True)

# Only rows with QC info (optional subset for comment/flag stats)
qc_mask = (df[QC_COL] != "") | (df[FLAG_COL] != "") | (df[COMMENT_COL] != "")
qc_df = df.loc[qc_mask].copy()

print(f"Rows with any qc_outcome/flag/comment: {len(qc_df):,}")

# ----------------------------
# 3) REPORTS
# ----------------------------
# 3a) qc_outcome counts
save_counts(
    qc_df[QC_COL],
    os.path.join(OUT_DIR, "SMPS_qc_outcome_counts.csv"),
    name_col=QC_COL
)

# 3b) flag token counts (split multi-flag cells)
flag_tokens = qc_df[FLAG_COL].apply(split_flags).explode()
flag_token_counts = (
    flag_tokens.dropna().replace("", "(blank)")
    .value_counts()
    .rename_axis("flag_token")
    .reset_index(name="n_rows_with_flag")
)
flag_token_counts.to_csv(os.path.join(OUT_DIR, "SMPS_flag_token_counts.csv"), index=False)

# 3c) flag combo counts (exact full cell strings)
save_counts(
    qc_df[FLAG_COL],
    os.path.join(OUT_DIR, "SMPS_flag_combo_counts.csv"),
    name_col="flag_combo"
)

# 3d) comment counts (exact and bucketed)
save_counts(
    qc_df[COMMENT_COL],
    os.path.join(OUT_DIR, "SMPS_comment_exact_counts.csv"),
    name_col="comment_exact"
)

comment_bucket_counts = (
    qc_df[COMMENT_COL].apply(bucket_comment)
    .value_counts()
    .rename_axis("comment_bucket")
    .reset_index(name="n_rows")
)
comment_bucket_counts.to_csv(os.path.join(OUT_DIR, "SMPS_comment_bucket_counts.csv"), index=False)

# 3e) status/error fields counts (if exist)
for col in STATUS_COLS:
    if col in df.columns:
        s = clean_text(df[col])
        save_counts(
            s,
            os.path.join(OUT_DIR, f"SMPS_{col}_counts.csv"),
            name_col=col
        )

# 3f) time-of-day distribution for QC rows (optional)
if DT_COL in qc_df.columns:
    qc_time_counts = (
        qc_df[DT_COL].dropna().dt.strftime("%H:%M:%S")
        .value_counts()
        .rename_axis("time_utc")
        .reset_index(name="n_rows")
    )
    qc_time_counts.to_csv(os.path.join(OUT_DIR, "SMPS_qc_time_of_day_counts.csv"), index=False)

print("\nDone. Reports saved to:", OUT_DIR)
print("Key outputs:")
print(" - SMPS_qc_outcome_counts.csv")
print(" - SMPS_flag_token_counts.csv")
print(" - SMPS_flag_combo_counts.csv")
print(" - SMPS_comment_exact_counts.csv")
print(" - SMPS_comment_bucket_counts.csv")
