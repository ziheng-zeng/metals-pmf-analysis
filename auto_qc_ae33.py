"""
Instrument (BC / multi-wavelength) 2024:
- Concatenate all 2024 CSVs in a folder
- Export a combined file
- Generate a QC report: qc_outcome + flags + comments + key status fields frequency

Header you gave includes:
status, cont_status, detect_status, led_status, valve_status, ... , flag, qc_outcome, comment
"""

import os
import glob
import re
import pandas as pd

# ----------------------------
# USER SETTINGS
# ----------------------------
IN_DIR = r"D:/Documents/PhD-Research/ASCENT-intercomparison/BC2024L1b"   # <-- change (folder with monthly files)
FILE_GLOB = "*2024*.csv"                                                # filenames include 2024 (adjust tighter if needed)

OUT_DIR = r"D:/Documents/PhD-Research/ASCENT-intercomparison/BC2024L2"   # <-- change
COMBINED_CSV = os.path.join(OUT_DIR, "ASCENT_BC_2024_all_files_concat.csv")

# Key columns
DT_COL = "sample_datetime_UTC"
QC_COL = "qc_outcome"
FLAG_COL = "flag"
COMMENT_COL = "comment"

# Status columns worth summarizing (if present)
STATUS_COLS = [
    "status",
    "cont_status",
    "detect_status",
    "led_status",
    "valve_status",
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
    parts = re.split(r"[;,|]\s*|\s+", x)  # split on ; , | or whitespace
    return [p.strip() for p in parts if p.strip()]

def bucket_comment(s: str) -> str:
    s = str(s)
    s = re.sub(r"\d{4}-\d{2}-\d{2}.*", "", s)  # strip dates and after
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

# Save combined
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

# Parse datetime (optional but nice)
if DT_COL in df.columns:
    df[DT_COL] = pd.to_datetime(df[DT_COL], errors="coerce", utc=True)

# QC subset
qc_mask = (df[QC_COL] != "") | (df[FLAG_COL] != "") | (df[COMMENT_COL] != "")
qc_df = df.loc[qc_mask].copy()
print(f"Rows with any qc_outcome/flag/comment: {len(qc_df):,}")

# ----------------------------
# 3) REPORTS
# ----------------------------
save_counts(qc_df[QC_COL], os.path.join(OUT_DIR, "BC_qc_outcome_counts.csv"), name_col=QC_COL)

# flag token counts (splitting multi-flags)
flag_tokens = qc_df[FLAG_COL].apply(split_flags).explode()
flag_token_counts = (
    flag_tokens.dropna().replace("", "(blank)")
    .value_counts()
    .rename_axis("flag_token")
    .reset_index(name="n_rows_with_flag")
)
flag_token_counts.to_csv(os.path.join(OUT_DIR, "BC_flag_token_counts.csv"), index=False)

# flag combo counts (exact string)
save_counts(qc_df[FLAG_COL], os.path.join(OUT_DIR, "BC_flag_combo_counts.csv"), name_col="flag_combo")

# comment counts (exact and bucketed)
save_counts(qc_df[COMMENT_COL], os.path.join(OUT_DIR, "BC_comment_exact_counts.csv"), name_col="comment_exact")

comment_bucket_counts = (
    qc_df[COMMENT_COL].apply(bucket_comment)
    .value_counts()
    .rename_axis("comment_bucket")
    .reset_index(name="n_rows")
)
comment_bucket_counts.to_csv(os.path.join(OUT_DIR, "BC_comment_bucket_counts.csv"), index=False)

# status fields counts (all rows, not just QC rows)
for col in STATUS_COLS:
    if col in df.columns:
        s = clean_text(df[col])
        save_counts(s, os.path.join(OUT_DIR, f"BC_{col}_counts.csv"), name_col=col)

# QC time-of-day (optional)
if DT_COL in qc_df.columns:
    qc_time_counts = (
        qc_df[DT_COL].dropna().dt.strftime("%H:%M:%S")
        .value_counts()
        .rename_axis("time_UTC")
        .reset_index(name="n_rows")
    )
    qc_time_counts.to_csv(os.path.join(OUT_DIR, "BC_qc_time_of_day_counts.csv"), index=False)

print("\nDone. Reports saved to:", OUT_DIR)
print("Key outputs:")
print(" - BC_qc_outcome_counts.csv")
print(" - BC_flag_token_counts.csv")
print(" - BC_flag_combo_counts.csv")
print(" - BC_comment_exact_counts.csv")
print(" - BC_comment_bucket_counts.csv")
print(" - BC_status*_counts.csv (if present)")
