import os
import glob
import re
import pandas as pd

# ----------------------------
# USER SETTINGS
# ----------------------------
IN_DIR = r"D:/Documents/PhD-Research/ASCENT-intercomparison/Xact2024L1b"   # folder with monthly CSVs
FILE_GLOB = "ASCENT_Xact_Lawrenceville_*_L1b.csv"                          # adjust if needed

OUT_DIR = r"D:/Documents/PhD-Research/ASCENT-intercomparison/Xact2024L2"
COMBINED_CSV = os.path.join(OUT_DIR, "ASCENT_Xact_Lawrenceville_2024_L1b_all_months.csv")

FLAG_COL = "flag"
QC_COL = "qc_outcome"
COMMENT_COL = "comment"
ALARM_COL = "alarm"
DT_COL = "sample_datetime_UTC"

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
    s = re.sub(r"\d{4}-\d{2}-\d{2}.*", "", s)       # remove dates and after
    s = re.sub(r"\d+(\.\d+)?", "#", s)              # replace numbers with #
    s = re.sub(r"\s+", " ", s).strip()
    return s if s else "(blank)"

# ----------------------------
# 1) LOAD + CONCATENATE
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

print(f"Files found: {len(paths)}")
print(f"Combined rows: {len(df):,}")
print("Example files:", [os.path.basename(p) for p in paths[:5]])

# Optional: save combined for reuse (handy)
df.to_csv(COMBINED_CSV, index=False)
print("Saved combined CSV:", COMBINED_CSV)

# ----------------------------
# 2) QC-PATTERN SUMMARY ON COMBINED
# ----------------------------
# Ensure cols exist
for c in [FLAG_COL, QC_COL, COMMENT_COL, ALARM_COL]:
    if c not in df.columns:
        df[c] = ""

df[FLAG_COL] = clean_text(df[FLAG_COL])
df[QC_COL] = clean_text(df[QC_COL])
df[COMMENT_COL] = clean_text(df[COMMENT_COL])
df[ALARM_COL] = clean_text(df[ALARM_COL])

# Parse datetime (helps time-of-day + sanity checks)
if DT_COL in df.columns:
    df[DT_COL] = pd.to_datetime(df[DT_COL], errors="coerce", utc=True)

mask_qc = (df[FLAG_COL] != "") | (df[QC_COL] != "") | (df[COMMENT_COL] != "") | (df[ALARM_COL] != "")
qc_df = df.loc[mask_qc].copy()

print(f"Rows with any QC/alarm/flag/comment info: {len(qc_df):,}")

# QC outcome counts
qc_outcome_counts = (
    qc_df[QC_COL].replace("", "(blank)")
    .value_counts(dropna=False)
    .rename_axis(QC_COL)
    .reset_index(name="n_rows")
)

# Flag token counts
flag_tokens = qc_df[FLAG_COL].apply(split_flags).explode()
flag_counts = (
    flag_tokens.dropna().replace("", "(blank)")
    .value_counts()
    .rename_axis("flag_token")
    .reset_index(name="n_rows_with_flag")
)

# Flag combo counts
flag_combo_counts = (
    qc_df[FLAG_COL].replace("", "(blank)")
    .value_counts()
    .rename_axis("flag_combo")
    .reset_index(name="n_rows")
)

# Comment exact + bucket counts
comment_counts = (
    qc_df[COMMENT_COL].replace("", "(blank)")
    .value_counts()
    .rename_axis("comment_exact")
    .reset_index(name="n_rows")
)

comment_bucket_counts = (
    qc_df[COMMENT_COL].apply(bucket_comment)
    .value_counts()
    .rename_axis("comment_bucket")
    .reset_index(name="n_rows")
)

# Alarm counts
alarm_counts = (
    qc_df[ALARM_COL].replace("", "(blank)")
    .value_counts()
    .rename_axis(ALARM_COL)
    .reset_index(name="n_rows")
)

# Time-of-day counts
time_counts = None
if DT_COL in qc_df.columns:
    time_counts = (
        qc_df[DT_COL].dropna().dt.strftime("%H:%M:%S")
        .value_counts()
        .rename_axis("time_UTC")
        .reset_index(name="n_rows")
    )

# Rare patterns: flags that occur only a few times (this is your “weird errors” detector)
RARE_N = 10
rare_flags = flag_counts[flag_counts["n_rows_with_flag"] <= RARE_N].copy()

# Rare comment buckets
rare_comment_buckets = comment_bucket_counts[comment_bucket_counts["n_rows"] <= RARE_N].copy()

# ----------------------------
# 3) EXPORT REPORTS
# ----------------------------
qc_outcome_counts.to_csv(os.path.join(OUT_DIR, "ALL_qc_outcome_counts.csv"), index=False)
flag_counts.to_csv(os.path.join(OUT_DIR, "ALL_flag_token_counts.csv"), index=False)
flag_combo_counts.to_csv(os.path.join(OUT_DIR, "ALL_flag_combo_counts.csv"), index=False)
comment_counts.to_csv(os.path.join(OUT_DIR, "ALL_comment_exact_counts.csv"), index=False)
comment_bucket_counts.to_csv(os.path.join(OUT_DIR, "ALL_comment_bucket_counts.csv"), index=False)
alarm_counts.to_csv(os.path.join(OUT_DIR, "ALL_alarm_counts.csv"), index=False)

rare_flags.to_csv(os.path.join(OUT_DIR, f"ALL_rare_flags_le_{RARE_N}.csv"), index=False)
rare_comment_buckets.to_csv(os.path.join(OUT_DIR, f"ALL_rare_comment_buckets_le_{RARE_N}.csv"), index=False)

if time_counts is not None:
    time_counts.to_csv(os.path.join(OUT_DIR, "ALL_qc_time_of_day_counts.csv"), index=False)

print("\nSaved combined + reports to:", OUT_DIR)
print(f"Also saved rare lists (<= {RARE_N} occurrences).")
