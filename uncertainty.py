import re
import pandas as pd
from pathlib import Path

INPUT_FILE = Path(r"Xact_EST_May2023_Oct2025_combined.csv")
TIME_COL = "TIME"

EXCLUDE_RANGES = [
    ("2024-01-09 00:00", "2024-02-13 23:59"),
    ("2024-07-02 00:00", "2024-08-08 23:59"),
]

EXCLUDE_METALS = ["Nb"]


def clean_metal_name(col):
    s = str(col).strip()
    s = re.sub(r"\(.*?\)", "", s)
    s = s.replace("Uncert", "").replace("uncert", "")
    s = re.sub(r"\d+", "", s)
    s = re.sub(r"[^A-Za-z]", "", s)
    return s


def find_xact_pairs(df):
    conc_cols = []
    uncert_cols = []

    for c in df.columns:
        cl = str(c).lower()
        if c == TIME_COL:
            continue
        if "uncert" in cl:
            uncert_cols.append(c)
        elif "ng/m3" in cl or "ng/m³" in cl or "(ng" in cl:
            conc_cols.append(c)

    conc_map = {clean_metal_name(c): c for c in conc_cols}
    uncert_map = {clean_metal_name(c): c for c in uncert_cols}

    pairs = {}
    for m in sorted(set(conc_map) & set(uncert_map)):
        if m not in EXCLUDE_METALS:
            pairs[m] = {"conc": conc_map[m], "unc": uncert_map[m]}

    return pairs


def parse_time(df):
    df = df.copy()
    t = pd.to_datetime(df[TIME_COL], errors="coerce", utc=True)
    df["time_local"] = t.dt.tz_convert("America/New_York").dt.tz_localize(None)
    df = df.dropna(subset=["time_local"])
    return df


def apply_exclusions(df):
    keep = pd.Series(True, index=df.index)
    for start, end in EXCLUDE_RANGES:
        s = pd.to_datetime(start)
        e = pd.to_datetime(end)
        keep &= ~df["time_local"].between(s, e)
    return df.loc[keep].reset_index(drop=True)


df = pd.read_csv(INPUT_FILE, low_memory=False)
df = parse_time(df)
df = apply_exclusions(df)

pairs = find_xact_pairs(df)

rows = []

for metal, cols in pairs.items():
    conc = pd.to_numeric(df[cols["conc"]], errors="coerce")
    unc = pd.to_numeric(df[cols["unc"]], errors="coerce")

    conc = conc.mask(conc < 0)
    unc = unc.mask(unc < 0)

    pass_1x = (
        conc.notna()
        & unc.notna()
        & (unc > 0)
        & (conc > unc)
    )

    rows.append({
        "metal": metal,
        "n_hours_total": len(df),
        "n_hours_pass_1x_uncertainty": int(pass_1x.sum()),
        "percent_pass_1x_uncertainty": 100 * pass_1x.mean(),
        "passes_1x_at_least_once": pass_1x.any(),
    })

summary = pd.DataFrame(rows).sort_values(
    "percent_pass_1x_uncertainty",
    ascending=False
)

passed = summary[summary["passes_1x_at_least_once"]]["metal"].tolist()

print("\nSpecies that pass C > 1σT at least once:")
print(", ".join(passed))

summary.to_csv("species_1x_uncertainty_summary.csv", index=False)

with open("species_pass_1x_uncertainty_list.txt", "w") as f:
    f.write(", ".join(passed))

print("\nSaved:")
print("species_1x_uncertainty_summary.csv")
print("species_pass_1x_uncertainty_list.txt")