
"""
summarize_same_source_family_similarity.py

Generic script to summarize how similar factors are when they were assigned
to the same source-family label.

This should be run AFTER:
  build_monthly_pmf_factor_decision_table_v2.py

It reads:
  pmf_monthly_factor_decision_tables/_ALL_monthly_factor_decision_table.csv
  pmf_outputs/<selected_run>/*profiles*.txt

It outputs:
  pmf_monthly_factor_decision_tables/_ALL_same_source_family_pairwise_similarity.csv
  pmf_monthly_factor_decision_tables/_ALL_same_source_family_similarity_summary.csv
  pmf_monthly_factor_decision_tables/plots/_ALL_same_family_median_profile_similarity.png

Important:
- Across different months, timestamp correlation is NOT physically meaningful.
- This compares profile similarity and timing metric similarity.
- Profile cosine similarity asks: are the chemical fingerprints similar?
- Timing differences ask: do they have similar night/day or weekday/weekend behavior?
"""

from pathlib import Path
import re
import itertools

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ================= USER SETTINGS =================

PMF_OUTPUTS_ROOT = Path(
    r"D:\Documents\PhD-Research\Xact python code\pmf_outputs"
)

DECISION_TABLE = Path(
    r"D:\Documents\PhD-Research\Xact python code\pmf_monthly_factor_decision_tables\_ALL_monthly_factor_decision_table.csv"
)

OUT_ROOT = Path(
    r"D:\Documents\PhD-Research\Xact python code\pmf_monthly_factor_decision_tables"
)

PLOTS_ROOT = OUT_ROOT / "plots"
PLOTS_ROOT.mkdir(parents=True, exist_ok=True)

SPECIES_ORDER = [
    "Si", "S", "Cl", "K", "Ca", "Ti", "Cr", "Mn", "Fe",
    "Cu", "Zn", "As", "Se", "Br", "Sr", "Ba", "Pb", "Bi"
]

# ==================================================


def safe_float(x):
    try:
        return float(x)
    except Exception:
        return np.nan


def factor_short(x):
    m = re.search(r"(\d+)", str(x))
    if not m:
        return str(x)
    return f"F{m.group(1)}"


def find_first(folder, patterns):
    folder = Path(folder)
    if not folder.exists():
        return None

    for pattern in patterns:
        hits = sorted(folder.glob(pattern))
        if hits:
            return hits[0]
    return None


def parse_profiles_pct_factor_total(profile_file):
    """
    Parse EPA PMF profiles file:
      Factor Profiles (% of factor total)

    Returns species x factor table with columns F1, F2, ...
    """
    profile_file = Path(profile_file)

    if not profile_file.exists():
        return pd.DataFrame()

    lines = profile_file.read_text(errors="replace").splitlines()

    start_idx = None
    for i, line in enumerate(lines):
        if "Factor Profiles (% of factor total)" in line:
            start_idx = i
            break

    if start_idx is None:
        return pd.DataFrame()

    data_lines = []
    started = False

    for line in lines[start_idx + 1:]:
        stripped = line.strip()

        if not stripped:
            if started:
                break
            continue

        if "Factor" in stripped and not re.match(r"^\d+\s+", stripped):
            continue

        if re.match(r"^\d+\s+[A-Za-z][A-Za-z0-9]*\s+", stripped):
            data_lines.append(stripped)
            started = True
        elif started:
            break

    rows = []

    for line in data_lines:
        parts = re.split(r"\s+", line.strip())

        if len(parts) < 4:
            continue

        species = parts[1]
        values = parts[2:]

        for i, val in enumerate(values, start=1):
            rows.append({
                "species": species,
                "factor": f"F{i}",
                "pct_factor_total": safe_float(val),
            })

    long = pd.DataFrame(rows)

    if long.empty:
        return pd.DataFrame()

    wide = long.pivot_table(
        index="species",
        columns="factor",
        values="pct_factor_total",
        aggfunc="first",
    )

    ordered = [s for s in SPECIES_ORDER if s in wide.index]
    remaining = [s for s in wide.index if s not in ordered]
    wide = wide.loc[ordered + remaining]

    cols = sorted(wide.columns, key=lambda x: int(re.search(r"(\d+)", x).group(1)))
    return wide[cols].fillna(0)


def cosine_similarity(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    species_mask = np.isfinite(a) & np.isfinite(b)
    a = a[species_mask]
    b = b[species_mask]

    if len(a) == 0:
        return np.nan

    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return np.nan

    return float(np.dot(a, b) / denom)


def get_profile_vector(run_name, factor):
    """
    Returns full selected profile vector using all parsed species.
    """
    profile_file = find_first(
        PMF_OUTPUTS_ROOT / run_name,
        ["*profiles*.txt", "*profile*.txt"]
    )

    if profile_file is None:
        raise FileNotFoundError(f"No profiles file found for {run_name}")

    wide = parse_profiles_pct_factor_total(profile_file)

    factor = factor_short(factor)

    if factor not in wide.columns:
        raise KeyError(f"{factor} not found in profile file for {run_name}")

    return wide[factor], profile_file


def profile_cosine_between(a_row, b_row):
    vec_a, file_a = get_profile_vector(a_row["selected_run"], a_row["factor"])
    vec_b, file_b = get_profile_vector(b_row["selected_run"], b_row["factor"])

    all_species = sorted(set(vec_a.index).union(set(vec_b.index)))
    va = vec_a.reindex(all_species).fillna(0).values
    vb = vec_b.reindex(all_species).fillna(0).values

    return cosine_similarity(va, vb), str(file_a), str(file_b)


def classify_similarity(profile_cosine):
    if pd.isna(profile_cosine):
        return "unavailable"
    if profile_cosine >= 0.95:
        return "very similar profile"
    if profile_cosine >= 0.85:
        return "similar profile"
    if profile_cosine >= 0.70:
        return "moderately similar profile"
    return "different profile despite same label"


def main():
    df = pd.read_csv(DECISION_TABLE)

    rows = []

    for source_family, g in df.groupby("source_family_label"):
        if len(g) < 2:
            continue

        for idx_a, idx_b in itertools.combinations(g.index, 2):
            a = df.loc[idx_a]
            b = df.loc[idx_b]

            try:
                profile_cos, file_a, file_b = profile_cosine_between(a, b)
            except Exception as e:
                profile_cos = np.nan
                file_a = ""
                file_b = ""
                error = str(e)
            else:
                error = ""

            nd_a = safe_float(a.get("night_day_ratio_fixed", np.nan))
            nd_b = safe_float(b.get("night_day_ratio_fixed", np.nan))
            ww_a = safe_float(a.get("weekend_weekday_ratio", np.nan))
            ww_b = safe_float(b.get("weekend_weekday_ratio", np.nan))

            rows.append({
                "source_family_label": source_family,

                "factor_a": f"{a['month_key']} {a['factor']}",
                "selected_run_a": a["selected_run"],
                "factor_only_a": a["factor"],
                "top_species_a": a.get("top_species_pct_factor_total", ""),
                "timing_label_a": a.get("timing_label", ""),
                "night_day_ratio_a": nd_a,
                "weekend_weekday_ratio_a": ww_a,

                "factor_b": f"{b['month_key']} {b['factor']}",
                "selected_run_b": b["selected_run"],
                "factor_only_b": b["factor"],
                "top_species_b": b.get("top_species_pct_factor_total", ""),
                "timing_label_b": b.get("timing_label", ""),
                "night_day_ratio_b": nd_b,
                "weekend_weekday_ratio_b": ww_b,

                "profile_cosine_similarity": profile_cos,
                "profile_similarity_class": classify_similarity(profile_cos),

                "abs_night_day_ratio_difference": abs(nd_a - nd_b) if pd.notna(nd_a) and pd.notna(nd_b) else np.nan,
                "abs_weekend_weekday_ratio_difference": abs(ww_a - ww_b) if pd.notna(ww_a) and pd.notna(ww_b) else np.nan,

                "profile_file_a": file_a,
                "profile_file_b": file_b,
                "parse_error": error,
            })

    pairwise = pd.DataFrame(rows)

    pairwise_out = OUT_ROOT / "_ALL_same_source_family_pairwise_similarity.csv"
    pairwise.to_csv(pairwise_out, index=False)

    if pairwise.empty:
        print("No pairwise same-family comparisons were generated.")
        return

    factor_counts = df.groupby("source_family_label").size().rename("n_selected_factors").reset_index()

    summary = pairwise.groupby("source_family_label").agg(
        n_pairs=("profile_cosine_similarity", "size"),
        median_profile_cosine=("profile_cosine_similarity", "median"),
        min_profile_cosine=("profile_cosine_similarity", "min"),
        max_profile_cosine=("profile_cosine_similarity", "max"),
        median_abs_night_day_ratio_difference=("abs_night_day_ratio_difference", "median"),
        median_abs_weekend_weekday_ratio_difference=("abs_weekend_weekday_ratio_difference", "median"),
    ).reset_index()

    summary = summary.merge(factor_counts, on="source_family_label", how="left")
    summary = summary[
        [
            "source_family_label",
            "n_selected_factors",
            "n_pairs",
            "median_profile_cosine",
            "min_profile_cosine",
            "max_profile_cosine",
            "median_abs_night_day_ratio_difference",
            "median_abs_weekend_weekday_ratio_difference",
        ]
    ].sort_values("median_profile_cosine", ascending=False)

    summary_out = OUT_ROOT / "_ALL_same_source_family_similarity_summary.csv"
    summary.to_csv(summary_out, index=False)

    # Plot median profile similarity
    plot_df = summary.sort_values("median_profile_cosine", ascending=True)

    plt.figure(figsize=(8, max(4, 0.5 * len(plot_df))))
    plt.barh(plot_df["source_family_label"], plot_df["median_profile_cosine"])
    plt.xlabel("Median profile cosine similarity")
    plt.ylabel("Source-family label")
    plt.title("Similarity among selected factors with the same source-family label")
    plt.xlim(0, 1.05)
    plt.tight_layout()

    plot_out = PLOTS_ROOT / "_ALL_same_family_median_profile_similarity.png"
    plt.savefig(plot_out, dpi=200)
    plt.close()

    print("Wrote:")
    print(pairwise_out)
    print(summary_out)
    print(plot_out)


if __name__ == "__main__":
    main()
