
"""
plot_monthly_source_family_profiles.py

Make stacked-bar PMF profile plots from the selected monthly factor table.

This script makes TWO useful plot styles:

1) By month:
   - one figure per month
   - x-axis = selected factors in that month
   - x tick label includes factor + source family
   - stacked bar segments = species % of factor total

2) By source family:
   - one figure per source-family label
   - x-axis = month (or month + factor if multiple factors in a month)
   - stacked bar segments = species % of factor total
   - good for comparing how "same-family" factors differ across months

Inputs:
  pmf_monthly_factor_decision_tables/_ALL_monthly_factor_decision_table.csv
  pmf_outputs/<selected_run>/*profiles*.txt

Outputs:
  pmf_monthly_factor_decision_tables/profile_plots/
    by_month/
      2023_06_Jun_selected_profiles_stacked.png
      ...
    by_source_family/
      S-rich_regional_secondary_sulfate_profiles_stacked.png
      ...
"""

from pathlib import Path
import re
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
    r"D:\Documents\PhD-Research\Xact python code\pmf_monthly_factor_decision_tables\profile_plots"
)

OUT_BY_MONTH = OUT_ROOT / "by_month"
OUT_BY_SOURCE = OUT_ROOT / "by_source_family"

OUT_BY_MONTH.mkdir(parents=True, exist_ok=True)
OUT_BY_SOURCE.mkdir(parents=True, exist_ok=True)

# Species to display in stacked bars.
# Keep this list short enough that the stacked bars remain readable.
DISPLAY_SPECIES = [
    "S", "K", "Cl", "Ca", "Ti", "Fe", "Mn", "Cu", "Zn",
    "As", "Se", "Br", "Sr", "Ba", "Pb", "Bi", "Si"
]

# If remaining species not in DISPLAY_SPECIES sum to >0, combine into "Other".
INCLUDE_OTHER = True

# Drop tiny segments from legend/plot if species contribution is always below this
# across a given figure. Units are % of factor total.
MIN_SPECIES_TO_SHOW = 0.5

# ==================================================


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
    Parse EPA PMF profile file:
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
            try:
                value = float(val)
            except Exception:
                value = np.nan

            rows.append({
                "species": species,
                "factor": f"F{i}",
                "pct_factor_total": value,
            })

    long = pd.DataFrame(rows)
    if long.empty:
        return pd.DataFrame()

    wide = long.pivot_table(
        index="species",
        columns="factor",
        values="pct_factor_total",
        aggfunc="first",
    ).fillna(0)

    cols = sorted(wide.columns, key=lambda x: int(re.search(r"(\d+)", x).group(1)))
    return wide[cols]


def sanitize_filename(s):
    s = str(s)
    s = re.sub(r"[^\w\-]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def load_selected_factor_profiles():
    df = pd.read_csv(DECISION_TABLE)

    rows = []

    for _, r in df.iterrows():
        run_name = r["selected_run"]
        factor = factor_short(r["factor"])

        profile_file = find_first(
            PMF_OUTPUTS_ROOT / run_name,
            ["*profiles*.txt", "*profile*.txt"]
        )

        if profile_file is None:
            continue

        wide = parse_profiles_pct_factor_total(profile_file)
        if wide.empty or factor not in wide.columns:
            continue

        series = wide[factor].copy()

        # Build a row with species values
        out = {
            "month_key": r["month_key"],
            "selected_run": run_name,
            "selected_n_factors": r["selected_n_factors"],
            "factor": factor,
            "source_family_label": r["source_family_label"],
            "timing_label": r.get("timing_label", ""),
            "event_dominated_flag": r.get("event_dominated_flag", False),
            "review_notes": r.get("review_notes", ""),
        }

        for sp in series.index:
            out[sp] = float(series.loc[sp])

        rows.append(out)

    prof = pd.DataFrame(rows)
    return prof


def build_plot_matrix(df_plot):
    # Determine which display species are present enough to show.
    species_cols = [c for c in DISPLAY_SPECIES if c in df_plot.columns]

    keep_species = []
    for sp in species_cols:
        if pd.to_numeric(df_plot[sp], errors="coerce").fillna(0).max() >= MIN_SPECIES_TO_SHOW:
            keep_species.append(sp)

    if not keep_species:
        keep_species = [c for c in species_cols]

    plot_df = df_plot.copy()

    if INCLUDE_OTHER:
        all_species_cols = [c for c in df_plot.columns if c not in [
            "month_key", "selected_run", "selected_n_factors", "factor",
            "source_family_label", "timing_label", "event_dominated_flag", "review_notes",
            "x_label"
        ]]
        other_cols = [c for c in all_species_cols if c not in keep_species]
        if other_cols:
            plot_df["Other"] = plot_df[other_cols].apply(pd.to_numeric, errors="coerce").fillna(0).sum(axis=1)
            if plot_df["Other"].max() >= MIN_SPECIES_TO_SHOW:
                keep_species = keep_species + ["Other"]

    return plot_df, keep_species


def make_stacked_bar(df_plot, x_col, title, out_file, rotation=35):
    if df_plot.empty:
        return

    plot_df, species_to_show = build_plot_matrix(df_plot)

    x = np.arange(len(plot_df))
    bottom = np.zeros(len(plot_df), dtype=float)

    plt.figure(figsize=(max(8, 1.3 * len(plot_df)), 6))

    for sp in species_to_show:
        vals = pd.to_numeric(plot_df[sp], errors="coerce").fillna(0).values
        plt.bar(x, vals, bottom=bottom, label=sp)
        bottom += vals

    plt.xticks(x, plot_df[x_col].tolist(), rotation=rotation, ha="right")
    plt.ylabel("% of factor total")
    plt.xlabel("")
    plt.title(title)
    plt.legend(title="Species", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()

    out_file = Path(out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_file, dpi=200)
    plt.close()


def plot_by_month(prof):
    for month_key, g in prof.groupby("month_key"):
        g = g.copy().sort_values("factor")

        # x tick label includes factor + source family
        g["x_label"] = (
            g["factor"].astype(str)
            + "\n"
            + g["source_family_label"].astype(str)
        )

        title = f"{month_key}: selected PMF factor profiles"
        out_file = OUT_BY_MONTH / f"{month_key}_selected_profiles_stacked.png"

        make_stacked_bar(g, "x_label", title, out_file, rotation=35)


def plot_by_source_family(prof):
    for source_family, g in prof.groupby("source_family_label"):
        g = g.copy().sort_values(["month_key", "factor"])

        # If multiple factors in one month share same label, keep both with factor ID.
        g["x_label"] = g["month_key"].astype(str) + "\n" + g["factor"].astype(str)

        title = f"{source_family}: profile comparison across months"
        out_file = OUT_BY_SOURCE / f"{sanitize_filename(source_family)}_profiles_stacked.png"

        make_stacked_bar(g, "x_label", title, out_file, rotation=35)


def main():
    prof = load_selected_factor_profiles()

    if prof.empty:
        raise RuntimeError("No selected factor profiles could be loaded.")

    prof_out = OUT_ROOT / "_selected_factor_profiles_long.csv"
    prof.to_csv(prof_out, index=False)

    plot_by_month(prof)
    plot_by_source_family(prof)

    print("Wrote:")
    print(prof_out)
    print(OUT_BY_MONTH)
    print(OUT_BY_SOURCE)


if __name__ == "__main__":
    main()
