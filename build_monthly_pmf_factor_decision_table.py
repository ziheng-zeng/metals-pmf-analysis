
"""
build_monthly_pmf_factor_decision_table.py

Generic PMF factor decision + labeling table builder.

This script reads:
1. PMF output folders:
   D:/.../pmf_outputs/<run_folder>/
      *_profiles.txt
      *_diagnostics.txt

2. Timestamp-merged PMF output folders:
   D:/.../pmf_outputs_merged/<run_folder>/
      *_contributions_merged_with_time.csv
      *_factor_timing_summary.csv
      *_factor_top_hours.csv

3. Time-aware factor-number comparison folders:
   D:/.../pmf_factor_number_comparisons_timeaware/<month_key>/
      timeaware_pairwise_factor_matches.csv
      timeaware_best_factor_matches.csv

It produces:
  pmf_monthly_factor_decision_tables/
    _ALL_monthly_model_selection_summary.csv
    _ALL_monthly_factor_decision_table.csv
    _ALL_source_family_by_month.csv
    <month_key>/
      <month_key>_model_selection_summary.csv
      <month_key>_factor_decision_table.csv
      plots

Main goal:
For each month, create a practical table:
  month
  selected factor number
  selected run
  factor
  source-family label
  timing label
  stability across 4f/5f/6f
  event-dominated flag
  notes

This is meant for first-pass interpretation and advisor discussion.
You should still manually review/adjust final source labels.
"""

from pathlib import Path
import re
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ================= USER SETTINGS =================

PMF_OUTPUTS_ROOT = Path(
    r"D:\Documents\PhD-Research\Xact python code\pmf_outputs"
)

PMF_MERGED_ROOT = Path(
    r"D:\Documents\PhD-Research\Xact python code\pmf_outputs_merged"
)

TIMEAWARE_COMPARISON_ROOT = Path(
    r"D:\Documents\PhD-Research\Xact python code\pmf_factor_number_comparisons_timeaware"
)

OUT_ROOT = Path(
    r"D:\Documents\PhD-Research\Xact python code\pmf_monthly_factor_decision_tables"
)

OUT_ROOT.mkdir(parents=True, exist_ok=True)

# Factor numbers to consider.
FACTOR_COUNTS_TO_COMPARE = [4, 5, 6]

# Manual override if you want to force selected factor number for a month.
# Example:
# SELECTED_FACTOR_OVERRIDE = {
#     "2025_01_Jan": 5,
#     "2023_08_Aug": 5,
# }
SELECTED_FACTOR_OVERRIDE = {}

# Model selection settings.
MIN_CONVERGED_FRACTION = 0.75

# Simplicity rule:
# If a simpler model is within this fraction of the best eligible Qtrue/Qexp,
# choose the simpler model.
# Example: 0.10 means within 10%.
SIMPLER_MODEL_Q_TOLERANCE = 0.10

# Stability thresholds.
VERY_STABLE_PROFILE_COSINE = 0.90
VERY_STABLE_TIME_CORR = 0.70

STABLE_PROFILE_COSINE = 0.80
STABLE_TIME_CORR = 0.50

# Event-dominated if max / median is very high.
EVENT_DOMINATED_MAX_MEDIAN_RATIO = 20

# Time behavior thresholds.
NIGHT_ENHANCED_RATIO = 1.30
DAY_ENHANCED_RATIO = 1 / NIGHT_ENHANCED_RATIO

WEEKEND_ENHANCED_RATIO = 1.30
WEEKDAY_ENHANCED_RATIO = 1 / WEEKEND_ENHANCED_RATIO

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


def parse_run_folder_name(name):
    """
    Parse names like:
      2025_01_Jan_5f
      2023_07_Jul_4f
    """
    m = re.search(r"(20\d{2})_(\d{2})_([A-Za-z]{3})_(\d+)f$", str(name))
    if not m:
        return None

    return {
        "year": int(m.group(1)),
        "month": int(m.group(2)),
        "month_name": m.group(3),
        "n_factors": int(m.group(4)),
        "month_key": f"{m.group(1)}_{m.group(2)}_{m.group(3)}",
    }


def factor_short(x):
    """
    Convert Factor_1 or factor 1 to F1.
    """
    m = re.search(r"(\d+)", str(x))
    if not m:
        return str(x)
    return f"F{m.group(1)}"


def factor_col(x):
    """
    Convert F1 to Factor_1.
    """
    m = re.search(r"(\d+)", str(x))
    if not m:
        return str(x)
    return f"Factor_{m.group(1)}"


def find_first(folder, patterns):
    folder = Path(folder)
    if not folder.exists():
        return None

    for pattern in patterns:
        hits = sorted(folder.glob(pattern))
        if hits:
            return hits[0]

    return None


def list_available_runs():
    rows = []

    for folder in sorted(PMF_OUTPUTS_ROOT.iterdir()):
        if not folder.is_dir():
            continue

        info = parse_run_folder_name(folder.name)
        if info is None:
            continue

        if info["n_factors"] not in FACTOR_COUNTS_TO_COMPARE:
            continue

        profile_file = find_first(folder, ["*profiles*.txt", "*profile*.txt"])
        diagnostics_file = find_first(folder, ["*diagnostics*.txt", "*diagnostic*.txt"])

        merged_folder = PMF_MERGED_ROOT / folder.name
        timing_file = find_first(merged_folder, ["*factor_timing_summary.csv", "*timing_summary.csv"])
        merged_contrib_file = find_first(merged_folder, ["*contributions_merged_with_time.csv", "*merged_with_time.csv"])
        top_hours_file = find_first(merged_folder, ["*factor_top_hours.csv", "*top_hours.csv"])

        rows.append({
            "run_folder": folder.name,
            "run_path": str(folder),
            "profile_file": str(profile_file) if profile_file else "",
            "diagnostics_file": str(diagnostics_file) if diagnostics_file else "",
            "timing_file": str(timing_file) if timing_file else "",
            "merged_contrib_file": str(merged_contrib_file) if merged_contrib_file else "",
            "top_hours_file": str(top_hours_file) if top_hours_file else "",
            **info,
        })

    return pd.DataFrame(rows)


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

        # Header line can look like Factor 1 Factor 2...
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


def species_pct(profile_wide, factor, species):
    if profile_wide.empty:
        return 0.0
    if species not in profile_wide.index:
        return 0.0
    if factor not in profile_wide.columns:
        return 0.0

    val = profile_wide.loc[species, factor]
    if pd.isna(val):
        return 0.0

    return float(val)


def top_species_string(profile_wide, factor, n=8):
    if profile_wide.empty or factor not in profile_wide.columns:
        return ""

    s = profile_wide[factor].dropna().sort_values(ascending=False)
    s = s[s > 0].head(n)

    return "; ".join([f"{sp}:{val:.1f}" for sp, val in s.items()])


def parse_datetime_safe(x):
    """Parse a datetime-like value safely."""
    if x is None:
        return pd.NaT
    if isinstance(x, float) and pd.isna(x):
        return pd.NaT
    try:
        return pd.to_datetime(x, errors="coerce")
    except Exception:
        return pd.NaT


def fireworks_timing_note(month_key, timing=None):
    """
    Identify whether a factor's top-hour timing overlaps obvious fireworks windows.

    This is intentionally conservative and only uses top-hour summary fields.
    For final event analysis, use the full factor_top_hours file and predefined event windows.
    """
    if timing is None or not isinstance(timing, dict):
        return ""

    date_fields = [
        "top5pct_first_time", "top5pct_last_time",
        "top10pct_first_time", "top10pct_last_time",
        "top9pct_first_time", "top9pct_last_time",
    ]

    dates = []
    for c in date_fields:
        dt = parse_datetime_safe(timing.get(c, None))
        if pd.notna(dt):
            dates.append(dt)

    if not dates:
        return ""

    # New Year fireworks: strongest evidence if top hours touch Jan 1 early morning/evening.
    for dt in dates:
        if dt.month == 1 and dt.day == 1:
            return "top hours overlap New Year fireworks window"
        if dt.month == 12 and dt.day == 31 and dt.hour >= 18:
            return "top hours overlap New Year fireworks window"

    # July 4 / Independence Day fireworks. Broad enough to catch city celebrations and delayed peaks.
    for dt in dates:
        if dt.month == 7 and 3 <= dt.day <= 5:
            return "top hours overlap July 4 fireworks window"

    return ""


def assign_source_family(profile_wide, factor, month_key=None, timing=None):
    """
    First-pass rule-based source family label using % of factor total + simple timing context.

    Important:
    - This is an initial label, not final source attribution.
    - Fireworks labeling is chemistry + timing aware to avoid falsely labeling
      winter K-Cl factors or Fe-rich factors as fireworks.
    """
    S = species_pct(profile_wide, factor, "S")
    Cl = species_pct(profile_wide, factor, "Cl")
    K = species_pct(profile_wide, factor, "K")
    Ca = species_pct(profile_wide, factor, "Ca")
    Ti = species_pct(profile_wide, factor, "Ti")
    Si = species_pct(profile_wide, factor, "Si")
    Fe = species_pct(profile_wide, factor, "Fe")
    Mn = species_pct(profile_wide, factor, "Mn")
    Cu = species_pct(profile_wide, factor, "Cu")
    Zn = species_pct(profile_wide, factor, "Zn")
    As = species_pct(profile_wide, factor, "As")
    Se = species_pct(profile_wide, factor, "Se")
    Sr = species_pct(profile_wide, factor, "Sr")
    Ba = species_pct(profile_wide, factor, "Ba")
    Pb = species_pct(profile_wide, factor, "Pb")
    Bi = species_pct(profile_wide, factor, "Bi")

    fw_note = fireworks_timing_note(month_key, timing)
    has_fireworks_timing = bool(fw_note)

    # Strong fireworks-like: K-rich plus clear Cu and Ba/Sr support, not dominated by Fe/Cl.
    strong_fireworks_chemistry = (
        (K >= 20)
        and (Cu >= 0.5)
        and ((Ba >= 0.5) or (Sr >= 0.5))
        and (Fe < 40)
        and (Cl < 35)
    )

    # Weaker fireworks influence: used only when timing supports fireworks.
    possible_fireworks_chemistry = (
        (K >= 10)
        and (Cu >= 0.5)
        and ((Ba + Sr) >= 0.05)
        and (Fe < 50)
    )

    if strong_fireworks_chemistry and has_fireworks_timing:
        return "Fireworks-like K/Cu/Ba/Sr event"

    if strong_fireworks_chemistry:
        return "K/Cu/Ba/Sr event-like factor; check fireworks timing"

    # Winter K-Cl / chloride-rich. If this overlaps New Year and has weak firework support,
    # preserve the K-Cl identity while flagging possible fireworks influence.
    if (K + Cl >= 45) and (K >= 10) and (Cl >= 10):
        if has_fireworks_timing and possible_fireworks_chemistry:
            return "K-Cl/chloride with possible fireworks influence"
        return "K-Cl winter combustion/chloride"

    # Sulfate/regional.
    if S >= 50:
        return "S-rich regional/secondary sulfate"

    # Dust/road salt/crustal.
    if (Ca >= 25) or ((Ca + Ti + Si) >= 35 and Ca >= 8):
        return "Ca-rich dust/road salt"

    # Fe-rich industrial/metal/brake/steel. If K/Cu/Ba/Sr are also present near fireworks,
    # this could be an event-influenced metal factor rather than a pure source.
    if (Fe >= 35) or ((Fe + Mn) >= 45):
        if has_fireworks_timing and possible_fireworks_chemistry:
            return "Fe/K mixed metal-event factor; possible fireworks influence"
        return "Fe-rich metal/industrial"

    # Zn/Pb industrial plume.
    if (Zn >= 20) or ((Zn + Pb) >= 20 and Zn >= 10):
        return "Zn/Pb-rich industrial metal plume"

    # K-rich without enough Cl/Cu/Ba for specific label.
    if K >= 20:
        if has_fireworks_timing and possible_fireworks_chemistry:
            return "K-rich event factor; possible fireworks influence"
        return "K-rich combustion/biomass-like"

    # Mixed sulfate + dust/metals.
    if (S >= 20) and ((Ca + Fe + Ti + Si) >= 20):
        return "mixed sulfate + dust/metal"

    # Trace-metal industrial/coke-like candidate.
    if ((As + Se + Pb + Zn) >= 15) and ((As + Se + Pb) >= 0.5):
        return "trace-metal industrial/coke candidate"

    return "mixed/uncertain"

def parse_diagnostics_q_summary(diagnostics_file):
    """
    Parse EPA PMF diagnostics base run summary table.
    Returns one-row summary.
    """
    diagnostics_file = Path(diagnostics_file)

    if not diagnostics_file.exists():
        return {
            "n_base_runs": np.nan,
            "n_converged": np.nan,
            "converged_fraction": np.nan,
            "min_q_robust": np.nan,
            "min_q_true": np.nan,
            "min_qtrue_qexp": np.nan,
            "best_run_number": np.nan,
            "diagnostics_parse_note": "missing diagnostics file",
        }

    lines = diagnostics_file.read_text(errors="replace").splitlines()

    in_table = False
    rows = []

    for line in lines:
        stripped = line.strip()

        if stripped.startswith("Run #") and "Q(Robust)" in stripped and "Q(True)" in stripped:
            in_table = True
            continue

        if in_table:
            if not stripped:
                if rows:
                    break
                continue

            parts = re.split(r"\s+", stripped)

            if len(parts) >= 6 and parts[0].isdigit():
                rows.append({
                    "run_number": int(parts[0]),
                    "q_robust": safe_float(parts[1]),
                    "q_true": safe_float(parts[2]),
                    "converged": parts[3],
                    "n_steps": safe_float(parts[4]),
                    "qtrue_qexp": safe_float(parts[5]),
                })
            elif rows:
                break

    if not rows:
        return {
            "n_base_runs": np.nan,
            "n_converged": np.nan,
            "converged_fraction": np.nan,
            "min_q_robust": np.nan,
            "min_q_true": np.nan,
            "min_qtrue_qexp": np.nan,
            "best_run_number": np.nan,
            "diagnostics_parse_note": "base run table not parsed",
        }

    df = pd.DataFrame(rows)
    conv = df["converged"].astype(str).str.lower().eq("yes")

    # Best by Qtrue/Qexp among converged if possible.
    candidates = df[conv].copy()
    if candidates.empty:
        candidates = df.copy()

    best = candidates.sort_values("qtrue_qexp").iloc[0]

    return {
        "n_base_runs": len(df),
        "n_converged": int(conv.sum()),
        "converged_fraction": float(conv.mean()),
        "min_q_robust": float(candidates["q_robust"].min()),
        "min_q_true": float(candidates["q_true"].min()),
        "min_qtrue_qexp": float(candidates["qtrue_qexp"].min()),
        "best_run_number": int(best["run_number"]),
        "diagnostics_parse_note": "",
    }


def load_timing_summary(timing_file):
    timing_file = Path(timing_file)

    if not timing_file.exists():
        return pd.DataFrame()

    df = pd.read_csv(timing_file)

    if "factor" not in df.columns:
        return pd.DataFrame()

    df["factor_short"] = df["factor"].apply(factor_short)
    return df


def get_timing_for_factor(timing_df, factor):
    if timing_df.empty:
        return {}

    hit = timing_df[timing_df["factor_short"] == factor_short(factor)]
    if hit.empty:
        return {}

    return hit.iloc[0].to_dict()


def col_value(row, col, default=np.nan):
    if isinstance(row, dict):
        return row.get(col, default)
    return default


def timing_label(timing):
    labels = []

    nd = safe_float(col_value(timing, "night_day_ratio_fixed"))
    ww = safe_float(col_value(timing, "weekend_weekday_ratio"))
    peak = safe_float(col_value(timing, "peak_hour_by_mean"))

    if pd.notna(nd):
        if nd >= NIGHT_ENHANCED_RATIO:
            labels.append("night-enhanced")
        elif nd <= DAY_ENHANCED_RATIO:
            labels.append("day-enhanced")
        else:
            labels.append("weak day/night contrast")
    else:
        labels.append("day/night unavailable")

    if pd.notna(ww):
        if ww >= WEEKEND_ENHANCED_RATIO:
            labels.append("weekend-enhanced")
        elif ww <= WEEKDAY_ENHANCED_RATIO:
            labels.append("weekday-enhanced")
        else:
            labels.append("weak weekday/weekend contrast")
    else:
        labels.append("weekday/weekend unavailable")

    if pd.notna(peak):
        labels.append(f"peak {int(peak):02d}:00")

    return "; ".join(labels)


def event_dominated_from_timing(timing):
    max_val = safe_float(col_value(timing, "max"))
    median_val = safe_float(col_value(timing, "median"))

    if pd.isna(max_val) or pd.isna(median_val) or median_val <= 0:
        return np.nan, False

    ratio = max_val / median_val
    return ratio, bool(ratio >= EVENT_DOMINATED_MAX_MEDIAN_RATIO)


def get_top_fraction(timing, preferred_cols):
    for c in preferred_cols:
        val = col_value(timing, c)
        if pd.notna(safe_float(val)):
            return safe_float(val)
    return np.nan


def build_run_fit_table(runs):
    rows = []

    for _, run in runs.iterrows():
        diag = parse_diagnostics_q_summary(run["diagnostics_file"])
        rows.append({
            "run_folder": run["run_folder"],
            "month_key": run["month_key"],
            "year": run["year"],
            "month": run["month"],
            "month_name": run["month_name"],
            "n_factors": run["n_factors"],
            **diag,
        })

    return pd.DataFrame(rows)


def choose_selected_run(month_key, fit_table):
    """
    Generic selection rule:
      1. If manual override exists, use it.
      2. Exclude poorly converged runs if possible.
      3. Choose lowest Qtrue/Qexp among eligible.
      4. If simpler model is within tolerance, choose simpler model.
    """
    g = fit_table[fit_table["month_key"] == month_key].copy()

    if g.empty:
        return None, "No fit table rows"

    g = g.sort_values("n_factors")

    if month_key in SELECTED_FACTOR_OVERRIDE:
        n = SELECTED_FACTOR_OVERRIDE[month_key]
        hit = g[g["n_factors"] == n]
        if not hit.empty:
            return hit.iloc[0]["run_folder"], f"Manual override selected {n}f"

    # Eligible based on convergence, if convergence info exists.
    has_conv = g["converged_fraction"].notna().any()

    if has_conv:
        eligible = g[g["converged_fraction"] >= MIN_CONVERGED_FRACTION].copy()
        if eligible.empty:
            eligible = g.copy()
            conv_note = "No runs met convergence threshold; selected from all runs"
        else:
            conv_note = f"Eligible runs have converged fraction >= {MIN_CONVERGED_FRACTION:.2f}"
    else:
        eligible = g.copy()
        conv_note = "Convergence unavailable; selected using Qtrue/Qexp only"

    # If Q unavailable, choose middle/5f if possible.
    if eligible["min_qtrue_qexp"].isna().all():
        hit = eligible[eligible["n_factors"] == 5]
        if not hit.empty:
            return hit.iloc[0]["run_folder"], "Q unavailable; selected 5f as default middle solution"
        row = eligible.iloc[len(eligible) // 2]
        return row["run_folder"], "Q unavailable; selected middle available solution"

    eligible_q = eligible.dropna(subset=["min_qtrue_qexp"]).copy()
    best_q_row = eligible_q.sort_values("min_qtrue_qexp").iloc[0]
    best_q = best_q_row["min_qtrue_qexp"]

    # Simplicity rule.
    simpler_candidates = eligible_q[
        eligible_q["min_qtrue_qexp"] <= best_q * (1 + SIMPLER_MODEL_Q_TOLERANCE)
    ].sort_values("n_factors")

    selected = simpler_candidates.iloc[0]

    if selected["run_folder"] == best_q_row["run_folder"]:
        reason = (
            f"Selected {int(selected['n_factors'])}f: lowest eligible Qtrue/Qexp "
            f"({selected['min_qtrue_qexp']:.3g}). {conv_note}."
        )
    else:
        reason = (
            f"Selected simpler {int(selected['n_factors'])}f: Qtrue/Qexp "
            f"({selected['min_qtrue_qexp']:.3g}) is within "
            f"{SIMPLER_MODEL_Q_TOLERANCE*100:.0f}% of best eligible "
            f"{int(best_q_row['n_factors'])}f ({best_q:.3g}). {conv_note}."
        )

    return selected["run_folder"], reason


def load_pairwise_matches(month_key):
    f = TIMEAWARE_COMPARISON_ROOT / month_key / "timeaware_pairwise_factor_matches.csv"

    if not f.exists():
        return pd.DataFrame()

    df = pd.read_csv(f)

    for col in ["factor_a", "factor_b"]:
        if col in df.columns:
            df[col] = df[col].apply(factor_short)

    return df


def classify_match(profile_cos, time_corr, score):
    if pd.notna(profile_cos) and pd.notna(time_corr):
        if (profile_cos >= VERY_STABLE_PROFILE_COSINE) and (time_corr >= VERY_STABLE_TIME_CORR):
            return "very stable"
        if (profile_cos >= STABLE_PROFILE_COSINE) and (time_corr >= STABLE_TIME_CORR):
            return "stable"
        if (profile_cos >= STABLE_PROFILE_COSINE) and (time_corr < STABLE_TIME_CORR):
            return "similar profile but different timing"
        if (profile_cos < STABLE_PROFILE_COSINE) and (time_corr >= STABLE_TIME_CORR):
            return "similar timing but different profile"

    if pd.notna(score) and score >= 0.80:
        return "stable by combined score"

    return "weak/uncertain"


def factor_stability_for_selected(pairwise, selected_run, selected_factor):
    """
    Find best match of one selected factor to each other factor-number run.
    Works whether selected factor is run_a/factor_a or run_b/factor_b.
    """
    if pairwise.empty:
        return {
            "stability_status": "not available",
            "n_stable_matches": np.nan,
            "n_other_runs_compared": np.nan,
            "stability_evidence": "",
        }

    selected_factor = factor_short(selected_factor)

    evidence_parts = []
    statuses = []
    other_runs = set()

    # Selected factor appears as run_a/factor_a.
    g1 = pairwise[
        (pairwise["run_a"] == selected_run) &
        (pairwise["factor_a"] == selected_factor)
    ].copy()

    for other_run, g in g1.groupby("run_b"):
        other_runs.add(other_run)
        best = g.sort_values("combined_profile_time_score", ascending=False).iloc[0]
        profile_cos = safe_float(best.get("profile_cosine_similarity"))
        time_corr = safe_float(best.get("timestamp_contribution_corr"))
        score = safe_float(best.get("combined_profile_time_score"))
        status = classify_match(profile_cos, time_corr, score)
        statuses.append(status)

        evidence_parts.append(
            f"{other_run} {best['factor_b']}: {status} "
            f"(profile={profile_cos:.3f}, time_r={time_corr:.3f}, score={score:.3f})"
        )

    # Selected factor appears as run_b/factor_b.
    g2 = pairwise[
        (pairwise["run_b"] == selected_run) &
        (pairwise["factor_b"] == selected_factor)
    ].copy()

    for other_run, g in g2.groupby("run_a"):
        other_runs.add(other_run)
        best = g.sort_values("combined_profile_time_score", ascending=False).iloc[0]
        profile_cos = safe_float(best.get("profile_cosine_similarity"))
        time_corr = safe_float(best.get("timestamp_contribution_corr"))
        score = safe_float(best.get("combined_profile_time_score"))
        status = classify_match(profile_cos, time_corr, score)
        statuses.append(status)

        evidence_parts.append(
            f"{other_run} {best['factor_a']}: {status} "
            f"(profile={profile_cos:.3f}, time_r={time_corr:.3f}, score={score:.3f})"
        )

    n_other = len(other_runs)
    n_stable = sum(s in ["very stable", "stable", "stable by combined score"] for s in statuses)

    if n_other == 0:
        stability_status = "not available"
    elif n_stable == n_other:
        stability_status = "stable across compared solutions"
    elif n_stable >= 1:
        stability_status = "partly stable / possible split-merge"
    else:
        stability_status = "unstable or solution-specific"

    return {
        "stability_status": stability_status,
        "n_stable_matches": n_stable,
        "n_other_runs_compared": n_other,
        "stability_evidence": "; ".join(evidence_parts),
    }


def build_month_factor_decision_table(month_key, runs, fit_table):
    selected_run, selection_reason = choose_selected_run(month_key, fit_table)

    if selected_run is None:
        return pd.DataFrame(), pd.DataFrame()

    selected_info = runs[runs["run_folder"] == selected_run].iloc[0].to_dict()
    selected_n = int(selected_info["n_factors"])

    profile_wide = parse_profiles_pct_factor_total(selected_info["profile_file"])
    timing_df = load_timing_summary(selected_info["timing_file"])
    pairwise = load_pairwise_matches(month_key)

    if profile_wide.empty:
        warnings.warn(f"No profile parsed for selected run {selected_run}")

    factors = sorted(
        list(profile_wide.columns),
        key=lambda x: int(re.search(r"(\d+)", x).group(1))
    )

    rows = []

    for factor in factors:
        timing = get_timing_for_factor(timing_df, factor)
        max_median_ratio, event_flag = event_dominated_from_timing(timing)
        stability = factor_stability_for_selected(pairwise, selected_run, factor)

        source_label = assign_source_family(profile_wide, factor, month_key=month_key, timing=timing)
        fw_note = fireworks_timing_note(month_key, timing)
        time_label = timing_label(timing)

        top5_night_fraction = get_top_fraction(
            timing,
            ["top5pct_night_fraction", "top05pct_night_fraction"]
        )
        top10_night_fraction = get_top_fraction(
            timing,
            ["top10pct_night_fraction", "top9pct_night_fraction", "top90pct_night_fraction"]
        )
        top5_weekend_fraction = get_top_fraction(
            timing,
            ["top5pct_weekend_fraction", "top05pct_weekend_fraction"]
        )
        top10_weekend_fraction = get_top_fraction(
            timing,
            ["top10pct_weekend_fraction", "top9pct_weekend_fraction", "top90pct_weekend_fraction"]
        )

        note_bits = []

        if event_flag:
            note_bits.append("event-dominated")
        if stability["stability_status"] == "partly stable / possible split-merge":
            note_bits.append("possible split/merge across factor numbers")
        if stability["stability_status"] == "unstable or solution-specific":
            note_bits.append("review carefully; weak stability")
        if fw_note:
            note_bits.append(fw_note)
        if ("candidate" in source_label) or ("possible" in source_label) or ("check" in source_label):
            note_bits.append("source label needs manual confirmation")

        rows.append({
            "month_key": month_key,
            "year": selected_info["year"],
            "month": selected_info["month"],
            "month_name": selected_info["month_name"],
            "selected_run": selected_run,
            "selected_n_factors": selected_n,
            "selection_reason": selection_reason,
            "factor": factor,

            "source_family_label": source_label,
            "fireworks_timing_note": fw_note,
            "timing_label": time_label,
            "top_species_pct_factor_total": top_species_string(profile_wide, factor, n=8),

            "stability_status": stability["stability_status"],
            "n_stable_matches": stability["n_stable_matches"],
            "n_other_runs_compared": stability["n_other_runs_compared"],
            "stability_evidence": stability["stability_evidence"],

            "event_dominated_flag": event_flag,
            "max_median_ratio": max_median_ratio,

            "night_day_ratio_fixed": safe_float(col_value(timing, "night_day_ratio_fixed")),
            "weekend_weekday_ratio": safe_float(col_value(timing, "weekend_weekday_ratio")),
            "peak_hour_by_mean": safe_float(col_value(timing, "peak_hour_by_mean")),

            "top5_night_fraction": top5_night_fraction,
            "top10_night_fraction": top10_night_fraction,
            "top5_weekend_fraction": top5_weekend_fraction,
            "top10_weekend_fraction": top10_weekend_fraction,

            "top5_first_time": col_value(timing, "top5pct_first_time", ""),
            "top5_last_time": col_value(timing, "top5pct_last_time", ""),
            "top10_first_time": col_value(timing, "top10pct_first_time", col_value(timing, "top9pct_first_time", "")),
            "top10_last_time": col_value(timing, "top10pct_last_time", col_value(timing, "top9pct_last_time", "")),

            "review_notes": "; ".join(note_bits),
        })

    factor_table = pd.DataFrame(rows)

    # Model selection table for this month.
    model_rows = fit_table[fit_table["month_key"] == month_key].copy()
    model_rows["selected_run"] = selected_run
    model_rows["is_selected"] = model_rows["run_folder"].eq(selected_run)
    model_rows["selection_reason"] = selection_reason

    return factor_table, model_rows


def plot_month_summary(month_key, factor_table, out_dir):
    if factor_table.empty:
        return

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Night/day ratio
    if "night_day_ratio_fixed" in factor_table.columns:
        d = factor_table.dropna(subset=["night_day_ratio_fixed"])
        if not d.empty:
            plt.figure(figsize=(7, 4))
            plt.bar(d["factor"], d["night_day_ratio_fixed"])
            plt.axhline(1, linestyle="--", linewidth=1)
            plt.ylabel("Night/day ratio")
            plt.xlabel("Factor")
            plt.title(f"{month_key}: selected run night/day behavior")
            plt.tight_layout()
            plt.savefig(out_dir / f"{month_key}_night_day_ratio.png", dpi=200)
            plt.close()

    # Weekend/weekday ratio
    if "weekend_weekday_ratio" in factor_table.columns:
        d = factor_table.dropna(subset=["weekend_weekday_ratio"])
        if not d.empty:
            plt.figure(figsize=(7, 4))
            plt.bar(d["factor"], d["weekend_weekday_ratio"])
            plt.axhline(1, linestyle="--", linewidth=1)
            plt.ylabel("Weekend/weekday ratio")
            plt.xlabel("Factor")
            plt.title(f"{month_key}: selected run weekday/weekend behavior")
            plt.tight_layout()
            plt.savefig(out_dir / f"{month_key}_weekend_weekday_ratio.png", dpi=200)
            plt.close()

    # Event domination
    if "max_median_ratio" in factor_table.columns:
        d = factor_table.dropna(subset=["max_median_ratio"])
        if not d.empty:
            plt.figure(figsize=(7, 4))
            plt.bar(d["factor"], d["max_median_ratio"])
            plt.axhline(EVENT_DOMINATED_MAX_MEDIAN_RATIO, linestyle="--", linewidth=1)
            plt.ylabel("Max / median contribution")
            plt.xlabel("Factor")
            plt.title(f"{month_key}: event-dominated behavior")
            plt.tight_layout()
            plt.savefig(out_dir / f"{month_key}_max_median_ratio.png", dpi=200)
            plt.close()



def plot_model_selection(month_key, model_table, out_dir):
    """Plot model-selection diagnostics for one month."""
    if model_table is None or model_table.empty:
        return

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    d = model_table.sort_values("n_factors").copy()

    if "min_qtrue_qexp" in d.columns and d["min_qtrue_qexp"].notna().any():
        plt.figure(figsize=(7, 4))
        plt.bar(d["n_factors"].astype(str), d["min_qtrue_qexp"])
        plt.ylabel("Minimum Qtrue/Qexp")
        plt.xlabel("Number of factors")
        plt.title(f"{month_key}: PMF fit by factor number")
        plt.tight_layout()
        plt.savefig(out_dir / f"{month_key}_model_selection_qtrue_qexp.png", dpi=200)
        plt.close()

    if "n_converged" in d.columns and d["n_converged"].notna().any():
        plt.figure(figsize=(7, 4))
        plt.bar(d["n_factors"].astype(str), d["n_converged"])
        plt.ylabel("Number of converged base runs")
        plt.xlabel("Number of factors")
        plt.title(f"{month_key}: convergence by factor number")
        plt.tight_layout()
        plt.savefig(out_dir / f"{month_key}_model_selection_convergence.png", dpi=200)
        plt.close()


def get_selected_run_info(factor_table, runs):
    if factor_table is None or factor_table.empty:
        return None
    selected_run = factor_table["selected_run"].iloc[0]
    hit = runs[runs["run_folder"] == selected_run]
    if hit.empty:
        return None
    return hit.iloc[0].to_dict()


def plot_selected_profile_heatmap(month_key, factor_table, runs, out_dir):
    """Plot profile % of factor total heatmap for the selected solution."""
    info = get_selected_run_info(factor_table, runs)
    if info is None:
        return

    profile_wide = parse_profiles_pct_factor_total(info["profile_file"])
    if profile_wide.empty:
        return

    # Keep species that contribute meaningfully in at least one selected factor.
    d = profile_wide.loc[(profile_wide.max(axis=1) >= 1)].copy()
    if d.empty:
        d = profile_wide.copy()

    plt.figure(figsize=(max(6, 0.7 * len(d.columns)), max(5, 0.25 * len(d.index))))
    plt.imshow(d.values, aspect="auto")
    plt.colorbar(label="% of factor total")
    plt.xticks(range(len(d.columns)), d.columns)
    plt.yticks(range(len(d.index)), d.index)
    plt.xlabel("Factor")
    plt.ylabel("Species")
    plt.title(f"{month_key}: selected-factor chemical profiles")
    plt.tight_layout()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_dir / f"{month_key}_selected_profile_heatmap.png", dpi=200)
    plt.close()


def plot_selected_factor_time_series(month_key, factor_table, runs, out_dir):
    """Plot timestamped contribution time series for each selected factor."""
    info = get_selected_run_info(factor_table, runs)
    if info is None:
        return

    f = Path(info.get("merged_contrib_file", ""))
    if not f.exists():
        return

    df = pd.read_csv(f)
    if "Date_Local" not in df.columns:
        return
    df["Date_Local"] = pd.to_datetime(df["Date_Local"], errors="coerce")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for _, row in factor_table.iterrows():
        factor = factor_short(row["factor"])
        col = factor_col(factor)
        if col not in df.columns:
            continue

        label = str(row.get("source_family_label", ""))
        timing = str(row.get("timing_label", ""))

        plt.figure(figsize=(9, 4))
        plt.plot(df["Date_Local"], pd.to_numeric(df[col], errors="coerce"), linewidth=1)
        plt.ylabel("Relative factor contribution")
        plt.xlabel("Date/time local")
        plt.title(f"{month_key} {factor}: {label}\n{timing}")
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        plt.savefig(out_dir / f"{month_key}_{factor}_time_series.png", dpi=200)
        plt.close()


def plot_factor_timing_scatter(month_key, factor_table, out_dir):
    """Scatter plot showing timing behavior of selected factors."""
    if factor_table is None or factor_table.empty:
        return

    d = factor_table.dropna(subset=["night_day_ratio_fixed", "weekend_weekday_ratio"]).copy()
    if d.empty:
        return

    plt.figure(figsize=(7, 5))
    plt.scatter(d["night_day_ratio_fixed"], d["weekend_weekday_ratio"])
    plt.axvline(1, linestyle="--", linewidth=1)
    plt.axhline(1, linestyle="--", linewidth=1)

    for _, row in d.iterrows():
        plt.annotate(str(row["factor"]), (row["night_day_ratio_fixed"], row["weekend_weekday_ratio"]))

    plt.xlabel("Night/day ratio")
    plt.ylabel("Weekend/weekday ratio")
    plt.title(f"{month_key}: selected-factor timing map")
    plt.tight_layout()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_dir / f"{month_key}_factor_timing_scatter.png", dpi=200)
    plt.close()


def plot_global_source_family_presence(all_factor_table, out_dir):
    """Plot source-family presence/count by month."""
    if all_factor_table is None or all_factor_table.empty:
        return

    counts = all_factor_table.pivot_table(
        index="source_family_label",
        columns="month_key",
        values="factor",
        aggfunc="count",
        fill_value=0,
    )

    if counts.empty:
        return

    plt.figure(figsize=(max(7, 0.7 * len(counts.columns)), max(5, 0.35 * len(counts.index))))
    plt.imshow(counts.values, aspect="auto")
    plt.colorbar(label="Number of selected factors")
    plt.xticks(range(len(counts.columns)), counts.columns, rotation=30, ha="right")
    plt.yticks(range(len(counts.index)), counts.index)
    plt.xlabel("Month")
    plt.ylabel("Source-family label")
    plt.title("Source-family recurrence across selected monthly PMF solutions")
    plt.tight_layout()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_dir / "_ALL_source_family_presence_heatmap.png", dpi=200)
    plt.close()


def plot_global_selected_factor_numbers(all_model_table, out_dir):
    """Plot selected number of factors by month."""
    if all_model_table is None or all_model_table.empty:
        return

    d = all_model_table[all_model_table["is_selected"] == True].copy()
    if d.empty:
        return

    d = d.sort_values(["year", "month"])

    plt.figure(figsize=(8, 4))
    plt.bar(d["month_key"], d["n_factors"])
    plt.ylabel("Selected number of factors")
    plt.xlabel("Month")
    plt.title("Selected PMF factor number by month")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_dir / "_ALL_selected_factor_numbers.png", dpi=200)
    plt.close()


def build_source_family_by_month(all_factor_table):
    if all_factor_table.empty:
        return pd.DataFrame()

    rows = []

    for (month_key, source_family), g in all_factor_table.groupby(["month_key", "source_family_label"]):
        rows.append({
            "month_key": month_key,
            "source_family_label": source_family,
            "factors": "; ".join(g["selected_run"] + " " + g["factor"]),
            "n_factors": len(g),
            "timing_labels": " | ".join(g["factor"] + ": " + g["timing_label"]),
            "stability_statuses": " | ".join(g["factor"] + ": " + g["stability_status"]),
            "event_flags": " | ".join(g["factor"] + ": " + g["event_dominated_flag"].astype(str)),
        })

    long = pd.DataFrame(rows)

    pivot = long.pivot_table(
        index="source_family_label",
        columns="month_key",
        values="factors",
        aggfunc=lambda x: " ; ".join(x)
    ).reset_index()

    return long, pivot


def main():
    runs = list_available_runs()

    if runs.empty:
        raise RuntimeError("No PMF run folders found. Check PMF_OUTPUTS_ROOT.")

    runs_out = OUT_ROOT / "_available_runs.csv"
    runs.to_csv(runs_out, index=False)
    print(f"Found {len(runs)} run folders.")
    print(f"Wrote {runs_out}")

    fit_table = build_run_fit_table(runs)
    fit_out = OUT_ROOT / "_ALL_run_fit_summary.csv"
    fit_table.to_csv(fit_out, index=False)
    print(f"Wrote {fit_out}")

    all_factor_tables = []
    all_model_tables = []

    for month_key, g in runs.groupby("month_key"):
        month_dir = OUT_ROOT / month_key
        month_dir.mkdir(parents=True, exist_ok=True)

        print()
        print("====================================")
        print(f"Building decision table for {month_key}")

        factor_table, model_table = build_month_factor_decision_table(month_key, runs, fit_table)

        if factor_table.empty:
            print(f"  No factor table generated for {month_key}")
            continue

        factor_out = month_dir / f"{month_key}_factor_decision_table.csv"
        model_out = month_dir / f"{month_key}_model_selection_summary.csv"

        factor_table.to_csv(factor_out, index=False)
        model_table.to_csv(model_out, index=False)

        plot_dir = month_dir / "plots"
        plot_month_summary(month_key, factor_table, plot_dir)
        plot_model_selection(month_key, model_table, plot_dir)
        plot_selected_profile_heatmap(month_key, factor_table, runs, plot_dir)
        plot_selected_factor_time_series(month_key, factor_table, runs, plot_dir / "factor_time_series")
        plot_factor_timing_scatter(month_key, factor_table, plot_dir)

        all_factor_tables.append(factor_table)
        all_model_tables.append(model_table)

        selected_run = factor_table["selected_run"].iloc[0]
        print(f"  Selected run: {selected_run}")
        print(f"  Wrote: {factor_out}")
        print(f"  Wrote: {model_out}")

    if all_factor_tables:
        all_factor = pd.concat(all_factor_tables, ignore_index=True)
    else:
        all_factor = pd.DataFrame()

    if all_model_tables:
        all_model = pd.concat(all_model_tables, ignore_index=True)
    else:
        all_model = pd.DataFrame()

    all_factor_out = OUT_ROOT / "_ALL_monthly_factor_decision_table.csv"
    all_model_out = OUT_ROOT / "_ALL_monthly_model_selection_summary.csv"

    all_factor.to_csv(all_factor_out, index=False)
    all_model.to_csv(all_model_out, index=False)

    if not all_factor.empty:
        source_long, source_pivot = build_source_family_by_month(all_factor)

        source_long_out = OUT_ROOT / "_ALL_source_family_by_month_long.csv"
        source_pivot_out = OUT_ROOT / "_ALL_source_family_by_month_pivot.csv"

        source_long.to_csv(source_long_out, index=False)
        source_pivot.to_csv(source_pivot_out, index=False)

        global_plot_dir = OUT_ROOT / "plots"
        plot_global_source_family_presence(all_factor, global_plot_dir)
        plot_global_selected_factor_numbers(all_model, global_plot_dir)

    print()
    print("====================================")
    print("DONE")
    print("====================================")
    print("Main outputs:")
    print(all_model_out)
    print(all_factor_out)
    if not all_factor.empty:
        print(source_long_out)
        print(source_pivot_out)


if __name__ == "__main__":
    main()
