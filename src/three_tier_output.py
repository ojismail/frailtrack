"""
three_tier_output.py — Clinical comparison layer and three-tier output.

Implements the threshold-based clinical interpretation of per-rep quality
features extracted in quality_assessment.py.

Three-tier structure:
  Tier 1: Rep Count             — number of detected STS repetitions
  Tier 2: Relative Power Score  — adapted Alcázar W/kg + threshold flag
  Tier 3: Movement Quality Flags — acceleration, CV, fatigue indicators

IMPORTANT: UCI HAPT is NOT a 30-second chair stand test.  Each subject
has only 2–3 sit-to-stand transitions. CV and fatigue slope are computed
to demonstrate the pipeline, but are not clinically meaningful with so
few repetitions.

Outputs:
    Results/fatigue_slope_example.png
    Results/three_tier_example.png
    Results/young_vs_elderly_comparison.png
    stdout — tables, three-tier reports, early-detection narrative

Usage:
    python -m src.three_tier_output    # from project root
    python src/three_tier_output.py    # also works
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from scipy.stats import linregress

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
_RESULTS_DIR = _PROJECT_ROOT / "Results"
_PER_REP_CSV = _RESULTS_DIR / "per_rep_features.csv"
_SUBJECT_CSV = _RESULTS_DIR / "subject_quality_summary.csv"
_SISFALL_CSV = _RESULTS_DIR / "sisfall_per_rep_features.csv"

_FIG_FATIGUE = _RESULTS_DIR / "fatigue_slope_example.png"
_FIG_THREE_TIER = _RESULTS_DIR / "three_tier_example.png"
_FIG_YOUNG_ELDERLY = _RESULTS_DIR / "young_vs_elderly_comparison.png"


# ===================================================================
# Step 1: Reference thresholds
# ===================================================================

THRESHOLDS = {
    "peak_dynamic_accel": {
        "frail": 2.7,           # m/s², Galán-Mercant 2013/2014
        "non_frail": 8.5,       # m/s², Galán-Mercant 2013/2014
        "source": "Galán-Mercant & Cuesta-Vargas (2013/2014)",
        "match_quality": (
            "Approximate — they used defined vertical axis"
        ),
    },
    "power": {
        # Alcázar et al. 2021, age 20-30 male (approximate for UCI HAPT pop)
        "low": 2.0,             # W/kg, below this suggests weakness
        "source": "Alcázar et al. (2021)",
        "match_quality": (
            "Approximate — validated on 5-rep STS, "
            "we use 30s CST adaptation"
        ),
    },
    "cv_accel": {
        "high": 0.30,           # above this → exhaustion / instability
        "source": "Park et al. (2021)",
        "match_quality": (
            "Approximate — they used 5-rep STS with 5 wearable sensors"
        ),
    },
    "fatigue_slope": {
        "declining": -0.0037,   # m/s per rep, Schwenk/Lindemann
        "source": "Schwenk/Lindemann",
        "match_quality": (
            "Exploratory — no validated threshold for this setup"
        ),
    },
}


# ===================================================================
# Step 2: Comparison functions
# ===================================================================

def classify_peak_accel(mean_peak_accel: float) -> str:
    """Classify mean peak dynamic acceleration into frailty range."""
    if mean_peak_accel >= THRESHOLDS["peak_dynamic_accel"]["non_frail"]:
        return "Within non-frail range"
    elif mean_peak_accel <= THRESHOLDS["peak_dynamic_accel"]["frail"]:
        return "Within frail range"
    else:
        return "Intermediate"


def classify_power(mean_power: float) -> str:
    """Classify mean relative power against weakness threshold."""
    if mean_power >= THRESHOLDS["power"]["low"]:
        return "Within expected range"
    else:
        return "Below expected range"


def classify_cv(cv_accel: float) -> str:
    """Classify coefficient of variation of peak acceleration."""
    if np.isnan(cv_accel):
        return "Insufficient data"
    if cv_accel > THRESHOLDS["cv_accel"]["high"]:
        return "Variable (exhaustion flag)"
    else:
        return "Stable"


def classify_fatigue(slope: float) -> str:
    """Classify fatigue slope of peak dynamic acceleration."""
    if np.isnan(slope):
        return "Insufficient data"
    if slope < THRESHOLDS["fatigue_slope"]["declining"]:
        return "Declining (fatigability flag)"
    else:
        return "Stable"


# ===================================================================
# Step 3: Three-tier output
# ===================================================================

def generate_three_tier(subj_row: pd.Series) -> dict:
    """Generate three-tier report for one subject.

    Parameters
    ----------
    subj_row : pd.Series
        One row from subject_quality_summary.csv.

    Returns
    -------
    report : dict with keys tier_1, tier_2, tier_3.
    """
    n_reps = int(subj_row["n_reps"])
    mean_accel = subj_row["mean_peak_accel"]
    mean_power = subj_row["mean_power"]
    cv_accel = subj_row["cv_accel"]
    slope = subj_row["fatigue_slope_accel"]

    tier_1 = {
        "rep_count": n_reps,
        "note": (
            "UCI HAPT is not a 30-second chair stand test "
            "(only 2–3 transitions per subject). Jones et al. (1999) "
            "normative ranges would apply in a real 30s CST."
        ),
    }

    tier_2 = {
        "power_w_kg": mean_power,
        "threshold": THRESHOLDS["power"]["low"],
        "status": classify_power(mean_power),
        "source": THRESHOLDS["power"]["source"],
    }

    tier_3 = {
        "accel": {
            "value": mean_accel,
            "status": classify_peak_accel(mean_accel),
            "dimension": "Force production capacity",
            "source": THRESHOLDS["peak_dynamic_accel"]["source"],
        },
        "cv": {
            "value": cv_accel,
            "status": classify_cv(cv_accel),
            "dimension": "Movement consistency / exhaustion",
            "source": THRESHOLDS["cv_accel"]["source"],
        },
        "fatigue": {
            "value": slope,
            "status": classify_fatigue(slope),
            "dimension": "Fatigability",
            "source": THRESHOLDS["fatigue_slope"]["source"],
        },
    }

    return {"tier_1": tier_1, "tier_2": tier_2, "tier_3": tier_3}


def print_three_tier(subject_id: int, report: dict) -> None:
    """Pretty-print one subject's three-tier report."""
    t1 = report["tier_1"]
    t2 = report["tier_2"]
    t3 = report["tier_3"]

    print(f"\n  ┌─────────────────────────────────────────────────────────────┐")
    print(f"  │  THREE-TIER REPORT — Subject {subject_id:<30}│")
    print(f"  ├─────────────────────────────────────────────────────────────┤")

    # Tier 1
    print(f"  │  TIER 1: Rep Count                                         │")
    print(f"  │    Detected reps: {t1['rep_count']:<40}│")
    print(f"  │    Note: {t1['note'][:50]:<50}│")
    if len(t1["note"]) > 50:
        print(f"  │          {t1['note'][50:]:<50}│")

    print(f"  ├─────────────────────────────────────────────────────────────┤")

    # Tier 2
    status_marker = "✓" if "expected" in t2["status"].lower() else "⚠"
    print(f"  │  TIER 2: Relative Power Score                              │")
    print(f"  │    Power:     {t2['power_w_kg']:.2f} W/kg{'':<38}│")
    print(f"  │    Threshold: {t2['threshold']:.1f} W/kg (Alcázar et al. 2021){'':<18}│")
    print(f"  │    Status:    {status_marker} {t2['status']:<43}│")

    print(f"  ├─────────────────────────────────────────────────────────────┤")

    # Tier 3
    print(f"  │  TIER 3: Movement Quality Flags                            │")
    for key, info in t3.items():
        marker = "✓" if "Stable" in info["status"] or "non-frail" in info["status"] else "⚠"
        val_str = f"{info['value']:.3f}" if not np.isnan(info["value"]) else "n/a"
        print(f"  │    {info['dimension']:<30}                             │")
        print(f"  │      Value: {val_str:<10}  Status: {marker} {info['status']:<22}│")
        print(f"  │      Source: {info['source'][:45]:<45}│")

    print(f"  └─────────────────────────────────────────────────────────────┘")


# ===================================================================
# Step 4: Early detection example
# ===================================================================

def find_early_detection_subject(summary_df: pd.DataFrame) -> int | None:
    """Find a subject where rep count looks normal but quality flags suggest risk.

    Criteria:
      - n_reps >= 2 (looks 'normal')
      - At least one quality flag is concerning (intermediate/frail accel,
        low power, high CV, or declining fatigue slope)
    Prefer subjects with multiple flags.
    """
    candidates: list[tuple[int, int]] = []  # (subject_id, n_flags)

    for _, row in summary_df.iterrows():
        flags = 0
        if classify_peak_accel(row["mean_peak_accel"]) != "Within non-frail range":
            flags += 1
        if classify_power(row["mean_power"]) != "Within expected range":
            flags += 1
        if classify_cv(row["cv_accel"]) == "Variable (exhaustion flag)":
            flags += 1
        if classify_fatigue(row["fatigue_slope_accel"]) == "Declining (fatigability flag)":
            flags += 1

        if flags >= 2 and int(row["n_reps"]) >= 2:
            candidates.append((int(row["subject_id"]), flags))

    if not candidates:
        # Fallback: any subject with ≥1 flag.
        for _, row in summary_df.iterrows():
            flags = 0
            if classify_peak_accel(row["mean_peak_accel"]) != "Within non-frail range":
                flags += 1
            if classify_power(row["mean_power"]) != "Within expected range":
                flags += 1
            if flags >= 1:
                candidates.append((int(row["subject_id"]), flags))

    if not candidates:
        return None

    # Pick the subject with the most flags.
    candidates.sort(key=lambda x: -x[1])
    return candidates[0][0]


def print_early_detection(
    subject_id: int,
    summary_df: pd.DataFrame,
    rep_df: pd.DataFrame,
) -> None:
    """Print the early detection narrative for one subject."""
    row = summary_df[summary_df["subject_id"] == subject_id].iloc[0]
    n_reps = int(row["n_reps"])
    report = generate_three_tier(row)

    print(f"\n{'=' * 72}")
    print("EARLY DETECTION EXAMPLE — Pre-frailty Risk Invisible to Rep Count")
    print("=" * 72)

    print(f"\n  Subject {subject_id} completed {n_reps} sit-to-stand repetitions.")
    print(f"\n  Standard assessment (rep count alone):")
    print(f"    → {n_reps} reps recorded.  In a 30-second CST, this would be")
    print(f"      evaluated against age/sex norms (Jones et al. 1999).")
    print(f"      With only {n_reps} reps from a protocol context, ")
    print(f"      no red flag is raised by count alone.")

    print(f"\n  Movement quality analysis reveals:")

    flags: list[str] = []
    accel_status = classify_peak_accel(row["mean_peak_accel"])
    power_status = classify_power(row["mean_power"])
    cv_status = classify_cv(row["cv_accel"])
    fatigue_status = classify_fatigue(row["fatigue_slope_accel"])

    if accel_status != "Within non-frail range":
        label = (
            f"Peak dynamic acceleration = {row['mean_peak_accel']:.2f} m/s²  "
            f"→ {accel_status}"
        )
        flags.append(label)
    if power_status != "Within expected range":
        label = (
            f"Relative power = {row['mean_power']:.2f} W/kg  "
            f"→ {power_status} (threshold: {THRESHOLDS['power']['low']:.1f} W/kg)"
        )
        flags.append(label)
    if cv_status != "Stable":
        label = (
            f"CV of peak accel = {row['cv_accel']:.3f}  "
            f"→ {cv_status} (threshold: {THRESHOLDS['cv_accel']['high']:.2f})"
        )
        flags.append(label)
    if fatigue_status != "Stable":
        label = (
            f"Fatigue slope = {row['fatigue_slope_accel']:.4f} m/s²/rep  "
            f"→ {fatigue_status}"
        )
        flags.append(label)

    for i, f in enumerate(flags, 1):
        print(f"    {i}. {f}")

    if flags:
        print(f"\n  → These {len(flags)} indicator(s) suggest pre-frailty risk that")
        print(f"    would be INVISIBLE to rep count alone.")
        print(f"    This is the core value of wearable-based movement quality")
        print(f"    analysis: catching decline before it manifests as inability")
        print(f"    to complete repetitions.")
    else:
        print(f"\n    (No quality flags raised for this subject.)")


# ===================================================================
# Step 5: Reference comparison table
# ===================================================================

def print_reference_table() -> None:
    """Print the full reference threshold table in checklist format."""
    print(f"\n{'=' * 110}")
    print("REFERENCE THRESHOLD TABLE")
    print("=" * 110)

    rows = [
        (
            "Peak dynamic accel",
            "Force production",
            f"≤{THRESHOLDS['peak_dynamic_accel']['frail']} frail / "
            f"≥{THRESHOLDS['peak_dynamic_accel']['non_frail']} non-frail (m/s²)",
            THRESHOLDS["peak_dynamic_accel"]["source"],
            THRESHOLDS["peak_dynamic_accel"]["match_quality"],
        ),
        (
            "Relative power",
            "Muscle power",
            f"<{THRESHOLDS['power']['low']} W/kg suggests weakness",
            THRESHOLDS["power"]["source"],
            THRESHOLDS["power"]["match_quality"],
        ),
        (
            "CV of peak accel",
            "Movement consistency",
            f">{THRESHOLDS['cv_accel']['high']:.2f} = exhaustion flag",
            THRESHOLDS["cv_accel"]["source"],
            THRESHOLDS["cv_accel"]["match_quality"],
        ),
        (
            "Fatigue slope",
            "Fatigability",
            f"<{THRESHOLDS['fatigue_slope']['declining']} m/s²/rep = declining",
            THRESHOLDS["fatigue_slope"]["source"],
            THRESHOLDS["fatigue_slope"]["match_quality"],
        ),
    ]

    hdr = (
        f"  {'Indicator':<22} {'Frailty Dimension':<22} "
        f"{'Threshold':<38} {'Source':<30} {'Match Quality'}"
    )
    print(hdr)
    print(f"  {'-' * 105}")

    for indicator, dim, thresh, src, mq in rows:
        # Truncate source for table fit.
        src_short = src[:28] + ".." if len(src) > 30 else src
        print(
            f"  {indicator:<22} {dim:<22} "
            f"{thresh:<38} {src_short:<30} {mq}"
        )


# ===================================================================
# Step 6: Figures
# ===================================================================

def fig_fatigue_slope(rep_df: pd.DataFrame, subject_id: int) -> None:
    """Figure 1: Per-rep peak dynamic accel with fatigue slope overlay."""
    sub = rep_df[rep_df["subject_id"] == subject_id].sort_values("rep_number")
    reps = sub["rep_number"].values.astype(float)
    accels = sub["peak_dynamic_accel_ms2"].values

    fig, ax = plt.subplots(figsize=(7, 4.5))

    # Bar chart.
    colors = ["#2196F3"] * len(reps)
    bars = ax.bar(reps, accels, color=colors, width=0.6, edgecolor="white",
                  linewidth=0.8, zorder=3)

    # Add value labels on bars.
    for bar, val in zip(bars, accels):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                f"{val:.1f}", ha="center", va="bottom", fontsize=9,
                fontweight="bold", color="#333")

    # Linear regression line.
    if len(reps) >= 2:
        slope, intercept, r_val, _, _ = linregress(reps, accels)
        x_fit = np.linspace(reps.min() - 0.3, reps.max() + 0.3, 50)
        y_fit = slope * x_fit + intercept
        ax.plot(x_fit, y_fit, "r--", linewidth=2.0, zorder=4,
                label=f"Fatigue slope: {slope:+.2f} m/s²/rep (r={r_val:.2f})")
        ax.legend(fontsize=9, loc="upper right")

    # Threshold bands.
    ax.axhline(y=THRESHOLDS["peak_dynamic_accel"]["non_frail"], color="#4CAF50",
               linestyle=":", linewidth=1.2, alpha=0.7)
    ax.text(reps.max() + 0.35, THRESHOLDS["peak_dynamic_accel"]["non_frail"],
            f"Non-frail ≥{THRESHOLDS['peak_dynamic_accel']['non_frail']}",
            fontsize=8, color="#4CAF50", va="bottom")

    ax.axhline(y=THRESHOLDS["peak_dynamic_accel"]["frail"], color="#F44336",
               linestyle=":", linewidth=1.2, alpha=0.7)
    ax.text(reps.max() + 0.35, THRESHOLDS["peak_dynamic_accel"]["frail"],
            f"Frail ≤{THRESHOLDS['peak_dynamic_accel']['frail']}",
            fontsize=8, color="#F44336", va="top")

    ax.set_xlabel("Repetition Number", fontsize=11)
    ax.set_ylabel("Peak Dynamic Acceleration (m/s²)", fontsize=11)
    ax.set_title(f"Subject {subject_id} — Per-Rep Peak Acceleration & Fatigue Slope",
                 fontsize=12, fontweight="bold")
    ax.set_xticks(reps)
    ax.set_xlim(reps.min() - 0.5, reps.max() + 0.8)
    ax.set_ylim(0, max(accels) * 1.3)
    ax.grid(axis="y", alpha=0.3, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(_FIG_FATIGUE, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {_FIG_FATIGUE}")


def fig_three_tier(subject_id: int, report: dict) -> None:
    """Figure 2: Three-tier output summary as a clean table figure."""
    t1 = report["tier_1"]
    t2 = report["tier_2"]
    t3 = report["tier_3"]

    fig, ax = plt.subplots(figsize=(10, 6.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")

    # Title.
    ax.text(5, 9.6, f"Three-Tier Clinical Report — Subject {subject_id}",
            ha="center", va="center", fontsize=14, fontweight="bold",
            fontfamily="monospace")

    y = 9.0

    # ── Tier 1 ──
    ax.add_patch(plt.Rectangle((0.3, y - 0.6), 9.4, 1.1,
                 facecolor="#E3F2FD", edgecolor="#1976D2",
                 linewidth=1.5, zorder=2))
    ax.text(0.5, y + 0.2, "TIER 1: REP COUNT", fontsize=11,
            fontweight="bold", color="#1976D2", zorder=3)
    ax.text(0.7, y - 0.2, f"Detected repetitions: {t1['rep_count']}",
            fontsize=10, zorder=3)

    y -= 1.5

    # ── Tier 2 ──
    power_color = "#E8F5E9" if "expected" in t2["status"].lower() else "#FFF3E0"
    edge_color = "#388E3C" if "expected" in t2["status"].lower() else "#F57C00"
    marker = "✓" if "expected" in t2["status"].lower() else "⚠"

    ax.add_patch(plt.Rectangle((0.3, y - 1.1), 9.4, 1.6,
                 facecolor=power_color, edgecolor=edge_color,
                 linewidth=1.5, zorder=2))
    ax.text(0.5, y + 0.2, "TIER 2: RELATIVE POWER SCORE", fontsize=11,
            fontweight="bold", color=edge_color, zorder=3)
    ax.text(0.7, y - 0.2,
            f"Power: {t2['power_w_kg']:.2f} W/kg   "
            f"(threshold: {t2['threshold']:.1f} W/kg)",
            fontsize=10, zorder=3)
    ax.text(0.7, y - 0.6,
            f"Status: {marker} {t2['status']}    "
            f"Source: {t2['source']}",
            fontsize=9, color="#555", zorder=3)

    y -= 2.1

    # ── Tier 3 ──
    ax.add_patch(plt.Rectangle((0.3, y - 3.5), 9.4, 4.0,
                 facecolor="#FFF8E1", edgecolor="#FFA000",
                 linewidth=1.5, zorder=2))
    ax.text(0.5, y + 0.2, "TIER 3: MOVEMENT QUALITY FLAGS", fontsize=11,
            fontweight="bold", color="#F57C00", zorder=3)

    row_y = y - 0.3
    for key, info in t3.items():
        is_ok = ("Stable" in info["status"] or "non-frail" in info["status"])
        marker = "✓" if is_ok else "⚠"
        color = "#388E3C" if is_ok else "#D32F2F"
        val_str = f"{info['value']:.3f}" if not np.isnan(info["value"]) else "n/a"

        ax.text(0.7, row_y,
                f"{info['dimension']}",
                fontsize=10, fontweight="bold", zorder=3)
        ax.text(5.0, row_y,
                f"Value: {val_str}",
                fontsize=9, zorder=3)
        ax.text(7.0, row_y,
                f"{marker} {info['status']}",
                fontsize=9, color=color, fontweight="bold", zorder=3)
        row_y -= 0.5
        ax.text(0.9, row_y,
                f"Source: {info['source'][:50]}",
                fontsize=8, color="#777", zorder=3)
        row_y -= 0.7

    # Footer note.
    ax.text(5, 0.3,
            "Note: UCI HAPT has only 2–3 reps/subject. CV and fatigue slope "
            "are demonstrated but not clinically meaningful.",
            ha="center", fontsize=8, style="italic", color="#999")

    fig.tight_layout()
    fig.savefig(_FIG_THREE_TIER, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {_FIG_THREE_TIER}")


def fig_young_vs_elderly(
    uci_rep: pd.DataFrame,
    sis_rep: pd.DataFrame,
) -> None:
    """Figure 3: Grouped bar chart comparing UCI HAPT (young) vs SisFall (elderly).

    Uses only SisFall D07 (slow STS) for a fairer comparison to UCI HAPT's
    natural-pace transitions.
    """
    # Separate SisFall by activity for additional context.
    sis_d07 = sis_rep[sis_rep["activity"] == "D07"]
    sis_d08 = sis_rep[sis_rep["activity"] == "D08"]

    groups = ["UCI HAPT\n(young, 19–48 y)", "SisFall D07\n(elderly, slow)",
              "SisFall D08\n(elderly, fast)"]
    accel_means = [
        uci_rep["peak_dynamic_accel_ms2"].mean(),
        sis_d07["peak_dynamic_accel_ms2"].mean(),
        sis_d08["peak_dynamic_accel_ms2"].mean(),
    ]
    accel_stds = [
        uci_rep["peak_dynamic_accel_ms2"].std(),
        sis_d07["peak_dynamic_accel_ms2"].std(),
        sis_d08["peak_dynamic_accel_ms2"].std(),
    ]
    gyro_means = [
        uci_rep["peak_gyro_rad_s"].mean(),
        sis_d07["peak_gyro_rad_s"].mean(),
        sis_d08["peak_gyro_rad_s"].mean(),
    ]
    gyro_stds = [
        uci_rep["peak_gyro_rad_s"].std(),
        sis_d07["peak_gyro_rad_s"].std(),
        sis_d08["peak_gyro_rad_s"].std(),
    ]

    x = np.arange(len(groups))
    width = 0.35

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # ── Panel A: Peak Dynamic Acceleration ──
    colors_accel = ["#2196F3", "#FF9800", "#F44336"]
    bars_a = ax1.bar(x, accel_means, width=0.5, yerr=accel_stds,
                     capsize=5, color=colors_accel, edgecolor="white",
                     linewidth=1, zorder=3)
    for bar, val in zip(bars_a, accel_means):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                 f"{val:.1f}", ha="center", va="bottom", fontsize=10,
                 fontweight="bold")

    # Add threshold lines.
    ax1.axhline(y=THRESHOLDS["peak_dynamic_accel"]["non_frail"],
                color="#4CAF50", linestyle="--", linewidth=1.2, alpha=0.7)
    ax1.text(2.4, THRESHOLDS["peak_dynamic_accel"]["non_frail"] + 0.2,
             "Non-frail", fontsize=8, color="#4CAF50")
    ax1.axhline(y=THRESHOLDS["peak_dynamic_accel"]["frail"],
                color="#D32F2F", linestyle="--", linewidth=1.2, alpha=0.7)
    ax1.text(2.4, THRESHOLDS["peak_dynamic_accel"]["frail"] + 0.2,
             "Frail", fontsize=8, color="#D32F2F")

    ax1.set_ylabel("Peak Dynamic Acceleration (m/s²)", fontsize=11)
    ax1.set_title("(A) Peak Dynamic Acceleration", fontsize=12,
                  fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(groups, fontsize=9)
    ax1.set_ylim(0, max(accel_means) + max(accel_stds) + 2)
    ax1.grid(axis="y", alpha=0.3, zorder=0)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # ── Panel B: Peak Gyroscope Magnitude ──
    colors_gyro = ["#2196F3", "#FF9800", "#F44336"]
    bars_g = ax2.bar(x, gyro_means, width=0.5, yerr=gyro_stds,
                     capsize=5, color=colors_gyro, edgecolor="white",
                     linewidth=1, zorder=3)
    for bar, val in zip(bars_g, gyro_means):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                 f"{val:.2f}", ha="center", va="bottom", fontsize=10,
                 fontweight="bold")

    ax2.set_ylabel("Peak Gyroscope Magnitude (rad/s)", fontsize=11)
    ax2.set_title("(B) Peak Gyroscope Magnitude", fontsize=12,
                  fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(groups, fontsize=9)
    ax2.set_ylim(0, max(gyro_means) + max(gyro_stds) + 1)
    ax2.grid(axis="y", alpha=0.3, zorder=0)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    fig.suptitle(
        "UCI HAPT (Young Adults) vs SisFall (Elderly) — Per-Rep Kinematics",
        fontsize=13, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    fig.savefig(_FIG_YOUNG_ELDERLY, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {_FIG_YOUNG_ELDERLY}")


# ===================================================================
# Main
# ===================================================================

def main() -> None:
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load data.
    print("Loading per-rep and subject summary data …\n")
    rep_df = pd.read_csv(_PER_REP_CSV)
    summary_df = pd.read_csv(_SUBJECT_CSV)
    sis_rep = pd.read_csv(_SISFALL_CSV)

    # ------------------------------------------------------------------
    # Step 1 & 2: Thresholds are defined above. Quick summary:
    # ------------------------------------------------------------------
    print("=" * 72)
    print("STEP 1–2: Reference Thresholds & Classification Functions")
    print("=" * 72)
    print(f"  Loaded {len(summary_df)} subjects from UCI HAPT")
    print(f"  Loaded {len(sis_rep)} SisFall trials")
    print(f"  Threshold definitions: {len(THRESHOLDS)} indicators\n")

    # Classify all subjects.
    for _, row in summary_df.iterrows():
        subj = int(row["subject_id"])
        accel_cl = classify_peak_accel(row["mean_peak_accel"])
        power_cl = classify_power(row["mean_power"])
        cv_cl = classify_cv(row["cv_accel"])
        fat_cl = classify_fatigue(row["fatigue_slope_accel"])

    # Count classifications across all subjects.
    accel_classes = summary_df["mean_peak_accel"].apply(classify_peak_accel)
    power_classes = summary_df["mean_power"].apply(classify_power)
    cv_classes = summary_df["cv_accel"].apply(classify_cv)
    fat_classes = summary_df["fatigue_slope_accel"].apply(classify_fatigue)

    print("  Classification distribution across 30 subjects:")
    print(f"    Peak accel:  {dict(accel_classes.value_counts())}")
    print(f"    Power:       {dict(power_classes.value_counts())}")
    print(f"    CV accel:    {dict(cv_classes.value_counts())}")
    print(f"    Fatigue:     {dict(fat_classes.value_counts())}")

    # ------------------------------------------------------------------
    # Step 3: Three-tier output for 3 example subjects
    # ------------------------------------------------------------------
    print(f"\n{'=' * 72}")
    print("STEP 3: Three-Tier Reports for 3 Example Subjects")
    print("=" * 72)

    # Pick 3 subjects with diverse profiles.
    # High accel: subject with highest mean_peak_accel.
    high_subj = int(
        summary_df.loc[summary_df["mean_peak_accel"].idxmax(), "subject_id"]
    )
    # Low accel: subject with lowest mean_peak_accel.
    low_subj = int(
        summary_df.loc[summary_df["mean_peak_accel"].idxmin(), "subject_id"]
    )
    # Mid: median by mean_peak_accel, excluding high and low.
    mid_df = summary_df[
        ~summary_df["subject_id"].isin([high_subj, low_subj])
    ].copy()
    mid_df["rank"] = mid_df["mean_peak_accel"].rank()
    mid_subj = int(
        mid_df.loc[(mid_df["rank"] - mid_df["rank"].median()).abs().idxmin(),
                   "subject_id"]
    )

    print(f"\n  Selected subjects: high={high_subj}, mid={mid_subj}, "
          f"low={low_subj}")

    for subj in [high_subj, mid_subj, low_subj]:
        row = summary_df[summary_df["subject_id"] == subj].iloc[0]
        report = generate_three_tier(row)
        print_three_tier(subj, report)

    # ------------------------------------------------------------------
    # Step 4: Early detection example
    # ------------------------------------------------------------------
    early_subj = find_early_detection_subject(summary_df)
    if early_subj is not None:
        print_early_detection(early_subj, summary_df, rep_df)
    else:
        print("\n  (No early detection candidate found.)")

    # ------------------------------------------------------------------
    # Step 5: Reference comparison table
    # ------------------------------------------------------------------
    print_reference_table()

    # ------------------------------------------------------------------
    # Step 6: Figures
    # ------------------------------------------------------------------
    print(f"\n{'=' * 72}")
    print("STEP 6: Generating Figures")
    print("=" * 72)

    # Figure 1: Fatigue slope example.
    # Use subject 8 (3 reps — most reps available, best for slope visual).
    fatigue_subj = 8
    if fatigue_subj not in rep_df["subject_id"].values:
        # Fallback: pick subject with most reps.
        rep_counts = rep_df.groupby("subject_id").size()
        fatigue_subj = int(rep_counts.idxmax())
    fig_fatigue_slope(rep_df, fatigue_subj)

    # Figure 2: Three-tier example (use the mid subject).
    mid_row = summary_df[summary_df["subject_id"] == mid_subj].iloc[0]
    mid_report = generate_three_tier(mid_row)
    fig_three_tier(mid_subj, mid_report)

    # Figure 3: Young vs elderly comparison.
    fig_young_vs_elderly(rep_df, sis_rep)

    # ------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------
    print(f"\n{'=' * 72}")
    print("SUMMARY OF OUTPUTS")
    print("=" * 72)
    outputs = [
        ("Three-tier reports",  "stdout (3 example subjects)"),
        ("Early detection example", "stdout"),
        ("Reference threshold table", "stdout"),
        ("Fatigue slope figure",  str(_FIG_FATIGUE)),
        ("Three-tier figure",     str(_FIG_THREE_TIER)),
        ("Young vs elderly figure", str(_FIG_YOUNG_ELDERLY)),
    ]
    for name, loc in outputs:
        print(f"  {name:<30} → {loc}")

    print(f"\nDone.")


if __name__ == "__main__":
    main()
