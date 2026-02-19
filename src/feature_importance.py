"""
feature_importance.py — Train a Random Forest on all subjects and
analyse which features matter most for sit-to-stand detection.

Outputs:
    Results/feature_importance.png   — horizontal bar chart of top-10 features
    stdout                           — full ranked list + channel-group analysis

Usage:
    python -m src.feature_importance     # from project root
    python src/feature_importance.py     # also works
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from src.models import LABEL_COL, SUBJECT_COL, _feature_cols

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
_FEATURES_CSV = _PROJECT_ROOT / "Data" / "uci_hapt" / "features.csv"
_RESULTS_DIR = _PROJECT_ROOT / "Results"
_IMPORTANCE_PNG = _RESULTS_DIR / "feature_importance.png"

# ---------------------------------------------------------------------------
# Channel-group classification
# ---------------------------------------------------------------------------
CHANNEL_GROUPS = {
    "ax": "Accel axes",
    "ay": "Accel axes",
    "az": "Accel axes",
    "gx": "Gyro axes",
    "gy": "Gyro axes",
    "gz": "Gyro axes",
    "accel_mag": "Accel magnitude",
    "gyro_mag": "Gyro magnitude",
}


def _channel_group(feature_name: str) -> str:
    """Map a feature name like 'accel_mag_max' → its channel group."""
    # Walk known channel prefixes longest-first so 'accel_mag' matches
    # before 'a'.
    for prefix in sorted(CHANNEL_GROUPS, key=len, reverse=True):
        if feature_name.startswith(prefix + "_"):
            return CHANNEL_GROUPS[prefix]
    return "Unknown"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    print(f"Loading features from {_FEATURES_CSV} …")
    df = pd.read_csv(_FEATURES_CSV)
    feat_cols = _feature_cols(df)

    X = df[feat_cols]
    y = df[LABEL_COL].values

    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    print(f"  {len(df):,} windows, {len(feat_cols)} features")
    print(f"  label=1: {n_pos}  |  label=0: {n_neg}\n")

    # ------------------------------------------------------------------
    # Train Random Forest on ALL data
    # ------------------------------------------------------------------
    rf = RandomForestClassifier(
        n_estimators=100, class_weight="balanced", random_state=42,
    )
    rf.fit(X, y)

    importances = rf.feature_importances_
    order = np.argsort(importances)[::-1]      # descending

    # ------------------------------------------------------------------
    # Full ranked list
    # ------------------------------------------------------------------
    print("=" * 65)
    print("Feature Importances (Gini, all 48 features)")
    print("=" * 65)
    print(f"{'Rank':<6} {'Feature':<28} {'Importance':>12} {'Group':<18}")
    print("-" * 65)
    for rank, idx in enumerate(order, 1):
        fname = feat_cols[idx]
        imp = importances[idx]
        group = _channel_group(fname)
        print(f"{rank:<6} {fname:<28} {imp:>12.4f} {group:<18}")

    # ------------------------------------------------------------------
    # Top-10 bar chart
    # ------------------------------------------------------------------
    top_n = 10
    top_idx = order[:top_n]
    top_names = [feat_cols[i] for i in top_idx]
    top_imp = importances[top_idx]
    top_groups = [_channel_group(n) for n in top_names]

    # Colour by channel group.
    group_colors = {
        "Accel axes": "#2196F3",
        "Gyro axes": "#FF9800",
        "Accel magnitude": "#4CAF50",
        "Gyro magnitude": "#E91E63",
    }
    bar_colors = [group_colors.get(g, "#999999") for g in top_groups]

    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    y_pos = np.arange(top_n)
    ax.barh(y_pos, top_imp[::-1], color=bar_colors[::-1], edgecolor="white")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_names[::-1], fontsize=10)
    ax.set_xlabel("Gini Importance")
    ax.set_title("Top-10 Feature Importances — Random Forest (all subjects)")

    # Legend by channel group (only groups present in top-10).
    from matplotlib.patches import Patch
    seen = []
    handles = []
    for g in top_groups:
        if g not in seen:
            seen.append(g)
            handles.append(Patch(facecolor=group_colors[g], label=g))
    ax.legend(handles=handles, loc="lower right", fontsize=9)

    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(_IMPORTANCE_PNG, dpi=150)
    plt.close(fig)
    print(f"\nTop-{top_n} bar chart saved to {_IMPORTANCE_PNG}")

    # ------------------------------------------------------------------
    # Channel-group dominance analysis
    # ------------------------------------------------------------------
    print(f"\n{'=' * 55}")
    print(f"Channel-Group Breakdown (Top {top_n})")
    print(f"{'=' * 55}")

    from collections import Counter
    group_counts = Counter(top_groups)
    group_imp_sum: dict[str, float] = {}
    for name, imp in zip(top_names, top_imp):
        g = _channel_group(name)
        group_imp_sum[g] = group_imp_sum.get(g, 0.0) + imp

    total_top_imp = top_imp.sum()
    print(f"{'Group':<20} {'Count':>6} {'Sum Importance':>16} {'% of Top-10':>12}")
    print("-" * 55)
    for g in sorted(group_imp_sum, key=group_imp_sum.get, reverse=True):
        cnt = group_counts[g]
        simp = group_imp_sum[g]
        pct = 100.0 * simp / total_top_imp
        print(f"{g:<20} {cnt:>6} {simp:>16.4f} {pct:>11.1f}%")

    dominant = max(group_imp_sum, key=group_imp_sum.get)
    print(f"\nDominant group: {dominant}")

    # ------------------------------------------------------------------
    # Literature alignment check
    # ------------------------------------------------------------------
    print(f"\n{'=' * 65}")
    print("Literature Alignment")
    print("=" * 65)

    # Check which of the "expected" features appear in the top 10.
    expected_high = {
        "accel_mag_max", "accel_mag_range", "accel_mag_std",
        "accel_mag_energy",
    }
    gyro_features_in_top = [n for n in top_names
                            if n.startswith("g") or n.startswith("gyro")]
    accel_mag_in_top = [n for n in top_names if n.startswith("accel_mag")]
    expected_found = expected_high & set(top_names)

    print(f"Expected high-importance features (from literature):")
    print(f"  accel_mag_max, accel_mag_range, accel_mag_std, accel_mag_energy")
    print(f"  Found in top-{top_n}: {sorted(expected_found) if expected_found else 'NONE'}")
    print()
    print(f"Accel-magnitude features in top-{top_n}: {accel_mag_in_top}")
    print(f"Gyroscope features in top-{top_n}:       {gyro_features_in_top}")
    print()

    if len(expected_found) >= 2:
        print("RESULT: Consistent with literature — acceleration magnitude")
        print("features (max, range, energy, std) are among the strongest")
        print("discriminators for sit-to-stand transitions, as reported by")
        print("Millor et al. (2014) and Galan-Mercant et al. (2014).")
    elif len(expected_found) == 1:
        print("RESULT: Partially consistent — one expected accel-magnitude")
        print("feature ranks high, but the model relies more on individual")
        print("axes or gyroscope channels than magnitude alone.")
    else:
        print("RESULT: Diverges from literature expectations — the model")
        print("relies primarily on raw axis-level or gyroscope features")
        print("rather than the acceleration magnitude features predicted")
        print("by Millor et al. and Galan-Mercant et al.")

    if gyro_features_in_top:
        print(f"\nGyroscope features ({len(gyro_features_in_top)} in top-{top_n})")
        print("confirm that rotational dynamics provide complementary")
        print("information beyond linear acceleration, consistent with")
        print("Galan-Mercant et al.'s finding that gyroscope signals")
        print("improve transition detection accuracy.")

    print("\nDone.")


if __name__ == "__main__":
    main()
