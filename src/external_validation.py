"""
external_validation.py — Test UCI HAPT-trained models on SisFall elderly data.

Pipeline:
  1. Train final models on ALL UCI HAPT windows (no held-out set)
  2. Predict on SisFall elderly D07/D08 windows
  3. Report per-trial event detection and per-subject breakdown

Models tested:
  - Threshold Baseline (48 features — uses only accel_mag_max/range)
  - Logistic Regression, 48 features
  - Logistic Regression, 24 features (accel-only)
  - Random Forest, 48 features
  - Random Forest, 24 features (accel-only)

Outputs:
    Results/external_validation_results.csv   — per-trial results
    stdout                                     — summary tables

Usage:
    python -m src.external_validation        # from project root
    python src/external_validation.py        # also works
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.models import LABEL_COL, ThresholdBaseline

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
_UCI_FEATURES = _PROJECT_ROOT / "Data" / "uci_hapt" / "features.csv"
_SIS_FEATURES = _PROJECT_ROOT / "Data" / "sisfall" / "features.csv"
_RESULTS_DIR = _PROJECT_ROOT / "Results"
_RESULTS_CSV = _RESULTS_DIR / "external_validation_results.csv"

# Minimal post-processing (best config from event_detection tuning).
MIN_EVENT_WINDOWS = 1
MAX_GAP_WINDOWS = 2

# Feature-set helpers
ACCEL_PREFIXES = ("ax_", "ay_", "az_", "accel_mag_")
META_COLS = {"label", "subject_id", "activity", "trial_id"}


def _all_feat_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in META_COLS]


def _accel_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns
            if c not in META_COLS
            and any(c.startswith(p) for p in ACCEL_PREFIXES)]


# ---------------------------------------------------------------------------
# Simple cluster counter for a single trial's binary predictions
# ---------------------------------------------------------------------------

def _count_event_clusters(
    binary: np.ndarray,
    min_len: int = MIN_EVENT_WINDOWS,
    max_gap: int = MAX_GAP_WINDOWS,
) -> int:
    """Count the number of positive clusters in a 1-D binary array."""
    if len(binary) == 0 or binary.sum() == 0:
        return 0

    # Identify runs of positives.
    clusters: list[list[int]] = []
    current: list[int] = []
    for i in range(len(binary)):
        if binary[i] == 1:
            current.append(i)
        else:
            if current:
                clusters.append(current)
                current = []
    if current:
        clusters.append(current)

    # Merge clusters separated by ≤ max_gap.
    merged: list[list[int]] = []
    for cluster in clusters:
        if merged and (cluster[0] - merged[-1][-1] - 1) <= max_gap:
            merged[-1].extend(cluster)
        else:
            merged.append(cluster)

    # Filter by min length.
    return sum(1 for c in merged if len(c) >= min_len)


# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------

ModelSpec = dict  # {name, feat_set_label, feat_cols, needs_scaling}


def _define_models(all_feat: list[str], accel_feat: list[str]) -> list[dict]:
    """Return a list of model specifications to train and evaluate."""
    return [
        {
            "name": "Threshold Baseline",
            "feat_label": "48",
            "feat_cols": all_feat,
            "needs_scaling": False,
            "build_fn": lambda: ThresholdBaseline(n_grid=20),
            "has_proba": False,
        },
        {
            "name": "Logistic Regression",
            "feat_label": "48",
            "feat_cols": all_feat,
            "needs_scaling": True,
            "build_fn": lambda: LogisticRegression(
                class_weight="balanced", max_iter=1000, random_state=42,
            ),
            "has_proba": True,
        },
        {
            "name": "Logistic Regression",
            "feat_label": "24 (accel)",
            "feat_cols": accel_feat,
            "needs_scaling": True,
            "build_fn": lambda: LogisticRegression(
                class_weight="balanced", max_iter=1000, random_state=42,
            ),
            "has_proba": True,
        },
        {
            "name": "Random Forest",
            "feat_label": "48",
            "feat_cols": all_feat,
            "needs_scaling": False,
            "build_fn": lambda: RandomForestClassifier(
                n_estimators=100, class_weight="balanced", random_state=42,
            ),
            "has_proba": True,
        },
        {
            "name": "Random Forest",
            "feat_label": "24 (accel)",
            "feat_cols": accel_feat,
            "needs_scaling": False,
            "build_fn": lambda: RandomForestClassifier(
                n_estimators=100, class_weight="balanced", random_state=42,
            ),
            "has_proba": True,
        },
    ]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # ------------------------------------------------------------------
    # Load datasets
    # ------------------------------------------------------------------
    print(f"Loading UCI HAPT features from {_UCI_FEATURES} …")
    uci = pd.read_csv(_UCI_FEATURES)
    all_feat = _all_feat_cols(uci)
    accel_feat = _accel_cols(uci)
    X_train_all = uci[all_feat]
    y_train = uci[LABEL_COL].values
    print(f"  UCI HAPT: {len(uci):,} windows, {len(all_feat)} features, "
          f"{int(y_train.sum())} positive\n")

    print(f"Loading SisFall features from {_SIS_FEATURES} …")
    sis = pd.read_csv(_SIS_FEATURES)
    print(f"  SisFall:  {len(sis):,} windows, "
          f"{sis[['subject_id','activity','trial_id']].drop_duplicates().shape[0]} trials\n")

    # ------------------------------------------------------------------
    # Train models and predict
    # ------------------------------------------------------------------
    models = _define_models(all_feat, accel_feat)

    # Store per-window predictions keyed by model tag.
    predictions: dict[str, np.ndarray] = {}

    for spec in models:
        tag = f"{spec['name']} ({spec['feat_label']})"
        feat = spec["feat_cols"]
        model = spec["build_fn"]()

        X_tr = uci[feat]
        X_te = sis[feat]

        if spec["needs_scaling"]:
            scaler = StandardScaler()
            X_tr = pd.DataFrame(
                scaler.fit_transform(X_tr), columns=feat, index=X_tr.index,
            )
            X_te = pd.DataFrame(
                scaler.transform(X_te), columns=feat, index=X_te.index,
            )

        # Threshold Baseline .fit() needs the full DataFrame (for column lookup).
        if spec["name"] == "Threshold Baseline":
            model.fit(uci, y_train)
            y_pred = model.predict(sis)
        else:
            model.fit(X_tr, y_train)
            y_pred = model.predict(X_te)

        predictions[tag] = y_pred
        n_pos = int(y_pred.sum())
        print(f"  {tag:<42} → {n_pos:>4} / {len(y_pred)} windows positive "
              f"({100 * n_pos / len(y_pred):.1f}%)")

    # ------------------------------------------------------------------
    # Per-trial analysis
    # ------------------------------------------------------------------
    trial_keys = sis[["subject_id", "activity", "trial_id"]].drop_duplicates()
    trial_keys = trial_keys.sort_values(
        ["subject_id", "activity", "trial_id"]
    ).reset_index(drop=True)

    result_rows: list[dict] = []

    for _, tk in trial_keys.iterrows():
        mask = (
            (sis["subject_id"] == tk["subject_id"])
            & (sis["activity"] == tk["activity"])
            & (sis["trial_id"] == tk["trial_id"])
        )
        idx = mask.values

        for spec in models:
            tag = f"{spec['name']} ({spec['feat_label']})"
            y_pred = predictions[tag][idx]
            n_pos_wins = int(y_pred.sum())
            detected = int(n_pos_wins >= 1)
            n_clusters = _count_event_clusters(y_pred)

            result_rows.append({
                "subject_id": tk["subject_id"],
                "activity": tk["activity"],
                "trial_id": int(tk["trial_id"]),
                "model": tag,
                "n_windows": int(idx.sum()),
                "n_pos_windows": n_pos_wins,
                "detected": detected,
                "n_pred_events": n_clusters,
            })

    results = pd.DataFrame(result_rows)
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(_RESULTS_CSV, index=False)
    print(f"\nPer-trial results saved to {_RESULTS_CSV}")

    # ------------------------------------------------------------------
    # Window-level activity summary
    # ------------------------------------------------------------------
    print(f"\n{'=' * 85}")
    print("Window-Level Positive Predictions per Trial")
    print("=" * 85)
    print(f"  {'Model':<42} {'Mean':>6} {'Std':>6} {'Min':>5} {'Max':>5}")
    print(f"  {'-' * 70}")

    model_tags = [f"{s['name']} ({s['feat_label']})" for s in models]
    for tag in model_tags:
        sub = results[results["model"] == tag]
        pw = sub["n_pos_windows"]
        print(f"  {tag:<42} {pw.mean():>6.2f} {pw.std():>6.2f} "
              f"{pw.min():>5} {pw.max():>5}")

    # ------------------------------------------------------------------
    # Event-level summary
    # ------------------------------------------------------------------
    n_trials = len(trial_keys)
    n_d07 = int((trial_keys["activity"] == "D07").sum())
    n_d08 = int((trial_keys["activity"] == "D08").sum())

    print(f"\n{'=' * 85}")
    print(f"Event-Level Detection ({n_trials} trials: {n_d07} D07 + {n_d08} D08)")
    print("=" * 85)
    print(f"  {'Model':<42} {'Recall':>7} {'D07':>7} {'D08':>7} "
          f"{'Mean evts':>10}")
    print(f"  {'-' * 78}")

    for tag in model_tags:
        sub = results[results["model"] == tag]
        recall_all = sub["detected"].mean()
        d07 = sub[sub["activity"] == "D07"]
        d08 = sub[sub["activity"] == "D08"]
        recall_d07 = d07["detected"].mean()
        recall_d08 = d08["detected"].mean()
        mean_evts = sub["n_pred_events"].mean()

        print(f"  {tag:<42} {recall_all:>7.3f} {recall_d07:>7.3f} "
              f"{recall_d08:>7.3f} {mean_evts:>10.2f}")

    # ------------------------------------------------------------------
    # Best model: per-subject breakdown
    # ------------------------------------------------------------------
    # Use RF-48 as the primary model; show per-subject.
    best_tag = "Random Forest (48)"
    print(f"\n{'=' * 85}")
    print(f"Per-Subject Breakdown — {best_tag}")
    print("=" * 85)

    best = results[results["model"] == best_tag]
    subjects = sorted(best["subject_id"].unique())

    print(f"  {'Subject':<10} {'Trials':>7} {'Detected':>9} {'Recall':>8} "
          f"{'D07 Rec':>8} {'D08 Rec':>8} {'Mean evts':>10}")
    print(f"  {'-' * 68}")

    for subj in subjects:
        s = best[best["subject_id"] == subj]
        n = len(s)
        det = int(s["detected"].sum())
        rec = s["detected"].mean()
        s07 = s[s["activity"] == "D07"]
        s08 = s[s["activity"] == "D08"]
        r07 = s07["detected"].mean() if len(s07) else float("nan")
        r08 = s08["detected"].mean() if len(s08) else float("nan")
        me = s["n_pred_events"].mean()
        print(f"  {subj:<10} {n:>7} {det:>9} {rec:>8.3f} "
              f"{r07:>8.3f} {r08:>8.3f} {me:>10.2f}")

    # ------------------------------------------------------------------
    # Comparison: accel-only impact on external data
    # ------------------------------------------------------------------
    print(f"\n{'=' * 85}")
    print("Feature-Set Impact on External Validation")
    print("=" * 85)
    print(f"  {'Model':<30} {'48-feat Recall':>15} {'24-feat Recall':>15} {'Δ':>8}")
    print(f"  {'-' * 72}")

    for base_name in ["Logistic Regression", "Random Forest"]:
        tag48 = f"{base_name} (48)"
        tag24 = f"{base_name} (24 (accel))"
        r48 = results[results["model"] == tag48]["detected"].mean()
        r24 = results[results["model"] == tag24]["detected"].mean()
        delta = r48 - r24
        print(f"  {base_name:<30} {r48:>15.3f} {r24:>15.3f} {delta:>+8.3f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
