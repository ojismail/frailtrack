"""
feature_ablation.py — Accel-only vs full feature-set ablation study.

Re-runs LOSO-CV with only accelerometer features (24 of 48) and compares
against the full-feature results from Results/loso_cv_results.csv.
Also runs RF event-level detection (minimal post-processing) on both
feature sets.

Outputs:
    Results/feature_ablation_results.csv   — per-fold accel-only metrics
    stdout                                  — side-by-side comparison tables

Usage:
    python -m src.feature_ablation       # from project root
    python src/feature_ablation.py       # also works
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import StandardScaler

from src.load_data import load_labels, load_signals
from src.models import LABEL_COL, SUBJECT_COL, ThresholdBaseline
from src.event_detection import (
    MODEL_NAMES,
    MATCH_TOLERANCE_SEC,
    PP_CONFIGS,
    _build_window_metadata,
    _extract_events,
    _gt_event_centers,
    _gt_rep_counts,
    _match_events,
    _smooth_probabilities,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
_FEATURES_CSV = _PROJECT_ROOT / "Data" / "uci_hapt" / "features.csv"
_FULL_LOSO_CSV = _PROJECT_ROOT / "Results" / "loso_cv_results.csv"
_FULL_EVENT_CSV = _PROJECT_ROOT / "Results" / "event_detection_results_tuned.csv"
_RESULTS_DIR = _PROJECT_ROOT / "Results"
_ABLATION_CSV = _RESULTS_DIR / "feature_ablation_results.csv"

# ---------------------------------------------------------------------------
# Feature-set definitions
# ---------------------------------------------------------------------------
ACCEL_PREFIXES = ("ax_", "ay_", "az_", "accel_mag_")


def _accel_cols(df: pd.DataFrame) -> list[str]:
    """Return the 24 accel-only feature columns."""
    return [c for c in df.columns
            if c not in (LABEL_COL, SUBJECT_COL, "exp_id", "win_center_sec")
            and any(c.startswith(p) for p in ACCEL_PREFIXES)]


def _all_feat_cols(df: pd.DataFrame) -> list[str]:
    """Return all 48 feature columns."""
    return [c for c in df.columns
            if c not in (LABEL_COL, SUBJECT_COL, "exp_id", "win_center_sec")]


# ---------------------------------------------------------------------------
# LOSO-CV for a given feature set
# ---------------------------------------------------------------------------

def _run_loso_window(
    df: pd.DataFrame,
    feat_cols: list[str],
    subjects: list[int],
) -> pd.DataFrame:
    """Full 30-fold LOSO-CV for all three models.  Returns per-fold metrics."""

    rows: list[dict] = []

    for i, held_out in enumerate(subjects, 1):
        test_mask = df[SUBJECT_COL] == held_out
        train_mask = ~test_mask

        X_train = df.loc[train_mask, feat_cols]
        y_train = df.loc[train_mask, LABEL_COL].values
        X_test = df.loc[test_mask, feat_cols]
        y_test = df.loc[test_mask, LABEL_COL].values

        n_pos = int(y_test.sum())
        n_neg = len(y_test) - n_pos

        # --- Threshold Baseline ---
        tb = ThresholdBaseline(n_grid=20)
        tb.fit(
            df.loc[train_mask],       # needs accel_mag columns by name
            y_train,
        )
        y_pred_tb = tb.predict(df.loc[test_mask])
        _record(rows, held_out, "Threshold Baseline",
                y_test, y_pred_tb, None, n_pos, n_neg)

        # --- Logistic Regression ---
        scaler = StandardScaler()
        X_tr_sc = scaler.fit_transform(X_train)
        X_te_sc = scaler.transform(X_test)

        lr = LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=42,
        )
        lr.fit(X_tr_sc, y_train)
        y_pred_lr = lr.predict(X_te_sc)
        y_prob_lr = lr.predict_proba(X_te_sc)[:, 1]
        _record(rows, held_out, "Logistic Regression",
                y_test, y_pred_lr, y_prob_lr, n_pos, n_neg)

        # --- Random Forest ---
        rf = RandomForestClassifier(
            n_estimators=100, class_weight="balanced", random_state=42,
        )
        rf.fit(X_train, y_train)
        y_pred_rf = rf.predict(X_test)
        y_prob_rf = rf.predict_proba(X_test)[:, 1]
        _record(rows, held_out, "Random Forest",
                y_test, y_pred_rf, y_prob_rf, n_pos, n_neg)

        print(f"  Fold {i:2d}/{len(subjects)} (subject {held_out:2d}) — "
              f"test {len(y_test):>4} ({n_pos} pos) ✓")

    return pd.DataFrame(rows)


def _record(
    rows: list[dict],
    subj: int,
    model: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None,
    n_pos: int,
    n_neg: int,
) -> None:
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    pr_auc = np.nan
    if y_prob is not None and n_pos > 0:
        pr_auc = average_precision_score(y_true, y_prob)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    rows.append(dict(
        subject_id=subj, model=model,
        precision=prec, recall=rec, f1=f1, pr_auc=pr_auc,
        tp=int(tp), fp=int(fp), fn=int(fn),
        n_test_pos=n_pos, n_test_neg=n_neg,
    ))


# ---------------------------------------------------------------------------
# Event-level detection for RF (minimal post-processing)
# ---------------------------------------------------------------------------

def _run_rf_event_detection(
    df: pd.DataFrame,
    feat_cols: list[str],
    subjects: list[int],
    exp_ids_col: np.ndarray,
    center_secs_col: np.ndarray,
    gt_centers: dict,
    gt_reps: dict,
) -> pd.DataFrame:
    """Run RF LOSO-CV and convert to event-level results using minimal PP."""

    pp = PP_CONFIGS["minimal"]
    smooth_k = pp["smooth_k"]
    smooth_thresh = pp["smooth_thresh"]
    min_dur = pp["min_event_windows"]
    max_gap = pp["max_gap_windows"]

    rows: list[dict] = []

    for i, held_out in enumerate(subjects, 1):
        test_mask = df[SUBJECT_COL] == held_out
        train_mask = ~test_mask

        X_train = df.loc[train_mask, feat_cols]
        y_train = df.loc[train_mask, LABEL_COL].values
        X_test = df.loc[test_mask, feat_cols]

        rf = RandomForestClassifier(
            n_estimators=100, class_weight="balanced", random_state=42,
        )
        rf.fit(X_train, y_train)
        y_prob = rf.predict_proba(X_test)[:, 1]

        # Post-process.
        if smooth_k > 1:
            smoothed = _smooth_probabilities(y_prob, k=smooth_k)
            binary = (smoothed >= smooth_thresh).astype(int)
        else:
            binary = (y_prob >= smooth_thresh).astype(int)

        exp_ids = exp_ids_col[test_mask.values]
        center_secs = center_secs_col[test_mask.values]

        pred_events = _extract_events(
            binary, exp_ids, center_secs,
            min_len=min_dur, max_gap=max_gap,
        )

        gt_events = gt_centers.get(held_out, [])
        n_gt = gt_reps.get(held_out, 0)
        n_pred = len(pred_events)
        tp, fp, fn = _match_events(pred_events, gt_events)

        evt_prec = tp / (tp + fp) if (tp + fp) else 0.0
        evt_rec = tp / (tp + fn) if (tp + fn) else 0.0
        evt_f1 = 2 * evt_prec * evt_rec / (evt_prec + evt_rec) if (evt_prec + evt_rec) else 0.0

        rows.append(dict(
            subject_id=held_out, gt_reps=n_gt, pred_reps=n_pred,
            abs_error=abs(n_pred - n_gt),
            event_tp=tp, event_fp=fp, event_fn=fn,
            event_precision=evt_prec, event_recall=evt_rec, event_f1=evt_f1,
        ))

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Printing helpers
# ---------------------------------------------------------------------------

def _window_summary(label: str, res: pd.DataFrame) -> None:
    """Print mean ± std for each model in a results DataFrame."""
    print(f"\n  {label}")
    print(f"  {'Model':<25} {'Precision':>16} {'Recall':>16} "
          f"{'F1':>16} {'PR-AUC':>16}")
    print(f"  {'-' * 84}")
    for name in MODEL_NAMES:
        sub = res[res["model"] == name]
        p_m, p_s = sub["precision"].mean(), sub["precision"].std()
        r_m, r_s = sub["recall"].mean(), sub["recall"].std()
        f_m, f_s = sub["f1"].mean(), sub["f1"].std()
        pa = sub["pr_auc"].dropna()
        pa_str = f"{pa.mean():.3f} ± {pa.std():.3f}" if len(pa) else "       —"
        print(f"  {name:<25} {p_m:.3f} ± {p_s:.3f}  "
              f"{r_m:.3f} ± {r_s:.3f}  "
              f"{f_m:.3f} ± {f_s:.3f}  "
              f"{pa_str}")


def _event_summary_row(label: str, ev: pd.DataFrame) -> dict:
    """Compute pooled event-level metrics from a per-subject DataFrame."""
    mae = ev["abs_error"].mean()
    tp = int(ev["event_tp"].sum())
    fp = int(ev["event_fp"].sum())
    fn = int(ev["event_fn"].sum())
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return dict(label=label, mae=mae, prec=prec, rec=rec, f1=f1,
                tp=tp, fp=fp, fn=fn)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # ------------------------------------------------------------------
    # Load data + metadata
    # ------------------------------------------------------------------
    print(f"Loading features from {_FEATURES_CSV} …")
    df = pd.read_csv(_FEATURES_CSV)
    subjects = sorted(df[SUBJECT_COL].unique())

    accel_feat = _accel_cols(df)
    all_feat = _all_feat_cols(df)
    print(f"  Accel-only features: {len(accel_feat)}")
    print(f"  Full features:       {len(all_feat)}")

    # Temporal metadata for event detection.
    print("Building temporal metadata …")
    signals = load_signals()
    labels_df = load_labels()
    win_meta = _build_window_metadata(signals, labels_df)
    assert len(win_meta) == len(df)
    exp_ids_col = win_meta["exp_id"].values.astype(int)
    center_secs_col = win_meta["win_center_sec"].values

    gt_centers = _gt_event_centers(labels_df)
    gt_reps = _gt_rep_counts(labels_df)

    # ------------------------------------------------------------------
    # Accel-only LOSO-CV (window-level)
    # ------------------------------------------------------------------
    print(f"\n{'=' * 80}")
    print("Running Accel-Only LOSO-CV (24 features) …")
    print("=" * 80)
    accel_window = _run_loso_window(df, accel_feat, subjects)

    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    accel_window.to_csv(_ABLATION_CSV, index=False)
    print(f"\nAccel-only per-fold results saved to {_ABLATION_CSV}")

    # ------------------------------------------------------------------
    # Load full-feature window-level results (already computed)
    # ------------------------------------------------------------------
    print(f"\nLoading full-feature LOSO-CV results from {_FULL_LOSO_CSV} …")
    full_window = pd.read_csv(_FULL_LOSO_CSV)

    # ------------------------------------------------------------------
    # Window-level comparison
    # ------------------------------------------------------------------
    print(f"\n{'=' * 80}")
    print("Window-Level Comparison (mean ± std across 30 LOSO folds)")
    print("=" * 80)
    _window_summary(f"Accel only ({len(accel_feat)} features)", accel_window)
    _window_summary(f"Full ({len(all_feat)} features)", full_window)

    # Side-by-side delta table.
    print(f"\n  {'— Δ (Full − Accel-only) —':^84}")
    print(f"  {'Model':<25} {'Δ Precision':>14} {'Δ Recall':>14} "
          f"{'Δ F1':>14} {'Δ PR-AUC':>14}")
    print(f"  {'-' * 84}")
    for name in MODEL_NAMES:
        a = accel_window[accel_window["model"] == name]
        f_ = full_window[full_window["model"] == name]
        dp = f_["precision"].mean() - a["precision"].mean()
        dr = f_["recall"].mean() - a["recall"].mean()
        df1 = f_["f1"].mean() - a["f1"].mean()
        pa_a = a["pr_auc"].dropna()
        pa_f = f_["pr_auc"].dropna()
        if len(pa_a) and len(pa_f):
            dpa = pa_f.mean() - pa_a.mean()
            dpa_str = f"{dpa:>+.3f}"
        else:
            dpa_str = "      —"
        print(f"  {name:<25} {dp:>+14.3f} {dr:>+14.3f} "
              f"{df1:>+14.3f} {dpa_str:>14}")

    # ------------------------------------------------------------------
    # Event-level: RF + minimal PP, accel-only
    # ------------------------------------------------------------------
    print(f"\n{'=' * 80}")
    print("Running RF Event Detection — Accel-Only (minimal PP) …")
    print("=" * 80)
    accel_events = _run_rf_event_detection(
        df, accel_feat, subjects,
        exp_ids_col, center_secs_col,
        gt_centers, gt_reps,
    )

    # Load full-feature event results.
    print(f"Loading full-feature event results from {_FULL_EVENT_CSV} …")
    full_events_all = pd.read_csv(_FULL_EVENT_CSV)
    full_events_rf = full_events_all[full_events_all["model"] == "Random Forest"]

    # ------------------------------------------------------------------
    # Event-level comparison
    # ------------------------------------------------------------------
    accel_ev = _event_summary_row(f"Accel only ({len(accel_feat)})", accel_events)
    full_ev = _event_summary_row(f"Full ({len(all_feat)})", full_events_rf)

    print(f"\n{'=' * 80}")
    print("Event-Level Comparison — Random Forest (minimal post-processing)")
    print("=" * 80)
    print(f"  {'Feature Set':<20} {'MAE':>6} {'Prec':>7} {'Rec':>7} "
          f"{'F1':>7} {'TP':>5} {'FP':>5} {'FN':>5}")
    print(f"  {'-' * 64}")
    for row in [accel_ev, full_ev]:
        print(f"  {row['label']:<20} {row['mae']:>6.2f} {row['prec']:>7.3f} "
              f"{row['rec']:>7.3f} {row['f1']:>7.3f} "
              f"{row['tp']:>5} {row['fp']:>5} {row['fn']:>5}")

    # Delta row.
    print(f"  {'Δ (Full − Accel)':<20} "
          f"{full_ev['mae'] - accel_ev['mae']:>+6.2f} "
          f"{full_ev['prec'] - accel_ev['prec']:>+7.3f} "
          f"{full_ev['rec'] - accel_ev['rec']:>+7.3f} "
          f"{full_ev['f1'] - accel_ev['f1']:>+7.3f} "
          f"{full_ev['tp'] - accel_ev['tp']:>+5} "
          f"{full_ev['fp'] - accel_ev['fp']:>+5} "
          f"{full_ev['fn'] - accel_ev['fn']:>+5}")

    # ------------------------------------------------------------------
    # Interpretation
    # ------------------------------------------------------------------
    f1_drop = full_ev["f1"] - accel_ev["f1"]
    print(f"\n{'=' * 80}")
    print("Interpretation")
    print("=" * 80)

    if abs(f1_drop) < 0.05:
        print("RESULT: Removing gyroscope features has MINIMAL impact")
        print(f"(event F1 Δ = {f1_drop:+.3f}).  The model is broadly deployable")
        print("on accel-only devices.  External validation on Marques (accel-only)")
        print("should be meaningful.")
    elif f1_drop > 0:
        print(f"RESULT: Gyroscope features provide a MODERATE-TO-LARGE benefit")
        print(f"(event F1 Δ = {f1_drop:+.3f}).  This confirms the feature-importance")
        print("finding that gyro axes accounted for 54% of RF's Gini importance.")
        print("External validation on Marques (accel-only) will reflect the")
        print("degraded accel-only performance, not the full model's capability.")
    else:
        print(f"RESULT: Accel-only OUTPERFORMS full features (event F1 Δ = {f1_drop:+.3f}).")
        print("This is surprising given the feature-importance analysis.")
        print("Possible explanations: gyro features introduce noise in")
        print("cross-subject generalisation, or the RF overfits to gyro patterns")
        print("that don't transfer across subjects in LOSO-CV.")

    print("\nDone.")


if __name__ == "__main__":
    main()
