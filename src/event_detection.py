"""
event_detection.py — Convert window-level predictions into event-level
(rep-level) sit-to-stand detection results.

For each held-out subject in LOSO-CV:
  1. Train all three models, get per-window predictions and probabilities
  2. Post-process into discrete events (smoothing, min duration, gap merge)
  3. Match predicted events to ground-truth segments (±1.0 s tolerance)
  4. Compute event-level precision, recall, and rep-count MAE

Supports multiple post-processing configurations to find settings that
best suit each model type (strict filtering helps noisy models; light
filtering preserves sparse-but-precise models like Random Forest).

Outputs:
    Results/event_detection_results.csv        — per-subject, current config
    Results/event_detection_results_tuned.csv   — per-subject, best config
    stdout                                      — side-by-side comparison

Usage:
    python -m src.event_detection        # from project root
    python src/event_detection.py        # also works
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.load_data import (
    SAMPLING_RATE_HZ,
    SIT_TO_STAND_ID,
    load_labels,
    load_signals,
)
from src.models import LABEL_COL, SUBJECT_COL, ThresholdBaseline, _feature_cols
from src.windowing import PURITY_THRESHOLD, STRIDE, WINDOW_SIZE, _build_sample_labels

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
_FEATURES_CSV = _PROJECT_ROOT / "Data" / "uci_hapt" / "features.csv"
_RESULTS_DIR = _PROJECT_ROOT / "Results"
_RESULTS_CSV = _RESULTS_DIR / "event_detection_results.csv"
_TUNED_CSV = _RESULTS_DIR / "event_detection_results_tuned.csv"

MODEL_NAMES = ["Threshold Baseline", "Logistic Regression", "Random Forest"]

MATCH_TOLERANCE_SEC = 1.0   # ±seconds for event-level TP matching

# ---------------------------------------------------------------------------
# Post-processing configurations to compare
# ---------------------------------------------------------------------------
PP_CONFIGS = {
    "strict": {
        "label": "Strict (min_dur=2, gap=1, smooth k=3)",
        "smooth_k": 3,
        "smooth_thresh": 0.5,
        "min_event_windows": 2,
        "max_gap_windows": 1,
    },
    "relaxed": {
        "label": "Relaxed (min_dur=1, gap=2, smooth k=3)",
        "smooth_k": 3,
        "smooth_thresh": 0.5,
        "min_event_windows": 1,
        "max_gap_windows": 2,
    },
    "minimal": {
        "label": "Minimal (min_dur=1, gap=2, no smoothing)",
        "smooth_k": 1,              # k=1 means no smoothing
        "smooth_thresh": 0.5,
        "min_event_windows": 1,
        "max_gap_windows": 2,
    },
}


# ---------------------------------------------------------------------------
# Build per-window metadata
# ---------------------------------------------------------------------------

def _build_window_metadata(
    signals: dict[tuple[int, int], np.ndarray],
    labels_df: pd.DataFrame,
) -> pd.DataFrame:
    """Return a DataFrame with one row per kept window, columns:
        exp_id, user_id, win_start_sample, win_center_sec
    in the same order as the feature matrix rows.
    """
    rows: list[dict] = []

    for (exp_id, user_id), signal in sorted(signals.items()):
        T = signal.shape[0]
        mask = (
            (labels_df["experiment_id"] == exp_id)
            & (labels_df["user_id"] == user_id)
        )
        sample_labels = _build_sample_labels(T, labels_df.loc[mask])

        start = 0
        while start + WINDOW_SIZE <= T:
            win = sample_labels[start : start + WINDOW_SIZE]
            unique, counts = np.unique(win, return_counts=True)
            majority_frac = counts[np.argmax(counts)] / WINDOW_SIZE

            if majority_frac >= PURITY_THRESHOLD:
                center_sample = start + WINDOW_SIZE / 2
                rows.append({
                    "exp_id": exp_id,
                    "user_id": user_id,
                    "win_start_sample": start,
                    "win_center_sec": center_sample / SAMPLING_RATE_HZ,
                })
            start += STRIDE

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Ground truth helpers
# ---------------------------------------------------------------------------

def _gt_event_centers(labels_df: pd.DataFrame) -> dict[int, list[float]]:
    """Return {user_id: [(exp_id, center_sec), …]} for all activity-8 segments."""
    sts = labels_df[labels_df["activity_id"] == SIT_TO_STAND_ID].copy()
    sts["center_sample"] = (sts["start_sample"] + sts["end_sample"]) / 2.0
    sts["center_sec"] = sts["center_sample"] / SAMPLING_RATE_HZ

    out: dict[int, list[float]] = {}
    for uid in sts["user_id"].unique():
        sub = sts[sts["user_id"] == uid]
        out[uid] = list(zip(
            sub["experiment_id"].values,
            sub["center_sec"].values,
        ))
    return out


def _gt_rep_counts(labels_df: pd.DataFrame) -> dict[int, int]:
    sts = labels_df[labels_df["activity_id"] == SIT_TO_STAND_ID]
    return sts.groupby("user_id").size().to_dict()


# ---------------------------------------------------------------------------
# Post-processing primitives
# ---------------------------------------------------------------------------

def _smooth_probabilities(probs: np.ndarray, k: int) -> np.ndarray:
    """Causal moving average of length k (pads front with first value)."""
    if k <= 1 or len(probs) == 0:
        return probs.copy()
    kernel = np.ones(k) / k
    padded = np.concatenate([np.full(k - 1, probs[0]), probs])
    return np.convolve(padded, kernel, mode="valid")


def _extract_events(
    binary_preds: np.ndarray,
    exp_ids: np.ndarray,
    center_secs: np.ndarray,
    min_len: int,
    max_gap: int,
) -> list[tuple[int, float]]:
    """Extract discrete events from a temporal sequence of binary predictions."""
    n = len(binary_preds)
    if n == 0:
        return []

    # Step 1: runs of consecutive positives within each experiment.
    clusters: list[list[int]] = []
    current: list[int] = []

    for i in range(n):
        if binary_preds[i] == 1:
            if current and exp_ids[i] != exp_ids[current[-1]]:
                clusters.append(current)
                current = [i]
            else:
                current.append(i)
        else:
            if current:
                clusters.append(current)
                current = []
    if current:
        clusters.append(current)

    # Step 2: merge clusters separated by ≤ max_gap within same experiment.
    merged: list[list[int]] = []
    for cluster in clusters:
        if (merged
                and exp_ids[cluster[0]] == exp_ids[merged[-1][-1]]
                and (cluster[0] - merged[-1][-1] - 1) <= max_gap):
            merged[-1].extend(cluster)
        else:
            merged.append(cluster)

    # Step 3: discard clusters shorter than min_len.
    events: list[tuple[int, float]] = []
    for cluster in merged:
        if len(cluster) >= min_len:
            c_secs = center_secs[cluster]
            c_exp = exp_ids[cluster[0]]
            events.append((int(c_exp), float(c_secs.mean())))

    return events


# ---------------------------------------------------------------------------
# Event-level matching
# ---------------------------------------------------------------------------

def _match_events(
    pred_events: list[tuple[int, float]],
    gt_events: list[tuple[int, float]],
    tolerance_sec: float = MATCH_TOLERANCE_SEC,
) -> tuple[int, int, int]:
    """Match predicted events to GT events (same experiment, ±tolerance).
    Each GT can be matched at most once.  Returns (tp, fp, fn)."""
    gt_matched = [False] * len(gt_events)
    tp = 0

    for p_exp, p_sec in pred_events:
        best_dist = float("inf")
        best_idx = -1
        for j, (g_exp, g_sec) in enumerate(gt_events):
            if gt_matched[j]:
                continue
            if p_exp != g_exp:
                continue
            dist = abs(p_sec - g_sec)
            if dist < best_dist:
                best_dist = dist
                best_idx = j
        if best_idx >= 0 and best_dist <= tolerance_sec:
            gt_matched[best_idx] = True
            tp += 1

    fp = len(pred_events) - tp
    fn = len(gt_events) - tp
    return tp, fp, fn


# ---------------------------------------------------------------------------
# Model training (once per fold)
# ---------------------------------------------------------------------------

def _run_fold_with_probs(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    feat_cols: list[str],
) -> dict[str, dict]:
    """Train all three models.  Return y_pred and y_prob for each."""
    results: dict[str, dict] = {}

    # --- Threshold Baseline ---
    tb = ThresholdBaseline(n_grid=20)
    tb.fit(X_train, y_train)
    results["Threshold Baseline"] = {
        "y_pred": tb.predict(X_test),
        "y_prob": None,
    }

    # --- Logistic Regression ---
    scaler = StandardScaler()
    X_tr_sc = pd.DataFrame(
        scaler.fit_transform(X_train[feat_cols]),
        columns=feat_cols, index=X_train.index,
    )
    X_te_sc = pd.DataFrame(
        scaler.transform(X_test[feat_cols]),
        columns=feat_cols, index=X_test.index,
    )
    lr = LogisticRegression(
        class_weight="balanced", max_iter=1000, random_state=42,
    )
    lr.fit(X_tr_sc, y_train)
    results["Logistic Regression"] = {
        "y_pred": lr.predict(X_te_sc),
        "y_prob": lr.predict_proba(X_te_sc)[:, 1],
    }

    # --- Random Forest ---
    rf = RandomForestClassifier(
        n_estimators=100, class_weight="balanced", random_state=42,
    )
    rf.fit(X_train[feat_cols], y_train)
    results["Random Forest"] = {
        "y_pred": rf.predict(X_test[feat_cols]),
        "y_prob": rf.predict_proba(X_test[feat_cols])[:, 1],
    }

    return results


# ---------------------------------------------------------------------------
# Apply a single post-processing config to stored fold outputs
# ---------------------------------------------------------------------------

def _apply_postprocessing(
    fold_outputs: list[dict],
    pp: dict,
    gt_centers: dict,
    gt_reps: dict,
) -> pd.DataFrame:
    """Given raw per-fold model outputs, apply a post-processing config
    and return a results DataFrame."""

    smooth_k = pp["smooth_k"]
    smooth_thresh = pp["smooth_thresh"]
    min_dur = pp["min_event_windows"]
    max_gap = pp["max_gap_windows"]

    rows: list[dict] = []

    for fold in fold_outputs:
        held_out = fold["subject_id"]
        exp_ids = fold["exp_ids"]
        center_secs = fold["center_secs"]
        gt_events = gt_centers.get(held_out, [])
        n_gt = gt_reps.get(held_out, 0)

        for name in MODEL_NAMES:
            y_pred = fold["models"][name]["y_pred"]
            y_prob = fold["models"][name]["y_prob"]

            # --- Post-process ---
            if y_prob is not None and smooth_k > 1:
                smoothed = _smooth_probabilities(y_prob, k=smooth_k)
                binary = (smoothed >= smooth_thresh).astype(int)
            elif y_prob is not None and smooth_k <= 1:
                # No smoothing: threshold raw probabilities
                binary = (y_prob >= smooth_thresh).astype(int)
            else:
                # Threshold baseline: raw binary predictions always
                binary = y_pred.copy()

            pred_events = _extract_events(
                binary, exp_ids, center_secs,
                min_len=min_dur, max_gap=max_gap,
            )

            n_pred = len(pred_events)
            tp, fp, fn = _match_events(pred_events, gt_events)

            evt_prec = tp / (tp + fp) if (tp + fp) else 0.0
            evt_rec = tp / (tp + fn) if (tp + fn) else 0.0
            evt_f1 = (2 * evt_prec * evt_rec / (evt_prec + evt_rec)
                       if (evt_prec + evt_rec) else 0.0)
            abs_err = abs(n_pred - n_gt)

            rows.append({
                "subject_id": held_out,
                "model": name,
                "gt_reps": n_gt,
                "pred_reps": n_pred,
                "abs_error": abs_err,
                "event_tp": tp,
                "event_fp": fp,
                "event_fn": fn,
                "event_precision": evt_prec,
                "event_recall": evt_rec,
                "event_f1": evt_f1,
            })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Pretty-print a summary table for one config
# ---------------------------------------------------------------------------

def _summarise_config(results_df: pd.DataFrame) -> dict[str, dict]:
    """Return {model_name: {mae, prec, rec, f1, tp, fp, fn}} (pooled)."""
    summary: dict[str, dict] = {}
    for name in MODEL_NAMES:
        sub = results_df[results_df["model"] == name]
        mae = sub["abs_error"].mean()
        tp = int(sub["event_tp"].sum())
        fp = int(sub["event_fp"].sum())
        fn = int(sub["event_fn"].sum())
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        summary[name] = dict(mae=mae, prec=prec, rec=rec, f1=f1,
                             tp=tp, fp=fp, fn=fn)
    return summary


def _print_config_table(label: str, summary: dict[str, dict]) -> None:
    """Print a compact summary table for one configuration."""
    print(f"\n  {label}")
    print(f"  {'Model':<25} {'MAE':>6} {'Prec':>7} {'Rec':>7} "
          f"{'F1':>7} {'TP':>5} {'FP':>5} {'FN':>5}")
    print(f"  {'-' * 72}")
    for name in MODEL_NAMES:
        s = summary[name]
        print(f"  {name:<25} {s['mae']:>6.2f} {s['prec']:>7.3f} "
              f"{s['rec']:>7.3f} {s['f1']:>7.3f} "
              f"{s['tp']:>5} {s['fp']:>5} {s['fn']:>5}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # ------------------------------------------------------------------
    # Load features + metadata
    # ------------------------------------------------------------------
    print(f"Loading features from {_FEATURES_CSV} …")
    df = pd.read_csv(_FEATURES_CSV)
    feat_cols = _feature_cols(df)
    subjects = sorted(df[SUBJECT_COL].unique())
    n_subjects = len(subjects)

    print("Building per-window temporal metadata …")
    signals = load_signals()
    labels_df = load_labels()
    win_meta = _build_window_metadata(signals, labels_df)

    assert len(win_meta) == len(df), (
        f"Metadata length {len(win_meta)} != feature matrix length {len(df)}"
    )
    df["exp_id"] = win_meta["exp_id"].values
    df["win_center_sec"] = win_meta["win_center_sec"].values

    gt_centers = _gt_event_centers(labels_df)
    gt_reps = _gt_rep_counts(labels_df)

    print(f"  {len(df):,} windows, {n_subjects} subjects, "
          f"{sum(gt_reps.values())} total GT reps\n")

    # ------------------------------------------------------------------
    # LOSO loop — train ONCE, store raw outputs for all configs
    # ------------------------------------------------------------------
    fold_outputs: list[dict] = []

    for i, held_out in enumerate(subjects, 1):
        test_mask = df[SUBJECT_COL] == held_out
        train_mask = ~test_mask

        X_train = df.loc[train_mask]
        y_train = df.loc[train_mask, LABEL_COL].values
        X_test = df.loc[test_mask].copy()

        fold_results = _run_fold_with_probs(
            X_train, y_train, X_test, feat_cols,
        )

        fold_outputs.append({
            "subject_id": held_out,
            "exp_ids": X_test["exp_id"].values.astype(int),
            "center_secs": X_test["win_center_sec"].values,
            "models": fold_results,
        })

        n_gt = gt_reps.get(held_out, 0)
        print(f"  Fold {i:2d}/{n_subjects} (subject {held_out:2d}) — "
              f"GT reps: {n_gt} ✓")

    # ------------------------------------------------------------------
    # Evaluate all three post-processing configs
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("Post-Processing Configuration Comparison")
    print("=" * 80)

    all_summaries: dict[str, dict[str, dict]] = {}

    for cfg_key, pp in PP_CONFIGS.items():
        results_df = _apply_postprocessing(
            fold_outputs, pp, gt_centers, gt_reps,
        )
        summary = _summarise_config(results_df)
        all_summaries[cfg_key] = summary

        _print_config_table(pp["label"], summary)

        # Save the original "strict" config to the standard CSV.
        if cfg_key == "strict":
            _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
            results_df.to_csv(_RESULTS_CSV, index=False)
            print(f"\n  → Saved to {_RESULTS_CSV}")

    # ------------------------------------------------------------------
    # Side-by-side comparison: one row per (model × config)
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("Side-by-Side Comparison")
    print("=" * 80)
    print(f"{'Model':<25} {'Config':<12} {'MAE':>6} {'Prec':>7} "
          f"{'Rec':>7} {'F1':>7} {'TP':>5} {'FP':>5} {'FN':>5}")
    print("-" * 80)

    for name in MODEL_NAMES:
        for cfg_key in PP_CONFIGS:
            s = all_summaries[cfg_key][name]
            tag = cfg_key.capitalize()
            print(f"  {name:<23} {tag:<12} {s['mae']:>6.2f} "
                  f"{s['prec']:>7.3f} {s['rec']:>7.3f} {s['f1']:>7.3f} "
                  f"{s['tp']:>5} {s['fp']:>5} {s['fn']:>5}")
        print()

    # ------------------------------------------------------------------
    # Find the best (config, model) pair by event-level F1
    # ------------------------------------------------------------------
    best_f1 = -1.0
    best_cfg = ""
    best_model = ""
    for cfg_key, summary in all_summaries.items():
        for name, s in summary.items():
            if s["f1"] > best_f1:
                best_f1 = s["f1"]
                best_cfg = cfg_key
                best_model = name

    print(f"Best event-level F1: {best_f1:.3f}  "
          f"({best_model}, {PP_CONFIGS[best_cfg]['label']})")

    # Save that configuration's full results.
    best_results_df = _apply_postprocessing(
        fold_outputs, PP_CONFIGS[best_cfg], gt_centers, gt_reps,
    )
    best_results_df["pp_config"] = best_cfg
    best_results_df.to_csv(_TUNED_CSV, index=False)
    print(f"Best config per-subject results saved to {_TUNED_CSV}")

    # ------------------------------------------------------------------
    # Per-subject detail for the best config (all models)
    # ------------------------------------------------------------------
    print(f"\n{'=' * 90}")
    print(f"Per-Subject Detail — {PP_CONFIGS[best_cfg]['label']}")
    print(f"{'=' * 90}")

    for name in MODEL_NAMES:
        sub = best_results_df[best_results_df["model"] == name]
        print(f"\n{'— ' + name + ' —':^90}")
        print(f"{'Subj':>5}  {'GT':>4}  {'Pred':>5}  {'|Err|':>5}  "
              f"{'EvtTP':>5}  {'EvtFP':>5}  {'EvtFN':>5}  "
              f"{'EvtPrec':>8}  {'EvtRec':>8}  {'EvtF1':>8}")
        print("-" * 75)
        for _, r in sub.iterrows():
            print(f"{int(r.subject_id):>5}  {int(r.gt_reps):>4}  "
                  f"{int(r.pred_reps):>5}  {int(r.abs_error):>5}  "
                  f"{int(r.event_tp):>5}  {int(r.event_fp):>5}  "
                  f"{int(r.event_fn):>5}  "
                  f"{r.event_precision:>8.3f}  {r.event_recall:>8.3f}  "
                  f"{r.event_f1:>8.3f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
