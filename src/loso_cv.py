"""
loso_cv.py — Leave-One-Subject-Out cross-validation for sit-to-stand detection.

Evaluates three models (Threshold Baseline, Logistic Regression, Random Forest)
across all 30 subjects, computes per-fold and aggregate metrics, and generates
a confusion-matrix plot plus a false-positive activity breakdown.

Outputs:
    Results/loso_cv_results.csv   — per-fold metrics
    Results/confusion_matrix.png  — pooled confusion matrix for best model

Usage:
    python -m src.loso_cv            # from project root
    python src/loso_cv.py            # also works
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import StandardScaler

from src.load_data import ACTIVITY_LABELS, load_labels, load_signals
from src.models import LABEL_COL, SUBJECT_COL, ThresholdBaseline, _feature_cols
from src.windowing import PURITY_THRESHOLD, STRIDE, WINDOW_SIZE, _build_sample_labels

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
_FEATURES_CSV = _PROJECT_ROOT / "Data" / "uci_hapt" / "features.csv"
_RESULTS_DIR = _PROJECT_ROOT / "Results"
_RESULTS_CSV = _RESULTS_DIR / "loso_cv_results.csv"
_CONFUSION_PNG = _RESULTS_DIR / "confusion_matrix.png"

MODEL_NAMES = ["Threshold Baseline", "Logistic Regression", "Random Forest"]


# ---------------------------------------------------------------------------
# Reconstruct per-window multi-class activity labels (for FP analysis)
# ---------------------------------------------------------------------------

def _get_multiclass_window_labels() -> np.ndarray:
    """Replay the windowing logic to recover the *original* majority
    activity_id (1–12, or 0 for unlabeled) for every window, in the
    same order that ``create_windows`` produces them.
    """
    signals = load_signals()
    labels_df = load_labels()

    mc_labels: list[int] = []

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
            majority_idx = int(np.argmax(counts))
            majority_frac = counts[majority_idx] / WINDOW_SIZE

            if majority_frac >= PURITY_THRESHOLD:
                mc_labels.append(int(unique[majority_idx]))

            start += STRIDE

    return np.asarray(mc_labels, dtype=np.int32)


# ---------------------------------------------------------------------------
# Single-fold evaluation
# ---------------------------------------------------------------------------

def _run_fold(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    feat_cols: list[str],
) -> dict[str, dict]:
    """Train all three models on one fold. Return dict[model_name → metrics]."""

    results: dict[str, dict] = {}

    # ---------- Model 1: Threshold Baseline ----------
    tb = ThresholdBaseline(n_grid=20)
    tb.fit(X_train, y_train)
    y_pred_tb = tb.predict(X_test)

    results["Threshold Baseline"] = {
        "y_pred": y_pred_tb,
        "y_prob": None,
    }

    # ---------- Model 2: Logistic Regression ----------
    scaler = StandardScaler()
    X_train_sc = pd.DataFrame(
        scaler.fit_transform(X_train[feat_cols]),
        columns=feat_cols, index=X_train.index,
    )
    X_test_sc = pd.DataFrame(
        scaler.transform(X_test[feat_cols]),
        columns=feat_cols, index=X_test.index,
    )

    lr = LogisticRegression(
        class_weight="balanced", max_iter=1000, random_state=42,
    )
    lr.fit(X_train_sc, y_train)
    y_pred_lr = lr.predict(X_test_sc)
    y_prob_lr = lr.predict_proba(X_test_sc)[:, 1]

    results["Logistic Regression"] = {
        "y_pred": y_pred_lr,
        "y_prob": y_prob_lr,
    }

    # ---------- Model 3: Random Forest ----------
    rf = RandomForestClassifier(
        n_estimators=100, class_weight="balanced", random_state=42,
    )
    rf.fit(X_train[feat_cols], y_train)
    y_pred_rf = rf.predict(X_test[feat_cols])
    y_prob_rf = rf.predict_proba(X_test[feat_cols])[:, 1]

    results["Random Forest"] = {
        "y_pred": y_pred_rf,
        "y_prob": y_prob_rf,
    }

    return results


# ---------------------------------------------------------------------------
# Main LOSO-CV loop
# ---------------------------------------------------------------------------

def main() -> None:
    # ------------------------------------------------------------------
    # Load feature matrix
    # ------------------------------------------------------------------
    print(f"Loading features from {_FEATURES_CSV} …")
    df = pd.read_csv(_FEATURES_CSV)
    feat_cols = _feature_cols(df)
    subjects = sorted(df[SUBJECT_COL].unique())
    n_subjects = len(subjects)
    print(f"  {df.shape[0]:,} windows, {len(feat_cols)} features, "
          f"{n_subjects} subjects\n")

    # ------------------------------------------------------------------
    # Storage for per-fold bookkeeping
    # ------------------------------------------------------------------
    fold_rows: list[dict] = []                # → per-fold results DataFrame
    pooled: dict[str, dict] = {               # aggregated y_true / y_pred
        name: {"y_true": [], "y_pred": []}
        for name in MODEL_NAMES
    }

    # ------------------------------------------------------------------
    # LOSO loop
    # ------------------------------------------------------------------
    for i, held_out in enumerate(subjects, 1):
        test_mask = df[SUBJECT_COL] == held_out
        train_mask = ~test_mask

        X_train = df.loc[train_mask]
        y_train = df.loc[train_mask, LABEL_COL].values
        X_test = df.loc[test_mask]
        y_test = df.loc[test_mask, LABEL_COL].values

        n_test_pos = int(y_test.sum())
        n_test_neg = len(y_test) - n_test_pos

        fold_results = _run_fold(X_train, y_train, X_test, y_test, feat_cols)

        for name in MODEL_NAMES:
            y_pred = fold_results[name]["y_pred"]
            y_prob = fold_results[name]["y_prob"]

            prec = precision_score(y_test, y_pred, zero_division=0)
            rec = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)

            # PR-AUC: only defined when test set has at least 1 positive.
            pr_auc = np.nan
            if y_prob is not None and n_test_pos > 0:
                pr_auc = average_precision_score(y_test, y_prob)

            tn, fp, fn, tp = confusion_matrix(
                y_test, y_pred, labels=[0, 1],
            ).ravel()

            fold_rows.append({
                "subject_id": held_out,
                "model": name,
                "precision": prec,
                "recall": rec,
                "f1": f1,
                "pr_auc": pr_auc,
                "tp": int(tp),
                "fp": int(fp),
                "fn": int(fn),
                "n_test_pos": n_test_pos,
                "n_test_neg": n_test_neg,
            })

            pooled[name]["y_true"].extend(y_test.tolist())
            pooled[name]["y_pred"].extend(y_pred.tolist())

        print(f"  Fold {i:2d}/{n_subjects} (subject {held_out:2d}) — "
              f"test {len(y_test):>4} windows "
              f"({n_test_pos} pos, {n_test_neg} neg) ✓")

    # ------------------------------------------------------------------
    # Build results DataFrame
    # ------------------------------------------------------------------
    results_df = pd.DataFrame(fold_rows)
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(_RESULTS_CSV, index=False)
    print(f"\nPer-fold results saved to {_RESULTS_CSV}")

    # ------------------------------------------------------------------
    # Per-fold table (all folds × all models)
    # ------------------------------------------------------------------
    print("\n" + "=" * 100)
    print("Per-Fold Results")
    print("=" * 100)

    for name in MODEL_NAMES:
        sub = results_df[results_df["model"] == name]
        print(f"\n{'— ' + name + ' —':^100}")
        print(f"{'Subj':>5} {'Prec':>8} {'Recall':>8} {'F1':>8} "
              f"{'PR-AUC':>8} {'TP':>5} {'FP':>5} {'FN':>5} "
              f"{'#Pos':>5} {'#Neg':>6}")
        print("-" * 100)
        for _, r in sub.iterrows():
            prauc_str = f"{r.pr_auc:.3f}" if not np.isnan(r.pr_auc) else "  n/a"
            print(f"{int(r.subject_id):>5} {r.precision:>8.3f} {r.recall:>8.3f} "
                  f"{r.f1:>8.3f} {prauc_str:>8} {int(r.tp):>5} {int(r.fp):>5} "
                  f"{int(r.fn):>5} {int(r.n_test_pos):>5} {int(r.n_test_neg):>6}")

    # ------------------------------------------------------------------
    # Summary table: mean ± std
    # ------------------------------------------------------------------
    print("\n" + "=" * 100)
    print("Aggregate Summary (mean ± std across 30 LOSO folds)")
    print("=" * 100)

    header = (f"{'Model':<25} {'Precision':>16} {'Recall (all)':>16} "
              f"{'Recall (pos>0)':>16} {'F1':>16} {'PR-AUC':>16}")
    print(header)
    print("-" * 107)

    for name in MODEL_NAMES:
        sub = results_df[results_df["model"] == name]

        prec_m, prec_s = sub["precision"].mean(), sub["precision"].std()
        rec_all_m, rec_all_s = sub["recall"].mean(), sub["recall"].std()

        # Recall over folds that actually have positive test examples.
        pos_folds = sub[sub["n_test_pos"] > 0]
        rec_pos_m = pos_folds["recall"].mean()
        rec_pos_s = pos_folds["recall"].std()
        n_pos_folds = len(pos_folds)

        f1_m, f1_s = sub["f1"].mean(), sub["f1"].std()

        prauc_valid = sub["pr_auc"].dropna()
        if len(prauc_valid):
            prauc_m, prauc_s = prauc_valid.mean(), prauc_valid.std()
            prauc_str = f"{prauc_m:.3f} ± {prauc_s:.3f}"
        else:
            prauc_str = "       —"

        print(f"{name:<25} {prec_m:.3f} ± {prec_s:.3f}  "
              f"{rec_all_m:.3f} ± {rec_all_s:.3f}  "
              f"{rec_pos_m:.3f} ± {rec_pos_s:.3f}  "
              f"{f1_m:.3f} ± {f1_s:.3f}  "
              f"{prauc_str}")

    # Note about recall variants.
    n_zero_pos = int((results_df.groupby("subject_id")["n_test_pos"]
                      .first() == 0).sum())
    print(f"\nNote: {n_zero_pos} of {n_subjects} folds have 0 positive test "
          f"examples (recall = 0 by definition).")
    print("  'Recall (all)' includes those folds; "
          "'Recall (pos>0)' excludes them.")

    # ------------------------------------------------------------------
    # Identify best model by mean F1
    # ------------------------------------------------------------------
    f1_means = {
        name: results_df[results_df["model"] == name]["f1"].mean()
        for name in MODEL_NAMES
    }
    best_model = max(f1_means, key=f1_means.get)  # type: ignore[arg-type]
    print(f"\nBest model by mean F1: {best_model} (F1 = {f1_means[best_model]:.3f})")

    # ------------------------------------------------------------------
    # Pooled confusion matrix for best model
    # ------------------------------------------------------------------
    y_true_all = np.asarray(pooled[best_model]["y_true"])
    y_pred_all = np.asarray(pooled[best_model]["y_pred"])

    cm = confusion_matrix(y_true_all, y_pred_all, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    print(f"\nPooled confusion matrix ({best_model}):")
    print(f"  TP={tp}  FP={fp}  FN={fn}  TN={tn}")

    fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Other (0)", "Sit-to-Stand (1)"],
    )
    disp.plot(ax=ax, cmap="Blues", values_format="d")
    ax.set_title(f"Pooled Confusion Matrix — {best_model}\n(LOSO-CV, {n_subjects} folds)")
    fig.savefig(_CONFUSION_PNG, dpi=150)
    plt.close(fig)
    print(f"Confusion matrix saved to {_CONFUSION_PNG}")

    # ------------------------------------------------------------------
    # False-positive activity breakdown
    # ------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print(f"False-Positive Activity Breakdown ({best_model})")
    print(f"{'=' * 60}")
    print("When the model incorrectly predicts sit-to-stand, what was")
    print("the true (multi-class) activity?\n")

    # Recover multi-class labels for every window.
    print("Reconstructing per-window multi-class labels …")
    mc_labels = _get_multiclass_window_labels()
    assert len(mc_labels) == len(y_true_all), (
        f"Length mismatch: {len(mc_labels)} vs {len(y_true_all)}"
    )

    fp_mask = (y_pred_all == 1) & (y_true_all == 0)
    fp_activities = mc_labels[fp_mask]

    if len(fp_activities) == 0:
        print("  No false positives — nothing to break down.")
    else:
        unique_acts, act_counts = np.unique(fp_activities, return_counts=True)
        order = np.argsort(-act_counts)
        print(f"{'Activity ID':<14} {'Activity Name':<22} {'FP Count':>9} {'% of FPs':>9}")
        print("-" * 58)
        for idx in order:
            aid = int(unique_acts[idx])
            cnt = int(act_counts[idx])
            name_str = ACTIVITY_LABELS.get(aid, f"UNLABELED ({aid})")
            pct = 100.0 * cnt / len(fp_activities)
            print(f"{aid:<14} {name_str:<22} {cnt:>9} {pct:>8.1f}%")
        print(f"{'':14} {'TOTAL':<22} {len(fp_activities):>9}")

    print("\nDone.")


if __name__ == "__main__":
    main()
