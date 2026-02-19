"""
models.py — Three binary classifiers for sit-to-stand detection.

Models
------
1. ThresholdBaseline  — dual-threshold on accel_mag_max & accel_mag_range
2. Logistic Regression — sklearn, all 48 features, StandardScaler
3. Random Forest       — sklearn, all 48 features, no scaling

Quick sanity test (train: users 1–25, test: users 26–30) is run when
this file is executed directly.

Usage:
    python -m src.models             # from project root
    python src/models.py             # also works
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
_FEATURES_CSV = _PROJECT_ROOT / "Data" / "uci_hapt" / "features.csv"

# The 48 feature columns (everything except label and subject_id).
LABEL_COL = "label"
SUBJECT_COL = "subject_id"
THRESHOLD_FEATURES = ["accel_mag_max", "accel_mag_range"]


# ---------------------------------------------------------------------------
# Model 1: Threshold Baseline
# ---------------------------------------------------------------------------

class ThresholdBaseline:
    """Predict sit-to-stand if **both** accel_mag_max > t_max AND
    accel_mag_range > t_range.

    Thresholds are chosen via grid search on training data to maximise
    F1 for the positive class (label=1).
    """

    def __init__(self, n_grid: int = 20) -> None:
        self.n_grid = n_grid
        self.threshold_max: float | None = None
        self.threshold_range: float | None = None

    # ---- helpers ----
    @staticmethod
    def _f1(tp: int, fp: int, fn: int) -> float:
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        return 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0

    # ---- sklearn-compatible interface ----
    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "ThresholdBaseline":
        """Grid-search for the best (t_max, t_range) on training data.

        Parameters
        ----------
        X : DataFrame with at least ``accel_mag_max`` and ``accel_mag_range``.
        y : 1-d array of binary labels.
        """
        mag_max = X["accel_mag_max"].values
        mag_range = X["accel_mag_range"].values
        y = np.asarray(y)

        # Candidate grids: 20 values between the 10th and 90th percentile.
        max_lo, max_hi = np.percentile(mag_max, 10), np.percentile(mag_max, 90)
        rng_lo, rng_hi = np.percentile(mag_range, 10), np.percentile(mag_range, 90)

        cands_max = np.linspace(max_lo, max_hi, self.n_grid)
        cands_rng = np.linspace(rng_lo, rng_hi, self.n_grid)

        best_f1 = -1.0
        best_tm, best_tr = cands_max[0], cands_rng[0]

        for tm in cands_max:
            pred_max = mag_max > tm
            for tr in cands_rng:
                pred = (pred_max & (mag_range > tr)).astype(int)
                tp = int(((pred == 1) & (y == 1)).sum())
                fp = int(((pred == 1) & (y == 0)).sum())
                fn = int(((pred == 0) & (y == 1)).sum())
                f1 = self._f1(tp, fp, fn)
                if f1 > best_f1:
                    best_f1 = f1
                    best_tm, best_tr = tm, tr

        self.threshold_max = float(best_tm)
        self.threshold_range = float(best_tr)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        assert self.threshold_max is not None, "Call .fit() first"
        mag_max = X["accel_mag_max"].values
        mag_range = X["accel_mag_range"].values
        return ((mag_max > self.threshold_max) &
                (mag_range > self.threshold_range)).astype(int)


# ---------------------------------------------------------------------------
# Helpers for the sklearn models
# ---------------------------------------------------------------------------

def _feature_cols(df: pd.DataFrame) -> list[str]:
    """Return the 48 feature column names (everything except label/subject)."""
    return [c for c in df.columns if c not in (LABEL_COL, SUBJECT_COL)]


def _split_train_test(
    df: pd.DataFrame,
    train_subjects: list[int] | range,
    test_subjects: list[int] | range,
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray]:
    """Split a feature DataFrame by subject_id."""
    feat_cols = _feature_cols(df)
    train_mask = df[SUBJECT_COL].isin(train_subjects)
    test_mask = df[SUBJECT_COL].isin(test_subjects)

    X_train = df.loc[train_mask, feat_cols]
    y_train = df.loc[train_mask, LABEL_COL].values
    X_test = df.loc[test_mask, feat_cols]
    y_test = df.loc[test_mask, LABEL_COL].values
    return X_train, y_train, X_test, y_test


# ---------------------------------------------------------------------------
# Evaluation helper
# ---------------------------------------------------------------------------

def _evaluate(name: str, y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """Print classification report + TP/FP/FN for the positive class."""
    print(f"\n{'— ' + name + ' —':^60}")

    # sklearn classification_report for label=1.
    print(classification_report(
        y_true, y_pred, target_names=["other (0)", "sit-to-stand (1)"], digits=3,
        zero_division=0,
    ))

    # Confusion-matrix derived counts.
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    print(f"  True Positives  (TP): {tp}")
    print(f"  False Positives (FP): {fp}")
    print(f"  False Negatives (FN): {fn}")
    print(f"  True Negatives  (TN): {tn}")


# ---------------------------------------------------------------------------
# Main sanity test
# ---------------------------------------------------------------------------

def main() -> None:
    # ------------------------------------------------------------------
    # Load features
    # ------------------------------------------------------------------
    print(f"Loading features from: {_FEATURES_CSV}\n")
    df = pd.read_csv(_FEATURES_CSV)
    feat_cols = _feature_cols(df)
    print(f"Feature matrix : {df.shape[0]} rows × {len(feat_cols)} features")

    # ------------------------------------------------------------------
    # Train / test split by subject
    # ------------------------------------------------------------------
    train_subjects = list(range(1, 26))   # users 1–25
    test_subjects = list(range(26, 31))   # users 26–30

    X_train, y_train, X_test, y_test = _split_train_test(
        df, train_subjects, test_subjects,
    )

    n_train_pos = int(y_train.sum())
    n_test_pos = int(y_test.sum())
    print(f"Train set      : {len(y_train):,} windows  "
          f"({n_train_pos} positive, {len(y_train) - n_train_pos} negative)")
    print(f"Test  set      : {len(y_test):,} windows  "
          f"({n_test_pos} positive, {len(y_test) - n_test_pos} negative)")

    # ==================================================================
    # Model 1: Threshold Baseline
    # ==================================================================
    tb = ThresholdBaseline(n_grid=20)
    tb.fit(X_train, y_train)
    print(f"\nThreshold Baseline — fitted thresholds: "
          f"accel_mag_max > {tb.threshold_max:.4f}, "
          f"accel_mag_range > {tb.threshold_range:.4f}")
    y_pred_tb = tb.predict(X_test)
    _evaluate("Threshold Baseline", y_test, y_pred_tb)

    # ==================================================================
    # Model 2: Logistic Regression (scaled)
    # ==================================================================
    scaler = StandardScaler()
    X_train_sc = pd.DataFrame(
        scaler.fit_transform(X_train), columns=feat_cols, index=X_train.index,
    )
    X_test_sc = pd.DataFrame(
        scaler.transform(X_test), columns=feat_cols, index=X_test.index,
    )

    lr = LogisticRegression(
        class_weight="balanced", max_iter=1000, random_state=42,
    )
    lr.fit(X_train_sc, y_train)
    y_pred_lr = lr.predict(X_test_sc)
    _evaluate("Logistic Regression", y_test, y_pred_lr)

    # ==================================================================
    # Model 3: Random Forest
    # ==================================================================
    rf = RandomForestClassifier(
        n_estimators=100, class_weight="balanced", random_state=42,
    )
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    _evaluate("Random Forest", y_test, y_pred_rf)


if __name__ == "__main__":
    main()
