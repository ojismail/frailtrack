"""
windowing.py — Sliding-window segmentation for sit-to-stand detection.

Imports raw 6-channel signals and labels from load_data, then:
  1. Slides a 128-sample window (2.56 s @ 50 Hz) with 50 % overlap (stride 64)
  2. Assigns each window the majority activity label
  3. Drops windows where < 50 % of samples share a single label (ambiguous)
  4. Remaps to binary: activity 8 (SIT_TO_STAND) → 1, everything else → 0
  5. Tracks user_id per window for leave-one-subject-out CV

Usage:
    python -m src.windowing          # from project root
    python src/windowing.py          # also works
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.load_data import (
    ACTIVITY_LABELS,
    SAMPLING_RATE_HZ,
    SIT_TO_STAND_ID,
    load_labels,
    load_signals,
)

# ---------------------------------------------------------------------------
# Windowing parameters
# ---------------------------------------------------------------------------
WINDOW_SIZE = 128          # samples per window (2.56 s at 50 Hz)
STRIDE = 64               # 50 % overlap
PURITY_THRESHOLD = 0.50   # minimum fraction of samples with the same label


# ---------------------------------------------------------------------------
# Helper: build a per-sample activity vector for one experiment
# ---------------------------------------------------------------------------

def _build_sample_labels(
    signal_length: int,
    exp_labels: pd.DataFrame,
) -> np.ndarray:
    """Create an int array of length *signal_length* where each element is
    the activity_id that sample belongs to, or 0 for unlabeled gaps.

    Parameters
    ----------
    signal_length : int
        Number of rows in the 6-channel signal for this experiment.
    exp_labels : pd.DataFrame
        Subset of the labels DataFrame for this experiment-user pair.
        Must contain columns: activity_id, start_sample, end_sample.
        start_sample / end_sample are **1-indexed** and inclusive.
    """
    sample_labels = np.zeros(signal_length, dtype=np.int32)

    for _, row in exp_labels.iterrows():
        # Convert 1-indexed inclusive range → 0-indexed slice.
        start = int(row["start_sample"]) - 1
        end = int(row["end_sample"])          # exclusive upper bound
        end = min(end, signal_length)         # guard against overshoot
        if start < 0:
            start = 0
        sample_labels[start:end] = int(row["activity_id"])

    return sample_labels


# ---------------------------------------------------------------------------
# Core windowing function
# ---------------------------------------------------------------------------

def create_windows(
    signals: dict[tuple[int, int], np.ndarray] | None = None,
    labels_df: pd.DataFrame | None = None,
    window_size: int = WINDOW_SIZE,
    stride: int = STRIDE,
    purity_threshold: float = PURITY_THRESHOLD,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Segment all experiment signals into fixed-length windows.

    Parameters
    ----------
    signals : dict, optional
        Output of ``load_signals()``.  Loaded automatically if *None*.
    labels_df : DataFrame, optional
        Output of ``load_labels()``.  Loaded automatically if *None*.
    window_size : int
        Samples per window.
    stride : int
        Step between consecutive windows.
    purity_threshold : float
        Minimum fraction of samples that must agree for a window to be kept.

    Returns
    -------
    X : np.ndarray, shape (N, window_size, 6)
        Windowed sensor data.
    y : np.ndarray, shape (N,)
        Binary labels — 1 = sit-to-stand, 0 = everything else.
    users : np.ndarray, shape (N,)
        Subject ID for each window.
    n_dropped : int
        Number of ambiguous windows that were discarded.
    """
    if signals is None:
        signals = load_signals()
    if labels_df is None:
        labels_df = load_labels()

    windows: list[np.ndarray] = []
    window_labels: list[int] = []
    window_users: list[int] = []
    n_dropped = 0

    for (exp_id, user_id), signal in sorted(signals.items()):
        T = signal.shape[0]

        # Labels for this experiment-user pair.
        mask = (
            (labels_df["experiment_id"] == exp_id)
            & (labels_df["user_id"] == user_id)
        )
        exp_labels = labels_df.loc[mask]

        sample_labels = _build_sample_labels(T, exp_labels)

        # Slide the window across the signal.
        start = 0
        while start + window_size <= T:
            win_labels = sample_labels[start : start + window_size]

            # Majority vote — find the most common label and its fraction.
            unique, counts = np.unique(win_labels, return_counts=True)
            majority_idx = np.argmax(counts)
            majority_label = unique[majority_idx]
            majority_frac = counts[majority_idx] / window_size

            if majority_frac < purity_threshold:
                n_dropped += 1
            else:
                windows.append(signal[start : start + window_size])
                # Binary remap: sit-to-stand → 1, else → 0.
                window_labels.append(
                    1 if majority_label == SIT_TO_STAND_ID else 0
                )
                window_users.append(user_id)

            start += stride

    X = np.stack(windows, axis=0)                 # (N, 128, 6)
    y = np.asarray(window_labels, dtype=np.int32) # (N,)
    users = np.asarray(window_users, dtype=np.int32)

    return X, y, users, n_dropped


# ---------------------------------------------------------------------------
# Pretty-print summary
# ---------------------------------------------------------------------------

def print_summary(
    X: np.ndarray,
    y: np.ndarray,
    users: np.ndarray,
    n_dropped: int,
) -> None:
    n_total = len(y)
    n_pos = int(y.sum())
    n_neg = n_total - n_pos
    pct_pos = 100.0 * n_pos / n_total if n_total else 0.0

    print("=" * 60)
    print("Windowing Summary")
    print("=" * 60)
    print(f"Window size           : {WINDOW_SIZE} samples "
          f"({WINDOW_SIZE / SAMPLING_RATE_HZ:.2f} s)")
    print(f"Stride                : {STRIDE} samples "
          f"({STRIDE / SAMPLING_RATE_HZ:.2f} s)  "
          f"({100 * (1 - STRIDE / WINDOW_SIZE):.0f}% overlap)")
    print(f"Purity threshold      : {PURITY_THRESHOLD:.0%}")
    print()
    print(f"Windows kept          : {n_total:,}")
    print(f"  Sit-to-stand (1)    : {n_pos:,}")
    print(f"  Other          (0)  : {n_neg:,}")
    print(f"  Positive class %    : {pct_pos:.2f}%")
    print(f"Windows dropped (amb) : {n_dropped:,}")
    print(f"X shape               : {X.shape}")
    print(f"Unique users          : {np.unique(users).tolist()}")

    # Per-user breakdown of positive windows.
    print(f"\n{'User':<8} {'Total':>8} {'Pos (1)':>8} {'Neg (0)':>8} {'Pos %':>8}")
    print("-" * 44)
    for uid in np.unique(users):
        mask = users == uid
        tot = int(mask.sum())
        pos = int(y[mask].sum())
        neg = tot - pos
        pct = 100.0 * pos / tot if tot else 0.0
        print(f"{uid:<8} {tot:>8} {pos:>8} {neg:>8} {pct:>7.2f}%")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("Loading signals and labels …")
    signals = load_signals()
    labels_df = load_labels()

    print("Creating windows …\n")
    X, y, users, n_dropped = create_windows(signals, labels_df)

    print_summary(X, y, users, n_dropped)


if __name__ == "__main__":
    main()
