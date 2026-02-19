"""
features.py — Extract per-window statistical features for sit-to-stand detection.

Pipeline:
  1. Import windowed data (128×6) from windowing.py
  2. Append 2 magnitude channels → 128×8
  3. Compute 6 statistical features per channel → 48 features per window
  4. Save to Data/uci_hapt/features.csv  (48 features + label + subject_id)
  5. Generate a sanity-check plot → Results/sanity_check.png

Usage:
    python -m src.features           # from project root
    python src/features.py           # also works
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")                       # headless backend — no display needed
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

from src.load_data import (
    ACTIVITY_LABELS,
    SAMPLING_RATE_HZ,
    SIT_TO_STAND_ID,
    load_labels,
    load_signals,
)
from src.windowing import STRIDE, WINDOW_SIZE, create_windows

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
_FEATURES_CSV = _PROJECT_ROOT / "Data" / "uci_hapt" / "features.csv"
_RESULTS_DIR = _PROJECT_ROOT / "Results"
_SANITY_PNG = _RESULTS_DIR / "sanity_check.png"

# ---------------------------------------------------------------------------
# Channel / feature names
# ---------------------------------------------------------------------------
RAW_CHANNELS = ["ax", "ay", "az", "gx", "gy", "gz"]
MAG_CHANNELS = ["accel_mag", "gyro_mag"]
ALL_CHANNELS = RAW_CHANNELS + MAG_CHANNELS          # 8 total

STAT_NAMES = ["mean", "std", "min", "max", "range", "energy"]

FEATURE_COLUMNS = [
    f"{ch}_{stat}" for ch in ALL_CHANNELS for stat in STAT_NAMES
]                                                     # 8 × 6 = 48


# ---------------------------------------------------------------------------
# Core feature extraction
# ---------------------------------------------------------------------------

def _augment_magnitudes(X: np.ndarray) -> np.ndarray:
    """Append accel_mag and gyro_mag channels.

    Parameters
    ----------
    X : (N, W, 6) — [ax, ay, az, gx, gy, gz]

    Returns
    -------
    X_aug : (N, W, 8) — [ax, ay, az, gx, gy, gz, accel_mag, gyro_mag]
    """
    accel_mag = np.sqrt(np.sum(X[:, :, :3] ** 2, axis=2, keepdims=True))
    gyro_mag = np.sqrt(np.sum(X[:, :, 3:6] ** 2, axis=2, keepdims=True))
    return np.concatenate([X, accel_mag, gyro_mag], axis=2)


def extract_features(X: np.ndarray) -> np.ndarray:
    """Compute 6 statistical features for each of 8 channels.

    Parameters
    ----------
    X : (N, W, 6)  — raw windowed data (6 channels).

    Returns
    -------
    feats : (N, 48)  — feature matrix.
    """
    X_aug = _augment_magnitudes(X)                    # (N, W, 8)

    means   = X_aug.mean(axis=1)                      # (N, 8)
    stds    = X_aug.std(axis=1, ddof=0)               # (N, 8)
    mins    = X_aug.min(axis=1)                       # (N, 8)
    maxs    = X_aug.max(axis=1)                       # (N, 8)
    ranges  = maxs - mins                             # (N, 8)
    energy  = (X_aug ** 2).mean(axis=1)               # (N, 8)

    # Interleave: for each channel, stack its 6 stats contiguously.
    parts = []
    for ch_idx in range(X_aug.shape[2]):
        parts.append(means[:, ch_idx : ch_idx + 1])
        parts.append(stds[:, ch_idx : ch_idx + 1])
        parts.append(mins[:, ch_idx : ch_idx + 1])
        parts.append(maxs[:, ch_idx : ch_idx + 1])
        parts.append(ranges[:, ch_idx : ch_idx + 1])
        parts.append(energy[:, ch_idx : ch_idx + 1])

    return np.hstack(parts)                           # (N, 48)


def build_feature_dataframe(
    X: np.ndarray,
    y: np.ndarray,
    users: np.ndarray,
) -> pd.DataFrame:
    """Return a tidy DataFrame with 48 features + label + subject_id."""
    feats = extract_features(X)
    df = pd.DataFrame(feats, columns=FEATURE_COLUMNS)
    df["label"] = y
    df["subject_id"] = users
    return df


# ---------------------------------------------------------------------------
# Sanity-check plot
# ---------------------------------------------------------------------------

def _sanity_check_plot(
    signals: dict[tuple[int, int], np.ndarray],
    labels_df: pd.DataFrame,
    df_features: pd.DataFrame,
    y: np.ndarray,
    users: np.ndarray,
) -> None:
    """Pick one SIT_TO_STAND and one SITTING segment, plot raw accel
    magnitude, overlay the window boundaries that were assigned label=1
    or label=0, and save to Results/sanity_check.png.
    """
    # --- Pick the first sit-to-stand segment ---
    sts_rows = labels_df[labels_df["activity_id"] == SIT_TO_STAND_ID]
    sts_seg = sts_rows.iloc[0]
    sts_exp, sts_user = int(sts_seg["experiment_id"]), int(sts_seg["user_id"])
    sts_start = int(sts_seg["start_sample"]) - 1      # 0-indexed
    sts_end = int(sts_seg["end_sample"])               # exclusive

    # --- Pick the first sitting (activity 4) segment ---
    sit_rows = labels_df[labels_df["activity_id"] == 4]
    sit_seg = sit_rows.iloc[0]
    sit_exp, sit_user = int(sit_seg["experiment_id"]), int(sit_seg["user_id"])
    sit_start = int(sit_seg["start_sample"]) - 1
    sit_end = int(sit_seg["end_sample"])

    # --- Helper to compute accel magnitude for a signal ---
    def accel_mag(signal: np.ndarray) -> np.ndarray:
        return np.sqrt(np.sum(signal[:, :3] ** 2, axis=1))

    # --- Determine which global window indices belong to each experiment ---
    # We reconstruct the window start positions per experiment to map back.
    def get_window_global_offsets(
        signals: dict,
        target_exp: int,
        target_user: int,
    ) -> tuple[int, int]:
        """Return (first_global_idx, n_windows) for the given experiment."""
        offset = 0
        for (exp_id, user_id), signal in sorted(signals.items()):
            T = signal.shape[0]
            n_win = max(0, (T - WINDOW_SIZE) // STRIDE + 1)
            if exp_id == target_exp and user_id == target_user:
                return offset, n_win
            offset += n_win
        raise ValueError(f"Experiment {target_exp}, user {target_user} not found")

    # --- Create figure ---
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), constrained_layout=True)

    for ax, (exp, usr, seg_start, seg_end, activity_name, target_label) in zip(
        axes,
        [
            (sts_exp, sts_user, sts_start, sts_end, "SIT_TO_STAND", 1),
            (sit_exp, sit_user, sit_start, sit_end, "SITTING", 0),
        ],
    ):
        signal = signals[(exp, usr)]
        amag = accel_mag(signal)
        T = len(amag)

        # Context: show 256 samples before and after the segment.
        ctx = 256
        plot_lo = max(0, seg_start - ctx)
        plot_hi = min(T, seg_end + ctx)
        time_ax = np.arange(plot_lo, plot_hi) / SAMPLING_RATE_HZ

        ax.plot(
            time_ax,
            amag[plot_lo:plot_hi],
            color="black",
            linewidth=0.7,
            label="accel magnitude",
        )

        # Shade the ground-truth segment.
        ax.axvspan(
            seg_start / SAMPLING_RATE_HZ,
            seg_end / SAMPLING_RATE_HZ,
            color="orange" if target_label == 1 else "steelblue",
            alpha=0.25,
            label=f"GT: {activity_name}",
        )

        # Overlay window boundaries + their assigned labels.
        glob_offset, n_win = get_window_global_offsets(signals, exp, usr)
        n_pos_shown = 0
        n_neg_shown = 0
        for w in range(n_win):
            w_start = w * STRIDE
            w_end = w_start + WINDOW_SIZE
            # Only draw windows that overlap the plot range.
            if w_end <= plot_lo or w_start >= plot_hi:
                continue
            gidx = glob_offset + w
            lbl = y[gidx]
            color = "red" if lbl == 1 else "green"
            alpha = 0.15
            if lbl == 1:
                n_pos_shown += 1
            else:
                n_neg_shown += 1
            ax.axvspan(
                w_start / SAMPLING_RATE_HZ,
                w_end / SAMPLING_RATE_HZ,
                ymin=0.92 if lbl == 1 else 0.85,
                ymax=1.0 if lbl == 1 else 0.92,
                color=color,
                alpha=0.6,
            )

        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Acceleration magnitude (g)")
        ax.set_title(
            f"Exp {exp}, User {usr} — GT: {activity_name} "
            f"(samples {seg_start}–{seg_end})  |  "
            f"Windows shown: {n_pos_shown} pos(red), {n_neg_shown} neg(green)"
        )

        # Manual legend.
        handles = [
            mpatches.Patch(
                color="orange" if target_label == 1 else "steelblue",
                alpha=0.35,
                label=f"Ground truth: {activity_name}",
            ),
            mpatches.Patch(color="red", alpha=0.6, label="Window label = 1 (STS)"),
            mpatches.Patch(color="green", alpha=0.6, label="Window label = 0 (other)"),
            plt.Line2D([0], [0], color="black", lw=0.8, label="Accel magnitude"),
        ]
        ax.legend(handles=handles, loc="upper left", fontsize=8)

    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(_SANITY_PNG, dpi=150)
    plt.close(fig)
    print(f"Sanity-check plot saved to: {_SANITY_PNG}")


# ---------------------------------------------------------------------------
# Summary printing
# ---------------------------------------------------------------------------

def print_summary(df: pd.DataFrame) -> None:
    n_total = len(df)
    n_pos = int((df["label"] == 1).sum())
    n_neg = n_total - n_pos
    pct_pos = 100.0 * n_pos / n_total if n_total else 0.0

    print("=" * 60)
    print("Feature Extraction Summary")
    print("=" * 60)
    print(f"DataFrame shape       : {df.shape}  "
          f"({df.shape[1] - 2} features + label + subject_id)")
    print(f"Feature columns       : {len(FEATURE_COLUMNS)}")
    print()
    print(f"Label distribution:")
    print(f"  label=0 (other)     : {n_neg:,}  ({100 - pct_pos:.2f}%)")
    print(f"  label=1 (sit-to-stand): {n_pos:,}  ({pct_pos:.2f}%)")
    print()
    print(f"First 5 feature names : {FEATURE_COLUMNS[:5]}")
    print(f"Last  5 feature names : {FEATURE_COLUMNS[-5:]}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("Loading signals and labels …")
    signals = load_signals()
    labels_df = load_labels()

    print("Creating windows …")
    X, y, users, n_dropped = create_windows(signals, labels_df)

    print("Extracting features …")
    df = build_feature_dataframe(X, y, users)

    # Save CSV.
    _FEATURES_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(_FEATURES_CSV, index=False)
    print(f"Features saved to: {_FEATURES_CSV}\n")

    print_summary(df)

    # Sanity-check plot.
    print("\nGenerating sanity-check plot …")
    _sanity_check_plot(signals, labels_df, df, y, users)


if __name__ == "__main__":
    main()
