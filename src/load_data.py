"""
load_data.py — Load and combine UCI HAPT raw sensor data.

Reads accelerometer + gyroscope files from Data/uci_hapt/RawData/,
parses the ground-truth labels, and returns a dictionary of 6-channel
signals keyed by (experiment_id, user_id) plus a labels DataFrame.

Usage:
    python -m src.load_data          # from project root
    python src/load_data.py          # also works
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths — resolve relative to this file so the script works from any cwd.
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
_RAW_DIR = _PROJECT_ROOT / "Data" / "uci_hapt" / "RawData"

ACTIVITY_LABELS = {
    1: "WALKING",
    2: "WALKING_UPSTAIRS",
    3: "WALKING_DOWNSTAIRS",
    4: "SITTING",
    5: "STANDING",
    6: "LAYING",
    7: "STAND_TO_SIT",
    8: "SIT_TO_STAND",
    9: "SIT_TO_LIE",
    10: "LIE_TO_SIT",
    11: "STAND_TO_LIE",
    12: "LIE_TO_STAND",
}

SIT_TO_STAND_ID = 8
SAMPLING_RATE_HZ = 50


# ---------------------------------------------------------------------------
# Core loaders
# ---------------------------------------------------------------------------

def load_labels(raw_dir: Path = _RAW_DIR) -> pd.DataFrame:
    """Read labels.txt → DataFrame with columns:
    experiment_id, user_id, activity_id, start_sample, end_sample
    """
    labels_path = raw_dir / "labels.txt"
    df = pd.read_csv(
        labels_path,
        sep=r"\s+",
        header=None,
        names=["experiment_id", "user_id", "activity_id",
               "start_sample", "end_sample"],
        dtype=int,
    )
    return df


def _parse_exp_user(filename: str) -> tuple[int, int] | None:
    """Extract (experiment_id, user_id) from a filename like
    acc_exp01_user01.txt or gyro_exp12_user06.txt.
    Returns None if the name doesn't match.
    """
    m = re.match(r"(?:acc|gyro)_exp(\d+)_user(\d+)\.txt", filename)
    if m is None:
        return None
    return int(m.group(1)), int(m.group(2))


def load_signals(
    raw_dir: Path = _RAW_DIR,
) -> dict[tuple[int, int], np.ndarray]:
    """Load all acc/gyro file pairs and combine into 6-channel arrays.

    Returns
    -------
    signals : dict[(exp_id, user_id)] → np.ndarray of shape (T, 6)
        Columns are [ax, ay, az, gx, gy, gz].
        Accelerometer units: g.  Gyroscope units: rad/s.
    """
    # Discover all accelerometer files and their (exp, user) keys.
    acc_files: dict[tuple[int, int], Path] = {}
    gyro_files: dict[tuple[int, int], Path] = {}

    for f in sorted(raw_dir.iterdir()):
        if not f.is_file():
            continue
        key = _parse_exp_user(f.name)
        if key is None:
            continue
        if f.name.startswith("acc_"):
            acc_files[key] = f
        elif f.name.startswith("gyro_"):
            gyro_files[key] = f

    # Sanity: every acc file should have a matching gyro file and vice versa.
    acc_only = set(acc_files) - set(gyro_files)
    gyro_only = set(gyro_files) - set(acc_files)
    if acc_only:
        print(f"[WARNING] Acc files with no matching gyro: {sorted(acc_only)}")
    if gyro_only:
        print(f"[WARNING] Gyro files with no matching acc: {sorted(gyro_only)}")

    # Load and combine.
    signals: dict[tuple[int, int], np.ndarray] = {}
    common_keys = sorted(set(acc_files) & set(gyro_files))

    for key in common_keys:
        acc = np.loadtxt(acc_files[key])   # (T, 3)
        gyro = np.loadtxt(gyro_files[key]) # (T, 3)

        if acc.shape[0] != gyro.shape[0]:
            print(
                f"[WARNING] Row mismatch for exp{key[0]:02d}_user{key[1]:02d}: "
                f"acc has {acc.shape[0]} rows, gyro has {gyro.shape[0]} rows"
            )

        # Truncate to the shorter length so concatenation still works.
        min_len = min(acc.shape[0], gyro.shape[0])
        combined = np.hstack([acc[:min_len], gyro[:min_len]])  # (T, 6)
        signals[key] = combined

    return signals


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------

def print_summary(
    signals: dict[tuple[int, int], np.ndarray],
    labels: pd.DataFrame,
) -> None:
    """Print human-readable summary of the loaded dataset."""

    # --- Experiments & users ---
    n_experiments = labels["experiment_id"].nunique()
    n_users = labels["user_id"].nunique()
    total_samples = sum(arr.shape[0] for arr in signals.values())

    print("=" * 60)
    print("UCI HAPT Dataset — Summary")
    print("=" * 60)
    print(f"Experiment-user pairs loaded : {len(signals)}")
    print(f"Unique experiments           : {n_experiments}")
    print(f"Unique users (subjects)      : {n_users}")
    print(f"Total samples across all files: {total_samples:,}  "
          f"({total_samples / SAMPLING_RATE_HZ:,.1f} s)")
    print(f"Channels per sample          : 6  (ax, ay, az, gx, gy, gz)")

    # --- Activity counts ---
    print(f"\n{'Activity ID':<14} {'Name':<22} {'Segments':>8}")
    print("-" * 48)
    activity_counts = labels["activity_id"].value_counts().sort_index()
    for aid, count in activity_counts.items():
        name = ACTIVITY_LABELS.get(aid, "UNKNOWN")
        marker = "  ◀ target" if aid == SIT_TO_STAND_ID else ""
        print(f"{aid:<14} {name:<22} {count:>8}{marker}")

    # --- Sit-to-stand detail ---
    sts = labels[labels["activity_id"] == SIT_TO_STAND_ID].copy()
    sts["duration_samples"] = sts["end_sample"] - sts["start_sample"] + 1
    sts["duration_sec"] = sts["duration_samples"] / SAMPLING_RATE_HZ

    print(f"\n{'— Sit-to-Stand (ID 8) Detail —':^60}")
    print(f"Total segments      : {len(sts)}")
    print(f"Duration (samples)  : "
          f"mean={sts['duration_samples'].mean():.1f}  "
          f"std={sts['duration_samples'].std():.1f}  "
          f"min={sts['duration_samples'].min()}  "
          f"max={sts['duration_samples'].max()}")
    print(f"Duration (seconds)  : "
          f"mean={sts['duration_sec'].mean():.2f}  "
          f"std={sts['duration_sec'].std():.2f}  "
          f"min={sts['duration_sec'].min():.2f}  "
          f"max={sts['duration_sec'].max():.2f}")

    # --- Per-subject breakdown ---
    per_user = (
        sts.groupby("user_id")["duration_samples"]
        .agg(["count", "mean"])
        .rename(columns={"count": "segments", "mean": "avg_dur_samples"})
    )
    per_user["avg_dur_sec"] = per_user["avg_dur_samples"] / SAMPLING_RATE_HZ

    print(f"\n{'User':<8} {'Segments':>8} {'Avg Duration (samples)':>24} {'Avg Duration (s)':>18}")
    print("-" * 62)
    for uid, row in per_user.iterrows():
        print(f"{uid:<8} {int(row['segments']):>8} {row['avg_dur_samples']:>24.1f} {row['avg_dur_sec']:>18.2f}")

    print("-" * 62)
    print(f"{'TOTAL':<8} {int(per_user['segments'].sum()):>8}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print(f"Loading data from: {_RAW_DIR}\n")
    labels = load_labels()
    signals = load_signals()
    print_summary(signals, labels)


if __name__ == "__main__":
    main()
