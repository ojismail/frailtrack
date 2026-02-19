"""
sisfall_loader.py — Load, convert, resample, window, and extract features
from SisFall elderly sit-to-stand trials for external validation.

Pipeline:
  1. Load D07/D08 trials for elderly subjects SE01–SE15 (9-col CSV)
  2. Extract ADXL345 accel (cols 0-2) + ITG3200 gyro (cols 3-5) → 6 channels
  3. Convert units: accel → g, gyro → rad/s
  4. Resample 200 Hz → 50 Hz  (scipy.signal.resample_poly, down=4)
  5. Window with 128-sample / 64-stride (same as UCI HAPT)
  6. Extract 48 features per window (same pipeline as UCI HAPT)
  7. Save to Data/sisfall/features.csv

Labeling strategy:
  All windows are labelled 0 ("other").  Event-level evaluation checks
  whether the model predicts ≥1 positive window per trial.

Usage:
    python -m src.sisfall_loader         # from project root
    python src/sisfall_loader.py         # also works
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import resample_poly

from src.features import FEATURE_COLUMNS, extract_features
from src.windowing import STRIDE, WINDOW_SIZE

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
_SISFALL_DIR = _PROJECT_ROOT / "Data" / "sisfall"
_FEATURES_CSV = _SISFALL_DIR / "features.csv"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ELDERLY_SUBJECTS = [f"SE{i:02d}" for i in range(1, 16)]   # SE01–SE15
STS_ACTIVITIES = ["D07", "D08"]                             # sit-to-stand
TRIALS_PER_ACTIVITY = 5                                     # R01–R05

SISFALL_SAMPLE_RATE = 200   # Hz
TARGET_SAMPLE_RATE = 50     # Hz (match UCI HAPT)
RESAMPLE_DOWN = SISFALL_SAMPLE_RATE // TARGET_SAMPLE_RATE   # 4

# Unit conversion factors
ADXL345_SCALE = 2 * 16 / (2 ** 13)       # ±16g / 13-bit → 0.00391 g/bit
ITG3200_SCALE_DEG = 2 * 2000 / (2 ** 16) # ±2000°/s / 16-bit → 0.0611 °/s/bit
DEG_TO_RAD = np.pi / 180.0
ITG3200_SCALE_RAD = ITG3200_SCALE_DEG * DEG_TO_RAD  # → rad/s/bit


# ---------------------------------------------------------------------------
# File parsing
# ---------------------------------------------------------------------------

def _parse_sisfall_file(filepath: Path) -> np.ndarray:
    """Read a SisFall CSV file (9 integer columns, trailing semicolons).

    Returns
    -------
    data : np.ndarray, shape (T, 9) — raw integer values.
    """
    rows: list[list[int]] = []
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip().rstrip(";").strip()
            if not line:
                continue
            vals = [int(v.strip()) for v in line.split(",")]
            rows.append(vals)
    return np.array(rows, dtype=np.int32)


def _parse_filename(name: str) -> tuple[str, str, int] | None:
    """Parse 'D07_SE01_R01.txt' → ('D07', 'SE01', 1).  None if no match."""
    m = re.match(r"(D\d{2})_(SE\d{2})_R(\d{2})\.txt", name)
    if m is None:
        return None
    return m.group(1), m.group(2), int(m.group(3))


# ---------------------------------------------------------------------------
# Load all elderly STS trials
# ---------------------------------------------------------------------------

def load_sisfall_sts(
    sisfall_dir: Path = _SISFALL_DIR,
) -> list[dict]:
    """Load and convert all elderly sit-to-stand trials.

    Returns a list of dicts, each with:
        subject, activity, trial, signal (T_resampled, 6)
    """
    trials: list[dict] = []
    missing: list[str] = []

    for subj in ELDERLY_SUBJECTS:
        subj_dir = sisfall_dir / subj
        if not subj_dir.is_dir():
            print(f"[WARNING] Subject directory not found: {subj_dir}")
            continue

        for act in STS_ACTIVITIES:
            for trial_num in range(1, TRIALS_PER_ACTIVITY + 1):
                fname = f"{act}_{subj}_R{trial_num:02d}.txt"
                fpath = subj_dir / fname
                if not fpath.exists():
                    missing.append(fname)
                    continue

                raw = _parse_sisfall_file(fpath)          # (T, 9) int

                # Extract ADXL345 accel (cols 0-2) + ITG3200 gyro (cols 3-5).
                accel_raw = raw[:, 0:3].astype(np.float64)
                gyro_raw = raw[:, 3:6].astype(np.float64)

                # Unit conversion.
                accel_g = accel_raw * ADXL345_SCALE       # → g
                gyro_rads = gyro_raw * ITG3200_SCALE_RAD  # → rad/s

                signal_200 = np.hstack([accel_g, gyro_rads])  # (T, 6) @ 200 Hz

                # Resample each channel 200 Hz → 50 Hz.
                n_channels = signal_200.shape[1]
                resampled_channels = []
                for ch in range(n_channels):
                    resampled_channels.append(
                        resample_poly(signal_200[:, ch], up=1, down=RESAMPLE_DOWN)
                    )
                signal_50 = np.column_stack(resampled_channels)  # (T', 6) @ 50 Hz

                trials.append({
                    "subject": subj,
                    "activity": act,
                    "trial": trial_num,
                    "signal": signal_50,
                })

    if missing:
        print(f"[WARNING] {len(missing)} missing files: {missing[:10]}"
              + ("…" if len(missing) > 10 else ""))

    return trials


# ---------------------------------------------------------------------------
# Windowing + feature extraction
# ---------------------------------------------------------------------------

def build_features(
    trials: list[dict],
    window_size: int = WINDOW_SIZE,
    stride: int = STRIDE,
) -> pd.DataFrame:
    """Window every trial and extract 48 features per window.

    All windows are labelled 0 ("other").  Event detection happens
    downstream by checking per-trial positive-window counts.
    """
    all_windows: list[np.ndarray] = []
    meta_rows: list[dict] = []

    for trial in trials:
        signal = trial["signal"]                  # (T, 6)
        T = signal.shape[0]

        start = 0
        while start + window_size <= T:
            win = signal[start : start + window_size]
            all_windows.append(win)
            meta_rows.append({
                "subject_id": trial["subject"],
                "activity": trial["activity"],
                "trial_id": trial["trial"],
            })
            start += stride

    if not all_windows:
        return pd.DataFrame()

    X = np.stack(all_windows, axis=0)            # (N, 128, 6)
    feats = extract_features(X)                  # (N, 48)

    df = pd.DataFrame(feats, columns=FEATURE_COLUMNS)
    df["label"] = 0                              # all windows labelled "other"
    df["subject_id"] = [m["subject_id"] for m in meta_rows]
    df["activity"] = [m["activity"] for m in meta_rows]
    df["trial_id"] = [m["trial_id"] for m in meta_rows]

    return df


# ---------------------------------------------------------------------------
# Summary + sanity check
# ---------------------------------------------------------------------------

def print_summary(trials: list[dict], df: pd.DataFrame) -> None:
    n_subjects = len(set(t["subject"] for t in trials))
    n_d07 = sum(1 for t in trials if t["activity"] == "D07")
    n_d08 = sum(1 for t in trials if t["activity"] == "D08")

    print("=" * 65)
    print("SisFall Elderly Sit-to-Stand — Summary")
    print("=" * 65)
    print(f"Elderly subjects loaded  : {n_subjects}")
    print(f"D07 (slow STS) trials    : {n_d07}")
    print(f"D08 (fast STS) trials    : {n_d08}")
    print(f"Total trials             : {len(trials)}")

    # Signal lengths after resampling.
    lengths = [t["signal"].shape[0] for t in trials]
    print(f"Samples per trial (50Hz) : "
          f"mean={np.mean(lengths):.0f}, "
          f"min={min(lengths)}, max={max(lengths)}")

    print(f"\nTotal windows            : {len(df):,}")
    print(f"Feature columns          : {len(FEATURE_COLUMNS)}")
    print(f"DataFrame shape          : {df.shape}")

    # Per-trial window counts.
    per_trial = (
        df.groupby(["subject_id", "activity", "trial_id"])
        .size()
        .reset_index(name="n_windows")
    )
    print(f"Windows per trial        : "
          f"mean={per_trial.n_windows.mean():.1f}, "
          f"min={per_trial.n_windows.min()}, "
          f"max={per_trial.n_windows.max()}")

    # Sanity check: compare feature scales to UCI HAPT.
    print(f"\n{'— Feature Scale Sanity Check (SisFall vs UCI HAPT) —':^65}")
    print(f"{'Feature':<22} {'SisFall mean':>14} {'SisFall std':>14}")
    print("-" * 52)
    check_feats = [
        "ax_mean", "ax_std", "accel_mag_mean", "accel_mag_max",
        "gx_mean", "gx_std", "gyro_mag_mean",
    ]
    for feat in check_feats:
        if feat in df.columns:
            print(f"{feat:<22} {df[feat].mean():>14.4f} {df[feat].std():>14.4f}")

    print("\nExpected UCI HAPT ranges for reference:")
    print("  accel_mag_mean ≈ 1.03,  accel_mag_max ≈ 1.42")
    print("  gx_mean ≈ 0.02,  gx_std ≈ 0.35,  gyro_mag_mean ≈ 0.51")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("Loading SisFall elderly sit-to-stand trials …\n")
    trials = load_sisfall_sts()

    print(f"\nExtracting features (window={WINDOW_SIZE}, stride={STRIDE}) …")
    df = build_features(trials)

    _SISFALL_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(_FEATURES_CSV, index=False)
    print(f"Features saved to {_FEATURES_CSV}\n")

    print_summary(trials, df)
    print("\nDone.")


if __name__ == "__main__":
    main()
