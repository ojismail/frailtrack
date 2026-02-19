"""
quality_assessment.py — Extract per-rep clinical features from sit-to-stand events.

Uses ground-truth segment boundaries (labels.txt, activity_id == 8) so that
Part 2 (quality assessment) is evaluated independently of Part 1's detection
accuracy.

Pipeline:
  1. Load ground-truth STS segments + raw 6-channel signals (src/load_data)
  2. Gravity removal via 4th-order high-pass Butterworth (0.3 Hz cutoff)
  3. Per-rep features: peak dynamic accel, time-per-rep, peak gyro, power
  4. Per-subject summaries: CV and fatigue slope across reps
  5. Sanity checks on expected value ranges
  6. SisFall elderly comparison (D07/D08)

IMPORTANT LIMITATION:
  UCI HAPT contains only 2–3 sit-to-stand transitions per subject (it is NOT
  a 30-second chair stand test). Consequently, coefficient-of-variation (CV)
  and fatigue-slope metrics are computed to demonstrate the pipeline, but are
  NOT clinically meaningful with so few repetitions.

Outputs:
    Results/per_rep_features.csv
    Results/subject_quality_summary.csv
    Results/sisfall_per_rep_features.csv
    stdout — tables and sanity checks

Usage:
    python -m src.quality_assessment    # from project root
    python src/quality_assessment.py    # also works
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from scipy.stats import linregress

from src.load_data import (
    SAMPLING_RATE_HZ,
    SIT_TO_STAND_ID,
    load_labels,
    load_signals,
)
from src.sisfall_loader import (
    ADXL345_SCALE,
    ITG3200_SCALE_RAD,
    ELDERLY_SUBJECTS,
    STS_ACTIVITIES,
    TRIALS_PER_ACTIVITY,
    SISFALL_SAMPLE_RATE,
    TARGET_SAMPLE_RATE,
    RESAMPLE_DOWN,
    _parse_sisfall_file,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
_RESULTS_DIR = _PROJECT_ROOT / "Results"
_PER_REP_CSV = _RESULTS_DIR / "per_rep_features.csv"
_SUBJECT_SUMMARY_CSV = _RESULTS_DIR / "subject_quality_summary.csv"
_SISFALL_PER_REP_CSV = _RESULTS_DIR / "sisfall_per_rep_features.csv"
_SISFALL_DIR = _PROJECT_ROOT / "Data" / "sisfall"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
G_TO_MS2 = 9.81            # 1 g in m/s²
ASSUMED_HEIGHT_M = 1.70     # assumed subject height
CHAIR_HEIGHT_M = 0.46       # standard chair seat height

# High-pass filter parameters for gravity removal
HP_ORDER = 4
HP_CUTOFF_HZ = 0.3         # Hz

# Expected ranges for sanity checks
EXPECTED_RANGES = {
    "peak_dynamic_accel_ms2": (1.0, 15.0),
    "time_per_rep_s":         (1.0, 5.0),
    "peak_gyro_rad_s":        (0.5, 10.0),
    "power_w_kg":             (0.5, 5.0),
}


# ---------------------------------------------------------------------------
# Step 2: Gravity removal
# ---------------------------------------------------------------------------

def _highpass_butter(
    data: np.ndarray,
    cutoff: float = HP_CUTOFF_HZ,
    fs: float = SAMPLING_RATE_HZ,
    order: int = HP_ORDER,
) -> np.ndarray:
    """Apply a zero-phase high-pass Butterworth filter to *data* (1-D array)."""
    nyq = 0.5 * fs
    normalized_cutoff = cutoff / nyq
    b, a = butter(order, normalized_cutoff, btype="high")
    return filtfilt(b, a, data)


def gravity_removal(
    signal: np.ndarray,
    fs: float = SAMPLING_RATE_HZ,
) -> tuple[np.ndarray, np.ndarray]:
    """Remove gravity from accelerometer channels and compute magnitudes.

    Parameters
    ----------
    signal : ndarray, shape (T, 6)
        Columns: [ax, ay, az, gx, gy, gz].
        Accelerometer in g's, gyroscope in rad/s.

    Returns
    -------
    dynamic_accel_mag : ndarray, shape (T,)
        sqrt(filtered_ax² + filtered_ay² + filtered_az²)  — in g's.
    gyro_mag : ndarray, shape (T,)
        sqrt(gx² + gy² + gz²)  — in rad/s.
    """
    # High-pass each accel axis separately.
    filtered_ax = _highpass_butter(signal[:, 0], fs=fs)
    filtered_ay = _highpass_butter(signal[:, 1], fs=fs)
    filtered_az = _highpass_butter(signal[:, 2], fs=fs)

    dynamic_accel_mag = np.sqrt(
        filtered_ax ** 2 + filtered_ay ** 2 + filtered_az ** 2
    )

    # Gyroscope does NOT need gravity removal.
    gyro_mag = np.sqrt(
        signal[:, 3] ** 2 + signal[:, 4] ** 2 + signal[:, 5] ** 2
    )

    return dynamic_accel_mag, gyro_mag


# ---------------------------------------------------------------------------
# Step 3: Per-rep feature extraction
# ---------------------------------------------------------------------------

def _compute_power(time_per_rep_s: float) -> float:
    """Relative muscle power (W/kg) using adapted Alcázar equation.

    Power = [0.9 × g × (height × 0.5 − chair_height)] / (time × 0.5)
    """
    displacement = ASSUMED_HEIGHT_M * 0.5 - CHAIR_HEIGHT_M
    if time_per_rep_s <= 0:
        return np.nan
    return (0.9 * G_TO_MS2 * displacement) / (time_per_rep_s * 0.5)


def extract_rep_features(
    dynamic_accel_mag: np.ndarray,
    gyro_mag: np.ndarray,
    start_sample: int,
    end_sample: int,
    fs: float = SAMPLING_RATE_HZ,
) -> dict:
    """Extract clinical features for one sit-to-stand segment.

    Parameters
    ----------
    dynamic_accel_mag, gyro_mag : 1-D arrays covering the full experiment.
    start_sample, end_sample : 1-indexed inclusive boundaries from labels.txt.
    """
    # Convert 1-indexed inclusive to 0-indexed slice.
    s = start_sample - 1
    e = end_sample           # exclusive upper bound

    seg_accel = dynamic_accel_mag[s:e]
    seg_gyro = gyro_mag[s:e]
    n_samples = e - s

    peak_accel_g = float(np.max(seg_accel))
    peak_accel_ms2 = peak_accel_g * G_TO_MS2

    time_per_rep = n_samples / fs

    peak_gyro = float(np.max(seg_gyro))

    power = _compute_power(time_per_rep)

    return {
        "peak_dynamic_accel_ms2": peak_accel_ms2,
        "time_per_rep_s":         time_per_rep,
        "peak_gyro_rad_s":        peak_gyro,
        "power_w_kg":             power,
    }


# ---------------------------------------------------------------------------
# Step 4: Per-subject summaries
# ---------------------------------------------------------------------------

def compute_subject_summary(rep_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-rep features into per-subject summary statistics.

    Input DataFrame must have columns: subject_id, peak_dynamic_accel_ms2,
    time_per_rep_s, peak_gyro_rad_s, power_w_kg.
    """
    rows: list[dict] = []

    for subj, grp in rep_df.groupby("subject_id"):
        n_reps = len(grp)

        mean_accel = grp["peak_dynamic_accel_ms2"].mean()
        mean_time = grp["time_per_rep_s"].mean()
        mean_gyro = grp["peak_gyro_rad_s"].mean()
        mean_power = grp["power_w_kg"].mean()

        # CV = std / mean (undefined for n=1, but compute anyway for pipeline demo)
        cv_accel = grp["peak_dynamic_accel_ms2"].std() / mean_accel if mean_accel > 0 and n_reps > 1 else np.nan
        cv_time = grp["time_per_rep_s"].std() / mean_time if mean_time > 0 and n_reps > 1 else np.nan
        cv_gyro = grp["peak_gyro_rad_s"].std() / mean_gyro if mean_gyro > 0 and n_reps > 1 else np.nan

        # Fatigue slope: linregress(rep_number, peak_accel).
        if n_reps >= 2:
            slope, _, _, _, _ = linregress(
                grp["rep_number"].values.astype(float),
                grp["peak_dynamic_accel_ms2"].values,
            )
        else:
            slope = np.nan

        rows.append({
            "subject_id":           subj,
            "n_reps":               n_reps,
            "mean_peak_accel":      mean_accel,
            "mean_time_per_rep":    mean_time,
            "mean_peak_gyro":       mean_gyro,
            "mean_power":           mean_power,
            "cv_accel":             cv_accel,
            "cv_time":              cv_time,
            "cv_gyro":              cv_gyro,
            "fatigue_slope_accel":  slope,
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Step 1 + 2 + 3: UCI HAPT pipeline
# ---------------------------------------------------------------------------

def run_uci_hapt() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Process all UCI HAPT sit-to-stand segments.

    Returns (per_rep_df, subject_summary_df).
    """
    print("Loading UCI HAPT signals and labels …")
    signals = load_signals()
    labels = load_labels()

    sts = labels[labels["activity_id"] == SIT_TO_STAND_ID].copy()
    print(f"  {len(sts)} ground-truth sit-to-stand segments\n")

    # Pre-compute gravity-removed magnitudes per experiment.
    print("Applying gravity removal (high-pass Butterworth, "
          f"{HP_ORDER}th order, {HP_CUTOFF_HZ} Hz cutoff) …")
    processed: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
    for key, sig in signals.items():
        processed[key] = gravity_removal(sig)

    # Extract per-rep features.
    rep_rows: list[dict] = []
    # Track rep number per subject (across experiments).
    subject_rep_counter: dict[int, int] = {}

    for _, row in sts.iterrows():
        exp_id = int(row["experiment_id"])
        user_id = int(row["user_id"])
        key = (exp_id, user_id)

        if key not in processed:
            print(f"  [WARNING] No signal for exp={exp_id}, user={user_id}")
            continue

        dyn_accel, gyro_m = processed[key]

        subject_rep_counter.setdefault(user_id, 0)
        subject_rep_counter[user_id] += 1
        rep_num = subject_rep_counter[user_id]

        feats = extract_rep_features(
            dyn_accel, gyro_m,
            int(row["start_sample"]),
            int(row["end_sample"]),
        )
        feats.update({
            "subject_id": user_id,
            "exp_id":     exp_id,
            "rep_number": rep_num,
        })
        rep_rows.append(feats)

    rep_df = pd.DataFrame(rep_rows)

    # Reorder columns.
    col_order = [
        "subject_id", "exp_id", "rep_number",
        "peak_dynamic_accel_ms2", "time_per_rep_s",
        "peak_gyro_rad_s", "power_w_kg",
    ]
    rep_df = rep_df[col_order]

    summary_df = compute_subject_summary(rep_df)

    return rep_df, summary_df


# ---------------------------------------------------------------------------
# Step 5: Sanity checks
# ---------------------------------------------------------------------------

def print_sanity_checks(rep_df: pd.DataFrame, summary_df: pd.DataFrame) -> None:
    """Validate per-rep features against expected clinical ranges."""
    print(f"\n{'=' * 80}")
    print("SANITY CHECKS")
    print("=" * 80)

    feature_cols = [
        "peak_dynamic_accel_ms2", "time_per_rep_s",
        "peak_gyro_rad_s", "power_w_kg",
    ]
    print(f"\n  {'Feature':<28} {'Min':>8} {'Max':>8} {'Mean':>8} "
          f"{'Expected Range':>18} {'OK?':>5}")
    print(f"  {'-' * 80}")

    n_flagged = 0
    for feat in feature_cols:
        vals = rep_df[feat]
        lo, hi = EXPECTED_RANGES[feat]
        fmin = vals.min()
        fmax = vals.max()
        fmean = vals.mean()
        # Flag if any value is outside expected range.
        outside = ((vals < lo) | (vals > hi)).sum()
        ok = "✓" if outside == 0 else f"✗ ({outside})"
        if outside > 0:
            n_flagged += outside
        print(f"  {feat:<28} {fmin:>8.3f} {fmax:>8.3f} {fmean:>8.3f} "
              f"  [{lo:.1f} – {hi:.1f}]{'':<4} {ok:>5}")

    if n_flagged > 0:
        print(f"\n  ⚠ {n_flagged} values outside expected ranges "
              f"(see per-rep CSV for details)")
    else:
        print(f"\n  ✓ All {len(rep_df)} rep values within expected ranges")

    # Correlation: time-per-rep vs peak acceleration.
    print(f"\n  Correlation: mean_time_per_rep vs mean_peak_accel across subjects")
    valid = summary_df.dropna(subset=["mean_time_per_rep", "mean_peak_accel"])
    if len(valid) >= 3:
        r, p = np.corrcoef(
            valid["mean_time_per_rep"], valid["mean_peak_accel"]
        )[0, 1], None
        from scipy.stats import pearsonr
        r, p = pearsonr(
            valid["mean_time_per_rep"].values,
            valid["mean_peak_accel"].values,
        )
        print(f"  Pearson r = {r:.3f},  p = {p:.4f}")
        if r < 0:
            print("  ✓ Negative correlation (faster reps → higher peak accel) "
                  "— expected pattern")
        else:
            print("  ⚠ Positive or zero correlation — unexpected; may reflect "
                  "limited sample or individual differences")
    else:
        print("  (Too few subjects to compute correlation)")


# ---------------------------------------------------------------------------
# Step 6: SisFall elderly
# ---------------------------------------------------------------------------

def run_sisfall() -> pd.DataFrame:
    """Extract per-rep features from SisFall elderly D07/D08 trials.

    Each trial file is treated as one sit-to-stand rep — the entire
    (resampled) signal is the rep boundary.

    LIMITATION: SisFall trials are ~12 s long but the actual transition is
    ~2–3 s.  Peak accel/gyro (max) are unaffected, but time-per-rep and
    power will be inflated because we use the full trial duration.
    """
    from scipy.signal import resample_poly

    print("\n" + "=" * 80)
    print("SisFall Elderly — Per-Rep Feature Extraction")
    print("=" * 80)
    print("  LIMITATION: Using full trial (~12 s) as rep boundary.")
    print("  Peak accel/gyro are valid; time-per-rep and power are inflated.\n")

    rep_rows: list[dict] = []
    missing: list[str] = []
    subject_rep_counter: dict[str, int] = {}

    for subj in ELDERLY_SUBJECTS:
        subj_dir = _SISFALL_DIR / subj
        if not subj_dir.is_dir():
            continue

        for act in STS_ACTIVITIES:
            for trial_num in range(1, TRIALS_PER_ACTIVITY + 1):
                fname = f"{act}_{subj}_R{trial_num:02d}.txt"
                fpath = subj_dir / fname
                if not fpath.exists():
                    missing.append(fname)
                    continue

                raw = _parse_sisfall_file(fpath)  # (T, 9) int

                # Extract ADXL345 accel + ITG3200 gyro → 6 channels.
                accel_g = raw[:, 0:3].astype(np.float64) * ADXL345_SCALE
                gyro_rads = raw[:, 3:6].astype(np.float64) * ITG3200_SCALE_RAD

                signal_200 = np.hstack([accel_g, gyro_rads])  # (T, 6) @ 200 Hz

                # Resample 200 Hz → 50 Hz.
                resampled = []
                for ch in range(signal_200.shape[1]):
                    resampled.append(
                        resample_poly(signal_200[:, ch], up=1, down=RESAMPLE_DOWN)
                    )
                signal_50 = np.column_stack(resampled)  # (T', 6) @ 50 Hz

                # Gravity removal at 50 Hz.
                dyn_accel, gyro_m = gravity_removal(signal_50, fs=TARGET_SAMPLE_RATE)

                # Use entire trial as the rep.
                T = signal_50.shape[0]
                peak_accel_g = float(np.max(dyn_accel))
                peak_accel_ms2 = peak_accel_g * G_TO_MS2
                time_per_rep = T / TARGET_SAMPLE_RATE
                peak_gyro = float(np.max(gyro_m))
                power = _compute_power(time_per_rep)

                subject_rep_counter.setdefault(subj, 0)
                subject_rep_counter[subj] += 1

                rep_rows.append({
                    "subject_id":               subj,
                    "activity":                 act,
                    "trial":                    trial_num,
                    "rep_number":               subject_rep_counter[subj],
                    "peak_dynamic_accel_ms2":   peak_accel_ms2,
                    "time_per_rep_s":           time_per_rep,
                    "peak_gyro_rad_s":          peak_gyro,
                    "power_w_kg":               power,
                })

    if missing:
        print(f"  [WARNING] {len(missing)} missing files: {missing[:5]}"
              + ("…" if len(missing) > 5 else ""))

    sis_df = pd.DataFrame(rep_rows)
    return sis_df


def print_comparison(uci_rep: pd.DataFrame, sis_rep: pd.DataFrame) -> None:
    """Print side-by-side comparison of UCI HAPT (young) vs SisFall (elderly)."""
    print(f"\n{'=' * 80}")
    print("UCI HAPT (young adults, 19–48 y) vs SisFall (elderly, 60+ y)")
    print("=" * 80)

    features = [
        ("peak_dynamic_accel_ms2", "Peak dynamic accel (m/s²)"),
        ("time_per_rep_s",         "Time per rep (s)"),
        ("peak_gyro_rad_s",        "Peak gyro magnitude (rad/s)"),
        ("power_w_kg",             "Relative power (W/kg)"),
    ]

    print(f"\n  {'Feature':<30} {'UCI HAPT':>12} {'SisFall':>12} {'Δ':>10} "
          f"{'Notes':>22}")
    print(f"  {'-' * 90}")

    for col, label in features:
        uci_mean = uci_rep[col].mean()
        # For SisFall time/power, note limitation.
        sis_mean = sis_rep[col].mean()
        delta = sis_mean - uci_mean

        note = ""
        if col == "time_per_rep_s":
            note = "(inflated — full trial)"
        elif col == "power_w_kg":
            note = "(deflated — full trial)"
        elif col in ("peak_dynamic_accel_ms2", "peak_gyro_rad_s"):
            if delta < 0:
                note = "✓ expected (elderly lower)"
            else:
                note = "⚠ unexpected"

        print(f"  {label:<30} {uci_mean:>12.3f} {sis_mean:>12.3f} "
              f"{delta:>+10.3f} {note:>22}")

    # Also show D07 (slow) vs D08 (fast) within SisFall.
    print(f"\n  SisFall: D07 (slow) vs D08 (fast)")
    print(f"  {'Feature':<30} {'D07':>12} {'D08':>12} {'Δ(D08-D07)':>12}")
    print(f"  {'-' * 70}")

    for col, label in features:
        d07 = sis_rep[sis_rep["activity"] == "D07"][col].mean()
        d08 = sis_rep[sis_rep["activity"] == "D08"][col].mean()
        delta = d08 - d07
        print(f"  {label:<30} {d07:>12.3f} {d08:>12.3f} {delta:>+12.3f}")


# ---------------------------------------------------------------------------
# Print helpers
# ---------------------------------------------------------------------------

def _print_per_rep_table(df: pd.DataFrame, subjects: list, title: str) -> None:
    """Print per-rep features for selected subjects."""
    print(f"\n{'=' * 80}")
    print(title)
    print("=" * 80)
    print(f"  {'Subject':>8} {'Exp':>5} {'Rep':>4} "
          f"{'Peak Accel':>12} {'Time (s)':>10} {'Peak Gyro':>11} {'Power':>10}")
    print(f"  {'':>8} {'':>5} {'#':>4} "
          f"{'(m/s²)':>12} {'':>10} {'(rad/s)':>11} {'(W/kg)':>10}")
    print(f"  {'-' * 66}")

    sub = df[df["subject_id"].isin(subjects)]
    for _, row in sub.iterrows():
        exp_col = "exp_id" if "exp_id" in row.index else "activity"
        exp_val = row.get("exp_id", row.get("activity", ""))
        print(f"  {row['subject_id']:>8} {exp_val:>5} {int(row['rep_number']):>4} "
              f"{row['peak_dynamic_accel_ms2']:>12.3f} "
              f"{row['time_per_rep_s']:>10.3f} "
              f"{row['peak_gyro_rad_s']:>11.3f} "
              f"{row['power_w_kg']:>10.3f}")


def _print_subject_summary(summary_df: pd.DataFrame) -> None:
    """Print full per-subject summary table."""
    print(f"\n{'=' * 110}")
    print("Per-Subject Quality Summary (all 30 subjects)")
    print("=" * 110)
    print("  NOTE: With only 2–3 reps per subject, CV and fatigue slope are NOT "
          "clinically meaningful.\n")

    print(f"  {'Subj':>5} {'Reps':>5} {'Accel':>8} {'Time':>8} {'Gyro':>8} "
          f"{'Power':>8}  |  {'CV_a':>7} {'CV_t':>7} {'CV_g':>7} "
          f"{'Slope_a':>9}")
    print(f"  {'':>5} {'':>5} {'(m/s²)':>8} {'(s)':>8} {'(rad/s)':>8} "
          f"{'(W/kg)':>8}  |  {'':>7} {'':>7} {'':>7} {'':>9}")
    print(f"  {'-' * 100}")

    for _, r in summary_df.iterrows():
        cv_a = f"{r['cv_accel']:.3f}" if not np.isnan(r["cv_accel"]) else "  n/a"
        cv_t = f"{r['cv_time']:.3f}" if not np.isnan(r["cv_time"]) else "  n/a"
        cv_g = f"{r['cv_gyro']:.3f}" if not np.isnan(r["cv_gyro"]) else "  n/a"
        sl = f"{r['fatigue_slope_accel']:>+9.3f}" if not np.isnan(r["fatigue_slope_accel"]) else "      n/a"

        print(f"  {int(r['subject_id']):>5} {int(r['n_reps']):>5} "
              f"{r['mean_peak_accel']:>8.3f} {r['mean_time_per_rep']:>8.3f} "
              f"{r['mean_peak_gyro']:>8.3f} {r['mean_power']:>8.3f}  |  "
              f"{cv_a:>7} {cv_t:>7} {cv_g:>7} {sl}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── UCI HAPT ──────────────────────────────────────────────────────────
    rep_df, summary_df = run_uci_hapt()

    rep_df.to_csv(_PER_REP_CSV, index=False)
    print(f"  Saved per-rep features to {_PER_REP_CSV}")

    summary_df.to_csv(_SUBJECT_SUMMARY_CSV, index=False)
    print(f"  Saved subject summary to {_SUBJECT_SUMMARY_CSV}")

    # Print per-rep table for first 3 subjects.
    first_3 = sorted(rep_df["subject_id"].unique())[:3]
    _print_per_rep_table(
        rep_df, first_3,
        f"Per-Rep Features — Subjects {first_3}",
    )

    # Print full subject summary.
    _print_subject_summary(summary_df)

    # Sanity checks.
    print_sanity_checks(rep_df, summary_df)

    # ── SisFall ───────────────────────────────────────────────────────────
    sis_rep = run_sisfall()

    sis_rep.to_csv(_SISFALL_PER_REP_CSV, index=False)
    print(f"\n  Saved SisFall per-rep features to {_SISFALL_PER_REP_CSV}")

    # Print SisFall table for first 3 subjects.
    sis_first_3 = sorted(sis_rep["subject_id"].unique())[:3]
    _print_per_rep_table(
        sis_rep, sis_first_3,
        f"SisFall Per-Rep Features — Subjects {sis_first_3}",
    )

    # Cross-dataset comparison.
    print_comparison(rep_df, sis_rep)

    print(f"\nDone.")


if __name__ == "__main__":
    main()
