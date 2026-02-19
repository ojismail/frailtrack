# Smartphone-Based Sit-to-Stand Power Assessment for Frailty Risk Screening

**Project Category:** Application
**Team Member:** Ozair Ismail (06966695)

---

## Motivation

Frailty affects approximately 10% of adults over 65 and is the strongest predictor of falls, hospitalization, and loss of independence. The 30-second chair stand test (30s CST) is a clinically validated assessment of lower-limb function, but traditional administration only counts repetitions using a stopwatch. Research demonstrates that sensor-derived parameters — peak acceleration, angular velocity, and movement variability — differentiate frailty levels even when repetition counts are identical (Millor et al., 2013). We aim to build a smartphone application that transforms the standard chair stand test into a comprehensive frailty screening tool by combining ML-based repetition detection with clinically validated movement quality indicators.

The input to our system is raw triaxial accelerometer and gyroscope data (6 channels at 50Hz) from a waist-mounted smartphone. We use a Random Forest classifier, a Logistic Regression classifier, and a threshold-based baseline to output a binary prediction per time window: sit-to-stand (1) or other activity (0). Consecutive positive windows are then clustered into discrete rep events, and per-rep features are extracted for clinical assessment.

## Dataset

We train and evaluate on the UCI HAPT dataset (Reyes-Ortiz et al., 2015), which contains raw inertial signals from 30 participants (ages 19–48) wearing a waist-mounted Samsung Galaxy S II. The dataset includes 12 activity classes; we frame the task as binary classification: sit-to-stand (activity ID 8) vs. everything else.

**Preprocessing:** Raw signals are segmented into 2.56-second windows (128 samples at 50Hz) with 50% overlap. Each window is labeled by majority vote — if >50% of samples belong to one activity, the window gets that label. We initially used an 80% purity threshold (per standard practice for removing ambiguous boundary windows), but analysis showed this was too aggressive: 16% of sit-to-stand segments (10/62) were shorter than the 103 samples needed to ever reach 80% purity within a 128-sample window, and 3 subjects lost all positive windows entirely. Lowering to 50% recovered these segments while maintaining that windows are labeled by their dominant activity.

**Feature extraction:** For each window, we compute acceleration and gyroscope magnitude (orientation-independent L2 norms), yielding 8 channels (ax, ay, az, gx, gy, gz, accel_mag, gyro_mag). We compute 6 summary statistics per channel (mean, std, min, max, range, energy), producing 48 features per window.

**Final dataset:** 17,453 windows total — 126 sit-to-stand (0.72%) and 17,327 other (99.28%). All 30 subjects have at least one positive window. For external validation, we use SisFall (Sucerquia et al., 2017): 15 elderly subjects (ages 60–75), activities D07 (slow sit-to-stand) and D08 (fast sit-to-stand), 149 trials total. SisFall signals were converted to matching units and resampled from 200Hz to 50Hz.

## Method

We compare three approaches representing a spectrum of model complexity:

**Threshold Baseline (no ML):** Predicts sit-to-stand if both `accel_mag_max` and `accel_mag_range` exceed thresholds found via grid search on training data. This represents what could be built without ML — simple rules on two hand-picked features.

**Logistic Regression:** Linear classifier on all 48 features with `class_weight='balanced'` and `StandardScaler`. Captures linear feature combinations but cannot model nonlinear interactions.

**Random Forest:** Ensemble of 100 decision trees on all 48 features with `class_weight='balanced'`. Can learn nonlinear feature interactions (e.g., "high acceleration AND high gyroscope = sit-to-stand, but high acceleration AND low gyroscope = walking").

All models are evaluated using 30-fold Leave-One-Subject-Out Cross-Validation (LOSO-CV), which ensures no data from the test subject appears in training. We report precision, recall, F1, and PR-AUC for the sit-to-stand class.

## Preliminary Experiments and Results

### Experiment 1: Internal Validation (LOSO-CV on UCI HAPT)

**Window-level results (mean ± std across 30 folds):**

| Model | Precision | Recall | F1 | PR-AUC |
|-------|-----------|--------|----|--------|
| Threshold Baseline | 0.011 ± 0.002 | 0.933 ± 0.135 | 0.022 ± 0.004 | — |
| Logistic Regression | 0.186 ± 0.070 | 0.933 ± 0.117 | 0.305 ± 0.096 | 0.675 ± 0.148 |
| Random Forest | 0.778 ± 0.377 | 0.454 ± 0.283 | 0.554 ± 0.301 | 0.742 ± 0.212 |

Random Forest achieves the best F1 (0.554) and PR-AUC (0.742). The pooled confusion matrix shows only 5 false positives across all folds, all of which were SITTING windows — the activity immediately preceding sit-to-stand, whose boundary windows share similar sensor patterns.

**Event-level results:** We convert window predictions into discrete events via post-processing (merging nearby positive clusters, filtering short clusters). We found that post-processing interacts differently with each model: strict filtering (minimum 2-window duration, smoothing) works for noisy models but destroys Random Forest's sparse, high-confidence predictions. With minimal post-processing (single positive window counts as an event), Random Forest achieves:

| Model | Rep Count MAE | Event Precision | Event Recall | Event F1 |
|-------|--------------|-----------------|--------------|----------|
| Threshold Baseline | 14.73 | 0.101 | 0.823 | 0.180 |
| Logistic Regression | 3.23 | 0.151 | 0.387 | 0.217 |
| Random Forest | 0.60 | 0.909 | 0.645 | 0.755 |

Random Forest with minimal post-processing achieves a mean absolute error of 0.60 reps — 20 of 30 subjects receive an exact rep count.

**Feature importance:** The top feature is `gx_energy` (gyroscope x-axis), and 5 of the top 10 features are gyroscope-based (54% of total Gini importance). This aligns with Millor et al.'s finding that angular velocity peaks are the strongest frailty differentiator.

### Experiment 2: External Validation (SisFall Elderly)

We trained final models on all 30 UCI HAPT subjects and tested on 149 SisFall elderly trials without retraining:

| Model | Event Recall (all) | Event Recall (D07 slow) | Event Recall (D08 fast) |
|-------|-------------------|------------------------|------------------------|
| Threshold Baseline | 0.960 | 0.919 | 1.000 |
| Logistic Regression | 0.926 | 0.892 | 0.960 |
| Random Forest | 0.000 | 0.000 | 0.000 |

Random Forest completely fails on external data — zero detections across all 1,192 windows. It learned UCI HAPT-specific decision boundaries so tightly that elderly kinematics from different sensors fall entirely outside them. Logistic Regression generalizes well (92.6% recall), and the threshold baseline generalizes best (96.0%). This is a textbook bias-variance tradeoff: the model with the strongest internal performance (RF) overfits to the training domain, while simpler models with weaker in-distribution performance transfer better to unseen populations.

### Experiment 3: Feature Ablation (Accelerometer-Only vs. Full)

| Feature Set | RF Event F1 | LR Event F1 |
|-------------|-------------|-------------|
| Accel + Gyro (48 features) | 0.755 | 0.217 |
| Accel only (24 features) | 0.341 | — |

Removing gyroscope features cuts RF event F1 from 0.755 to 0.341, confirming the feature importance finding. This has practical implications: datasets and devices without gyroscope will see substantially degraded performance.

## Next Steps

For the final report, we will complete the following:

1. **Quality assessment layer:** Extract six clinically meaningful per-rep features from detected rep boundaries — peak dynamic acceleration magnitude (weakness), relative muscle power via the adapted Alcázar equation (weakness), time-per-rep (slowness), peak gyroscope magnitude (slowness), coefficient of variation (exhaustion), and fatigue slope (fatigability). Each feature maps to a Fried frailty dimension and is compared against published reference thresholds. This layer is entirely rule-based, not ML.

2. **Three-tier output:** Present rep count (Tier 1), relative power score (Tier 2), and movement quality flags (Tier 3) — demonstrating that the system captures pre-frailty indicators invisible to rep count alone.

3. **Full discussion** of the generalization findings, clinical implications, and limitations.

## Contributions

This is a solo project. All work — data processing, feature engineering, model implementation, evaluation, and writing — was performed by Ozair Ismail.

## References

Alcázar, J., et al. (2021). Relative sit-to-stand power: aging trajectories, functionally relevant cut-off points, and normative data in a large European cohort. *Journal of Cachexia, Sarcopenia and Muscle*, 12(4), 1013–1028.

Galán-Mercant, A., & Cuesta-Vargas, A. I. (2014). Mobile Romberg test assessment (mRomberg). *BMC Research Notes*, 7, 640.

Millor, N., Lecumberri, P., Gómez, M., Martínez-Ramírez, A., & Izquierdo, M. (2013). An evaluation of the 30-s chair stand test in older adults: frailty detection based on kinematic parameters from a single inertial unit. *Journal of NeuroEngineering and Rehabilitation*, 10, 86.

Park, C., et al. (2021). Optimal sensor-based frailty phenotype assessment using wearable inertial sensors. *IEEE Journal of Biomedical and Health Informatics*, 25(8), 3057–3067.

Reyes-Ortiz, J. L., Oneto, L., Samà, A., Parra, X., & Anguita, D. (2015). Transition-aware human activity recognition using smartphones. *Neurocomputing*, 171, 754–767.

Sucerquia, A., López, J. D., & Vargas-Bonilla, J. F. (2017). SisFall: A fall and movement dataset. *Sensors*, 17(1), 198.

Van Lummel, R. C., et al. (2016). The instrumented sit-to-stand test (iSTS) has greater clinical relevance than the manually recorded sit-to-stand test in older adults. *PLoS ONE*, 11(7), e0157968.
