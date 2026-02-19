# Smartphone-Based Sit-to-Stand Power Assessment for Frailty Risk Screening

**Project Category:** Application
**Team Member:** Ozair Ismail (06966695)

## Abstract

Frailty affects ~10% of adults over 65 and is the strongest predictor of falls and loss of independence. The 30-second chair stand test is clinically validated but only captures repetition count, missing movement quality indicators that better predict adverse outcomes. We develop a two-stage system: (1) an ML-based sit-to-stand event detector trained on smartphone inertial data, and (2) a rule-based quality assessment layer that extracts six clinically meaningful features mapped to Fried frailty dimensions. The input is raw 6-axis accelerometer and gyroscope data at 50Hz from a waist-mounted smartphone. We compare a threshold baseline, Logistic Regression, and Random Forest using 30-fold Leave-One-Subject-Out Cross-Validation on the UCI HAPT dataset. Random Forest achieves the best internal performance (event F1 = 0.755, rep count MAE = 0.60), but completely fails on external validation with elderly subjects from SisFall (0% recall), while Logistic Regression generalizes well (92.6% recall). Feature ablation confirms gyroscope features are critical, contributing 54% of Random Forest's feature importance. The quality assessment layer demonstrates extraction of per-rep power, acceleration, angular velocity, and fatigue indicators that capture pre-frailty risk invisible to repetition counting alone.

## 1. Introduction

Frailty is a clinical syndrome of reduced physiological reserve that dramatically increases vulnerability to adverse health outcomes. The 30-second chair stand test (30s CST) is widely used for lower-limb functional assessment, but standard administration only counts repetitions with a stopwatch. Research shows that sensor-derived kinematic parameters — peak acceleration, angular velocity, and movement variability — differentiate frailty levels even when repetition counts are identical [1]. This motivates automating and enriching the test using ubiquitous smartphone sensors.

The input to our system is raw triaxial accelerometer and gyroscope data (6 channels at 50Hz) from a waist-mounted smartphone. We use a Random Forest classifier, Logistic Regression, and a threshold-based baseline to output a binary prediction per 2.56-second time window: sit-to-stand (1) or other activity (0). Consecutive positive windows are clustered into discrete rep events. A rule-based quality assessment layer then extracts six per-rep features — peak dynamic acceleration magnitude, relative muscle power, time-per-rep, peak gyroscope magnitude, coefficient of variation, and fatigue slope — each mapped to a specific Fried frailty dimension (weakness, slowness, or exhaustion) and compared against published reference ranges. The system outputs three tiers: rep count, relative power score, and movement quality flags.

## 2. Related Work

Prior work on instrumented sit-to-stand analysis falls into three categories.

**Clinical IMU studies.** Millor et al. [1] used a single lumbar IMU during the 30s CST and found that kinematic parameters (particularly angular velocity peaks) differentiated three frailty levels where rep count could not (p < 0.001). Van Lummel et al. [2] demonstrated that instrumented STS durations were more strongly associated with health status than manually recorded durations. These studies validate the clinical value of sensor-derived features but use dedicated research-grade hardware inaccessible for home use.

**Smartphone-based approaches.** Galán-Mercant & Cuesta-Vargas [3] used an iPhone 4 at the waist and found that frail elderly produced peak vertical acceleration of ~2.7 m/s² vs. ~8.5 m/s² for non-frail — the only smartphone study with frailty-specific thresholds. However, their dataset is not public. Sher et al. [4] achieved 99% cycle detection accuracy on 660 sit-to-stand cycles using rule-based signal processing with a waist-mounted smartphone, demonstrating that well-engineered heuristics can perform well on this task.

**Power-based assessment.** Alcázar et al. [5] validated a relative sit-to-stand power equation with the strongest association with frailty among available STS power formulas, and published age- and sex-stratified normative cut-off points. Park et al. [6] identified three optimal sensor-derived features for frailty phenotypes (mapping to slowness, weakness, and exhaustion) using wearable sensors and logistic regression, achieving AUC 0.86.

Our work differs in combining ML-based event detection with a clinically grounded quality assessment layer using a single smartphone, approximating Park et al.'s multi-sensor framework with a single device.

## 3. Dataset

**UCI HAPT** [7] (training/internal validation): Raw accelerometer and gyroscope data from 30 participants (ages 19–48) wearing a Samsung Galaxy S II at the waist, sampled at 50Hz. Contains 12 activity/transition classes including sit-to-stand (activity ID 8). We identified 62 sit-to-stand segments averaging 2.59 seconds (range 1.48–3.66s).

**Preprocessing:** Signals are segmented into 2.56-second windows (128 samples) with 50% overlap. Windows are labeled by majority vote with a 50% purity threshold — if >50% of samples belong to one activity, the window receives that label; otherwise it is dropped. We initially used 80% purity per standard practice, but found this too aggressive: 16% of sit-to-stand segments (10/62) were too short to ever reach 80% purity, eliminating all positive examples for 3 subjects. The 50% threshold recovered these while maintaining majority-vote labeling. Labels are remapped to binary: sit-to-stand = 1, everything else = 0.

**Feature extraction:** We compute acceleration magnitude (sqrt(ax² + ay² + az²)) and gyroscope magnitude (sqrt(gx² + gy² + gz²)) as orientation-independent channels, yielding 8 channels total. For each window, we compute 6 statistics per channel (mean, std, min, max, range, energy), producing **48 features per window**. Final dataset: 17,453 windows — 126 sit-to-stand (0.72%), 17,327 other (99.28%).

**SisFall** [8] (external validation): 15 elderly subjects (ages 60–75) performing slow (D07) and fast (D08) sit-to-stand trials. 149 trials total, each a self-contained 12-second recording. Signals were converted to matching units (g's and rad/s), resampled from 200Hz to 50Hz using `resample_poly` with anti-aliasing, and processed through the identical feature pipeline.

## 4. Methods

We compare three approaches representing a complexity spectrum:

**Threshold Baseline (no ML):** Predicts sit-to-stand if `accel_mag_max > t₁` AND `accel_mag_range > t₂`, where thresholds are found via grid search on training data maximizing F1. This uses 2 features and no learned parameters.

**Logistic Regression:** Learns a weight vector **w** ∈ ℝ⁴⁸ and bias b, predicting P(y=1|x) = σ(w·x + b) where σ is the sigmoid function. Uses `class_weight='balanced'` to upweight the minority class by a factor proportional to class_size_ratio, and `StandardScaler` for feature normalization. Captures linear feature combinations.

**Random Forest:** Ensemble of 100 decision trees, each trained on a bootstrap sample with random feature subsets at each split. Final prediction is majority vote. Uses `class_weight='balanced'`. Can learn nonlinear feature interactions — e.g., high acceleration combined with high gyroscope activity indicates sit-to-stand, while high acceleration alone may indicate walking upstairs.

**Evaluation:** 30-fold LOSO-CV where each fold holds out one subject for testing. We report precision, recall, F1, and PR-AUC for the sit-to-stand class. We also report event-level metrics by clustering consecutive positive predictions into events and matching against ground truth within ±1.0 second tolerance. Rep count MAE is computed per subject.

**Quality Assessment Layer:** For each detected rep, we extract features from the raw signal within rep boundaries after removing gravity via a 4th-order high-pass Butterworth filter (0.3Hz cutoff). Six indicators are computed: (1) peak dynamic acceleration magnitude, (2) relative muscle power via adapted Alcázar equation: Power (W/kg) = [0.9 × 9.81 × (height × 0.5 − chair_height)] / (mean_time_per_rep × 0.5), (3) time-per-rep, (4) peak gyroscope magnitude, (5) coefficient of variation across reps, and (6) fatigue slope (linear regression of per-rep metrics against rep number). Each maps to a Fried frailty dimension and is compared against published thresholds. This layer is entirely rule-based.

## 5. Experiments, Results, and Discussion

### 5.1 Experiment 1: Internal Validation (LOSO-CV)

**Table 1: Window-level results (mean ± std, 30 folds)**

| Model | Precision | Recall | F1 | PR-AUC |
|-------|-----------|--------|----|--------|
| Threshold Baseline | 0.011 ± 0.002 | 0.933 ± 0.135 | 0.022 ± 0.004 | — |
| Logistic Regression | 0.186 ± 0.070 | 0.933 ± 0.117 | 0.305 ± 0.096 | 0.675 ± 0.148 |
| Random Forest | 0.778 ± 0.377 | 0.454 ± 0.283 | 0.554 ± 0.301 | 0.742 ± 0.212 |

Random Forest achieves the best F1 and PR-AUC. The pooled confusion matrix reveals only 5 false positives, all SITTING windows — the activity immediately preceding sit-to-stand, whose boundary windows share similar sensor patterns.

**Post-processing interaction.** We found that post-processing parameters must match model characteristics. Random Forest produces sparse, high-confidence predictions (often a single positive window per event). Standard post-processing — smoothing probabilities and requiring minimum 2-window duration — destroyed these correct predictions. With minimal post-processing (single positive window counts as an event), Random Forest's event F1 jumped from 0.187 to 0.755:

**Table 2: Event-level results (RF, minimal post-processing)**

| Model | Rep Count MAE | Event Precision | Event Recall | Event F1 |
|-------|--------------|-----------------|--------------|----------|
| Threshold Baseline | 14.73 | 0.101 | 0.823 | 0.180 |
| Logistic Regression | 3.23 | 0.151 | 0.387 | 0.217 |
| Random Forest | **0.60** | **0.909** | **0.645** | **0.755** |

20 of 30 subjects received an exact rep count from RF. Five subjects (9, 10, 17, 23, 25) received 0 detections — these have the shortest sit-to-stand segments producing noisy boundary windows.

**Feature importance.** The top feature is `gx_energy` (gyroscope x-axis energy), and gyroscope features occupy 5 of the top 10 positions (54% of total Gini importance). This aligns with Millor et al.'s finding that angular velocity peaks are the strongest frailty differentiator [1].

### 5.2 Experiment 2: External Validation (SisFall Elderly)

Final models trained on all 30 UCI HAPT subjects were tested on 149 SisFall elderly trials without retraining.

**Table 3: External validation event recall**

| Model | Overall | D07 (slow) | D08 (fast) |
|-------|---------|------------|------------|
| Threshold Baseline | 0.960 | 0.919 | 1.000 |
| Logistic Regression | 0.926 | 0.892 | 0.960 |
| Random Forest | 0.000 | 0.000 | 0.000 |

Random Forest fails completely — zero detections across 1,192 windows. Its nonlinear decision boundaries overfit to UCI HAPT-specific patterns (young adults, specific sensor) such that elderly kinematics from different hardware fall entirely outside them. Logistic Regression generalizes well (92.6% recall), demonstrating that its linear decision boundary captures more universal biomechanical signatures of sit-to-stand. This is the bias-variance tradeoff in practice: the model with the strongest internal performance generalizes worst, while simpler models transfer better.

D08 (fast) trials are detected more reliably than D07 (slow) across all models, confirming that faster movements produce more salient sensor signals.

### 5.3 Experiment 3: Feature Ablation

**Table 4: Accel-only vs. full features (RF, event-level)**

| Feature Set | Event F1 | Event Recall |
|-------------|----------|-------------|
| Accel + Gyro (48 features) | 0.755 | 0.645 |
| Accel only (24 features) | 0.341 | 0.226 |

Removing gyroscope cuts event F1 by 0.41, confirming the feature importance finding and consistent with Millor et al.'s emphasis on angular velocity.

### 5.4 Quality Assessment Layer

All six indicators produce physiologically plausible values across 62 UCI HAPT reps: peak dynamic acceleration 2.07–9.06 m/s² (mean 4.54), time-per-rep 1.48–3.66s (mean 2.59), peak gyroscope magnitude 1.01–4.92 rad/s (mean 2.07), and relative power 1.88–4.65 W/kg (mean 2.79). The correlation between time-per-rep and peak acceleration is r = −0.40 (p = 0.029), confirming the expected biomechanical relationship.

When compared against published thresholds: all 30 subjects fall in the intermediate range for peak acceleration (between Galán-Mercant's frail threshold of 2.7 m/s² and non-frail threshold of 8.5 m/s²), 29/30 show power within the expected range (one subject at 1.92 W/kg), and 7/30 trigger the exhaustion flag (CV > 0.30). Individual variation is substantial — Subject 17's peak acceleration (2.80 m/s²) is near the frail threshold despite being a young adult, while Subject 10 reaches 6.99 m/s². This demonstrates that rep count alone misses clinically relevant individual differences in movement quality.

**Limitations.** UCI HAPT is not a 30s CST (only 2–3 reps per subject), so CV and fatigue slope values are illustrative rather than clinically meaningful. Published thresholds were derived from different protocols, sensors, and populations; comparisons are approximate. The Alcázar power equation is adapted from its validated 5-rep form. Acceleration magnitude is a proxy for the vertical acceleration used in the literature. No direct validation against Fried phenotype scores was performed.

## 6. Conclusion and Future Work

We demonstrated a two-stage smartphone-based system for sit-to-stand assessment. Random Forest achieved the best internal event detection (F1 = 0.755, MAE = 0.60 reps) but completely failed on external elderly data, while Logistic Regression generalized well (92.6% event recall). Gyroscope features are critical, contributing 54% of feature importance and improving event F1 by 0.41. The quality assessment layer extracts six clinically meaningful indicators that capture pre-frailty risk invisible to standard rep counting.

For future work, we would: (1) train and validate on a dataset collected under the actual 30s CST protocol with elderly participants and Fried phenotype labels, (2) explore domain adaptation techniques to improve Random Forest's cross-population generalization, (3) extend to continuous daily monitoring where sit-to-stands are detected opportunistically, and (4) conduct a clinical validation study comparing app-derived frailty indicators against clinician-administered assessments.

## References

[1] Millor, N., et al. (2013). An evaluation of the 30-s chair stand test in older adults: frailty detection based on kinematic parameters from a single inertial unit. *J. NeuroEng. Rehab.*, 10, 86.

[2] Van Lummel, R. C., et al. (2016). The instrumented sit-to-stand test has greater clinical relevance than the manually recorded sit-to-stand test in older adults. *PLoS ONE*, 11(7), e0157968.

[3] Galán-Mercant, A., & Cuesta-Vargas, A. I. (2014). Mobile Romberg test assessment. *BMC Research Notes*, 7, 640.

[4] Sher, T., et al. (2025). Waist-mounted smartphone 30s CST cycle detection. (rule-based, 99% accuracy on 660 cycles).

[5] Alcázar, J., et al. (2021). Relative sit-to-stand power: aging trajectories, functionally relevant cut-off points, and normative data. *J. Cachexia, Sarcopenia and Muscle*, 12(4), 1013–1028.

[6] Park, C., et al. (2021). Optimal sensor-based frailty phenotype assessment using wearable sensors. *IEEE J. Biomed. Health Inform.*, 25(8), 3057–3067.

[7] Reyes-Ortiz, J. L., et al. (2015). Transition-aware human activity recognition using smartphones. *Neurocomputing*, 171, 754–767.

[8] Sucerquia, A., López, J. D., & Vargas-Bonilla, J. F. (2017). SisFall: A fall and movement dataset. *Sensors*, 17(1), 198.

## Contributions

This is a solo project. All work — data processing, feature engineering, model implementation, evaluation, quality assessment pipeline, and writing — was performed by Ozair Ismail.
