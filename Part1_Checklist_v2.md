# Part 1 Checklist: Sit-to-Stand Event Detection ML Experiment

> **Key clarification:** The model classifies individual time windows as "sit-to-stand" or "not." Rep count is derived by post-processing consecutive positive windows into discrete events (smoothing → minimum duration → minimum gap → count clusters). The ML task is window classification; rep detection is a post-processing step on top of it.

## Phase 1: Setup & Data Understanding (Days 1-2)

### Environment Setup
- [ ] Set up Python environment (Python 3.10+)
- [ ] Install packages: `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `scipy`
- [ ] Create project folder structure:
  ```
  project/
  ├── data/
  │   ├── uci_hapt/
  │   ├── marques/
  │   └── sisfall/
  ├── notebooks/
  ├── src/
  └── results/
  ```

### Download & Explore UCI HAPT
- [ ] Download UCI HAPT dataset: https://archive.ics.uci.edu/dataset/341
- [ ] Read the README — understand what files are included
- [ ] Locate the **raw inertial signals** (not the pre-computed 561 features)
  - These are the accelerometer (x,y,z) and gyroscope (x,y,z) files
  - Sampled at 50Hz
- [ ] Locate the **labels file** — this tells you which time ranges correspond to which activity
- [ ] Locate the **subject IDs** — you need to know which data belongs to which of the 30 subjects
- [ ] Understand the 12 activity labels:
  - 6 static/dynamic: walking, walking upstairs, walking downstairs, sitting, standing, laying
  - 6 transitions: stand-to-sit, **sit-to-stand**, sit-to-lie, lie-to-sit, stand-to-lie, lie-to-stand
  - **You only care about sit-to-stand (label 5 in UCI HAPT)**
- [ ] Count how many sit-to-stand instances exist in the dataset
  - Note: this will be a small number relative to total windows — this is the class imbalance problem

### Download External Datasets (for later, but grab now)
- [ ] Download Marques dataset (search: "Accelerometer data sit-to-stand elderly Marques 2020")
  - 40 elderly participants, accelerometer only, waist smartphone
- [ ] Download SisFall dataset (search: "SisFall dataset")
  - 15 elderly participants, accelerometer+gyroscope, waist device
- [ ] For each: skim the documentation, note the sampling rate, sensor types, and label format
  - If sampling rate differs from 50Hz, you will need to resample using `scipy.signal.resample_poly` with anti-aliasing to avoid corrupted signals (especially SisFall at 200Hz → 50Hz)

---

## Phase 2: Feature Extraction Pipeline (Days 3-5)

### Windowing
- [ ] Write a function that takes a raw signal and cuts it into 2.56-second windows with 50% overlap
  - At 50Hz, each window = 128 samples
  - 50% overlap means each window starts 64 samples after the previous one
- [ ] Assign a label to each window based on which activity occupies the majority of that window
- [ ] **Drop ambiguous windows:** If a window straddles two different activity/transition boundaries (e.g., less than 80% of the window belongs to one label), drop it from the dataset. Transitions are exactly where labels are noisiest, and training on ambiguous labels hurts the model on the exact class you care about.
- [ ] Remap remaining labels to binary: sit-to-stand = 1, everything else = 0
- [ ] Track which subject each window belongs to (needed for LOSO-CV)

### Feature Computation
- [ ] First, compute **acceleration magnitude** per sample: `accel_mag = sqrt(ax² + ay² + az²)`. This is orientation-independent — "accel Z" is not guaranteed to be vertical because phone orientation varies in the waistband. Magnitude works regardless of how the phone is positioned.
- [ ] Similarly, compute **gyroscope magnitude**: `gyro_mag = sqrt(gx² + gy² + gz²)`
- [ ] For each window, compute 6 features for each of **8 channels** (accel x/y/z, gyro x/y/z, accel magnitude, gyro magnitude):

| Feature | Code | What it captures |
|---------|------|-----------------|
| Mean | `np.mean(window)` | Average signal level |
| Std | `np.std(window)` | Signal variability |
| Min | `np.min(window)` | Lowest value |
| Max | `np.max(window)` | Highest value |
| Range | `np.max(window) - np.min(window)` | Total swing |
| Energy | `np.mean(window**2)` | Movement intensity |

- [ ] This gives you **48 features per window** (6 features × 8 channels)
- [ ] Store as a DataFrame: each row = one window, columns = 48 features + label + subject_id
- [ ] Sanity check: print the shape — you should have thousands of rows, 50 columns
- [ ] Sanity check: print label distribution — confirm sit-to-stand (label=1) is a small minority

### Verify Your Pipeline
- [ ] Pick one known sit-to-stand segment from the labels file
- [ ] Look at the raw signal for that segment — you should see a clear spike in acceleration magnitude
- [ ] Confirm the windows covering that segment got labeled as 1
- [ ] Pick one known sitting/standing segment — confirm those windows got labeled as 0
- [ ] **This step catches bugs early. Do not skip it.**

---

## Phase 3: Build the Three Models (Days 5-7)

### Model 1: Threshold Baseline (Non-ML)
- [ ] This is your "no learning" approach
- [ ] Logic: use **acceleration magnitude** features (not a single axis — orientation-independent)
- [ ] If `max_accel_mag > threshold` and `range_accel_mag > threshold`, predict sit-to-stand
- [ ] Tune thresholds by eyeballing the data or using a simple grid search on training data
- [ ] Note: this baseline should be simple and dumb on purpose — it represents what you'd build without ML

### Model 2: Logistic Regression
- [ ] `from sklearn.linear_model import LogisticRegression`
- [ ] Use `class_weight='balanced'` to handle class imbalance
- [ ] Train on your 48-feature matrix with binary labels
- [ ] This is your "simple linear ML" approach

### Model 3: Random Forest
- [ ] `from sklearn.ensemble import RandomForestClassifier`
- [ ] Use `class_weight='balanced'` to handle class imbalance
- [ ] Start with default hyperparameters (100 trees)
- [ ] This is your "ensemble ML" approach

### Quick Test (before full LOSO-CV)
- [ ] Do a quick sanity check: train on subjects 1-25, test on subjects 26-30
- [ ] Print precision, recall, F1 for class 1 (sit-to-stand)
- [ ] If all three models give 0% recall, something is wrong with your pipeline — debug before proceeding
- [ ] If numbers look reasonable (>50% recall), proceed to full LOSO-CV

---

## Phase 4: LOSO-CV Evaluation — Experiment 1 (Days 7-9)

### Implement LOSO-CV
- [ ] Loop through subjects 1 to 30
- [ ] In each fold:
  - Train set = all windows NOT from this subject
  - Test set = all windows from this subject
  - Train each of the 3 models on train set
  - Predict on test set
  - Record precision, recall, F1 for sit-to-stand class
  - Record PR-AUC (precision-recall area under curve) for ML models — `from sklearn.metrics import average_precision_score`. Use predicted probabilities, not hard labels. PR-AUC is more informative than F1 for rare classes.
- [ ] Store results for each fold in a list/DataFrame

### Compute and Report Window-Level Results
- [ ] Calculate mean ± std across 30 folds for precision, recall, F1, PR-AUC — for each of the 3 models
- [ ] Create a comparison table:

| Model | Precision | Recall | F1 | PR-AUC |
|-------|-----------|--------|----|--------|
| Threshold baseline | X ± Y | X ± Y | X ± Y | — |
| Logistic regression | X ± Y | X ± Y | X ± Y | X ± Y |
| Random Forest | X ± Y | X ± Y | X ± Y | X ± Y |

- [ ] Generate a confusion matrix (aggregated across all folds) for the best model
- [ ] Check: what are sit-to-stand windows most commonly confused with? (probably stand-to-sit)

### Rep-Level Evaluation (Event Detection)
- [ ] For each held-out subject in each fold, apply post-processing to turn window predictions into events:
  1. **Smooth predictions:** Apply a moving average (0.5–1.0s window) to predicted probabilities, then threshold
  2. **Minimum event duration:** Discard any positive cluster shorter than 0.5s (too short to be a real sit-to-stand)
  3. **Minimum gap between events:** Merge positive clusters separated by less than 1.0s (likely the same transition)
  4. **Count remaining clusters** = predicted rep count
- [ ] Define event-level matching: a predicted event is a true positive if it falls within ±0.5s of a labeled sit-to-stand transition time. This gives you event-level precision and recall.
- [ ] Compare predicted rep count against ground truth rep count per subject
- [ ] Report mean absolute error between predicted and actual rep counts
- [ ] Report event-level precision and recall alongside window-level metrics
- [ ] This is the metric that connects directly to the app — and is more meaningful for your story

### Feature Importance (RF only)
- [ ] After full LOSO-CV, train a final RF on all 30 subjects
- [ ] Print `model.feature_importances_`
- [ ] Plot top 10 features
- [ ] Check: do the top features align with the literature? (expect acceleration magnitude and vertical features to rank high, per Millor and Galán-Mercant)
- [ ] Discuss in writeup

---

## Phase 5: Feature Ablation — Experiment 3 (Days 9-10)

### Accelerometer-Only vs. Full
- [ ] Re-run LOSO-CV using only accelerometer features (24 features: 6 stats × 3 accel axes + accel magnitude)
- [ ] Compare against the full 48-feature results
- [ ] Create comparison table:

| Feature Set | Precision | Recall | F1 | PR-AUC |
|-------------|-----------|--------|----|--------|
| Accel only (24 features) | X ± Y | X ± Y | X ± Y | X ± Y |
| Accel + Gyro (48 features) | X ± Y | X ± Y | X ± Y | X ± Y |

- [ ] This answers: does gyroscope add value?
- [ ] Practically important because Marques only has accelerometer data

---

## Phase 6: External Validation — Experiment 2 (Days 10-14)

### Define Ground Truth Extraction (DO THIS FIRST)
- [ ] **This is the step that takes longer than you think.** Before writing any code, open each external dataset and answer:
  - How are sit-to-stand events annotated? (start/end timestamps? protocol structure? video labels?)
  - Are labels provided per-sample, per-segment, or per-trial?
  - Do you need to derive event times from the protocol description (e.g., "trial D07 is sit-to-stand")?
- [ ] **For Marques:** Document exactly how their annotations map to sit-to-stand event start/end times
- [ ] **For SisFall:** Document exactly which activity codes correspond to sit-to-stand (e.g., D07, D08) and how to extract event boundaries from the trial structure
- [ ] Write this down before coding. If ground truth is ambiguous, note it as a limitation.

### Prepare External Data
- [ ] Apply the **exact same** windowing and feature extraction pipeline to Marques and SisFall
- [ ] Handle differences:
  - **Sampling rate:** If not 50Hz, resample using `scipy.signal.resample_poly` with anti-aliasing filter. Do NOT use simple interpolation — it introduces aliasing artifacts, especially going from 200Hz → 50Hz.
  - **Sensor availability:** Marques is accel-only → use your accel-only model from Experiment 3 (24 features)
  - **Axis orientation:** May differ from UCI HAPT — this is why your magnitude-based features are valuable (orientation-independent). Document any known differences.
  - **Label format:** Map their labels to your binary scheme using the ground truth extraction rules you defined above

### Train Final Model
- [ ] Train RF, logistic regression, and threshold baseline on ALL 30 UCI HAPT subjects
- [ ] This is the model you test externally — no retraining on external data

### Run and Report
- [ ] Run all three models on Marques data → report precision, recall, F1, PR-AUC
- [ ] Run all three models on SisFall data → report precision, recall, F1, PR-AUC
- [ ] Apply the same event post-processing (smoothing, min duration, min gap) and report event-level metrics
- [ ] Compare against Experiment 1 numbers — how much did performance drop?
- [ ] If big drop: discuss domain shift (young vs. elderly, different movement patterns, different sensor characteristics)
- [ ] If small drop: the model generalizes well — strong result
- [ ] Also report rep-level accuracy on external datasets

---

## Phase 7: Writeup & Figures (Days 14-16)

### Required Tables
- [ ] Experiment 1 results: 3 models × 4 metrics (precision, recall, F1, PR-AUC) — window-level
- [ ] Experiment 1 results: event-level precision, recall, and rep count MAE
- [ ] Experiment 2 results: 3 models × 2 datasets × metrics
- [ ] Experiment 3 results: accel-only vs. full feature set

### Required Figures
- [ ] Confusion matrix for best model (Experiment 1)
- [ ] Feature importance plot (top 10 features from RF)
- [ ] Optional: example raw signal showing sit-to-stand with detected events highlighted

### Key Points to Discuss
- [ ] Which model performed best and why?
- [ ] Did RF justify its complexity over logistic regression?
- [ ] Did the threshold baseline perform surprisingly well or poorly?
- [ ] Did gyroscope features help significantly?
- [ ] How well did models generalize to elderly populations?
- [ ] What were the most common classification errors?
- [ ] Do feature importances align with the literature (Millor, Galán-Mercant)?
- [ ] How did event-level metrics compare to window-level metrics? (This shows whether post-processing rescued noisy window predictions)

---

## Milestone Deadline: March 6

**Minimum needed for milestone:**
- [ ] Feature extraction pipeline complete
- [ ] All three models trained and evaluated with LOSO-CV (Experiment 1)
- [ ] Window-level results table with precision, recall, F1
- [ ] Brief discussion of results

**Nice to have for milestone:**
- [ ] Event-level post-processing and rep count accuracy
- [ ] Feature ablation (Experiment 3)
- [ ] Feature importance analysis
- [ ] PR-AUC reported

**Can wait for final report (March 17):**
- [ ] Complete external validation with discussion (Experiment 2)
- [ ] Full event-level analysis across all datasets
- [ ] Quality assessment layer demonstration
- [ ] Polished figures and full discussion
