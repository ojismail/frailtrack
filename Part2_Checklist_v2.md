# Part 2 Checklist: Quality Assessment Layer

> **Key clarification:** This layer is entirely rule-based, not ML. It takes the rep boundaries detected by the Part 1 model and extracts clinically meaningful features from the raw sensor data within those boundaries. Each feature is aligned with a specific Fried frailty dimension (weakness, slowness, or exhaustion) and is compared against published reference ranges. Be explicit about this distinction in the writeup — the ML contribution is Part 1; Part 2 demonstrates why accurate rep detection matters.

> **Test protocol:** The 30-second chair stand test (30s CST). The user performs as many sit-to-stand repetitions as possible in 30 seconds. This protocol (not the 5-rep STS) is used because: (1) it's what the proposal specifies, (2) it's what Millor et al. used, (3) fatigue/exhaustion indicators (CV, fatigue slope) require 10+ reps to be meaningful — 5 reps is too few for reliable slope estimation.

> **Output is educational, not diagnostic.** The app surfaces indicators aligned with published frailty research. It does not diagnose frailty. Interpretation is left to the user or their clinician. Use language like "below expected range for age group" rather than "you are frail." State this explicitly in the writeup.

---

## Phase 1: Extract Per-Rep Features (Days 1-2)

### Prerequisites
- [ ] Part 1 event detection is working — you have start and end timestamps for each detected rep
- [ ] You have access to the raw sensor signals (not just window features) for the corresponding time ranges

### Gravity Removal
- [ ] Before extracting per-rep features, separate gravity from dynamic acceleration
- [ ] Apply a high-pass Butterworth filter (cutoff ~0.3Hz, matching UCI HAPT's preprocessing) to the raw accelerometer signal. This removes the constant gravitational component and leaves only body-movement acceleration.
- [ ] Compute **dynamic acceleration magnitude**: `dynamic_accel_mag = sqrt(filtered_ax² + filtered_ay² + filtered_az²)`
- [ ] This is orientation-independent and represents movement intensity without gravity bias
- [ ] For gyroscope signals, no gravity removal is needed — gyroscope measures rotation directly

### For Each Detected Rep, Go Back to the Raw Signal and Extract:

**Weakness indicators:**

- [ ] **Peak dynamic acceleration magnitude**
  - Within the rep boundary, find `max(dynamic_accel_mag)`
  - This captures how forcefully the person moves during the transition
  - Note: this is NOT "vertical acceleration" — it is total dynamic movement intensity, orientation-independent. Rename from "peak vertical acceleration" throughout to avoid confusion with literature that assumes a defined vertical axis.
  - Reference: Galán-Mercant & Cuesta-Vargas (2013/2014) found frail elderly ~2.7 m/s² vs. ~8.5 m/s² non-frail using iPhone at waist. Their measurements used a defined vertical axis, so direct numeric comparison is approximate — note this as a limitation.

- [ ] **Relative muscle power (adapted Alcázar equation)**
  - Published formula (validated on 5-rep STS): `Power (W/kg) = [0.9 × 9.81 × (height × 0.5 − chair_height)] / (t_5reps / 5 × 0.5)`
  - **Our adaptation for 30s CST:** Use mean time-per-rep from detected reps as the time input: `Power (W/kg) = [0.9 × 9.81 × (height × 0.5 − chair_height)] / (mean_time_per_rep × 0.5)`
  - **State explicitly in the writeup:** "We adapt Alcázar's equation by using mean time-per-rep derived from the 30s CST rather than total 5-rep time. This is an adaptation, not the original validated formula. The adaptation is reasonable because the equation's core relationship — power is inversely proportional to movement time — holds regardless of protocol, but the published normative cut-off points were derived from the 5-rep version."
  - Requires user input: height (m) and chair height (m) — these would be entered in the app
  - Reference: Alcázar et al. (2021) published age- and sex-stratified normative cut-off points (derived from 5-rep STS)
  - Compare output against their thresholds as an approximate reference, noting the protocol difference

**Slowness indicators:**

- [ ] **Time-per-rep**
  - Simply: `rep_end_time − rep_start_time` in seconds
  - Longer duration = slower movement
  - This uses the full rep boundary (sit-to-stand complete cycle), not just the rising phase — reliably segmenting the rising phase alone would require additional signal processing beyond the scope of this project

- [ ] **Peak gyroscope magnitude**
  - Within the rep boundary, find `max(gyro_magnitude)` where `gyro_magnitude = sqrt(gx² + gy² + gz²)`
  - This captures how quickly the trunk rotates during the transition
  - Using magnitude rather than a single gyroscope axis for the same orientation-independence reason as acceleration
  - Reference: Millor et al. (2013) found Z-velocity peaks during stand-up were the strongest differentiator across frailty levels (p < 0.001). Their measurements used a defined axis; our magnitude is a proxy.
  - Reference: Van Lummel et al. (2013) found lower trunk flexion angular velocity in older adults
  - **Fallback if gyroscope unavailable** (e.g., Marques dataset): skip this feature, note it as a limitation, and rely on the remaining five indicators. Do not substitute a fake value.

**Exhaustion / fatigability indicators:**

- [ ] **Coefficient of variation (CV) across all reps in the session**
  - For any per-rep metric (e.g., peak dynamic acceleration magnitude), compute: `CV = std(values) / mean(values)`
  - High CV = inconsistent performance across the 30 seconds = fatigue or instability
  - With 30s CST, you expect 10-20 reps — enough data points for CV to be stable
  - Reference: Park et al. (2021) found CV of vertical power range was significantly higher in pre-frail/frail group
  - Compute CV for: peak dynamic acceleration magnitude, time-per-rep, peak gyro magnitude, and power

- [ ] **Fatigue slope**
  - Plot any per-rep metric (e.g., peak dynamic acceleration magnitude) against rep number (1st rep, 2nd rep, 3rd rep...)
  - Fit a simple linear regression: `metric = slope × rep_number + intercept`
  - Negative slope = performance declining across the session = fatigability
  - Reference: Schwenk/Lindemann found mean velocity decrease of −0.0037 m/s per repetition in older women, with 10/15/20% fatigue thresholds at approximately 8/14/21 reps
  - This is the most novel indicator — it captures something invisible to a stopwatch
  - Note: the slope value itself is exploratory — no published threshold exists for smartphone-derived fatigue slope during 30s CST. Present it as a trend indicator, not a validated diagnostic.

---

## Phase 2: Compare Against Published Thresholds (Day 3)

### Build a Reference Table
- [ ] Create a table of published thresholds you'll compare against:

| Indicator | Frailty dimension | Threshold / reference value | Source | Match quality |
|-----------|------------------|---------------------------|--------|---------------|
| Peak dynamic accel magnitude | Weakness | Frail: ~2.7 m/s², Non-frail: ~8.5 m/s² | Galán-Mercant (2013/2014) | Approximate — they used defined vertical axis |
| Relative power (adapted Alcázar) | Weakness | Age/sex-stratified cut-off points | Alcázar et al. (2021) | Approximate — validated on 5-rep STS, we use 30s CST mean |
| Time-per-rep | Slowness | Longer = slower (contextual, compare within session and against session mean) | Millor (2013) | Good — time is protocol-independent |
| Peak gyro magnitude | Slowness | Lower = slower; frail showed significantly lower peaks (p < 0.001) | Millor (2013) | Approximate — they used defined Z-axis |
| CV of per-rep metrics | Exhaustion | Higher = more variable = more fatigued; significantly higher in frail group | Park et al. (2021) | Approximate — they used 5-rep STS with 5 wearable sensors |
| Fatigue slope | Exhaustion | Negative slope = declining; ~−0.0037 m/s per rep | Schwenk/Lindemann | Exploratory — no validated threshold for this setup |

- [ ] The "Match quality" column is important for the writeup — it shows you understand the limitations of comparing across different protocols and sensor setups
- [ ] Look up the specific Alcázar normative tables — they provide cut-off points by age decade and sex
- [ ] Note: you are NOT building a frailty classifier. You are flagging where individual indicators fall relative to published ranges.

### Implement the Comparison
- [ ] For each indicator, write a simple function that takes the extracted value and returns a status:
  - "Within expected range" / "Below expected range" / "Above expected range"
  - Base the thresholds on the published values above
  - For fatigue slope: "Stable" (slope near zero) / "Declining" (negative slope beyond a threshold)
- [ ] This is just if/else logic — no ML, no training

---

## Phase 3: Build the Three-Tier Output (Day 4)

### Tier 1: Rep Count
- [ ] Total number of detected reps from Part 1
- [ ] This is the standard clinical measure — what a stopwatch gives you
- [ ] Present alongside normative rep count ranges by age/sex (from Jones et al., 1999 — the original 30s CST norms)

### Tier 2: Relative Power Score
- [ ] Adapted Alcázar power value computed from mean time-per-rep, height, and chair height
- [ ] Flag against age/sex normative cut-off points (noting the protocol adaptation)
- [ ] This goes beyond what a stopwatch provides — same rep count can yield different power scores depending on speed

### Tier 3: Movement Quality Flags
- [ ] Present the remaining indicators as individual flags:
  - Peak dynamic acceleration magnitude: within range / low (weakness flag)
  - Peak gyro magnitude: within range / low (slowness flag) — or "unavailable" if no gyroscope
  - CV across reps: within range / high (exhaustion flag)
  - Fatigue slope: stable / declining (fatigability flag — exploratory)
- [ ] Each flag is aligned with a specific Fried frailty dimension
- [ ] Each flag has a published reference supporting its clinical relevance (with match quality noted)

### The Early Detection Argument
- [ ] Demonstrate with an example (real or constructed):
  - "Subject X completed 12 reps — normal by standard test"
  - "But peak dynamic acceleration declined 30% from first to last rep, CV was 0.35 (above Park's threshold for frail group), and adapted power was below Alcázar's age-stratified cut-off"
  - "These indicators suggest pre-frailty risk that would be invisible to rep count alone"
- [ ] This is the core narrative of your project — rep count misses pre-frailty, your app catches it

---

## Phase 4: Validate the Pipeline on Your Data (Days 4-5)

### Sanity Checks
- [ ] Run the quality assessment pipeline on UCI HAPT subjects using Part 1's detected reps
- [ ] Do the extracted values look physiologically plausible?
  - Peak dynamic accel magnitude should be in the range of 1–15 m/s² (not 0.001 or 500)
  - Time-per-rep should be 1–5 seconds (not 0.01 or 30)
  - Power values should be in the range of 0.5–5 W/kg for adults
  - CV should be between 0 and 1 for most subjects (above 1 means extreme variability)
- [ ] If values are wildly off, debug the unit conversion, gravity removal, or signal processing

### Compare Across Subjects
- [ ] Do subjects who perform fewer reps also tend to have lower peak acceleration and higher CV?
- [ ] This isn't a formal experiment — it's a sanity check that the indicators behave as expected
- [ ] If a subject with 20 reps has worse quality metrics than a subject with 8 reps, something is probably wrong

### Run on External Data (if available from Part 1)
- [ ] Apply the same pipeline to Marques and/or SisFall detected reps
- [ ] Marques subjects are elderly — do their quality metrics look different from UCI HAPT (younger)?
  - Expect: lower peak dynamic accel magnitude, lower peak gyro magnitude, higher CV, more negative fatigue slope
- [ ] If you see this pattern, it supports the clinical validity of the indicators
- [ ] If you don't, discuss why (sensor differences, small sample, protocol differences, etc.)

---

## Phase 5: Writeup (Days 5-6)

### What to Include
- [ ] Clear statement that this layer is rule-based, not ML
- [ ] Clear statement that the test protocol is 30s CST, not 5-rep STS
- [ ] Clear statement that output is educational/screening, not diagnostic
- [ ] Table of all six indicators with their frailty dimension alignment, published references, and match quality
- [ ] The adapted Alcázar equation written out with variable definitions, with explicit note about protocol adaptation
- [ ] Gravity removal method described (high-pass Butterworth, cutoff 0.3Hz)
- [ ] Example output showing all three tiers for one or more subjects
- [ ] The early detection argument with a concrete example
- [ ] Limitations:
  - Thresholds from literature were derived with different protocols (5-rep STS vs. 30s CST), sensor setups, and populations
  - Acceleration magnitude is a proxy for vertical acceleration — direct numeric comparison with literature is approximate
  - Alcázar power formula is adapted, not used in its published validated form
  - No direct validation against Fried phenotype scores (this is future work)
  - Gyroscope-dependent features unavailable with accel-only data
  - Fatigue slope threshold is exploratory — no published cut-off for this setup
  - Small sample sizes in external validation datasets

### Figures
- [ ] Per-rep feature plot: e.g., peak dynamic accel magnitude across reps for one subject, showing the fatigue slope line
- [ ] Three-tier output summary for one subject (could be a simple table or mockup)
- [ ] Optional: comparison of quality metrics between UCI HAPT (younger) and Marques (elderly) subjects

### Connection to Part 1
- [ ] Explicitly state: accurate rep segmentation from Part 1 is the prerequisite for all of Part 2
- [ ] If Part 1 misses a rep, you lose that rep's quality data
- [ ] If Part 1 hallucinates a rep, you get garbage quality values for a non-existent transition
- [ ] This is why rep detection accuracy (especially event-level) matters beyond just counting

---

## Timeline Relative to Part 1

**Part 2 depends entirely on Part 1 being done.** Don't start Part 2 until you have working event detection with reasonable accuracy.

**For the milestone (March 6):**
- Part 2 is NOT required — focus on Part 1
- If Part 1 is done early, implement Tier 1 (rep count) and Tier 2 (adapted Alcázar power) as a preview

**For the final report (March 17):**
- Full three-tier output with all six indicators
- Published threshold comparisons with match quality noted
- Early detection example
- Discussion of clinical implications and limitations
- "Educational, not diagnostic" framing throughout
