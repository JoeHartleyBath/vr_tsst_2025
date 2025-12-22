# Feature Computation Analysis: R Pipeline

## Summary
The R preprocessing script (`preproccess_for_xgb.R`) performs significant **feature engineering** that should be moved to Python feature extraction for proper separation of concerns.

---

## Current R Pipeline Operations

### 1. **Feature Computation in `load_and_prepare_data()`**
**Location:** `scripts/utils/data_prep_helpers.R` lines 44-99

#### Computed Features:
```r
# Pupil features (aggregation)
Full_Pupil_Dilation_Mean = (Left + Right) / 2
Full_Pupil_Dilation_Median = (Left + Right) / 2
Full_Pupil_Dilation_SD = (Left + Right) / 2
Full_Pupil_Asymmetry = abs(Left - Right)
# Note: MIN/MAX are computed but immediately dropped by clean_feature_duplicates()

# EEG ratio features
Full_Alpha_Beta_Ratio = Alpha / Beta (FrontalMidline)
Full_Theta_Beta_Ratio = Theta / Beta (FrontalMidline)
Full_Frontal_Alpha_Asymmetry = log(Right_Alpha) - log(Left_Alpha)
```

**Action Required:** Move to Python feature extraction
- ✅ EEG ratios already in MATLAB (extract_eeg_features_refactored.m lines computing ratios)
- ❌ Pupil aggregation currently in R - should move to Python
- ❌ Pupil asymmetry currently in R - should move to Python

---

### 2. **Baseline Normalization in `prepare_full_window_data()`**
**Location:** `scripts/utils/data_prep_helpers.R` lines 109-177

#### Operations:
```r
# Compute two types of baselines
precond_bl = Relaxation period (per round)
glob_bl = Pre_Baseline period (global)

# Compute deltas for each feature
change_precond = task_value - precond_bl
change_glob = task_value - glob_bl

# Creates 3 variants per feature:
- feature_full_raw
- feature_full_change_precond  
- feature_full_change_glob
```

**Analysis:**
- This is **data preprocessing**, not feature extraction
- Appropriate to keep in R (preprocessing layer)
- Python should provide raw features; R computes deltas

**Decision:** ✅ Keep in R - this is preprocessing logic

---

### 3. **Z-Score Normalization in `transform_deltas()`**
**Location:** `scripts/utils/transform_functions.R` lines 41-91

#### Operations:
```r
# Per participant, per feature:
1. Apply transform (log, signed_log, or none)
2. Compute MAD-based robust z-score:
   Z = (x - median) / MAD

# Capped at [-3, +3]
```

**Analysis:**
- This is **data scaling**, not feature extraction
- Standard ML preprocessing step
- Appropriate to keep in R

**Decision:** ✅ Keep in R - this is preprocessing logic

---

### 4. **Feature Cleanup in `clean_feature_duplicates()`**
**Location:** `scripts/utils/data_prep_helpers.R` lines 238-281

#### Dropped features:
- Features with `_ABS_` or `_NK_` suffixes (if raw version exists)
- Connectivity features (`_wpli`, `aperiodic`)
- Per-channel pupil (`_left`, `_right`)
- RR interval features
- Slope features
- Blink features  
- Resistance features
- SD features for HR/RR
- HRV features (SDNN, pNN50)
- Skin conductance (unless part of `_eda`)

**Analysis:**
- This is **feature selection**, not computation
- Appropriate in R preprocessing
- But Python should NOT generate features that will be dropped

**Decision:** ✅ Keep in R, but **update Python to not generate dropped features**

---

## Required Changes to Python Pipeline

### Priority 1: Move Feature Computation from R to Python

#### A. **Pupil Features** (Currently in R line 76-86)
**New module:** `scripts/physio_features/private/compute_pupil_features.py`

```python
def compute_pupil_aggregates(left_series, right_series):
    """Aggregate left/right pupil into bilateral features"""
    return {
        'Pupil_Dilation_Mean': (left + right) / 2,
        'Pupil_Dilation_Median': (left_med + right_med) / 2,
        'Pupil_Dilation_SD': (left_sd + right_sd) / 2,
        'Pupil_Asymmetry': abs(left_mean - right_mean)
    }
    # Note: MIN/MAX explicitly dropped by R scripts - don't compute
```

#### B. **EEG Ratio Features** (Currently in R line 89-97)
**Location:** Already in MATLAB - verify output

EEG ratios should be in `eeg_features.csv`:
- `Frontal_Alpha_Asymmetry`
- `Alpha_Beta_Ratio`
- `Theta_Beta_Ratio`
- `RightFrontal_Alpha`
- `Theta_Alpha_Ratio`

**Action:** Verify these columns exist in current extraction

---

### Priority 2: Optimize Feature Generation

#### Features to NOT Generate (will be dropped by R)

**Summary statistics to skip:**
- ❌ **MIN features** - explicitly dropped by all ML scripts (XGBoost, SVM, correlations)
- ❌ **MAX features** - explicitly dropped by all ML scripts  
- ✅ Keep: Mean, Median, SD only

**Physio cleaning:**
- ❌ Don't create `_ABS_` suffixed columns if raw version exists
- ❌ Don't create `_NK_` suffixed columns if raw version exists
- ❌ Don't create per-eye pupil features (Left/Right separately)
- ❌ Don't create RR interval features
- ❌ Don't create SDNN, pNN50 (HRV - but keep RMSSD!)
- ❌ Don't create GSR resistance features
- ❌ Don't create skin conductance unless it's `_eda` variant

**EEG features:**
- ❌ Don't create slope features
- ❌ Don't create SD features (except ratios)
- ❌ Don't create connectivity features (wpli, aperiodic)

**Eye tracking:**
- ❌ Don't create blink features
- ❌ Don't create UnrestPower (spectral feature - marked as "garbage")

**Drop patterns from ML scripts:**
```r
# From xgb_nested.R, svm.R, conditionwise_correlations.R:
drop_pattern = "(_min_|_max_)|response|eeg_o|head|totalscrs|_glob|_raw|_delta"
```

---

### Priority 3: Essential Features to Generate

Based on `canonical_features` from config and R pipeline:

#### **HRV (HR cleaning):**
✅ `Full_RMSSD` - root mean square of successive differences

#### **HR Features:**
✅ `Full_Polar_HeartRate_BPM_Mean`
✅ `Full_Polar_HeartRate_BPM_Median` (canonical feature)
✅ `Full_Polar_HeartRate_BPM_SD`
❌ `Full_Polar_HeartRate_BPM_MIN` - DROPPED by all ML scripts
❌ `Full_Polar_HeartRate_BPM_MAX` - DROPPED by all ML scripts

#### **GSR Features:**
✅ `Full_Shimmer_GSR_Skin_Conductance_EDA_Tonic_Mean` (NeuroKit2 processed - canonical)
✅ `Full_Shimmer_GSR_Skin_Conductance_EDA_PeakHeight_Mean` (SCR - canonical)
✅ `Full_Shimmer_GSR_Mean` (raw)
✅ `Full_Shimmer_GSR_Median`
✅ `Full_Shimmer_GSR_SD`
❌ `Full_Shimmer_GSR_MIN` - DROPPED by all ML scripts
❌ `Full_Shimmer_GSR_MAX` - DROPPED by all ML scripts

#### **Pupil Features:**
✅ `Full_Pupil_Dilation_Mean` (aggregated)
✅ `Full_Pupil_Dilation_Median` (canonical feature)
✅ `Full_Pupil_Dilation_SD`
✅ `Full_Pupil_Asymmetry`
❌ `Full_Pupil_Dilation_MIN` - DROPPED by all ML scripts
❌ `Full_Pupil_Dilation_MAX` - DROPPED by all ML scripts
❌ `Full_Pupil_UnrestPower` - DROPPED as "garbage" feature

#### **EEG Features:**
All features from `eeg_features.csv` (127 columns) - already generated

---

## Recommended Python Architecture

### Module Structure
```
scripts/physio_features/
├── extract_physio_features.py          # Main CLI
├── private/
│   ├── __init__.py
│   ├── load_data.py                    # Data loading (already exists)
│   ├── clean_hr_data.py                # HR artifact removal → RMSSD, BPM stats
│   ├── clean_gsr_data.py               # GSR filtering → EDA decomposition  
│   ├── clean_eye_data.py               # Blink removal → pupil aggregates
│   ├── compute_hr_features.py          # Extract HR features per condition
│   ├── compute_gsr_features.py         # Extract GSR features per condition
│   ├── compute_pupil_features.py       # Extract pupil features per condition
│   └── merge_features.py               # Combine all modalities
```

### Data Flow
```
1. Raw Data (data/raw/metadata/P*.csv)
   ↓
2. Clean Data (HR, GSR, Pupil) 
   - Artifact removal
   - Outlier detection
   - Signal processing
   ↓
3. Extract Features (per condition)
   - Time domain (mean, median, min, max)
   - Frequency domain (UnrestPower for pupil)
   - HRV metrics (RMSSD only)
   - EDA decomposition (tonic, phasic, SCRs)
   ↓
4. Merge with EEG + Subjective
   - Load eeg_features.csv
   - Load subjective ratings
   - Align by Participant_ID + Condition
   ↓
5. Output: all_data_aggregated.csv
   Format: [Participant_ID, Condition, <features>, Stress, Workload]
   ↓
6. R Preprocessing
   - Compute baseline deltas (change_precond, change_glob)
   - Z-score normalization
   - Feature selection
   ↓
7. Final: final_data.csv (ready for XGBoost)
```

---

## Implementation Plan

### Phase 1: Minimal Viable Product (MVP)
**Goal:** Get end-to-end pipeline working on P01

1. ✅ EEG features extracted (DONE)
2. ⏳ Create simplified HR features (mean, median, min, max, RMSSD)
3. ⏳ Create simplified GSR features (mean, median, min, max, EDA)
4. ⏳ Create simplified Pupil features (aggregated bilaterally)
5. ⏳ Merge all features + subjective ratings
6. ⏳ Validate with R preprocessing script

### Phase 2: Full Pipeline
**Goal:** Production-ready pipeline with proper cleaning

1. Implement full HR cleaning (artifact removal, outlier detection)
2. Implement full GSR cleaning (NeuroKit2 EDA decomposition)
3. Implement full eye tracking cleaning (blink removal, interpolation)
4. Add frequency domain features (UnrestPower)
5. Add proper error handling and logging
6. Add data quality checks

### Phase 3: Optimization
**Goal:** Efficient, maintainable code

1. Parallelize feature extraction across participants
2. Add caching for intermediate steps
3. Add comprehensive tests
4. Document all functions
5. Create user-facing CLI

---

## Key Principles

1. **Separation of Concerns:**
   - Python: Data cleaning + Feature extraction
   - R: Preprocessing (baseline adjustment, normalization, selection)

2. **Don't Generate Redundant Features:**
   - Check R's `clean_feature_duplicates()` before implementing
   - Only create features that will actually be used

3. **Match Output Format:**
   - Column names must match R expectations exactly
   - Use `Full_` prefix for full-window features
   - Use proper naming: `Polar_HeartRate_BPM_Mean` not `hr_mean`

4. **Testable Components:**
   - Each module should be independently testable
   - Validate against legacy notebook outputs
   - Use P01 as test case throughout

---

## Next Steps

1. **Verify EEG ratios** are in current `eeg_features.csv`
2. **Create MVP merge script** to test R pipeline integration
3. **Implement HR features** (simplified first, then full cleaning)
4. **Implement GSR features** (using NeuroKit2)
5. **Implement Pupil features** (bilateral aggregation)
6. **Test end-to-end** on P01
7. **Expand to all participants** once validated
