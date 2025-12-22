# Required Features Analysis for R Preprocessing Pipeline

## Summary
Analysis of `preproccess_for_xgb.R` and helper functions to identify essential features needed.

## Key Findings

### 1. Feature Filtering Rules (from clean_feature_duplicates)

**DROPPED Features (garbage_patterns):**
- `_wpli` - Connectivity measures
- `aperiodic` - Aperiodic EEG components
- `_dilation_(left|right)` - Individual eye dilations (averaged to bilateral)
- `rr_` - RR interval features
- `slope` - Slope features  
- `meaningful` - Meaningful SCR counts
- `unrest` - Unrest power
- `blink` - Blink features
- `resistance` - GSR resistance (conductance used instead)
- `bpm_sd` - BPM standard deviation
- `interval_sd` - Interval standard deviation
- `sdnn` - SDNN HRV metric
- `pnn50` - pNN50 HRV metric
- `conductance(?!_eda)` - Raw conductance (EDA-processed version kept)
- `(?=.*eeg)(?=.*sd)` - EEG SD features

**DROPPED Duplicates:**
- Features with `_abs_` or `_nk_` suffixes when raw equivalent exists

### 2. Required Feature Categories

#### **A. Heart Rate & HRV (KEEP)**
- `Full_Polar_HeartRate_BPM_Mean` ✓
- `Full_Polar_HeartRate_BPM_Median` ✓  
- `Full_Polar_HeartRate_BPM_MIN` ✓
- `Full_Polar_HeartRate_BPM_MAX` ✓
- `Full_RMSSD` ✓ (canonical feature)
- ~~`Full_SDNN`~~ ❌ (dropped by garbage_patterns)
- ~~`Full_pNN50`~~ ❌ (dropped by garbage_patterns)
- ~~`Full_Polar_HeartRate_BPM_SD`~~ ❌ (dropped by bpm_sd pattern)
- ~~`Full_Polar_HeartRate_RR_Interval_*`~~ ❌ (dropped by rr_ pattern)

#### **B. GSR/EDA (KEEP)**
- `Full_Shimmer_GSR_Skin_Conductance_uS_CLEANED_ABS_CLEANED_NK_Median` ✓
- `Full_Shimmer_GSR_Skin_Conductance_uS_CLEANED_ABS_CLEANED_NK_Mean` ✓
- `Full_Shimmer_GSR_Skin_Conductance_uS_CLEANED_ABS_CLEANED_NK_SD` ✓
- `Full_Shimmer_GSR_Skin_Conductance_uS_CLEANED_ABS_CLEANED_NK_MIN` ✓
- `Full_Shimmer_GSR_Skin_Conductance_uS_CLEANED_ABS_CLEANED_NK_MAX` ✓
- `Full_Shimmer_GSR_Skin_Conductance_uS_CLEANED_ABS_CLEANED_NK_EDA_Tonic_Mean` ✓ (canonical)
- `Full_Shimmer_GSR_Skin_Conductance_uS_CLEANED_ABS_CLEANED_NK_EDA_Tonic_SD` ✓
- `Full_Shimmer_GSR_Skin_Conductance_uS_CLEANED_ABS_CLEANED_NK_EDA_PeakRate` ✓
- `Full_Shimmer_GSR_Skin_Conductance_uS_CLEANED_ABS_CLEANED_NK_EDA_PeakHeight_Mean` ✓ (canonical)
- `Full_Shimmer_GSR_Skin_Conductance_uS_CLEANED_ABS_CLEANED_NK_EDA_PeakHeight_Max` ✓
- `Full_Shimmer_GSR_Skin_Conductance_uS_CLEANED_ABS_CLEANED_NK_EDA_PeakHeight_Median` ✓
- `Full_Shimmer_GSR_Skin_Conductance_uS_CLEANED_ABS_CLEANED_NK_EDA_PeakArea` ✓
- `Full_Shimmer_GSR_Skin_Conductance_uS_CLEANED_ABS_CLEANED_NK_EDA_TotalSCRs` ✓
- ~~`Full_Shimmer_GSR_Skin_Conductance_uS_CLEANED_ABS_CLEANED_NK_EDA_MeaningfulSCRs`~~ ❌ (meaningful)
- ~~`Full_Shimmer_GSR_Skin_Conductance_uS_CLEANED_ABS_CLEANED_NK_EDA_ProportionMeaningful`~~ ❌ (meaningful)

#### **C. Pupil Dilation (KEEP - with transformations)**
- `Full_Foveal_Corrected_Dilation_Left_CLEANED_ABS_Mean` → used to compute bilateral ✓
- `Full_Foveal_Corrected_Dilation_Right_CLEANED_ABS_Mean` → used to compute bilateral ✓
- **Computed features (in R pipeline):**
  - `Full_Pupil_Dilation_Mean` = average of left/right ✓
  - `Full_Pupil_Dilation_Min` = average of left/right min ✓
  - `Full_Pupil_Dilation_Max` = average of left/right max ✓
  - `Full_Pupil_Dilation_Median` = average of left/right median ✓
  - `Full_Pupil_Dilation_SD` = average of left/right SD ✓
  - `Full_Pupil_Asymmetry` = abs(left - right) ✓
- ~~Individual left/right features~~ ❌ (dropped after bilateral computation)
- ~~`Full_Pupil_UnrestPower`~~ ❌ (dropped by unrest pattern)
- ~~`Full_Current_Blink_Duration_*`~~ ❌ (dropped by blink pattern)
- ~~`Full_Inter_Blink_Interval_*`~~ ❌ (dropped by blink pattern)

#### **D. EEG Features (KEEP - Mean only, NO SD)**
**Band Power (Mean only):**
- All `*_Delta_Mean` ✓
- All `*_Theta_Mean` ✓ (FM_Theta canonical)
- All `*_LowAlpha_Mean` ✓
- All `*_HighAlpha_Mean` ✓
- All `*_Alpha_Mean` ✓ (Parietal canonical)
- All `*_LowBeta_Mean` ✓
- All `*_HighBeta_Mean` ✓
- All `*_Beta_Mean` ✓ (Frontal canonical)

**Entropy Features (KEEP):**
- All `*_SampleEntropy` ✓
- All `*_SpectralEntropy` ✓

**Ratio Features (Computed in R):**
- `Full_Alpha_Beta_Ratio` = FM_Alpha / FM_Beta ✓
- `Full_Theta_Beta_Ratio` = FM_Theta / FM_Beta ✓
- `Full_Frontal_Alpha_Asymmetry` = log(Right) - log(Left) ✓

**DROPPED EEG Features:**
- ~~All `*_SD` features~~ ❌ (dropped by (?=.*eeg)(?=.*sd) pattern)
- ~~All `*_Slope` features~~ ❌ (dropped by slope pattern)
- ~~All `*_aperiodic_*` features~~ ❌ (dropped by aperiodic pattern)
- ~~All `*_wpli_*` features~~ ❌ (dropped by wpli pattern)

### 3. Canonical Features (Priority)
From `config/general.yaml` - these are the key features:
1. `hrv_rmssd` - HRV RMSSD
2. `hr_heartrate_bpm_med_abs` - Heart Rate (Median BPM)
3. `gsr_skin_conductance_eda_tonic_mean_nk` - EDA Tonic
4. `gsr_skin_conductance_eda_pkht_mean_nk` - EDA Peak Height
5. `eeg_fm_theta_mean` - EEG FM Theta
6. `eeg_f_beta_mean` - EEG Beta
7. `eeg_p_alpha_mean` - EEG Parietal Alpha
8. `pupil_dilation_med` - Pupil Dilation

### 4. Required Metadata Columns
- `Participant_ID` ✓
- `Condition` ✓
- `Round` (from counterbalance mapping) ✓
- `Stress` (subjective rating) ✓
- `Workload` (subjective rating) ✓

### 5. Processing Pipeline Requirements

**Step 1: Condition Mapping**
- Rename conditions to simplified format:
  - `HighStress_HighCog*_Task` → `High Stress - High Cog`
  - `HighStress_LowCog*_Task` → `High Stress - Low Cog`
  - `LowStress_HighCog*_Task` → `Low Stress - High Cog`
  - `LowStress_LowCog*_Task` → `Low Stress - Low Cog`

**Step 2: Baseline Computation**
- Extract Pre-exposure baselines (Pre_Exposure_*_Baseline)
- Extract Relaxation (Forest1-4) as pre-condition baselines
- Compute: `change_precond = task_value - relaxation_baseline`

**Step 3: Feature Naming**
- All features need `_full` suffix
- Applied features get `_full_change_precond` suffix

**Step 4: Transformation**
- Only `*_change_precond` features are transformed
- Uses MAD-based robust scaling

## Implementation Priority

### Phase 1: MVP (Minimal Viable Pipeline)
1. ✓ EEG features (127 columns) - DONE
2. Subjective ratings (2 columns: Stress, Workload)
3. Essential HR features (5 columns)
4. Essential GSR/EDA features (13 columns)
5. Essential Pupil features (6 computed bilateral features)

**Total Phase 1: ~153 columns**

### Phase 2: Full Pipeline  
1. Raw physiological data loading
2. HR cleaning (artifact removal, outlier detection)
3. GSR cleaning (NK2 eda_clean + threshold cleaning)
4. Pupil cleaning (blink removal, bilateral averaging)
5. Feature extraction per condition
6. Merge with EEG + subjective

### Phase 3: Validation
1. Load into R preproccess_for_xgb.R
2. Verify baseline adjustment works
3. Verify transformation works
4. Verify final_data.csv produced successfully

## Minimum Required Feature Set

To make R script work with minimal features:

```python
essential_features = {
    # Metadata
    'Participant_ID': int,
    'Condition': str,
    
    # Subjective
    'Stress': float,
    'Workload': float,
    
    # HR (5 features)
    'Full_Polar_HeartRate_BPM_Mean': float,
    'Full_Polar_HeartRate_BPM_Median': float,
    'Full_Polar_HeartRate_BPM_MIN': float,
    'Full_Polar_HeartRate_BPM_MAX': float,
    'Full_RMSSD': float,
    
    # GSR (13 features)
    'Full_Shimmer_GSR_*_CLEANED_ABS_CLEANED_NK_[Mean|Median|SD|MIN|MAX]': float,
    'Full_Shimmer_GSR_*_EDA_Tonic_[Mean|SD]': float,
    'Full_Shimmer_GSR_*_EDA_Peak*': float,
    
    # Pupil (6 bilateral features - computed in merge step)
    'Full_Pupil_Dilation_[Mean|Min|Max|Median|SD]': float,
    'Full_Pupil_Asymmetry': float,
    
    # EEG (127 features from MATLAB extraction)
    # All *_Power, *_Entropy, ratio features
}
```

## Recommendations

1. **Start with canonical features only** - Get 8 features working end-to-end first
2. **Test with placeholder data** - Verify R script accepts the structure
3. **Build cleaning modules incrementally** - HR → GSR → Pupil
4. **Validate against legacy** - Compare feature values on P01
5. **Skip deprecated features** - Don't implement features that get dropped

This focuses effort on ~150 essential features instead of 563.
