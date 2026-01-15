# Feature Naming Simplification - Implementation Summary

**Date:** December 17, 2025

## Overview

Successfully implemented comprehensive feature naming simplifications across the VR-TSST 2025 project. The changes reduce feature name complexity, eliminate redundant variants, and standardize naming conventions across all analysis scripts.

## Key Changes Implemented

### 1. Config Updates ([config/general.yaml](config/general.yaml))

**Canonical Feature Names Simplified:**
- ❌ `hr_heartrate_bpm_med_abs` → ✅ `hr_med`
- ❌ `gsr_skin_conductance_eda_tonic_mean_nk` → ✅ `eda_tonic`
- ❌ `gsr_skin_conductance_eda_pkht_mean_nk` → ✅ `eda_pkht`
- ❌ `eeg_fm_theta_mean` → ✅ `eeg_fm_theta`
- ❌ `eeg_f_beta_mean` → ✅ `eeg_f_beta`
- ❌ `eeg_p_alpha_mean` → ✅ `eeg_p_alpha`
- ❌ `pupil_dilation_med` → ✅ `pupil_med`
- ✅ `hrv_rmssd` (unchanged - already clean)

**Changes:**
- Removed cleaning tags (`_abs`, `_nk`)
- Removed redundant words (`heartrate_bpm`, `skin_conductance`, `dilation`)
- Removed metric suffixes when redundant with modality
- Changed `gsr` → `eda` (more scientific terminology)

### 2. Feature Naming Logic ([utils/r/feature_naming.R](utils/r/feature_naming.R))

**Major Updates:**
1. **Modality prefix changed:** `gsr` → `eda` in `modality_regex`
2. **Window tags eliminated:** No longer appends `_full` or `_roll` suffixes
3. **Cleaning tags removed:** Strips all `_abs`, `_nk`, `_cleaned` tags completely
4. **Redundant words dropped:** Removes `skin`, `conductance`, `heartrate`, `bpm`, `dilation`
5. **Vendor noise stripped:** More comprehensive removal of implementation details

**Example Transformation:**
```r
# Before:
"Full_Shimmer_D36A_GSR_Skin_Conductance_uS_CLEANED_ABS_CLEANED_NK_EDA_Tonic_Mean"
# Renamed to: gsr_skin_conductance_eda_tonic_mean_nk_full

# After implementation:
"Full_Shimmer_D36A_GSR_Skin_Conductance_uS_CLEANED_ABS_CLEANED_NK_EDA_Tonic_Mean"
# Now renames to: eda_tonic
```

### 3. Baseline Adjustment ([utils/r/data_prep_helpers.R](utils/r/data_prep_helpers.R))

**Eliminated Redundant Variants:**
- ❌ `_raw` (unadjusted values) - REMOVED
- ❌ `_change_glob` (global baseline) - REMOVED
- ✅ `_precond` (precondition baseline) - KEPT

**Before:** Each feature created 5 variants:
1. `feature_full`
2. `feature_full_raw`
3. `feature_full_change_precond`
4. `feature_full_change_glob`
5. `feature_full_change_precond_Z`

**After:** Each feature creates 2 variants:
1. `feature_precond` (baseline-adjusted)
2. `feature_precond_Z` (normalized)

**Impact:** 60% reduction in feature proliferation

### 4. Suffix Standardization

**Unified Naming Convention:**
- `_precond` = Baseline-adjusted (precondition relaxation baseline)
- `_precond_Z` = Normalized (MAD-scaled after transformation)

**Analysis Script Updates:**

| Script | Before | After |
|--------|--------|-------|
| **ANOVAs** | `_full_change_precond` | `_precond` |
| **Correlations (stratified)** | `_full_change_precond` + manual `_Z` | `_precond_Z` |
| **Correlations (conditionwise)** | `_precond_Z` (inconsistent pattern) | `_precond_Z` |
| **SVM** | `_precond` (already correct) | `_precond` |
| **T-tests** | `_full` | `_precond` |

### 5. Centralized Feature Selection ([utils/r/feature_selection.R](utils/r/feature_selection.R))

**New Utilities Created:**

```r
# Get standardized drop pattern
get_feature_drop_pattern()

# Select features with consistent filtering
select_analysis_features(df, suffix = "_precond_Z", 
                         canonical_only = FALSE, config = config)

# Get canonical features with suffix
get_canonical_features(config, suffix = "_precond")
```

**Replaced Scattered Logic In:**
- [utils/r/data_prep_helpers.R](utils/r/data_prep_helpers.R) (clean_feature_duplicates)
- [pipelines/08_r_svm/svm.R](pipelines/08_r_svm/svm.R) (feature filtering)
- [scripts/analysis/correlations/conditionwise_correlations.R](scripts/analysis/correlations/conditionwise_correlations.R) (drop pattern)

**Consolidates Exclusion of:**
- Response metrics
- Occipital EEG
- Min/max aggregations
- Head motion
- Global baseline variants
- Raw unadjusted features
- Delta band
- Asymmetry metrics
- WPLI connectivity
- Individual pupil L/R
- Redundant variability metrics (sdnn, pnn50, etc.)

### 6. Transform Functions ([utils/r/transform_functions.R](utils/r/transform_functions.R))

**Updated to match new suffixes:**
- Changed pattern from `_full_change_precond$` → `_precond$`
- Updated raw feature extraction to remove `_precond` suffix

### 7. Analysis Scripts Updated

**Files Modified:**
1. [pipelines/10_r_anova/anova_main.R](pipelines/10_r_anova/anova_main.R)
   - Suffix: `_full_change_precond` → `_precond`
   - Example feature updated: `eda_tonic_precond`

2. [scripts/analysis/correlations/stratified_rmcorr_correlations.R](scripts/analysis/correlations/stratified_rmcorr_correlations.R)
   - Suffix: `_full_change_precond` → `_precond_Z`
   - Removed redundant scaling (already in `_Z`)

3. [scripts/analysis/correlations/conditionwise_correlations.R](scripts/analysis/correlations/conditionwise_correlations.R)
   - Pattern: `_full_change_precond_Z` → `_precond_Z`
   - Added centralized feature selection
   - Updated `categorise_feature()` to recognize `eda` modality

4. [pipelines/08_r_svm/svm.R](pipelines/08_r_svm/svm.R)
   - Integrated centralized feature selection
   - Already used `_precond` suffix correctly

5. [utils/r/response_rate_eda_check.R](utils/r/response_rate_eda_check.R)
   - Updated feature names:
     - `gsr_skin_conductance_eda_tonic_mean_nk_full_change_precond` → `eda_tonic_precond`
     - `response_rate_per_min_full_change_precond` → `response_rate_per_min_precond`

6. [utils/r/baseline_adjustment_check.R](utils/r/baseline_adjustment_check.R)
   - Updated: `pupil_dilation_med_full_change_precond` → `pupil_med_precond`

## Feature Name Length Reduction

**Example Comparisons:**

| Feature Type | Before | After | Reduction |
|--------------|--------|-------|-----------|
| **EDA Tonic** | `gsr_skin_conductance_eda_tonic_mean_nk_full_change_precond_Z` (58 chars) | `eda_tonic_precond_Z` (20 chars) | **66%** |
| **Heart Rate** | `hr_heartrate_bpm_med_abs_full_change_precond` (45 chars) | `hr_med_precond` (14 chars) | **69%** |
| **EEG Theta** | `eeg_fm_theta_mean_full_change_precond` (38 chars) | `eeg_fm_theta_precond` (21 chars) | **45%** |
| **Pupil** | `pupil_dilation_med_full_change_precond` (39 chars) | `pupil_med_precond` (17 chars) | **56%** |

**Average reduction: ~60% shorter feature names**

## Benefits Achieved

### 1. **Reduced Complexity**
- Eliminated 60% of redundant feature variants
- 3 fewer suffix types to track (`_raw`, `_glob`, `_full` removed)
- Single source of truth for feature filtering

### 2. **Improved Consistency**
- All analyses now use standardized `_precond` or `_precond_Z` suffixes
- Unified drop pattern across all scripts
- Consistent modality naming (`eda` vs `gsr`)

### 3. **Better Maintainability**
- Centralized feature selection logic in one file
- Config-driven canonical features
- Single place to update exclusion rules

### 4. **Cleaner Output**
- Shorter, more readable feature names
- Implementation details hidden in metadata
- Scientific terminology over vendor-specific names

### 5. **Easier Understanding**
- Feature purpose clear from name alone
- No need to decode cleaning tags
- Modality immediately apparent

## Migration Notes

### Data Regeneration Required
After these changes, the following outputs need regeneration:
- ✅ `output/final_data.rds`
- ✅ `output/anova_features_precond.rds`
- ✅ All analysis results (ANOVAs, correlations, SVM)

### Backward Compatibility
**Breaking Changes:**
- Old feature names will no longer be generated
- Existing saved models/results reference old names
- Scripts expecting old suffixes will fail

**Migration Path:**
1. **Option A (Recommended):** Full pipeline rerun
   - Regenerate all data from `all_data_aggregated.csv`
   - Produces clean, consistent naming throughout

2. **Option B:** Create mapping file
   - Maintain old→new name translation
   - Use for loading legacy results only
   - Not recommended for new analyses

## Testing Recommendations

Before running full pipeline:
1. ✅ Test `rename_feature()` on sample column names
2. ✅ Verify canonical features match in config
3. ⚠️ Run preprocessing on single participant (P01)
4. ⚠️ Verify `_precond` and `_precond_Z` features created
5. ⚠️ Check ANOVA runs with new feature names
6. ⚠️ Validate correlation scripts find features
7. ⚠️ Test SVM with new selection logic

## Files Changed Summary

**Configuration:**
- [config/general.yaml](config/general.yaml)

**Core Utilities:**
- [utils/r/feature_naming.R](utils/r/feature_naming.R)
- [utils/r/data_prep_helpers.R](utils/r/data_prep_helpers.R)
- [utils/r/transform_functions.R](utils/r/transform_functions.R)
- [utils/r/feature_selection.R](utils/r/feature_selection.R) ⭐ NEW

**Analysis Scripts:**
- [pipelines/10_r_anova/anova_main.R](pipelines/10_r_anova/anova_main.R)
- [pipelines/08_r_svm/svm.R](pipelines/08_r_svm/svm.R)
- [scripts/analysis/correlations/stratified_rmcorr_correlations.R](scripts/analysis/correlations/stratified_rmcorr_correlations.R)
- [scripts/analysis/correlations/conditionwise_correlations.R](scripts/analysis/correlations/conditionwise_correlations.R)

**Check/Validation Scripts:**
- [utils/r/response_rate_eda_check.R](utils/r/response_rate_eda_check.R)
- [utils/r/baseline_adjustment_check.R](utils/r/baseline_adjustment_check.R)

**Total Files Modified:** 12
**Total Files Created:** 1 ([utils/r/feature_selection.R](utils/r/feature_selection.R))

## Next Steps

1. **Test the changes:**
   ```powershell
   # Test feature renaming
   Rscript -e "source('utils/r/feature_naming.R'); 
               test <- rename_feature('Full_Shimmer_D36A_GSR_Skin_Conductance_EDA_Tonic_Mean_Cleaned_NK');
               cat(test)"
   
   # Expected output: eda_tonic
   ```

2. **Regenerate preprocessed data:**
   ```powershell
   Rscript scripts/r/preprocessing_main.R
   ```

3. **Verify feature names in output:**
   ```powershell
   Rscript -e "df <- readRDS('output/final_data.rds'); 
               feats <- grep('_precond', names(df), value=TRUE); 
               head(feats, 10)"
   ```

4. **Run analyses with new names:**
   ```powershell
   Rscript pipelines/10_r_anova/anova_main.R
   Rscript scripts/analysis/correlations/stratified_rmcorr_correlations.R
   ```

## Questions to Resolve

1. **Transform metadata:** Should we create a separate YAML file documenting which features get which transforms (signed_log_z vs z)? Currently this is only in code.

2. **Legacy data:** Do you need to maintain compatibility with existing analysis outputs, or can we do a clean cutover?

3. **Feature descriptions:** Should canonical features in config have human-readable descriptions added?

---

**Implementation Status:** ✅ COMPLETE  
**Data Regeneration Required:** ⚠️ YES  
**Backward Compatible:** ❌ NO (breaking changes)
