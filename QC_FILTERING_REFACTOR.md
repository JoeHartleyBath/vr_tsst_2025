# QC Filtering Refactor: EEG-Only Exclusion

**Date:** December 19, 2025  
**Branch:** `feature/qc-eeg-only`  
**Status:** In Progress

---

## Objective

Modify the data preprocessing pipeline to apply QC-based participant exclusions **only** to EEG-dependent analyses, while retaining all 47 participants for subjective/behavioral analyses.

---

## Current Behavior (Baseline)

### QC Failures
Three participants flagged for EEG quality issues in `output/qc/qc_failures_summary.csv`:
- **P02**: EEG_HIGH_ICA_REMOVAL_30cs
- **P08**: EEG_HIGH_ICA_REMOVAL_21cs
- **P46**: EEG_HIGH_ICA_REMOVAL_23cs

### Current Implementation
**Location:** `utils/r/data_prep_helpers.R:12-24` in `load_and_prepare_data()`

```r
# 2. Drop QC failures
failed_ids <- readr::read_csv(
  file.path(config$paths$failed_qc, "qc_failures_summary.csv"),
  show_col_types = FALSE
) %>%
  dplyr::pull(Participant_ID) %>%
  unique()

# Convert "P02" format to numeric 2 for matching
failed_ids_numeric <- as.numeric(gsub("P", "", failed_ids))

data <- raw_data %>%
  dplyr::filter(!Participant_ID %in% failed_ids_numeric)
```

**Impact:**
- All downstream datasets (`output/final_data.csv`, `output/anova_features_precond.csv`) exclude P02, P08, P46
- Current N = 44 for all analyses (subjective and EEG)

---

## Target Behavior

### Principle
**QC exclusions apply ONLY when EEG features are involved in the analysis.**

### Implementation Strategy

1. **Load full dataset with QC flag** (`utils/r/data_prep_helpers.R`)
   - Add `qc_failed` boolean column instead of dropping rows
   - Preserve all 47 participants in base datasets

2. **Selective filtering in preprocessing** (`pipelines/06_r_preprocessing/preproccess_for_xgb.R`)
   - Create two data objects:
     - `final_data_full` (N=47): For subjective-only analyses
     - `final_data_eeg_valid` (N=44): For EEG-dependent analyses
   - Save both versions with clear naming

3. **Update analysis scripts**
   - **Subjective ANOVAs** (`scripts/analysis/subjective/subjective_anovas_and_plots.R`): Use N=47
   - **EEG ANOVAs** (`pipelines/10_r_anova/anova_main.R`): Filter to `qc_failed == FALSE`
   - **Correlations** (`scripts/analysis/correlations/*.R`):
     - Subjective-only correlations: N=47
     - EEG-involved correlations: N=44
   - **Machine learning** (`pipelines/08_r_svm/`, `pipelines/09_r_xgboost/`): Use EEG-valid subset

---

## Expected Outcomes

### Sample Sizes
| Analysis Type | Current N | New N | Change |
|---------------|-----------|-------|--------|
| Subjective ANOVAs (stress/workload ratings) | 44 | 47 | +3 |
| Subjective t-tests | 44 | 47 | +3 |
| Subjective-only correlations | 44 | 47 | +3 |
| EEG feature ANOVAs | 44 | 44 | No change |
| EEG-subjective correlations | 44 | 44 | No change |
| Machine learning (XGBoost, SVM) | 44 | 44 | No change |

### Benefits
- **Increased statistical power** for behavioral findings
- **Maintained data quality** for EEG analyses
- **Transparency**: Clear documentation of which analyses use which sample
- **Reproducibility**: Easy to verify N in each analysis from output logs

---

## Files to Modify

### Core Pipeline
1. ✅ `utils/r/data_prep_helpers.R` - Add QC flag logic
2. ✅ `pipelines/06_r_preprocessing/preproccess_for_xgb.R` - Create dual datasets

### Analysis Scripts
3. ✅ `scripts/analysis/subjective/subjective_anovas_and_plots.R` - Use full dataset
4. ✅ `pipelines/10_r_anova/anova_main.R` - Filter EEG features
5. ✅ `scripts/analysis/correlations/conditionwise_correlations.R` - Selective filtering
6. ✅ `scripts/analysis/correlations/stratified_rmcorr_correlations.R` - Selective filtering

### Validation
7. ✅ Create `validate_qc_filtering.R` to compare old vs. new results

---

## Validation Plan

1. **Backup current results** to `output/results_backup_pre_qc_refactor/`
2. **Run full preprocessing pipeline** with new code
3. **Compare outputs:**
   - Subjective ANOVA N should increase 44→47
   - EEG ANOVA N should remain 44
   - Correlation matrices should show correct N for each pairing
4. **Verify machine learning results unchanged** (should use same N=44 EEG-valid subset)
5. **Document any numerical differences** in effect sizes due to increased power

---

## Rollback Plan

If issues arise:
1. `git checkout eeg-refactor` (returns to pre-refactor state)
2. Results backed up in `output/results_backup_pre_qc_refactor/`
3. Stashed changes available: `git stash list`

---

## Next Steps

- [x] Create feature branch `feature/qc-eeg-only`
- [x] Document baseline behavior
- [ ] Implement core changes (data_prep_helpers.R, preproccess_for_xgb.R)
- [ ] Update analysis scripts
- [ ] Create validation script
- [ ] Run pipeline and compare results
- [ ] Commit with detailed message
- [ ] Merge to main branch after validation

---

**Log:**
- 2025-12-19 14:30: Branch created, baseline documented
