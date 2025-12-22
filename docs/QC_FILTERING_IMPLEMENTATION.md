# QC Filtering Implementation Guide

**Last Updated:** December 19, 2025  
**Branch:** `feature/qc-eeg-only`

---

## Overview

This document explains how QC-based participant exclusions are applied in the VR-TSST analysis pipeline following the refactor to selective filtering.

## Principle

**QC failures are excluded ONLY when EEG features are involved in the analysis.**

- **Non-EEG analyses** (subjective ratings, behavioral measures): Use full N=47
- **EEG-dependent analyses** (EEG features, mixed EEG-subjective): Use N=44 (QC-valid subset)

---

## QC Failures

Three participants flagged for excessive ICA component removal (>20 components):

| Participant | Reason | Status |
|-------------|--------|--------|
| P02 | EEG_HIGH_ICA_REMOVAL_30cs | Retained for subjective analyses |
| P08 | EEG_HIGH_ICA_REMOVAL_21cs | Retained for subjective analyses |
| P46 | EEG_HIGH_ICA_REMOVAL_23cs | Retained for subjective analyses |

---

## Implementation Details

### 1. Data Loading (`utils/r/data_prep_helpers.R`)

```r
# Adds qc_failed flag instead of dropping participants
data <- raw_data %>%
  dplyr::mutate(
    qc_failed = Participant_ID %in% failed_ids_numeric
  )
```

**Key Change:** Participants are flagged, not filtered.

### 2. Dataset Creation (`pipelines/06_r_preprocessing/preproccess_for_xgb.R`)

Two datasets are created:

1. **`final_data.csv/rds`** (N=47)
   - Contains `qc_failed` column
   - Use for: Subjective ANOVAs, subjective-only correlations, EDA

2. **`final_data_eeg_valid.csv/rds`** (N=44)
   - Filters `!qc_failed`
   - Use for: EEG ANOVAs, ML pipelines, mixed correlations

### 3. Analysis-Specific Filtering

#### Subjective ANOVAs (`scripts/analysis/subjective/subjective_anovas_and_plots.R`)
```r
# Uses final_data.csv (N=47) directly - no filtering
df <- readr::read_csv("output/final_data.csv")
```
**N=47** (all participants)

#### EEG ANOVAs (`pipelines/10_r_anova/anova_main.R`)
```r
df <- readRDS("output/anova_features_precond.rds")
if ("qc_failed" %in% names(df)) {
  df <- df %>% filter(!qc_failed)
}
```
**N=44** (QC-valid only)

#### Correlations (`scripts/analysis/correlations/*.R`)
```r
# Load full dataset
df_full <- load_obj("final_data")

# Selective filtering per feature
if (feature %in% eeg_features) {
  df_for_corr <- df_full %>% filter(!qc_failed)  # N=44
} else {
  df_for_corr <- df_full  # N=47
}
```
**Mixed N:** 47 for non-EEG features, 44 for EEG features

#### Machine Learning (`pipelines/08_r_svm/`, `pipelines/09_r_xgboost/`)
```r
# Use EEG-valid subset
data <- readRDS("output/final_data_eeg_valid.rds")
```
**N=44** (QC-valid only)

---

## How to Identify Which Dataset to Use

| Analysis Type | Use Dataset | N | Rationale |
|---------------|-------------|---|-----------|
| Subjective ratings ANOVA | `final_data.csv` | 47 | No EEG features |
| Subjective t-tests | `final_data.csv` | 47 | No EEG features |
| Subjective-only correlations | `final_data.csv` | 47 | No EEG features |
| EEG feature ANOVA | `anova_features_precond.rds` (filtered) | 44 | All EEG features |
| EEG-subjective correlations | `final_data.csv` (filtered per feature) | 44 | EEG quality affects results |
| ML classification | `final_data_eeg_valid.csv` | 44 | EEG features in model |
| Descriptive stats (demographics) | `final_data.csv` | 47 | No EEG involvement |

---

## Verification

### Check Sample Size in Your Analysis

```r
# At the start of your script:
message(sprintf("Analysis using N=%d participants", n_distinct(data$participant_id)))

# Check QC status distribution:
if ("qc_failed" %in% names(data)) {
  table(data$qc_failed)  # Should show FALSE=44, TRUE=3 for full dataset
}
```

### Expected Ns
- **Full dataset:** 47 participants total (44 pass + 3 fail QC)
- **EEG-valid:** 44 participants (QC failures excluded)

---

## Common Patterns

### Load Full Dataset
```r
df <- readr::read_csv("output/final_data.csv")
# or
df <- readRDS("output/final_data.rds")
```

### Load EEG-Valid Subset
```r
df <- readr::read_csv("output/final_data_eeg_valid.csv")
# or
df <- readRDS("output/final_data_eeg_valid.rds")
```

### Selective Filtering (for mixed analyses)
```r
eeg_features <- names(df)[str_detect(names(df), "^eeg_")]

# Inside correlation/analysis loop:
if (feature %in% eeg_features) {
  df_subset <- df %>% filter(!qc_failed)
} else {
  df_subset <- df  # Use full N
}
```

---

## Logging Best Practices

Always log sample sizes for transparency:

```r
message(sprintf("[Analysis Name] Using N=%d participants", n_distinct(df$participant_id)))

if ("qc_failed" %in% names(df)) {
  n_failed <- sum(df %>% distinct(participant_id, qc_failed) %>% pull(qc_failed))
  message(sprintf("[Analysis Name] Dataset includes %d QC failures", n_failed))
}
```

---

## Troubleshooting

### Problem: "Too few participants" error
**Solution:** Check if you're using the correct dataset. Subjective analyses should use `final_data.csv` (N=47), not `final_data_eeg_valid.csv` (N=44).

### Problem: Missing `qc_failed` column
**Solution:** Re-run preprocessing: `Rscript pipelines/06_r_preprocessing/preproccess_for_xgb.R`

### Problem: Inconsistent sample sizes across analyses
**Solution:** Verify each script explicitly loads the appropriate dataset and logs N at the start.

---

## References

- QC failures list: `output/qc/qc_failures_summary.csv`
- Preprocessing pipeline: `pipelines/06_r_preprocessing/preproccess_for_xgb.R`
- Data loading functions: `utils/r/data_prep_helpers.R`
- Refactor documentation: `QC_FILTERING_REFACTOR.md`
