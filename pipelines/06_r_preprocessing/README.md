# Stage 06: R Final Preprocessing

Performs final data transformations, baseline normalization, and prepares datasets for machine learning and ANOVA.

## Entry Point
- **Main**: `preproccess_for_xgb.R`

## Inputs
- **File**: `output/aggregated/all_data_aggregated.csv` (from Stage 05)
- **Config**: `scripts/utils/config.yaml` (R-specific configuration)

## Outputs
- **File 1**: `output/final_data.rds` (for ML: SVM, XGBoost)
- **File 2**: `output/anova_features_precond.rds` (for ANOVA: pre-condition aggregation)
- **Logs**: Preprocessing steps, transformations applied, data summary

## Processing Steps
1. Load merged dataset
2. Apply baseline normalization (subtract baseline from task conditions)
3. Remove outliers (>3 SD from mean)
4. Feature scaling/standardization
5. Handle missing values (imputation or exclusion)
6. Split into ML-ready and ANOVA-ready formats
7. Save as R data structures (.rds)

## Output Formats

### final_data.rds
- **Structure**: Tidy dataframe with one row per observation
- **Use Case**: SVM and XGBoost training/testing
- **Features**: Baseline-normalized, scaled features
- **Labels**: stress_label (high/low), workload_label (high/low)

### anova_features_precond.rds
- **Structure**: Wide format with condition as separate columns
- **Use Case**: ANOVA and statistical comparisons
- **Features**: Pre-aggregated by condition
- **Labels**: Participant-level metadata

## Dependencies
- R 4.3+ with tidyverse, data.table

## Shared Utilities
Sources from `utils/r/`:
- `data_prep_helpers.R` - Baseline adjustment, outlier removal
- `transform_functions.R` - Scaling, normalization
- `save_helpers.R` - RDS export with metadata

## Usage
```r
Rscript preproccess_for_xgb.R
```

## Notes
- Critical preprocessing step before ML
- Ensures consistent feature scaling across models
- Preserves ANOVA structure for statistical analyses
