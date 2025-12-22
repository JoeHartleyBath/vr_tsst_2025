# Stage 09: R XGBoost Classification

XGBoost gradient boosting with leave-one-subject-out cross-validation.

## Entry Point
- **Main**: `xgboost_loso_classification.R`

## Inputs
- **File**: `output/final_data.rds` (from Stage 06)
- **Config**: `scripts/utils/config.yaml`

## Outputs
- **Directory**: `results/xgb/`
- **Files**:
  - `xgb_results_*.rds` (predictions and metrics)
  - `xgb_feature_importance_*.csv` (feature rankings)
  - `xgb_confusion_matrix_*.csv`
  - `xgb_hyperparams_*.rds` (tuned parameters)
  - Plots: ROC curves, feature importance bar charts

## Classification Tasks
- **Stress Label**: High stress vs Low stress
- **Workload Label**: High workload vs Low workload

## Method
- **Validation**: Leave-one-subject-out (LOSO)
- **Hyperparameter Tuning**: Bayesian optimization (via `utils/r/bayes_opt.R`)
- **Objective**: Binary logistic regression
- **Evaluation Metric**: AUC, accuracy, F1-score

## Hyperparameters Tuned
- **nrounds**: Number of boosting iterations (50-500)
- **max_depth**: Tree depth (3-10)
- **eta**: Learning rate (0.01-0.3)
- **subsample**: Row subsampling (0.5-1.0)
- **colsample_bytree**: Feature subsampling (0.5-1.0)
- **gamma**: Minimum loss reduction (0-5)

## Dependencies
- R 4.3+ with xgboost, caret, doParallel, pROC

## Shared Utilities
Sources from `utils/r/`:
- `data_prep_helpers.R` - Data loading
- `prune_feats.R` - Optional feature selection
- `bayes_opt.R` - Bayesian hyperparameter optimization
- `xgb_nested.R` - XGBoost-specific utilities
- `save_helpers.R` - Result export

## Usage
```r
Rscript xgboost_loso_classification.R
```

## Performance
- Processing time: 4-8 hours (Bayesian optimization is thorough)
- Generally outperforms SVM on this dataset
- Provides interpretable feature importance

## Notes
- XGBoost handles non-linear relationships and interactions well
- Feature importance useful for understanding physiological correlates
- Bayesian optimization more efficient than grid search
