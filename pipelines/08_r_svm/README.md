# Stage 08: R SVM (Nested LOSO)

Nested leave-one-subject-out cross-validation with hyperparameter tuning for stress and workload classification.

## Entry Points
- **Main**: `svm.R` (nested LOSO with e1071 package)
- **Optimizer**: `svm_best_model_finder.R` (find optimal hyperparameters)

## Inputs
- **File**: `output/final_data.rds` (from Stage 06)
- **Config**: `scripts/utils/config.yaml`

## Outputs
- **Directory**: `results/svm/`
- **Files**:
  - `svm_progress_*.csv` (iteration results)
  - `svm_hyperparams_*.rds` (best hyperparameters per fold)
  - `svm_predictions_*.rds` (predicted labels)
  - `svm_summary_*.txt` (overall performance)

## Classification Tasks
- **Stress Label**: High stress vs Low stress
- **Workload Label**: High workload vs Low workload

## Method
- **Outer Loop**: Leave-one-subject-out (LOSO)
  - Train on N-1 participants, test on 1 held-out participant
  - Repeat for all N participants
- **Inner Loop**: Hyperparameter tuning on training set
  - Grid search over cost (C) and gamma
  - 10-fold cross-validation within training data
- **Kernel**: Radial (RBF)
- **Feature Selection**: Optional pruning with `utils/r/prune_feats.R`

## Hyperparameter Grid
- **Cost**: 0.01, 0.1, 1, 10, 100
- **Gamma**: 0.001, 0.01, 0.1, 1

## Dependencies
- R 4.3+ with e1071, caret, doParallel

## Shared Utilities
Sources from `utils/r/`:
- `data_prep_helpers.R` - Data loading and preparation
- `prune_feats.R` - Feature selection
- `save_helpers.R` - Result export

## Usage
```r
Rscript svm.R
```

## Performance
- Processing time: 2-4 hours (depends on feature count and parallelization)
- Nested CV is computationally expensive but provides unbiased estimates

## Notes
- Differs from Python SVM (Stage 07): uses aggregated features, not rolling windows
- Nested CV prevents hyperparameter overfitting
- Produces publication-ready performance metrics
