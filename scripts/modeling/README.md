# Machine Learning Models

Classification and prediction models for stress/workload detection.

## Structure

```
modeling/
├── svm/               # Support Vector Machine classifiers
├── xgboost/           # Gradient boosting classifiers
└── optimization/      # Hyperparameter tuning and feature selection
```

## Scripts

### SVM (`svm/`)
- **svm.R**: Main SVM training and evaluation
- **svm_best_model_finder.R**: Grid search for best SVM parameters
- Classification: Stress/workload prediction from physiological features
- Output: `results/svm/`

### XGBoost (`xgboost/`)
- **xgboost_loso_classification.R**: Leave-one-subject-out cross-validation
- Handles imbalanced classes and missing data well
- Output: `results/xgb/`

### Optimization (`optimization/`)
- **bayes_opt.R**: Bayesian hyperparameter optimization
- **prune_feats.R**: Feature selection and dimensionality reduction
- Improves model performance and interpretability

## Workflow

1. **Feature Preparation**: `optimization/prune_feats.R`
2. **Hyperparameter Tuning**: `optimization/bayes_opt.R` or `svm/svm_best_model_finder.R`
3. **Model Training**: `svm/svm.R` or `xgboost/xgboost_loso_classification.R`
4. **Evaluation**: LOSO cross-validation, accuracy, F1-score

## Dependencies

- Preprocessed data: `output/aggregated/all_data_aggregated.csv`
- R utilities: `scripts/utils/r/data_prep_helpers.R`
- R packages: `caret`, `e1071` (SVM), `xgboost`, `rBayesianOptimization`
