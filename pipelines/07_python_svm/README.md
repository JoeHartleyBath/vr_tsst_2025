# Stage 07: Python SVM (Rolling Windows)

Scikit-learn SVM classification with rolling window approach for temporal dynamics.

## Entry Points
- **Main**: `svm_rolling_windows.py` (standard rolling window classification)
- **All Conditions**: `svm_rolling_windows_all.py` (all condition pairs)
- **Task vs Forest**: `svm_rolling_windows_task_vs_forest.py` (specific comparison)
- **High Stress**: `svm_highstress_bandpower.py` (high-stress classification only)

## Inputs
- **File**: `output/final_data.rds` (from Stage 06, converted to CSV/pickle)
- **Config**: `scripts/modeling/model_class_labels.yaml`

## Outputs
- **Directory**: `results/svm/`
- **Files**: 
  - Classification reports (accuracy, precision, recall, F1)
  - Confusion matrices
  - Feature importance rankings
  - ROC curves
  - Model parameters

## Classification Tasks
1. **Stress**: High stress vs Low stress
2. **Workload**: High workload vs Low workload
3. **Condition Pairs**: All pairwise comparisons

## Method
- **Approach**: Rolling time windows (e.g., 5-second windows with 1-second overlap)
- **Validation**: Leave-one-subject-out (LOSO) cross-validation
- **Kernel**: RBF (radial basis function)
- **Hyperparameters**: Grid search for C and gamma
- **Features**: All EEG + physio features (dimensionality reduction optional)

## Dependencies
- Python 3.x with scikit-learn, pandas, numpy, matplotlib, joblib

## Usage
```python
python svm_rolling_windows.py
```

## Performance
- Processing time: ~30-60 minutes per classification task
- Depends on: number of participants, window size, hyperparameter grid

## Notes
- Complements R SVM (Stage 08) which uses aggregated features
- Rolling windows capture temporal dynamics within conditions
- Useful for deep learning dataset preparation (similar structure)
