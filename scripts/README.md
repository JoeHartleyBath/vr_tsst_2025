# Scripts Folder Organization

This folder contains all analysis and processing scripts organized by data science workflow stages.

## Structure

```
scripts/
├── preprocessing/     # Data cleaning and feature extraction
├── analysis/          # Statistical analysis
├── modeling/          # Machine learning models
├── utils/             # Shared utilities and helpers
├── notebooks/         # Exploratory Jupyter notebooks
├── tests/             # Unit tests
└── deprecated/        # Legacy code (kept for reference)
```

## Workflow

1. **Preprocessing**: Raw data → cleaned data → features
2. **Analysis**: Statistical tests, correlations, visualizations
3. **Modeling**: ML model training, optimization, evaluation

## Key Entry Points

- **EEG Feature Extraction**: `preprocessing/eeg/feature_extraction/extract_eeg_feats.py`
- **Physio Feature Extraction**: `preprocessing/physio/feature_extraction/extract_physio_features.py`
- **Raw Conversion**: `preprocessing/raw_conversion/run/run_xdf_to_set_end2end.py`
- **ANOVA Analysis**: `analysis/anova/anova_main.R`
- **SVM Classification**: `modeling/svm/svm.R`
- **XGBoost Classification**: `modeling/xgboost/xgboost_loso_classification.R`

## Guidelines

- All production code should be in `preprocessing/`, `analysis/`, or `modeling/`
- Exploratory work goes in `notebooks/`
- Shared functions go in `utils/`
- Tests go in `tests/`
- Old/unused code goes in `deprecated/` (never deleted for reproducibility)
