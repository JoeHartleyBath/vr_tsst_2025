# VR-TSST Pipeline Architecture

This directory contains the canonical 10-stage processing pipeline for the VR-TSST project, reorganized from scattered scripts into a professional structure.

## Pipeline Stages

### Data Acquisition & Preprocessing

**Stage 01: XDF to SET Conversion** (`01_xdf_to_set/`)
- Converts raw XDF files to EEGLAB .set format
- Entry point: `run_xdf_to_set_parallel.py`
- Input: `data/RAW/{participant}/`
- Output: `output/sets/*.set`

**Stage 02: MATLAB EEG Cleaning** (`02_matlab_cleaning/`)
- Runs AMICA ICA decomposition and artifact removal
- Entry point: `run_clean_eeg_pipeline.m`
- Input: `output/sets/*.set`
- Output: `output/cleaned_eeg/*.mat`

**Stage 03: MATLAB EEG Feature Extraction** (`03_matlab_eeg_features/`)
- Extracts spectral features from cleaned EEG
- Entry point: `extract_eeg_features.m`
- Input: `output/cleaned_eeg/*.mat`
- Output: `output/aggregated/eeg_features.csv`

**Stage 04: Python Physio Feature Extraction** (`04_python_physio_features/`)
- Extracts heart rate, GSR, pupil features from XDF
- Entry point: `extract_physio_features.py`
- Input: `data/RAW/{participant}/`
- Output: Internal (merged in next stage)

**Stage 05: Merge EEG + Physio Features** (`05_merge_features/`)
- Combines EEG and physiological features
- Entry point: `mvp_merge_pipeline.py`
- Input: Stage 3 + Stage 4 outputs
- Output: `output/aggregated/all_data_aggregated.csv`

**Stage 06: R Final Preprocessing** (`06_r_preprocessing/`)
- Baseline normalization, feature transformation
- Entry point: `preproccess_for_xgb.R`
- Input: `output/aggregated/all_data_aggregated.csv`
- Output: `output/final_data.rds`, `output/anova_features_precond.rds`

### Machine Learning & Analysis

**Stage 07: Python SVM (Rolling Windows)** (`07_python_svm/`)
- Scikit-learn SVM with rolling window classification
- Entry point: `svm_rolling_windows.py`
- Input: `output/final_data.rds` (converted)
- Output: `results/svm/`

**Stage 08: R SVM (Nested LOSO)** (`08_r_svm/`)
- Nested leave-one-subject-out cross-validation
- Entry point: `svm.R`
- Input: `output/final_data.rds`
- Output: `results/svm/`

**Stage 09: R XGBoost** (`09_r_xgboost/`)
- XGBoost with LOSO cross-validation
- Entry point: `xgboost_loso_classification.R`
- Input: `output/final_data.rds`
- Output: `results/xgb/`

**Stage 10: R ANOVA** (`10_r_anova/`)
- Statistical analysis of experimental conditions
- Entry point: `anova_main.R`
- Input: `output/anova_features_precond.rds`
- Output: `results/classic_analyses/`

## Running the Pipeline

### Individual Stages
```bash
# Stage 1: XDF to SET
python pipelines/01_xdf_to_set/run_xdf_to_set_parallel.py

# Stage 2: MATLAB Cleaning
matlab -batch "run('pipelines/02_matlab_cleaning/run_clean_eeg_pipeline.m')"

# Stage 3: EEG Features
matlab -batch "run('pipelines/03_matlab_eeg_features/extract_eeg_features.m')"

# Stage 4: Physio Features
python pipelines/04_python_physio_features/extract_physio_features.py

# Stage 5: Merge
python pipelines/05_merge_features/mvp_merge_pipeline.py

# Stage 6: R Preprocessing
Rscript pipelines/06_r_preprocessing/preproccess_for_xgb.R

# Stages 7-10: See individual README files
```

### Full Pipeline (deprecated, being updated)
```bash
python scripts/run_pipeline_master.py
```

## Shared Resources

- **Utilities**: See `utils/` directory
  - `utils/python/` - Python helpers and QC
  - `utils/matlab/` - MATLAB YAML parser
  - `utils/r/` - R data preparation and optimization helpers

- **Configuration**: `config/` directory
  - `general.yaml` - Participant lists, paths
  - `eeg_feature_extraction.yaml` - Feature extraction settings
  - `eeg_metadata.yaml` - Event markers and conditions

## Migration Notes

This structure was created on 2025-12-16 by consolidating scattered scripts from:
- `scripts/preprocessing/`
- `scripts/modeling/`
- `scripts/analysis/`

Legacy stage runners in `scripts/stages/` are being deprecated in favor of direct pipeline invocation.

See `MIGRATION_GUIDE.md` for detailed transition documentation.
