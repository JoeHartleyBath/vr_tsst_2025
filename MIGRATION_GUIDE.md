# Pipeline Reorganization Migration Guide

**Date**: December 16, 2025  
**Status**: Complete

## Overview

The VR-TSST codebase has been reorganized from a scattered `scripts/` structure into a professional `pipelines/` architecture with 10 numbered canonical stages. This guide helps you navigate the changes.

## What Changed

### New Directory Structure

```
pipelines/                          # NEW: Canonical 10-stage pipeline
├── 01_xdf_to_set/                  # Stage 1: XDF → SET conversion
├── 02_matlab_cleaning/             # Stage 2: AMICA cleaning
├── 03_matlab_eeg_features/         # Stage 3: EEG feature extraction
├── 04_python_physio_features/      # Stage 4: Physio feature extraction
├── 05_merge_features/              # Stage 5: Merge EEG + physio
├── 06_r_preprocessing/             # Stage 6: Final R preprocessing
├── 07_python_svm/                  # Stage 7: Python SVM
├── 08_r_svm/                       # Stage 8: R SVM (nested LOSO)
├── 09_r_xgboost/                   # Stage 9: XGBoost
└── 10_r_anova/                     # Stage 10: ANOVA

utils/                              # NEW: Consolidated shared utilities
├── python/                         # Python QC and helpers
├── matlab/                         # MATLAB YAML parser
└── r/                              # R helpers + optimization

dev/                                # NEW: Development/test scripts
```

### Files Moved

#### From `scripts/preprocessing/` → `pipelines/`

| Old Location | New Location | Stage |
|-------------|-------------|-------|
| `scripts/preprocessing/raw_conversion/xdf_to_set/xdf_to_set.py` | `pipelines/01_xdf_to_set/xdf_to_set.py` | 1 |
| `scripts/preprocessing/raw_conversion/run/run_xdf_to_set_parallel.py` | `pipelines/01_xdf_to_set/run_xdf_to_set_parallel.py` | 1 |
| `scripts/preprocessing/eeg/cleaning/run_clean_eeg_pipeline.m` | `pipelines/02_matlab_cleaning/run_clean_eeg_pipeline.m` | 2 |
| `scripts/preprocessing/eeg/cleaning/clean_eeg.m` | `pipelines/02_matlab_cleaning/clean_eeg.m` | 2 |
| `scripts/preprocessing/eeg/feature_extraction/extract_eeg_features.m` | `pipelines/03_matlab_eeg_features/extract_eeg_features.m` | 3 |
| `scripts/preprocessing/eeg/feature_extraction/private/*` | `pipelines/03_matlab_eeg_features/private/*` | 3 |
| `scripts/preprocessing/physio/feature_extraction/extract_physio_features.py` | `pipelines/04_python_physio_features/extract_physio_features.py` | 4 |
| `scripts/preprocessing/physio/feature_extraction/private/*` | `pipelines/04_python_physio_features/private/*` | 4 |
| `scripts/preprocessing/physio/feature_extraction/mvp_merge_pipeline.py` | `pipelines/05_merge_features/mvp_merge_pipeline.py` | 5 |
| `scripts/preprocessing/physio/preproccess_for_xgb.R` | `pipelines/06_r_preprocessing/preproccess_for_xgb.R` | 6 |

#### From `scripts/modeling/` → `pipelines/`

| Old Location | New Location | Stage |
|-------------|-------------|-------|
| `scripts/modeling/svm_rolling_windows.py` | `pipelines/07_python_svm/svm_rolling_windows.py` | 7 |
| `scripts/modeling/svm_rolling_windows_all.py` | `pipelines/07_python_svm/svm_rolling_windows_all.py` | 7 |
| `scripts/modeling/svm/svm.R` | `pipelines/08_r_svm/svm.R` | 8 |
| `scripts/modeling/svm/svm_best_model_finder.R` | `pipelines/08_r_svm/svm_best_model_finder.R` | 8 |
| `scripts/modeling/xgboost/xgboost_loso_classification.R` | `pipelines/09_r_xgboost/xgboost_loso_classification.R` | 9 |
| `scripts/modeling/optimization/prune_feats.R` | `utils/r/prune_feats.R` | Shared |
| `scripts/modeling/optimization/bayes_opt.R` | `utils/r/bayes_opt.R` | Shared |

#### From `scripts/analysis/` → `pipelines/`

| Old Location | New Location | Stage |
|-------------|-------------|-------|
| `scripts/analysis/anova/anova_main.R` | `pipelines/10_r_anova/anova_main.R` | 10 |

#### From `scripts/utils/` → `utils/`

| Old Location | New Location |
|-------------|-------------|
| `scripts/utils/r/*.R` | `utils/r/*.R` |
| `scripts/utils/matlab/*` | `utils/matlab/*` |
| `scripts/utils/*.py` | `utils/python/*.py` |

#### From Project Root → `dev/`

All test/diagnostic scripts moved to `dev/`:
- `assign_p01_chanlocs.m`
- `check_fdt_data.m`
- `check_original_amplitude.m`
- `debug_region_match.m`
- `diagnose_chanlocs.m`
- `fix_p01_chanlocs.m`
- `quick_data_check.m`
- `run_p01_complete.m`
- `test_feature_compute.m`
- `test_p01_features.m`
- `validate_pilot_conversion.m`
- `verify_rescale.m`
- `check_p01_events.py`
- `test_pipeline.py`

## How to Update Your Workflow

### If You Run Individual Stages

**Old Way:**
```bash
python scripts/preprocessing/raw_conversion/run/run_xdf_to_set_parallel.py
matlab -batch "run('scripts/preprocessing/eeg/cleaning/run_clean_eeg_pipeline.m')"
```

**New Way:**
```bash
python pipelines/01_xdf_to_set/run_xdf_to_set_parallel.py
matlab -batch "run('pipelines/02_matlab_cleaning/run_clean_eeg_pipeline.m')"
```

### If You Import Python Modules

**Old Way:**
```python
from scripts.preprocessing.physio.feature_extraction.private import extract_features
```

**New Way:**
```python
from pipelines.04_python_physio_features.private import extract_features
```

### If You Source R Scripts

**Old Way:**
```r
source("scripts/utils/r/data_prep_helpers.R")
source("scripts/modeling/optimization/prune_feats.R")
```

**New Way:**
```r
source("utils/r/data_prep_helpers.R")
source("utils/r/prune_feats.R")
```

### If You Use MATLAB addpath

**Old Way:**
```matlab
addpath(genpath('scripts'))
```

**New Way:**
```matlab
addpath(genpath('pipelines'))
addpath(genpath('utils'))
```

## Breaking Changes

1. **Python Imports**: Update any imports from `scripts.preprocessing` or `scripts.modeling` to `pipelines.*`

2. **R source() Paths**: Update paths to utilities from `scripts/utils/r/` to `utils/r/`

3. **MATLAB Paths**: Update `addpath()` statements to include `pipelines/` and `utils/`

4. **Stage Runners**: Old stage runners in `scripts/stages/` are deprecated. Use direct pipeline invocation.

5. **Master Pipeline**: `scripts/run_pipeline_master.py` needs path updates (TODO: update this file)

## What Stayed the Same

- **Configuration Files**: `config/` directory unchanged
- **Data Directories**: `data/`, `output/`, `results/` unchanged
- **Documentation**: `docs/` unchanged
- **Notebooks**: `scripts/notebooks/` unchanged (exploratory work)
- **File Formats**: All input/output formats unchanged

## Testing the Migration

### Quick Verification

```bash
# Check that canonical scripts exist
ls pipelines/01_xdf_to_set/xdf_to_set.py
ls pipelines/02_matlab_cleaning/run_clean_eeg_pipeline.m
ls pipelines/03_matlab_eeg_features/extract_eeg_features.m
ls pipelines/08_r_svm/svm.R
ls pipelines/09_r_xgboost/xgboost_loso_classification.R

# Check utilities moved
ls utils/r/prune_feats.R
ls utils/r/data_prep_helpers.R
ls utils/matlab/+yaml/

# Check dev scripts moved
ls dev/test_pipeline.py
ls dev/quick_data_check.m
```

### Run Individual Stages

Test each stage independently:

```bash
# Stage 1 (if you have new XDF data)
python pipelines/01_xdf_to_set/run_xdf_to_set_parallel.py

# Stage 3 (re-extract features from existing cleaned data)
matlab -batch "run('pipelines/03_matlab_eeg_features/extract_eeg_features.m')"

# Stage 6 (re-run R preprocessing)
Rscript pipelines/06_r_preprocessing/preproccess_for_xgb.R

# Stage 9 (re-run XGBoost)
Rscript pipelines/09_r_xgboost/xgboost_loso_classification.R
```

## Rollback Plan (If Needed)

If you encounter issues, you can temporarily restore old structure:

```powershell
# This is a DESTRUCTIVE operation - only use if migration fails
git reset --hard HEAD~1  # Reverts to pre-migration state
```

**Note**: This will lose all migration changes. Only use as last resort.

## Benefits of New Structure

1. **Clarity**: Clear numbered pipeline stages
2. **Documentation**: Each stage has its own README
3. **Modularity**: Easier to run individual stages
4. **Professional**: Follows data science best practices (Cookiecutter Data Science)
5. **Maintenance**: Utilities consolidated in one place
6. **Onboarding**: New team members can understand pipeline flow immediately

## Next Steps

1. ✅ **Reorganization Complete** (Dec 16, 2025)
2. ⏳ **Update master pipeline script** (`scripts/run_pipeline_master.py`)
3. ⏳ **Test full pipeline end-to-end**
4. ⏳ **Update documentation references**
5. ⏳ **Train team on new structure**

## Questions?

See individual stage README files:
- `pipelines/01_xdf_to_set/README.md`
- `pipelines/02_matlab_cleaning/README.md`
- ... (and so on for all 10 stages)

Or refer to main pipeline documentation:
- `pipelines/README.md`
