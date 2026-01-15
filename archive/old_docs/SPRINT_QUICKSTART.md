# 48-Hour Sprint Quick Start Guide

**Date**: December 16, 2025  
**Status**: Reorganization Complete ✅

## Sprint Objectives

1. **Physio Feature Extraction** (4-6 hours)
2. **Deep Learning Dataset Preparation** (8-16 hours)
3. **SVM + XGBoost Analysis** (8-12 hours)
4. **Validation & Documentation** (4-6 hours)

**Total**: ~24-40 hours of compute time + monitoring

## New Pipeline Structure

The repository is now professionally organized:

```
pipelines/
├── 01_xdf_to_set/              ✅ Complete (Stage 1)
├── 02_matlab_cleaning/         ✅ Complete (Stage 2, except P10/P14/P23)
├── 03_matlab_eeg_features/     ✅ Complete (Stage 3)
├── 04_python_physio_features/  ⏳ TO DO (Stage 4)
├── 05_merge_features/          ⏳ TO DO (Stage 5)
├── 06_r_preprocessing/         ⏳ TO DO (Stage 6)
├── 07_python_svm/              ⏳ TO DO (Stage 7)
├── 08_r_svm/                   ⏳ TO DO (Stage 8)
├── 09_r_xgboost/               ⏳ TO DO (Stage 9)
└── 10_r_anova/                 ⏳ TO DO (Stage 10)
```

## Sprint Task Breakdown

### Task 1: Physio Feature Extraction (Priority 1)

**Estimated Time**: 4-6 hours (parallelized)

```bash
# Activate environment
.\venv\Scripts\Activate.ps1

# Run Stage 4: Extract physio features
python pipelines\04_python_physio_features\extract_physio_features.py

# This will extract HR, HRV, GSR, pupil features from raw XDF files
# Output: Internal dataframes for Stage 5 merge
```

**What it does**:
- Loads XDF files for all participants
- Extracts heart rate (HR) and heart rate variability (HRV)
- Extracts galvanic skin response (GSR) features
- Extracts pupil dilation and blink metrics
- Handles missing data gracefully

**Monitor**: Check logs for data quality warnings

---

### Task 2: Merge EEG + Physio Features (Priority 1)

**Estimated Time**: 30 minutes

```bash
# Run Stage 5: Merge features
python pipelines\05_merge_features\mvp_merge_pipeline.py

# Output: output/aggregated/all_data_aggregated.csv
```

**What it does**:
- Combines EEG features from Stage 3
- Merges with physio features from Stage 4
- Aligns by participant and condition
- Creates unified dataset

---

### Task 3: R Final Preprocessing (Priority 1)

**Estimated Time**: 15-30 minutes

```bash
# Run Stage 6: R preprocessing
Rscript pipelines\06_r_preprocessing\preproccess_for_xgb.R

# Output: 
#   - output/final_data.rds (for ML)
#   - output/anova_features_precond.rds (for ANOVA)
```

**What it does**:
- Baseline normalization
- Feature scaling
- Outlier removal
- Splits into ML-ready and ANOVA-ready formats

---

### Task 4: Deep Learning Dataset Preparation (Priority 2)

**Estimated Time**: 8-16 hours

**Decision Required**: Choose DL format

**Option A: Rolling Windows (like Python SVM)**
- Use existing `pipelines/07_python_svm/svm_rolling_windows.py` structure
- Extract 5-second windows with 1-second overlap
- Label: stress (high/low) or workload (high/low)

**Option B: Time-Series Sequences**
- Full condition sequences (e.g., entire Math task)
- Preserve temporal ordering
- Label: condition type

**For CTNet (5-fold CV)**: Need 2 files per participant
- File 1: Low stress, High workload (e.g., Math tasks)
- File 2: Low stress, Low workload (e.g., Forest baseline)

**TO DO**: Create `scripts/dl_dataset_preparation.py`

```python
# Pseudocode structure
# 1. Load output/final_data.rds (convert to pandas)
# 2. Extract rolling windows or sequences
# 3. Create 2 files per participant:
#    - p{id}_low_stress_high_workload.npy
#    - p{id}_low_stress_low_workload.npy
# 4. Save labels separately
# 5. Create train/val/test splits (5-fold CV indices)
```

---

### Task 5: Python SVM (Priority 3)

**Estimated Time**: 2-4 hours

```bash
# Run Stage 7: Python SVM with rolling windows
python pipelines\07_python_svm\svm_rolling_windows.py

# Output: results/svm/
```

**What it does**:
- Classifies stress (high vs low)
- Classifies workload (high vs low)
- Uses rolling time windows
- LOSO cross-validation
- Outputs confusion matrices, ROC curves

---

### Task 6: R SVM (Priority 3)

**Estimated Time**: 2-4 hours

```r
# Run Stage 8: R SVM with nested LOSO
Rscript pipelines\08_r_svm\svm.R

# Output: results/svm/svm_progress_*.csv
```

**What it does**:
- Nested cross-validation
- Hyperparameter tuning (cost, gamma)
- Aggregated features (not rolling windows)
- Publication-ready metrics

---

### Task 7: XGBoost (Priority 3)

**Estimated Time**: 4-8 hours (Bayesian optimization)

```r
# Run Stage 9: XGBoost
Rscript pipelines\09_r_xgboost\xgboost_loso_classification.R

# Output: results/xgb/xgb_results_*.rds
```

**What it does**:
- Gradient boosting classification
- Bayesian hyperparameter optimization
- Feature importance rankings
- Best-performing model in previous tests

---

### Task 8: ANOVA (Priority 4)

**Estimated Time**: 30 minutes

```r
# Run Stage 10: Statistical analysis
Rscript pipelines\10_r_anova\anova_main.R

# Output: results/classic_analyses/anova_results_*.csv
```

**What it does**:
- Repeated-measures ANOVA
- Validates experimental manipulation
- Post-hoc pairwise comparisons
- Effect sizes

---

## Parallel Execution Strategy

Run these in parallel to save time:

**Terminal 1**: Physio extraction (Stage 4)
```bash
python pipelines\04_python_physio_features\extract_physio_features.py
```

**Terminal 2**: Monitor progress
```powershell
Get-Content logs\physio_extraction.log -Wait
```

**After Stage 6 completes, run Stages 7-10 in parallel**:

```powershell
# Terminal 1: Python SVM
python pipelines\07_python_svm\svm_rolling_windows.py

# Terminal 2: R SVM
Rscript pipelines\08_r_svm\svm.R

# Terminal 3: XGBoost (most time-consuming)
Rscript pipelines\09_r_xgboost\xgboost_loso_classification.R

# Terminal 4: ANOVA
Rscript pipelines\10_r_anova\anova_main.R
```

---

## Data Exclusions

**Known Issues**: P10, P14, P23 fail EEG cleaning
- Exclude from analysis
- Document in results
- N = 45 participants (48 - 3 excluded)

---

## Monitoring Progress

```powershell
# Check logs
Get-ChildItem logs\ -File | Sort-Object LastWriteTime -Descending | Select-Object -First 5

# Check outputs
Get-ChildItem output\aggregated\ -File
Get-ChildItem results\svm\ -File
Get-ChildItem results\xgb\ -File

# Check disk space
Get-PSDrive C | Select-Object Used,Free
```

---

## Validation Checklist

- [ ] Physio features extracted for 45 participants
- [ ] EEG + Physio merged successfully
- [ ] Final preprocessing complete (2 RDS files created)
- [ ] Python SVM results generated
- [ ] R SVM results generated
- [ ] XGBoost results generated
- [ ] ANOVA results generated
- [ ] DL datasets created (if applicable)
- [ ] All results documented

---

## Documentation TODO

After completing analyses:
1. Update `pipeline_quality_report.md` with results
2. Document DL dataset format
3. Update `READY_FOR_PILOT.txt` if applicable
4. Create results summary markdown

---

## Emergency Contacts

- **Git Issues**: See `MIGRATION_GUIDE.md`
- **Pipeline Errors**: Check individual stage `README.md` files
- **Dependencies**: Run `pip install -r requirements.txt` or `setup_environment.ps1`

---

## Quick Reference

| Stage | Script | Time | Status |
|-------|--------|------|--------|
| 1 | XDF → SET | N/A | ✅ Complete |
| 2 | MATLAB Cleaning | N/A | ✅ Complete |
| 3 | EEG Features | N/A | ✅ Complete |
| 4 | Physio Features | 4-6h | ⏳ TODO |
| 5 | Merge Features | 30m | ⏳ TODO |
| 6 | R Preprocessing | 30m | ⏳ TODO |
| 7 | Python SVM | 2-4h | ⏳ TODO |
| 8 | R SVM | 2-4h | ⏳ TODO |
| 9 | XGBoost | 4-8h | ⏳ TODO |
| 10 | ANOVA | 30m | ⏳ TODO |

**Total Compute Time**: ~16-26 hours (parallelizable to ~8-16 hours wall time)

---

**Ready to start the sprint!** 🚀
