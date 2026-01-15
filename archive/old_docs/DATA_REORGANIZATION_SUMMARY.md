# Data Reorganization Summary - P14/P16 Fix

## Issue Discovered

**Date**: December 18, 2025

**Problem**: P14 data missing from aggregated features; P16 appeared duplicated

### Investigation

1. **Physio features**: P14 had 0 rows, P16 had 42 rows (should be ~21 per participant)
2. **Root cause**: P14.csv raw file had `Participant_ID=16` instead of 14 (mislabeling)
3. **Counterbalance verification**: P14.csv follows P16's counterbalance order (stress subtraction second)
4. **Session comparison**: 
   - P16.csv (Session 1): 105,117 rows, 10.04% missing data
   - P14.csv (Session 2): 142,456 rows, 8.90% missing data (better quality)

### Decision

**Keep P14.csv as the correct P16 data** (Session 2 - better quality)

**Archive P16.csv** (Session 1 - lower quality)

**P14 remains empty** (never collected)

## Files Changed

### Raw Data

**Archived** (moved to `data/RAW/_archived_duplicate_sessions/`):
- `data/RAW/metadata/P16.csv` → `P16_old.csv`
- `data/RAW/eeg/P16.xdf` → `P16_old.xdf`

**Renamed**:
- `data/RAW/metadata/P14.csv` → `data/RAW/metadata/P16.csv`
- `data/RAW/eeg/P14.xdf` → `data/RAW/eeg/P16.xdf`

### Aggregated Features

**Physio features** (`output/aggregated/physio_features.csv`):
- **Before**: 1100 rows (P16 with 42 rows)
- **After**: 1077 rows (P16 with 19 rows)
- **Action**: Removed 21 duplicate P16 rows (Session 1), re-extracted all participants
- **Final**: Fresh extraction for all 48 participants from corrected raw files

**EEG features** (`output/aggregated/eeg_features.csv`):
- **Before**: 572 rows (P14=8, P16=12)
- **After**: 560 rows (P16=8, P14 absent)
- **Action**: Removed 12 old P16 rows, renamed 8 P14 rows to P16

### Cache

**Cleared**: `output/cache/physio_cleaned/` (all participants)

## Final Dataset Structure

### Participant Count
- **Total participants**: P01-P48 (48 IDs)
- **Data collected**: 47 participants (P14 never collected)
- **After QC filtering**: 44 participants (exclude P02, P08, P14, P46)

### P16 Data
- **Source**: Corrected P14.csv raw file (Session 2, better quality)
- **Counterbalance**: Follows P16 assignment
- **Quality**: 8.90% missing data (improved from 10.04%)

### P14 Data
- **Status**: No data collected
- **Expected N**: 47 participants (not 48)

## Verification Steps

### 1. Raw File Reorganization ✅
- [x] Archive old P16 files
- [x] Rename P14 files to P16
- [x] Verify file integrity

### 2. Aggregated Features Cleanup ✅
- [x] Remove duplicate P16 physio rows
- [x] Clean EEG features (remove old P16, rename P14→P16)
- [x] Clear physio cache

### 3. Full Re-extraction ✅
- [x] Re-extract physio features for all 48 participants
- [x] Output: 1077 rows total (P16 with corrected data from P14.csv)

### 4. Pending Validation ⏳
- [ ] Re-run merge pipeline
- [ ] Re-run R preprocessing
- [ ] Run final ANOVA validation
- [ ] Verify P16 appears in all analyses
- [ ] Confirm P14 absent from final dataset

## Impact on Analyses

### Before Fix
- **N**: 47 participants (P14 missing, P16 duplicated)
- **Data quality**: Mixed (Session 1 + Session 2 data for P16)
- **After QC**: 44 participants

### After Fix
- **N**: 47 participants (P14 never collected, P16 corrected)
- **Data quality**: Consistent (all participants use best available session)
- **After QC**: 44 participants (same exclusions)

## Documentation References

- **Raw data location**: `data/RAW/`
- **Archive location**: `data/RAW/_archived_duplicate_sessions/`
- **Counterbalance file**: `data/experimental_counterbalance.xlsx`
- **QC failures**: `output/qc/qc_failures_summary.csv`

## Notes

1. **P14 appears in counterbalance sheet** but was never recorded
2. **P16 was recorded twice** - second session (P14.csv) was better quality
3. **File naming mismatch** in raw data caused confusion
4. **Session selection** based on data quality metrics (% missing)
5. **All downstream analyses** will automatically use corrected P16 data

## Timeline

- **Issue discovered**: During ANOVA validation (missing P14, duplicate P16)
- **Root cause identified**: P14.csv contains Participant_ID=16
- **Counterbalance verified**: P14.csv follows P16's task order
- **Quality compared**: P14.csv (Session 2) superior to P16.csv (Session 1)
- **Files reorganized**: Raw files archived and renamed
- **Features re-extracted**: Full physio pipeline for 48 participants
- **Status**: Pending merge and preprocessing validation

---

**Prepared by**: GitHub Copilot  
**Date**: December 18, 2025
