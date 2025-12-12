# Physiological Feature Extraction - Refactoring Plan

## Overview
Refactor 1928-line notebook into modular Python scripts following the same architecture as the EEG feature extraction pipeline.

## Goals
1. **Modularity**: Separate concerns into focused, testable modules
2. **Maintainability**: Clear structure, documented functions
3. **Reliability**: Robust error handling and QC logging
4. **Performance**: Optimize I/O and enable parallel processing
5. **Validation**: Ensure time alignment with EEG pipeline

## Architecture

```
physio_features/
├── extract_physio_features.py       # Main orchestration (150 lines)
├── private/
│   ├── __init__.py
│   ├── load_data.py                 # Data loading with caching (~200 lines)
│   ├── validate_inputs.py           # Input validation (~100 lines)
│   ├── clean_hr_data.py             # HR/RR interval cleaning (~250 lines)
│   ├── clean_gsr_data.py            # GSR/EDA cleaning + resampling (~200 lines)
│   ├── clean_eye_data.py            # Pupil/blink cleaning (~300 lines)
│   ├── extract_hr_features.py       # HR + HRV features (~150 lines)
│   ├── extract_gsr_features.py      # GSR tonic/phasic features (~150 lines)
│   ├── extract_eye_features.py      # Pupil + blink features (~150 lines)
│   ├── extract_response_metrics.py  # Task response features (~100 lines)
│   ├── compute_features.py          # Feature orchestration (~200 lines)
│   ├── merge_with_eeg.py            # Merge physio + EEG + subjective (~150 lines)
│   └── qc_logging.py                # Quality control logging (~100 lines)
├── legacy/
│   └── step05_preprocess_physio_and_merge_legacy.ipynb
└── README.md
```

## Module Breakdown

### 1. `load_data.py`
**Purpose**: Load and cache raw data
- `load_physio_data()`: Load raw physio CSVs with pickle caching
- `load_eeg_data()`: Load EEG features CSV
- `load_subjective_data()`: Load and reshape subjective ratings
- `fix_participant_ids()`: Ensure consistent ID format

**Key improvements**:
- Remove hardcoded paths (use config)
- Robust column presence checking
- Clear error messages for missing files

### 2. `validate_inputs.py`
**Purpose**: Validate data before processing
- `validate_physio_columns()`: Check required columns exist
- `validate_time_alignment()`: Verify LSL timestamp alignment
- `check_participant_data_quality()`: Pre-filter bad participants

### 3. `clean_hr_data.py`
**Purpose**: Clean heart rate and RR interval signals
- `clean_hr_pipeline()`: Main HR cleaning orchestration
- `threshold_rr_intervals()`: Clamp to valid range (200-2000 ms)
- `correct_ectopic_beats()`: NeuroKit2 ectopic correction
- `remove_hr_outliers_mad()`: MAD-based outlier removal
- `interpolate_small_gaps()`: Fill small missing segments

**From legacy lines**: 847-943

### 4. `clean_gsr_data.py`
**Purpose**: Clean and resample GSR/EDA signals
- `resample_gsr_to_10hz()`: Resample GSR from 30 Hz to 10 Hz
- `clean_gsr_pipeline()`: Apply NeuroKit2 EDA cleaning
- `merge_gsr_metadata()`: Re-attach condition labels after resampling

**From legacy lines**: 606-723, 1287-1325

### 5. `clean_eye_data.py`
**Purpose**: Clean pupil dilation and blink data
- `clean_eye_pipeline()`: Main eye cleaning orchestration
- `threshold_eye_features()`: Absolute threshold cleaning
- `detect_blink_closures()`: Classify blink types (short/medium/prolonged)
- `interpolate_short_blinks()`: Interpolate brief closures
- `remove_prolonged_closures()`: Mask extended closures

**From legacy lines**: 999-1171

### 6. `extract_hr_features.py`
**Purpose**: Extract HR and HRV features
- `extract_hr_features()`: Mean, median, SD, min, max BPM
- `extract_hrv_features()`: RMSSD, SDNN, pNN50 using NeuroKit2
- `compute_rr_interval_stats()`: RR interval statistics

**From legacy lines**: 297-335

### 7. `extract_gsr_features.py`
**Purpose**: Extract GSR tonic and phasic features
- `extract_gsr_features()`: Main GSR feature extraction
- `compute_tonic_features()`: Mean, SD of tonic component
- `compute_phasic_features()`: SCR peak rate, amplitude, count
- `check_flat_segment()`: Flag low-variance segments

**From legacy lines**: 91-188

### 8. `extract_eye_features.py`
**Purpose**: Extract pupil and blink features
- `extract_pupil_features()`: Dilation statistics (L/R, foveal-corrected)
- `extract_blink_features()`: Duration, rate, inter-blink intervals
- `compute_pupil_unrest()`: Spectral power in 0.05-0.3 Hz band

**From legacy lines**: 189-196, 1328-1720

### 9. `extract_response_metrics.py`
**Purpose**: Extract task response features
- `extract_response_metrics()`: Response count, rate, latency, accuracy

**From legacy lines**: 358-392

### 10. `compute_features.py`
**Purpose**: Orchestrate feature extraction across all modalities
- `compute_features_all_participants()`: Main feature extraction loop
- `calculate_rolling_stats()`: Rolling window features (if enabled)
- `calculate_full_condition_stats()`: Full-condition aggregates
- `calculate_stats()`: Shared statistics computation

**From legacy lines**: 197-296, 1328-1720

### 11. `merge_with_eeg.py`
**Purpose**: Merge physio features with EEG and subjective data
- `merge_physio_with_eeg()`: Main merge orchestration
- `reshape_subjective_data()`: Transform subjective ratings format
- `align_time_windows()`: Ensure proper time alignment
- `validate_merge_results()`: Check for unexpected NaN patterns

**From legacy lines**: 1818-1924

### 12. `qc_logging.py`
**Purpose**: Quality control logging utilities
- `setup_qc_loggers()`: Initialize per-participant QC loggers
- `log_cleaning_stats()`: Log dropped/retained samples
- `log_feature_extraction()`: Log feature computation success/failure
- `generate_qc_summary()`: Create summary report

**From legacy lines**: 60-79

## Key Improvements

### 1. Remove Hardcoded Paths
**Current** (line 35):
```python
with open('D:/PhD_Projects/TSST_Stress_Workload_Pipeline/scripts/config.yaml', 'r') as f:
```

**Refactored**:
```python
config_path = os.path.join(os.path.dirname(__file__), '../../config/general.yaml')
with open(config_path, 'r') as f:
```

### 2. Enable Rolling Stats (Currently Commented Out)
Decide: Do we need rolling stats or just full-condition aggregates?
- If yes: Uncomment and validate time alignment
- If no: Remove dead code

### 3. Time Alignment Validation
**Critical**: Add validation that LSL timestamps align with EEG sample frames
```python
def validate_time_alignment(physio_df, eeg_df, participant_id):
    """Verify physio timestamps overlap with EEG windows."""
    # Check for gaps, verify synchronization
    pass
```

### 4. Simplified Feature Selection
Focus on most reliable features:
- HR: BPM (cleaned), HRV metrics (RMSSD, SDNN)
- GSR: Tonic mean/SD, phasic peak rate/amplitude
- Eye: Foveal-corrected dilation (cleaned), blink rate, pupil unrest
- Head: 3D velocity

### 5. Clear Separation of Concerns
- **Loading**: Just load data, no processing
- **Cleaning**: Only signal cleaning, no feature extraction
- **Extraction**: Only compute features, no merging
- **Merging**: Only combine datasets, no computation

## Validation Checklist

Before considering refactoring complete:

1. ✅ Load data successfully for all participants
2. ⏳ Verify cleaning effectiveness (plot raw vs cleaned)
3. ⏳ Validate time alignment with EEG pipeline
4. ⏳ Confirm feature ranges are reasonable
5. ⏳ Check for excessive NaN/missing data
6. ⏳ Compare output with legacy notebook (small subset)
7. ⏳ Run on 3 test participants end-to-end
8. ⏳ Profile performance vs legacy
9. ⏳ Generate QC summary report
10. ⏳ Full 48-participant run

## Timeline Estimate

- **Phase 1** (Day 1): Create module structure, implement load_data.py
- **Phase 2** (Day 2): Implement cleaning modules (HR, GSR, eye)
- **Phase 3** (Day 3): Implement feature extraction modules
- **Phase 4** (Day 4): Implement merging + validation
- **Phase 5** (Day 5): Testing + QC validation on subset
- **Phase 6** (Day 6): Full run + comparison with legacy

## Open Questions

1. **Rolling stats**: Keep or remove? (Currently commented out)
2. **Time alignment**: How to validate LSL timestamp synchronization?
3. **Feature selection**: Which features are actually used in downstream analyses?
4. **Baseline measures**: How are they computed and merged?
5. **GSR resampling**: Is 10 Hz the optimal rate, or use native 30 Hz?

## Next Steps

1. Review this plan with team
2. Make architectural decisions (rolling stats, feature selection)
3. Begin implementation starting with load_data.py
4. Test each module incrementally
5. Validate against legacy output
