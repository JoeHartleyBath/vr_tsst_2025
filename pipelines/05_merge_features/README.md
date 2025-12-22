# Stage 05: Merge EEG + Physio Features

Combines EEG features from Stage 03 with physiological features from Stage 04.

## Entry Point
- **Main**: `mvp_merge_pipeline.py`

## Inputs
- **EEG Features**: `output/aggregated/eeg_features.csv` (from Stage 03)
- **Physio Features**: Output from Stage 04 (in-memory or temporary files)
- **Config**: `config/general.yaml`

## Outputs
- **Directory**: `output/aggregated/`
- **File**: `all_data_aggregated.csv` (merged EEG + physio features)
- **Logs**: Merge statistics, missing data warnings

## Processing Steps
1. Load EEG features from CSV
2. Load or compute physio features
3. Align by participant and condition
4. Merge on common keys (participant_id, condition, timepoint)
5. Handle missing values
6. Validate feature completeness
7. Export merged dataset

## Output Schema
- **Participant**: participant_id
- **Condition**: baseline, math1, math2, math3, social, recovery, forest
- **EEG Features**: ~100-200 features (spectral power, ratios, entropy per channel)
- **Physio Features**: ~20-30 features (HR, HRV, GSR, pupil, blinks)
- **Metadata**: timestamp, event markers, exclusion flags

## Dependencies
- Python 3.x with pandas, numpy

## Usage
```python
python mvp_merge_pipeline.py
```

## Notes
- Performs inner join by default (only participants with both EEG and physio)
- Logs participants excluded due to missing data
- Output is ready for R preprocessing (Stage 06)
