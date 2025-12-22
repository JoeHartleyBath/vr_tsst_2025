# Stage 04: Python Physio Feature Extraction

Extracts heart rate, galvanic skin response, and eye tracking features from raw XDF data.

## Entry Point
- **Main**: `extract_physio_features.py`

## Core Logic
Located in `private/` subdirectory:
- `load_data.py` - XDF loading and stream extraction
- `clean_hr_data.py` - Heart rate ectopic beat correction
- `clean_gsr_data.py` - GSR filtering and resampling
- `clean_eye_data.py` - Pupil dilation and blink processing
- `extract_features.py` - Feature computation per condition
- `merge_with_eeg.py` - Merge physio with EEG features

## Inputs
- **Directory**: `data/RAW/{participant_id}/`
- **Files**: `*.xdf` (raw LSL recordings)
- **Config**: `config/general.yaml`, `config/eeg_metadata.yaml`

## Outputs
- Internal dataframes passed to Stage 05 (merge pipeline)
- **Logs**: Feature extraction progress, data quality warnings

## Features Extracted

### Heart Rate Variability (HRV)
- Mean HR, Std HR
- RMSSD (root mean square of successive differences)
- pNN50 (percentage of successive intervals > 50ms)
- LF/HF ratio (frequency domain HRV)

### Galvanic Skin Response (GSR)
- Mean GSR, Std GSR
- Number of SCRs (skin conductance responses)
- Mean SCR amplitude
- SCR frequency

### Eye Tracking
- Mean pupil diameter (left, right, average)
- Pupil diameter variability
- Blink rate
- Fixation duration

## Dependencies
- Python 3.x with pyxdf, scipy, numpy, pandas, neurokit2

## Usage
```python
python extract_physio_features.py
```

## Notes
- Runs independently of EEG processing
- Can be parallelized across participants
- Handles missing data gracefully (logs warnings)
- Requires clean physio streams in XDF
