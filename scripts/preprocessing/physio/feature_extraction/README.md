# Physiological Data Processing Pipeline

This folder contains the refactored physiological data processing pipeline for extracting HR, GSR, pupil, and blink features from VR-TSST experiment data.

## Directory Structure

```
physio_features/
├── extract_physio_features.py        # Main orchestration script
├── private/                           # Helper modules (similar to EEG pipeline)
│   ├── load_data.py
│   ├── clean_hr_data.py
│   ├── clean_gsr_data.py
│   ├── clean_eye_data.py
│   ├── extract_features.py
│   ├── merge_with_eeg.py
│   └── ...
├── legacy/                            # Original notebook-based pipeline
│   └── step05_preprocess_physio_and_merge_legacy.ipynb
└── README.md                          # This file
```

## Features Extracted

### Heart Rate (HR) Metrics
- BPM statistics (mean, median, SD, min, max)
- RR interval statistics
- HRV metrics (RMSSD, SDNN, pNN50)

### Galvanic Skin Response (GSR) Metrics
- Tonic measures (mean conductance, SD)
- Phasic measures (SCR peak rate, amplitude, count)
- NeuroKit2-based EDA decomposition

### Eye Tracking Metrics
- Pupil dilation (left, right, foveal-corrected)
- Blink duration and rate
- Inter-blink intervals
- Pupil unrest power (0.05-0.3 Hz)

### Head Movement
- 3D translational velocity

## Usage

**Basic usage** (all participants, full pipeline):
```bash
python extract_physio_features.py
```

**With options**:
```bash
python extract_physio_features.py --participants 1 2 3 --parallel --output custom_output.csv
```

## Data Flow

1. **Load**: Raw physio CSV files from `data/raw/`
2. **Clean**: Signal cleaning per modality (HR, GSR, eye)
3. **Resample**: GSR to 10 Hz with proper time alignment
4. **Extract**: Compute features per condition window
5. **Merge**: Combine with EEG features and subjective ratings
6. **Export**: Final aggregated dataset

## Critical Validation Steps

Before using extracted features:
1. Verify LSL timestamp alignment with EEG sample frames
2. Check cleaning effectiveness (plot raw vs cleaned signals)
3. Validate feature ranges (no extreme outliers/NaNs)
4. Confirm condition mapping matches EEG pipeline

## Refactoring Notes

- **Legacy notebook**: 1928 lines → modular architecture
- **Modular design**: Separate cleaning functions per signal type
- **Quality control**: Per-participant QC logs maintained
- **Time alignment**: Verified synchronization with EEG pipeline
- **Feature selection**: Focuses on most reliable metrics

## Dependencies

- pandas, numpy, scipy
- neurokit2 (for HR/GSR cleaning)
- matplotlib (for QC plots)
- PyYAML (for configuration)

## Configuration

Physio processing settings are in `config/general.yaml`:
- Sampling rates per modality
- Cleaning thresholds
- Feature extraction windows
- Output paths
