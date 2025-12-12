# Preprocessing Scripts

Data preparation pipeline: raw data → cleaned data → ML-ready features

## Structure

```
preprocessing/
├── raw_conversion/    # Convert XDF to EEGLAB .set format
├── eeg/               # EEG cleaning and feature extraction
├── physio/            # Physiological data cleaning and features
└── subjective/        # Self-report ratings processing
```

## Pipeline Flow

1. **Raw Conversion** (`raw_conversion/`):
   - Input: `.xdf` files from LSL recordings
   - Output: EEGLAB `.set` files + event markers
   - Entry: `run/run_xdf_to_set_end2end.py`

2. **EEG Processing** (`eeg/`):
   - **Cleaning** (`cleaning/`): Artifact rejection, ICA, epoch rejection
   - **Feature Extraction** (`feature_extraction/`): Power bands, entropy, ratios
   - Entry: `feature_extraction/extract_eeg_feats.py`

3. **Physio Processing** (`physio/`):
   - **Cleaning**: HR ectopic correction, GSR filtering, eye blink handling
   - **Feature Extraction**: HRV, EDA, pupil metrics, task responses
   - Entry: `feature_extraction/extract_physio_features.py`

4. **Subjective Ratings** (`subjective/`):
   - Processing self-report questionnaires (stress, workload)
   - Entry: `step04_preprocess_subjective_ratings.py`

## Output

- Cleaned data: `output/processed/`
- Features: `output/aggregated/all_data_aggregated.csv`
- QC logs: `output/qc/`
