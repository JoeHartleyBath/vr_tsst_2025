# Stage 03: MATLAB EEG Feature Extraction

Extracts spectral power, band ratios, and entropy features from cleaned EEG data.

## Entry Point
- **Main**: `extract_eeg_features.m` (condition-based extraction)
- **Alternative**: `extract_eeg_features_rolling_windows.m` (rolling window approach)

## Core Logic
Located in `private/` subdirectory:
- `compute_band_power.m` - Delta, theta, alpha, beta, gamma power
- `compute_ratios.m` - Theta/beta, alpha/theta ratios
- `compute_spectral_entropy.m` - Shannon entropy of PSD
- `compute_sample_entropy.m` - Time-series complexity
- `calc_psd.m` - Power spectral density computation
- Additional helpers for parallel processing and file management

## Inputs
- **Directory**: `output/cleaned_eeg/`
- **Files**: `P{id:02d}_cleaned.mat` (from Stage 02)
- **Config**: `config/eeg_feature_extraction.yaml`, `config/eeg_metadata.yaml`

## Outputs
- **Directory**: `output/aggregated/`
- **File**: `eeg_features.csv` (all participants × conditions × features)
- **Logs**: Feature extraction progress, errors, timing

## Features Extracted
- **Spectral Power**: Delta (0.5-4 Hz), Theta (4-8 Hz), Alpha (8-12 Hz), Beta (12-30 Hz), Gamma (30-50 Hz)
- **Power Ratios**: Theta/Beta, Alpha/Theta
- **Entropy**: Spectral entropy, Sample entropy
- **Per Channel**: All features computed for each EEG channel
- **Conditions**: Baseline, Math1, Math2, Math3, Social, Recovery, Forest

## Dependencies
- MATLAB R2021b+ with Signal Processing Toolbox
- EEGLAB (for data loading)
- Parallel Computing Toolbox (optional, for speed)

## Usage
```matlab
run('extract_eeg_features.m')
```

## Performance
- ~2-5 minutes per participant (parallel mode)
- ~10-20 minutes per participant (sequential mode)
- Supports resumption if interrupted
