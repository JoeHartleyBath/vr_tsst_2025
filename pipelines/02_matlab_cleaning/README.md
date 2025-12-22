# Stage 02: MATLAB EEG Cleaning

Performs artifact removal and ICA decomposition using AMICA on EEGLAB .set files.

## Entry Point
- **Main**: `run_clean_eeg_pipeline.m`

## Core Logic
- `clean_eeg.m` - Core cleaning function with ASR and AMICA
- `fix_eeglab_set_files.m` - Utility to convert embedded .set to .set+.fdt format

## Inputs
- **Directory**: `output/sets/`
- **Files**: `P{id:02d}.set` (from Stage 01)
- **Config**: `config/eeglab_template.yaml`

## Outputs
- **Directory**: `output/cleaned_eeg/`
- **Files**: `P{id:02d}_cleaned.mat` (cleaned EEG data with ICA)
- **QC**: Visual plots, ICLabel classifications, rejection logs

## Dependencies
- MATLAB R2021b+ with EEGLAB 2025.1.0
- EEGLAB plugins: clean_rawdata, AMICA, ICLabel, dipfit, firfilt

## Processing Steps
1. Load .set file
2. Remove bad channels (clean_rawdata)
3. Artifact Subspace Reconstruction (ASR)
4. Re-reference to average
5. AMICA ICA decomposition (30 iterations)
6. ICLabel component classification
7. Reject non-brain components
8. Save cleaned data

## Usage
```matlab
run('run_clean_eeg_pipeline.m')
```

## Known Issues
- P10, P14, P23 fail with "Maximum variable size exceeded" errors
- These participants may need to be excluded from analysis
- Memory-intensive: requires 16+ GB RAM
