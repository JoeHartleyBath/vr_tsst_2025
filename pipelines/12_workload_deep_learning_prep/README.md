# Pipeline 12: Workload Deep Learning Preparation (Data Export)

## Purpose
This pipeline transforms the cleaned EEG data (EEGLAB .set format) into MNE-Python compliant Epochs (`.fif` files) ready for deep learning training. It serves as the bridge between the preprocessing/cleaning stages and the deep learning training stage.

## Key Steps
1.  **Loading**: Loads cleaned `.set` files for each participant.
2.  **Epoching**: Segments continuous data into fixed-length windows (e.g., 2 seconds) with specific overlap.
3.  **Labeling**: Assigns workload labels (low/medium/high) to each epoch based on experimental markers.
4.  **Export**: Saves the epoched data as `.fif` files.

## Scripts
- `export_mne_epochs.py`: The main script that performs the conversion.

## Input
- `output/sets/*_clean.set` (Cleaned EEG data from Stage 2/10)

## Output
- `output/adaptive_workload/mne_epochs/*.fif` (MNE Epochs files)

## Dependencies
- **MNE-Python**: Used for data handling and export.
- **EEGLAB**: Source data format.

## Next Stage
- **Pipeline 13**: Deep Learning Training (Consumes the `.fif` files generated here).
