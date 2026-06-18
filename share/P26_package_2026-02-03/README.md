# P26 Share Package (VR-TSST)

This folder contains a single-participant dataset (P26) prepared for external testing of affective-state feedback.

## Contents

### Cleaned EEG (recommended starting point)
- `cleaned/P26_cleaned.set` + `cleaned/P26_cleaned.fdt`
  - Load in **EEGLAB** by opening the `.set` (it will automatically read the `.fdt`).
  - Load in **Python (MNE)** with `mne.io.read_raw_eeglab('P26_cleaned.set', preload=False)`.

### Raw (source) data
- `raw/P26.xdf` (EEG + other streams as captured)
- `raw/P26.csv` (synchronized metadata/physio table used for condition assignment)

### Raw EEGLAB with channel locations (metadata-only copy)
- `raw/P26_raw_chanlocs.set` + `raw/P26_raw_chanlocs.fdt` (if present)
  - This is a NEW copy created specifically for sharing so the raw EEGLAB file includes channel locations.

### Label / condition ontology
- `config/conditions.yaml`
  - Defines the condition filters and the exported event codes.
  - Note: some task labels intentionally share the same event code (e.g., `...1022_Task` and `...2043_Task`).
- `config/NA-271.elc` (ANT Neuro 128 layout file used for chanlocs)

### QC + sanity checks
- `reports/QC_P26.txt` (summary QC)
- `reports/P26_cleaned_condition_psd.png` (minor condition-level PSD sanity plot)
- `reports/manifest_verify_psd.json` (file presence + basic metadata)

## Event / label mapping
Event codes are stored as annotations in the EEG `.set` file (e.g., `101`, `102`, `103`, `104`, `200`, `201`).
See `config/conditions.yaml` → `export_event_labels` for the authoritative mapping.

## How to regenerate the raw-with-chanlocs file (if needed)
MATLAB batch script:
- `scripts/sharing/run_add_chanlocs_p26.m`

## Notes
- The `.fdt` contains samples; the `.set` contains metadata/events and points to the `.fdt`.
- The raw `.csv` includes many columns/streams; downstream users usually only need the condition-related columns.
