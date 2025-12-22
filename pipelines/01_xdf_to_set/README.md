# Stage 01: XDF to SET Conversion

Converts raw XDF (Extensible Data Format) files from Lab Streaming Layer to EEGLAB .set format.

## Entry Point
- **Main**: `run_xdf_to_set_parallel.py` (parallel processing)
- **Alternative**: `run_xdf_to_set_sequential.py` (sequential processing)

## Core Logic
- `xdf_to_set.py` - Main conversion function
- `xdf_to_set_legacy.py` - Original implementation (backup)

## Inputs
- **Directory**: `data/RAW/{participant_id}/`
- **Files**: `*.xdf` (raw LSL recordings)
- **Config**: `config/general.yaml`, `config/eeg_metadata.yaml`

## Outputs
- **Directory**: `output/sets/`
- **Files**: `P{id:02d}.set` + `P{id:02d}.fdt` (EEGLAB format)
- **Logs**: Processing status, event extraction, channel info

## Dependencies
- Python 3.x with pyxdf, mne, numpy, pandas
- EEGLAB channel locations: `config/chanlocs/NA-271.elc`

## Usage
```bash
python run_xdf_to_set_parallel.py
```

## Notes
- Processes all participants listed in `config/general.yaml`
- Extracts EEG, heart rate, GSR, eye tracking streams
- Creates EEGLAB event structure from LSL markers
- Parallel version recommended for batch processing
