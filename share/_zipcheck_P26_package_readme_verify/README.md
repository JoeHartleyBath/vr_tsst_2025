# P26 Package

This package contains:
- Raw EEG (EEGLAB): `raw/`
- Cleaned EEG (EEGLAB): `cleaned/`
- Event label mapping + descriptions: `labels/`

## Files
- Raw: `raw/P26_raw_chanlocs.set` + `raw/P26_raw_chanlocs.fdt`
- Cleaned: `cleaned/P26_cleaned.set` + `cleaned/P26_cleaned.fdt`
- Event labels: `labels/event_labels.md`

## EEG Info
- Sampling rate: 500 Hz (raw), 125 Hz (cleaned)
- Channels: 128; channel locations included
- Reference: raw = common; cleaned = average (after ICA)
- Cleaned preprocessing:
	- Bandpass 1–49 Hz
	- Notch 50 Hz
	- AMICA + ICLabel; artifactual ICs removed
	- Bad channels interpolated (spherical)

## Event codes
Use the mapping in `labels/`.

