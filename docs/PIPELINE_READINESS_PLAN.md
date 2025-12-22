# Pipeline Readiness and Run Guide

Use this checklist from the very start (raw intake) through features. Follow in order; do not skip.

## 0) Workspace and Git Hygiene
- `git status` clean or only intentional changes; stash/commit unrelated edits.
- No running MATLAB/Python jobs; stop stray processes.
- Activate the correct Python venv; note MATLAB/EEGLAB/AMICA versions.

### 0a) Repo Audit and Layout Sanity
- Audit directory structure: raw (`data/RAW`), converted (`output/sets`), cleaned (`output/cleaned_eeg`), QC (`output/qc`), visuals (`output/vis`), features (`output/eeg_features`), logs (`output/logs`), manifests (`output/run_manifests`).
- Identify any processed outputs that must be redone (e.g., unit/threshold fixes). Either remove them from outputs or mark them as stale (e.g., move to `output/_stale`) so reruns don’t consume bad artifacts.
- Ensure configs and scripts are not mixed with outputs; keep outputs out of version control.

## 1) Inputs and Configs
- Raw inventory: verify all expected raw files exist (paths, sizes) in `data/RAW` (or source). Do not modify raw.
- Configs present and correct: `config/general.yaml`, feature configs, `config/chanlocs/NA-271.elc`.
- `startup.m` current (toolbox discovery, SimpleYAML fallback).

## 2) Tooling Sanity
- MATLAB toolboxes: `matlab -batch "run('startup.m'); run('scripts/preprocessing/eeg/cleaning/test_toolbox_init.m')"` must pass.
- Python: venv active; required packages installed (pyxdf, scipy, pandas, tqdm, etc.).

## 3) Pilot Conversion (XDF → SET)
- Run a small subset (e.g., P01–P03): `python scripts/preprocessing/raw_conversion/run/run_xdf_to_set_parallel.py --processes N`.
- Confirm outputs in `output/sets`; check channel count/srate and unit scaling (µV).
- If pilot is good, proceed to full conversion; log commands and commit hash.

## 4) Pilot Cleaning
- Run a few subjects: `matlab -batch "run('startup.m'); run('scripts/preprocessing/eeg/cleaning/run_clean_eeg_parallel.m')"` with limited workers (fit RAM; AMICA ~8GB/subject).
- Check QC for pilot: bad channels, % samples retained, ASR repair %, ICs removed; spot-check PSD/plots.
- If QC acceptable, proceed to full cleaning; otherwise, tune thresholds and rerun pilot.

## 5) Full Conversion
- Convert all raw → `output/sets`. Verify count matches participants. Keep conversion logs.

## 6) Full Cleaning
- Clean all, with resume-safe skipping of already cleaned subjects. Use worker cap appropriate to RAM (e.g., 3–4 on 32GB).
- QC artifacts per subject:
  - Cleaned set: `output/cleaned_eeg/PXX_cleaned.set`
  - QC mat: `output/qc/PXX_qc.mat`
  - Log: `output/cleaned_eeg/PXX_processing_log.txt`
  - Visuals: `output/vis/PXX/`
- Batch checks: cleaned count vs expected; QC metrics; spot PSDs.
- If systemic high bad-channel counts, stop and retune before proceeding.

## 7) Feature Extraction
- Run features on cleaned data (example runner): `python scripts/preprocessing/features/run_features.py --subjects all`.
- Outputs: per-subject features, aggregated tables in `output/eeg_features` or `output/aggregated`.
- QC: check for NaNs, distribution sanity, expected bandpower ranges.

## 8) Provenance and Logging
- Each stage writes logs with timestamp, config, commit hash (store in `output/logs` or stage folders).
- Record commit + config names in run manifests (e.g., `output/run_manifests/{stage}_{timestamp}.json`).
- Maintain a short CHANGELOG for major pipeline changes (unit fixes, thresholds).

## 9) Git Commit Protocol
- Do not commit outputs. Stage only scripts/configs/docs.
- Commit message: `stage: summary` (e.g., `cleaning: add yaml fallback and skip cleaned subjects`).
- Tag after validated end-to-end runs (e.g., `v0.2-pipeline-ready`).

## 10) Run Order (full refresh)
1) Hygiene + tool sanity (sections 0–2).
2) Pilot conversion → review → full conversion.
3) Pilot cleaning → QC → tune → full cleaning.
4) Feature extraction → QC → aggregate.
5) Finalize: run manifests, commit, optional tag.

## 11) Triage Rules
- Stop if outputs missing participants, QC shows excessive bad channels across many subjects, or NaNs/flatlines appear post-cleaning.
- Fix, document, rerun only the affected stage; downstream stages depend on cleaned data.

## 12) Quick Commands
- Count converted: `Get-ChildItem output/sets -Filter P*.set | Measure-Object`
- Count cleaned: `Get-ChildItem output/cleaned_eeg -Filter P*_cleaned.set | Measure-Object`
- Tail cleaning log: `Get-Content output/cleaning_batch_fresh.log -Tail 40`
- List QC files: `Get-ChildItem output/qc -Filter P*_qc.mat`

Refer to this guide before every major run; avoid ad-hoc edits mid-run. If configs or thresholds change, rerun from the earliest affected stage and record in a run manifest and git commit.
