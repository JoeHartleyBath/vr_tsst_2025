# Subjective Ratings Preprocessing

This folder contains the pipeline for aggregating VR-TSST subjective ratings. The main entry point is [scripts/preprocessing/subjective/step04_preprocess_subjective_ratings.py](scripts/preprocessing/subjective/step04_preprocess_subjective_ratings.py).

## What it does
- Loads participant questionnaire CSVs, baseline measures, and counterbalance order.
- Computes baseline-adjusted affect ratings, NASA-TLX items, and composite IMI/MPS scores.
- Aligns rows to each participant's round order, then appends results to a single aggregated CSV.

## Required inputs (under `base_directory`)
- `Main_Study_Data_Raw/In_VR_Questions/PQs_<PN>_compiled.csv` for each participant (PN = 1..48).
- `Main_Study_Data_Processed/VR-TSST Baseline Measures.xlsx` (baseline affect).
- `Main_Study_Data_Processed/VR-TSST Counterbalance sheet.xlsx` (round order per participant).

## Outputs
- Appends to `Main_Study_Subjective_Aggregated.csv` in `base_directory` (creates file if missing).

## How to run
1. Open the script and set `base_directory` to the root containing the `Main_Study_Data_Raw` and `Main_Study_Data_Processed` folders.
2. (Optional) Adjust `participants` if you want a subset.
3. From the repo root, with Python 3.9+ and pandas installed:
```powershell
python scripts/preprocessing/subjective/step04_preprocess_subjective_ratings.py
```

## Key assumptions
- Participant files are named `PQs_<PN>_compiled.csv` with columns for affect (Stress, Calm, Happy, Sad, Pleasure, Arousal), NASA (`NASA_Mental`, `NASA_Performance`, `NASA_Effort`), and IMI/MPS items used in the script.
- Counterbalance sheet columns are `Participant`, `Round 1`..`Round 4` with values matching the condition labels: Calm Addition, Calm Subtraction, Stress Addition, Stress Subtraction.
- The script appends to the aggregated CSV; delete or move the file if you need a fresh rebuild.

## Troubleshooting
- If you see `File not found` errors, verify `base_directory` and the input file names/locations.
- To avoid mixed paths, keep `base_directory` as an absolute path.
- If columns are missing in participant CSVs, the script will log which ones are absent per participant.
