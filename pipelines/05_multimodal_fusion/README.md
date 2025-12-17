# Stage 5: Multimodal Feature Fusion

Merges EEG, physiological, and subjective features into a unified dataset for analysis.

## Overview

This stage combines data from three preprocessing pipelines:
- **EEG features** (Stage 3): Band power, asymmetry, entropy metrics
- **Physio features** (Stage 4): Heart rate, GSR, pupil dilation, blink metrics  
- **Subjective ratings** (Stage 4b): Affect, workload, motivation, presence scores

## Usage

From project root:
```bash
python pipelines/05_multimodal_fusion/merge_all_features.py
```

Or using the stage runner:
```bash
python scripts/stages/run_stage_5_merge.py
```

## Arguments

```bash
--output PATH       # Output file path (default: output/aggregated/all_data_aggregated.csv)
--force             # Force merge even if output exists
--eeg PATH          # Custom EEG features input path
--physio PATH       # Custom physio features input path
--subjective PATH   # Custom subjective ratings input path
```

## Input Files

- `output/aggregated/eeg_features.csv` - EEG features (127 columns per participant-condition)
- `output/aggregated/physio_features.csv` - Physio features (~40 columns per participant-condition)
- `output/aggregated/subjective.csv` - Subjective ratings (17 columns per participant-condition)

## Output

- `output/aggregated/all_data_aggregated.csv` - Full merged dataset
  - One row per participant-condition
  - ~180+ feature columns
  - Long format: Participant_ID, Condition, [all features]

## Merge Strategy

- **Key**: `[Participant_ID, Condition]`
- **Type**: Outer join (includes all participants from any dataset)
- **Validation**: Checks for missing data, type consistency, overlapping participants

## Example

```bash
# Standard merge
python pipelines/05_multimodal_fusion/merge_all_features.py

# Force regenerate
python pipelines/05_multimodal_fusion/merge_all_features.py --force

# Custom output location
python pipelines/05_multimodal_fusion/merge_all_features.py --output results/full_dataset.csv
```

## Notes

- Run this after all feature extraction pipelines complete
- Fast operation (~seconds) - just loads and merges CSVs
- Can be re-run anytime to regenerate merged dataset
- Use `--force` to overwrite existing output
