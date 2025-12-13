# Pipeline Validation Workflow

This document outlines the P01–P03 pilot test and full 48-participant validation process.

## Quick Start: Run the Full Pipeline

```bash
# Terminal: PowerShell or Git Bash
cd c:\vr_tsst_2025

# Run all 6 stages for P01-P03
python scripts/run_pipeline_master.py

# Or run specific stages
python scripts/run_pipeline_master.py --stages 1 4 5
python scripts/run_pipeline_master.py --stages 1-3
```

## Stage Runners (Individual)

For debugging individual stages:

```bash
# Stage 1: XDF → SET (Python)
python scripts/stages/run_stage_1_xdf_to_set.py

# Stage 2: EEG Cleaning (MATLAB)
powershell -ExecutionPolicy Bypass scripts/stages/run_stage_2_eeg_cleaning.ps1

# Stage 3: EEG Features (MATLAB)
powershell -ExecutionPolicy Bypass scripts/stages/run_stage_3_eeg_features.ps1

# Stage 4: Physio Features (Python)
python scripts/stages/run_stage_4_physio_features.py

# Stage 5: Merge Features (Python)
python scripts/stages/run_stage_5_merge_features.py

# Stage 6: R Preprocessing (R)
powershell -ExecutionPolicy Bypass scripts/stages/run_stage_6_r_preprocessing.ps1
```

## Validation: Compare Old vs. New Results

After running the pilot (or full pipeline), compare outputs:

```bash
# Default: compares results/baseline_old/final_data.csv → output/final_data.csv
python scripts/validation/compare_pipelines.py

# Custom paths
python scripts/validation/compare_pipelines.py \
  --old results/old_pipeline_output/final_data.csv \
  --new output/final_data.csv \
  --tolerance 0.05 \
  --output output/comparison_report.json
```

This generates a report with:
- **Correlation (r)**: How well new features match old (r > 0.95 = good)
- **RMSE**: Absolute differences
- **% Change**: Relative shift in feature values
- **Pass/Fail**: Green ✓ or Red ⚠ per feature

## Workflow: Pilot → Validation → Full Run

### Phase 1: Pilot (P01–P03, ~3–4 hours)
1. Ensure MATLAB/EEGLAB/AMICA installed
2. Run: `python scripts/run_pipeline_master.py`
3. Check output:
   - `output/logs/pipeline_YYYYMMDD_HHMMSS.log` (full log)
   - `output/final_data.csv` (merged features)
   - `output/aggregated/eeg_features.csv` (EEG features)
4. Spot-check a few participants manually

### Phase 2: Validation (5–10 mins)
1. If pilot succeeded, compare to old results:
   ```bash
   python scripts/validation/compare_pipelines.py
   ```
2. Review `output/comparison_report.json`:
   - Match rate > 95% → ✓ Proceed to full run
   - Match rate 80–95% → ⚠ Investigate divergences
   - Match rate < 80% → ✗ Debug immediately (see troubleshooting below)

### Phase 3: Full Run (All 48 participants, ~1 day)
1. Update `run_pipeline_master.py` to include all participants (lines ~20–23)
2. Run overnight/weekend: `python scripts/run_pipeline_master.py`
3. Final validation: `python scripts/validation/compare_pipelines.py`

## Output Structure

After running the pipeline:

```
output/
├── logs/
│   └── pipeline_20250120_153000.log       # Full execution log
├── final_data.csv                          # Master merged features
├── final_data.rds                          # R-compatible format
├── aggregated/
│   ├── eeg_features.csv                   # EEG-only features
│   └── all_data_aggregated.csv            # With metadata
├── qc/
│   └── summary/
│       └── qc_failures_summary.csv        # ICA/cleaning QC report
└── comparison_report.json                  # Validation report (after compare_pipelines.py)
```

## Troubleshooting

### Issue: MATLAB not found
```
ERROR: MATLAB not found in PATH
```
**Solution**: Install MATLAB, ensure it's in system PATH. Verify with:
```bash
matlab -v
```

### Issue: EEGLAB/AMICA not configured
```
Error: EEGLAB toolbox not found
```
**Solution**: Add paths in MATLAB startup:
```matlab
addpath(genpath('C:/MATLAB/toolboxes/eeglab'))
addpath(genpath('C:/MATLAB/toolboxes/amica'))
```

### Issue: Low correlation (r < 0.90)
**Potential causes**:
1. Different preprocessing thresholds (check `config/eeg_metadata.yaml`)
2. ICA seed changed (AMICA non-deterministic—set seed in script)
3. Different feature definitions (check `config/eeg_feature_extraction.yaml`)

**Fix**: 
- Review configs and comments in old pipeline scripts
- Run old pipeline in parallel for comparison
- Check if features are truly different or just scaled/transformed

### Issue: Missing output files
1. Check `output/logs/pipeline_*.log` for errors
2. Run individual stage: `python scripts/stages/run_stage_X_*.py`
3. Look for intermediate outputs:
   - Stage 1: `output/sets/P01.set`, `P01.fdt`
   - Stage 2: `output/sets/P01_clean.set`
   - Stage 3: `output/eeg_features/P01_features.csv`
   - Stage 4: `output/physio_features/P01_physio.csv`

## Key Files

| File | Purpose |
|------|---------|
| `scripts/run_pipeline_master.py` | Master orchestrator (all stages) |
| `scripts/stages/run_stage_*.py` | Individual stage runners (debug) |
| `scripts/validation/compare_pipelines.py` | Validation comparison |
| `output/logs/` | Pipeline execution logs |
| `output/final_data.csv` | Final merged feature matrix |
| `output/comparison_report.json` | Validation report |

## Monitoring Long Runs

For the full 48-participant run, you can monitor progress in a separate terminal:

```bash
# Terminal 1: Start pipeline
python scripts/run_pipeline_master.py

# Terminal 2: Monitor logs (watch for errors)
Get-Content -Path output/logs/pipeline_*.log -Wait

# Terminal 3: Check intermediate outputs
Get-ChildItem output/sets/*.set | Measure-Object
Get-ChildItem output/eeg_features/*.csv | Measure-Object
```

## Next Steps

- [ ] Finish MATLAB/EEGLAB/AMICA installation
- [ ] Run P01–P03 pilot test
- [ ] Validate results (correlation, RMSE)
- [ ] Fix any divergences
- [ ] Run full 48-participant pipeline
- [ ] Archive results and commit to git

---

*Last updated: 2025*
*Contact: Joe Hartley <jh3968@bath.ac.uk>*
