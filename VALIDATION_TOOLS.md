# Validation Tools & Workflow Summary

## What's New

You now have a complete **validation infrastructure** for the P01–P03 pilot test and full 48-participant run:

### Core Tools

| Tool | Purpose | Command |
|------|---------|---------|
| **Master Orchestrator** | Run pipeline stages 1–6 with logging | `python scripts/run_pipeline_master.py` |
| **Health Check** | Validate all dependencies before running | `python scripts/check_pipeline_health.py` |
| **Comparison Validator** | Compare new vs. old results | `python scripts/validation/compare_pipelines.py` |
| **Stage Runners** | Debug individual stages | `python scripts/stages/run_stage_*.py` |

### Documentation

| Doc | Use Case |
|-----|----------|
| [PILOT_TEST_GUIDE.md](../PILOT_TEST_GUIDE.md) | Quick-start for P01–P03 |
| [scripts/validation/README.md](validation/README.md) | Detailed validation workflow |
| [scripts/preprocessing/subjective/README.md](preprocessing/subjective/README.md) | Subjective data pipeline |
| [data/raw/README.md](../../data/raw/README.md) | Raw data structure & staging |

---

## Validation Workflow (Step-by-Step)

### Phase 1: Pre-Run Validation (5 min)

```powershell
cd c:\vr_tsst_2025
python scripts/check_pipeline_health.py
```

**Checks**:
- ✓ Python packages (pyxdf, mne, neurokit2, etc.)
- ✓ R installation (Rscript in PATH)
- ✓ MATLAB installation
- ✓ EEGLAB & AMICA toolboxes
- ✓ Raw data staged (48x EEG, metadata, subjective)

**Status**: Green ✓ → proceed; Red ✗ → fix issues

### Phase 2: Pilot Execution (3–4 hours)

```powershell
# Activate Python venv
venv\Scripts\Activate.ps1

# Run P01-P03 full pipeline
python scripts/run_pipeline_master.py
```

**Outputs**:
- `output/logs/pipeline_YYYYMMDD_HHMMSS.log` — Full execution trace
- `output/final_data.csv` — Merged feature matrix (3 participants × ~250 features)
- `output/aggregated/eeg_features.csv` — EEG features only
- `output/qc/summary/qc_failures_summary.csv` — ICA cleaning report

### Phase 3: Pilot Validation (5–10 min)

```powershell
# Compare new results to old baseline
python scripts/validation/compare_pipelines.py

# Review detailed report
Get-Content output/comparison_report.json
```

**Metrics Reported**:
- **Correlation (r)**: How well new features match old (r > 0.95 = excellent)
- **RMSE**: Absolute differences across participants
- **% Change**: Relative shift in feature values
- **Status**: ✓ OK (match) or ⚠ DIVERGE (mismatch)

**Decision Gate**:
- **Match rate > 95%** → ✓ Proceed to full 48-participant run
- **Match rate 80–95%** → ⚠ Investigate divergences (check configs, ICA seed)
- **Match rate < 80%** → ✗ Stop; debug immediately (likely pipeline change needed)

### Phase 4: Full Run (≤24 hours, can run overnight)

Once validated:

```powershell
# Update run_pipeline_master.py line ~20 to include all 48 participants
# participants = list(range(1, 49))  # Change from [1, 2, 3]

# Run full pipeline (will take ~16–24 hours)
python scripts/run_pipeline_master.py

# Monitor in separate terminal
Get-Content output/logs/pipeline_*.log -Wait
```

### Phase 5: Full Validation & Archival

```powershell
# Validate all 48 participants
python scripts/validation/compare_pipelines.py

# Check results
Get-Content output/final_data.csv -TotalCount 5  # View first 5 rows

# Commit to git with full results
git add output/ scripts/ config/
git commit -m "Full 48-participant pipeline run with validation: $(Get-Date -Format 'yyyy-MM-dd')"
```

---

## Debugging: Individual Stages

If a stage fails, run it independently:

```powershell
# Stage 1: XDF → SET (5–10 min)
python scripts/stages/run_stage_1_xdf_to_set.py

# Stage 2: EEG Cleaning / AMICA (2–3 hours) ← Usually slowest
powershell -ExecutionPolicy Bypass scripts/stages/run_stage_2_eeg_cleaning.ps1

# Stage 3: EEG Features (15–30 min)
powershell -ExecutionPolicy Bypass scripts/stages/run_stage_3_eeg_features.ps1

# Stage 4: Physio Features (5–10 min)
python scripts/stages/run_stage_4_physio_features.py

# Stage 5: Merge Features (2–5 min)
python scripts/stages/run_stage_5_merge_features.py

# Stage 6: R Preprocessing (5–15 min, optional)
powershell -ExecutionPolicy Bypass scripts/stages/run_stage_6_r_preprocessing.ps1
```

Check logs:
```powershell
Get-Content output/logs/pipeline_*.log | Select-String "ERROR|WARN" -Context 2
```

---

## Key Success Criteria

### Pilot Test
- [ ] All 6 stages complete without fatal errors
- [ ] Output files exist (final_data.csv, EEG features, physio features)
- [ ] Correlation with old results > 95%
- [ ] No missing participants

### Full Run
- [ ] All 48 participants process successfully
- [ ] Final feature matrix: 48 rows × ~250 features
- [ ] Correlation maintained > 95%
- [ ] Results committed to git with timestamped log

---

## File Manifest

**New Files Created**:

```
scripts/
├── run_pipeline_master.py                      # Main orchestrator
├── check_pipeline_health.py                    # Pre-run validation
├── stages/
│   ├── run_stage_1_xdf_to_set.py             # Individual stage runners
│   ├── run_stage_2_eeg_cleaning.ps1
│   ├── run_stage_3_eeg_features.ps1
│   ├── run_stage_4_physio_features.py
│   ├── run_stage_5_merge_features.py
│   └── run_stage_6_r_preprocessing.ps1
└── validation/
    ├── compare_pipelines.py                   # Comparison validator
    └── README.md                              # Detailed workflow docs

PILOT_TEST_GUIDE.md                            # Quick-start guide
```

---

## Next Steps

### Immediate (Today/Tomorrow)
1. ✓ **Pre-check**: `python scripts/check_pipeline_health.py`
2. ⏳ **Pilot run**: `python scripts/run_pipeline_master.py` (3–4 hours)
3. ⏳ **Validate**: `python scripts/validation/compare_pipelines.py` (5 min)

### Follow-Up
4. ⏳ **Full run**: Update script for P01–P48, run overnight (16–24 hours)
5. ⏳ **Final validation**: Compare all 48 results to old pipeline
6. ⏳ **Archive**: Commit to git with full provenance

---

## Important Notes

### MATLAB/EEGLAB Setup
Stage 2 (EEG Cleaning) requires MATLAB with EEGLAB and AMICA installed. If not done:
```matlab
% In MATLAB console:
addpath(genpath('c:/MATLAB/toolboxes/eeglab'))
addpath(genpath('c:/MATLAB/toolboxes/amica'))
savepath  % Persist paths
```

### ICA Non-Determinism
AMICA ICA is **not fully deterministic** (varies with OS/CPU). If you see r = 0.97 instead of r = 0.99, that's expected. Set tolerance threshold to 0.05 (r > 0.95).

### Disk Space
- Raw data: ~300 GB (data/raw/)
- Processed outputs: ~50–100 GB (output/ + staging)
- **Total**: ~400–500 GB needed

### Monitoring
For long runs, create a separate terminal to watch progress:
```powershell
# Terminal 1: Run pipeline
python scripts/run_pipeline_master.py

# Terminal 2: Monitor logs
Get-Content output/logs/pipeline_*.log -Wait

# Terminal 3: Check intermediate outputs
Watch { Get-ChildItem output/eeg_features/*.csv | Measure-Object -Property Length -Sum }
```

---

## Quick Reference

| Task | Command |
|------|---------|
| Health check | `python scripts/check_pipeline_health.py` |
| Run all 6 stages | `python scripts/run_pipeline_master.py` |
| Run stages 1–3 only | `python scripts/run_pipeline_master.py --stages 1-3` |
| Run stage 4 only | `python scripts/stages/run_stage_4_physio_features.py` |
| Compare results | `python scripts/validation/compare_pipelines.py` |
| View latest log | `Get-Content output/logs/pipeline_*.log \| tail` |
| Check output files | `Get-ChildItem output/final_data.csv, output/aggregated/` |

---

**Created**: January 2025  
**Author**: Joe Hartley  
**Contact**: jh3968@bath.ac.uk  
**Status**: Ready for pilot test
