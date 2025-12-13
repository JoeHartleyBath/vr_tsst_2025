# VR-TSST Pipeline: Quick Start

## Prerequisites Checklist

Before running the pipeline, ensure:

- [ ] Python 3.9+ with venv activated: `venv\Scripts\Activate.ps1`
- [ ] All Python packages installed: `pip list | grep pyxdf,mne,neurokit2`
- [ ] R 4.5.2 installed: `R --version`
- [ ] MATLAB R2020a+ installed: `matlab -v`
- [ ] EEGLAB toolbox in `c:/MATLAB/toolboxes/eeglab`
- [ ] AMICA toolbox in `c:/MATLAB/toolboxes/amica`
- [ ] Raw data staged in `data/raw/eeg/`, `data/raw/metadata/`, `data/raw/subjective/`

## Run Pilot Test (P01–P03)

**Estimated Time**: 3–4 hours (mostly AMICA cleaning)

### Step 1: Activate Python Environment
```powershell
cd c:\vr_tsst_2025
venv\Scripts\Activate.ps1
```

### Step 2: Run All 6 Pipeline Stages
```powershell
python scripts/run_pipeline_master.py
```

**What This Does**:
- Stage 1: Converts XDF (raw EEG) to SET (EEGLAB)
- Stage 2: Cleans EEG via ICA/AMICA (slowest)
- Stage 3: Extracts EEG features (spectral, temporal, complexity)
- Stage 4: Extracts physio features (EDA, HR, GSR)
- Stage 5: Merges all features into single CSV
- Stage 6: R preprocessing and analysis

### Step 3: Monitor Progress
```powershell
# In another terminal, watch the log
Get-Content -Path output/logs/pipeline_*.log -Wait

# Or check intermediate outputs
Get-ChildItem output/sets/ -Filter "*.set" | Measure-Object
Get-ChildItem output/eeg_features/ -Filter "*.csv" | Measure-Object
```

### Step 4: Validate Results
```powershell
# Compare new pipeline to old results
python scripts/validation/compare_pipelines.py

# Opens a report at: output/comparison_report.json
# Check: Match rate > 95% = success
```

## If Something Fails

### Option A: Restart from Failure Point
```powershell
# Run only stages 4, 5, 6 (skip EEG cleaning, which is slowest)
python scripts/run_pipeline_master.py --stages 4 5 6
```

### Option B: Debug Single Stage
```powershell
# Run Stage 4 directly
python scripts/stages/run_stage_4_physio_features.py

# View error message and fix
```

### Option C: Check Logs
```powershell
# View full log
Get-Content output/logs/pipeline_*.log | tail -50
```

## Expected Outputs

**✓ Success**: You should see these files:

```
output/
├── final_data.csv                  # Master feature matrix (48 rows × 200+ features)
├── aggregated/
│   ├── eeg_features.csv           # EEG-only (spectral, complexity, etc.)
│   └── all_data_aggregated.csv    # With metadata (age, condition, etc.)
├── qc/
│   └── summary/
│       └── qc_failures_summary.csv # ICA failures (if any)
└── logs/
    └── pipeline_*.log              # Execution log
```

**✓ Validation**: Run comparison:
```powershell
python scripts/validation/compare_pipelines.py
# Reports: correlation, RMSE per feature
# Pass = r > 0.95 (or your tolerance threshold)
```

## Next Steps

| Step | Command | Purpose |
|------|---------|---------|
| 1. Pilot test | `python scripts/run_pipeline_master.py` | Test P01–P03 |
| 2. Validate | `python scripts/validation/compare_pipelines.py` | Compare to old |
| 3. Full run | Update script for P01–P48, re-run | Process all 48 |
| 4. Archive | `git add output/ && git commit` | Save results |

## Key Commands Reference

```bash
# Run full pipeline
python scripts/run_pipeline_master.py

# Run specific stages (1=raw, 4=physio, 5=merge, 6=R analysis)
python scripts/run_pipeline_master.py --stages 4 5 6

# Run range of stages
python scripts/run_pipeline_master.py --stages 1-3

# Compare results
python scripts/validation/compare_pipelines.py

# View logs
Get-Content output/logs/pipeline_*.log

# Check raw data
Get-ChildItem data/raw/eeg/ | Measure-Object
```

## Troubleshooting

| Problem | Solution |
|---------|----------|
| "MATLAB not found" | Install MATLAB, add to PATH, restart terminal |
| "EEGLAB not found" | Verify path: `c:/MATLAB/toolboxes/eeglab` |
| Stage 4 fails | Ensure `output/sets/P01_clean.set` exists (check Stage 2) |
| Low correlation | Review configs; ICA is non-deterministic (set seed) |
| Out of disk space | Check `data/raw/` size (~300GB); use external drive |

---

**Quick Help**: `python scripts/run_pipeline_master.py --help`
