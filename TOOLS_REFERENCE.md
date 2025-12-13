# Pipeline Tools & Commands: Complete Reference

## Overview

Your VR-TSST pipeline is now **fully instrumented** for validation and debugging. This document is your command reference.

---

## 🚀 Quick Start (Copy-Paste Commands)

### 1. Pre-Flight Check (5 minutes)
```powershell
cd c:\vr_tsst_2025
python scripts/check_pipeline_health.py
```

### 2. Run Pilot Test: P01–P03 (3–4 hours)
```powershell
venv\Scripts\Activate.ps1
python scripts/run_pipeline_master.py
```

### 3. Validate Results (5 minutes)
```powershell
python scripts/validation/compare_pipelines.py
```

---

## 📋 All Available Tools

### Master Orchestrator
**File**: `scripts/run_pipeline_master.py`  
**Purpose**: Run pipeline stages 1–6 with logging and error handling  
**Usage**:
```bash
# Run all stages (default: P01-P03)
python scripts/run_pipeline_master.py

# Run specific stages
python scripts/run_pipeline_master.py --stages 1 4 5

# Run range
python scripts/run_pipeline_master.py --stages 1-3

# Custom participants (requires manual edit)
python scripts/run_pipeline_master.py
```

**Output**:
- `output/logs/pipeline_YYYYMMDD_HHMMSS.log` — Execution log
- `output/final_data.csv` — Master feature matrix
- `output/aggregated/` — Aggregated features (EEG, all data)

---

### Health Check
**File**: `scripts/check_pipeline_health.py`  
**Purpose**: Validate all dependencies before running  
**Usage**:
```bash
python scripts/check_pipeline_health.py
```

**Checks**:
- ✓ Python packages (pyxdf, mne, neurokit2, etc.)
- ✓ R installation (Rscript)
- ✓ MATLAB installation
- ✓ EEGLAB & AMICA toolboxes
- ✓ Raw data staging (48 participants)

**Output**: Pass/fail status for each check

---

### Comparison Validator
**File**: `scripts/validation/compare_pipelines.py`  
**Purpose**: Compare new pipeline results vs. old baseline  
**Usage**:
```bash
# Default: results/baseline_old/final_data.csv vs output/final_data.csv
python scripts/validation/compare_pipelines.py

# Custom paths
python scripts/validation/compare_pipelines.py \
  --old results/old_pipeline_output/final_data.csv \
  --new output/final_data.csv \
  --tolerance 0.05 \
  --output output/comparison_report.json
```

**Output**: `output/comparison_report.json`  
**Metrics**:
- Correlation (r) per feature
- RMSE (absolute difference)
- % Change (relative difference)
- Pass/fail status (r > 0.95 = OK)

---

### Individual Stage Runners

#### Stage 1: XDF → SET
**File**: `scripts/stages/run_stage_1_xdf_to_set.py`  
**Purpose**: Convert raw XDF (EEG) to EEGLAB SET format  
**Duration**: 5–10 min  
**Usage**:
```bash
python scripts/stages/run_stage_1_xdf_to_set.py
```

#### Stage 2: EEG Cleaning (ICA/AMICA)
**File**: `scripts/stages/run_stage_2_eeg_cleaning.ps1`  
**Purpose**: Remove artifacts via Independent Component Analysis  
**Duration**: 2–3 hours (AMICA is slow)  
**Usage**:
```powershell
powershell -ExecutionPolicy Bypass scripts/stages/run_stage_2_eeg_cleaning.ps1
```

#### Stage 3: EEG Feature Extraction
**File**: `scripts/stages/run_stage_3_eeg_features.ps1`  
**Purpose**: Extract spectral, temporal, and complexity features  
**Duration**: 15–30 min  
**Usage**:
```powershell
powershell -ExecutionPolicy Bypass scripts/stages/run_stage_3_eeg_features.ps1
```

#### Stage 4: Physio Feature Extraction
**File**: `scripts/stages/run_stage_4_physio_features.py`  
**Purpose**: Extract EDA, HR, GSR features from CSV  
**Duration**: 5–10 min  
**Usage**:
```bash
python scripts/stages/run_stage_4_physio_features.py
```

#### Stage 5: Feature Merge
**File**: `scripts/stages/run_stage_5_merge_features.py`  
**Purpose**: Combine EEG and physio features  
**Duration**: 2–5 min  
**Usage**:
```bash
python scripts/stages/run_stage_5_merge_features.py
```

#### Stage 6: R Preprocessing
**File**: `scripts/stages/run_stage_6_r_preprocessing.ps1`  
**Purpose**: Statistical preprocessing in R (optional)  
**Duration**: 5–15 min  
**Usage**:
```powershell
powershell -ExecutionPolicy Bypass scripts/stages/run_stage_6_r_preprocessing.ps1
```

---

## 📊 Common Workflows

### Workflow 1: Quick Sanity Check (15 min)
```powershell
cd c:\vr_tsst_2025

# Check dependencies
python scripts/check_pipeline_health.py

# Run just stages 1, 4, 5 (skip slow AMICA)
python scripts/run_pipeline_master.py --stages 1 4 5

# Check outputs exist
Get-ChildItem output/final_data.csv
```

### Workflow 2: Debug Individual Stage (Varies)
```powershell
# Stage 2 failing? Run it alone
powershell -ExecutionPolicy Bypass scripts/stages/run_stage_2_eeg_cleaning.ps1

# Check logs for errors
Get-Content output/logs/pipeline_*.log | Select-String "ERROR"

# Fix issue, then retry stages 2-6
python scripts/run_pipeline_master.py --stages 2-6
```

### Workflow 3: Full Pilot → Validation (4 hours + 5 min)
```powershell
# Activate Python
venv\Scripts\Activate.ps1

# Run all 6 stages (default P01-P03)
python scripts/run_pipeline_master.py

# Wait for completion...

# Validate results
python scripts/validation/compare_pipelines.py

# Review report
notepad output/comparison_report.json
```

### Workflow 4: Full 48-Participant Run (Overnight)
```powershell
# 1. Edit run_pipeline_master.py line ~20:
#    participants = list(range(1, 49))  # Change from [1, 2, 3]

# 2. Run (will take 16–24 hours)
python scripts/run_pipeline_master.py

# 3. Monitor in separate terminal
Get-Content output/logs/pipeline_*.log -Wait

# 4. After completion, validate
python scripts/validation/compare_pipelines.py

# 5. Commit to git
git add output/ scripts/
git commit -m "Full 48-participant pipeline: $(Get-Date -Format 'yyyy-MM-dd')"
```

---

## 🔍 Monitoring & Debugging

### View Current Logs
```powershell
# Latest log (real-time)
Get-Content output/logs/pipeline_*.log -Wait

# Search for errors
Get-Content output/logs/pipeline_*.log | Select-String "ERROR|FAIL" -Context 2

# View last 50 lines
Get-Content output/logs/pipeline_*.log | tail -50
```

### Check Intermediate Outputs
```powershell
# Count processed participants
Get-ChildItem output/sets/*.set | Measure-Object
Get-ChildItem output/eeg_features/*.csv | Measure-Object
Get-ChildItem output/physio_features/*.csv | Measure-Object

# View first 5 rows of final data
Get-Content output/final_data.csv | Select-Object -First 5

# File sizes
Get-ChildItem output/ -Recurse | Select-Object Name, Length | Sort Length -Desc | head -20
```

### Restart Failed Pipeline
```powershell
# If stage 3 fails, restart from stage 3
python scripts/run_pipeline_master.py --stages 3-6

# Or manually run stage
powershell -ExecutionPolicy Bypass scripts/stages/run_stage_3_eeg_features.ps1
```

---

## 📂 Output Files & Locations

| File | Purpose | Created By |
|------|---------|-----------|
| `output/final_data.csv` | Master feature matrix (all participants × ~250 features) | Stage 5 |
| `output/final_data.rds` | R-compatible version of final_data | Stage 6 (optional) |
| `output/aggregated/eeg_features.csv` | EEG features only | Stage 3 |
| `output/aggregated/all_data_aggregated.csv` | With metadata (age, condition) | Stage 5 |
| `output/qc/summary/qc_failures_summary.csv` | ICA cleaning report (failures) | Stage 2 |
| `output/logs/pipeline_*.log` | Full execution log | Master orchestrator |
| `output/comparison_report.json` | Validation report (vs. old pipeline) | Comparison validator |
| `output/sets/PXX.set` | EEGLAB SET format (raw) | Stage 1 |
| `output/sets/PXX_clean.set` | EEGLAB SET format (cleaned) | Stage 2 |
| `output/eeg_features/PXX_features.csv` | EEG features per participant | Stage 3 |
| `output/physio_features/PXX_physio.csv` | Physio features per participant | Stage 4 |

---

## ⚙️ Configuration & Customization

### Change Participants (Master Orchestrator)
Edit `scripts/run_pipeline_master.py` line ~20:
```python
# Default (pilot)
participants = [1, 2, 3]

# For full run
participants = list(range(1, 49))  # P01–P48

# Or specific set
participants = [1, 2, 5, 10, 15]
```

### Adjust Validation Tolerance
```bash
# Default: r > 0.95 (5% tolerance)
python scripts/validation/compare_pipelines.py --tolerance 0.05

# Stricter: r > 0.99 (1% tolerance)
python scripts/validation/compare_pipelines.py --tolerance 0.01

# Looser: r > 0.90 (10% tolerance)
python scripts/validation/compare_pipelines.py --tolerance 0.10
```

### Change Feature Thresholds
Edit `config/` files (YAML):
- `config/eeg_feature_extraction.yaml` — EEG settings
- `config/eeg_metadata.yaml` — Cleaning/filtering
- `config/general.yaml` — Global settings

---

## 🐛 Troubleshooting

### Problem: "MATLAB not found"
```
ERROR: MATLAB not found in PATH
```

**Fix**:
1. Install MATLAB (if not done)
2. Add to PATH: `C:/Program Files/MATLAB/R2023a/bin`
3. Restart terminal and verify: `matlab -v`

### Problem: "EEGLAB not found"
```
Error: EEGLAB toolbox not found
```

**Fix**: Verify EEGLAB at `c:/MATLAB/toolboxes/eeglab`  
If missing, download from: http://sccn.ucsd.edu/eeglab/download.html

### Problem: Low Correlation (r < 0.90)
Likely cause: ICA is non-deterministic. Expected variation: r = 0.95–0.99

**Check**:
1. Verify config files haven't changed: `git diff config/`
2. Review AMICA seed settings: `scripts/preprocessing/eeg/cleaning/run_clean_eeg_pipeline.m`
3. Compare old vs. new ICA components manually

### Problem: Out of Disk Space
```
Error: not enough space
```

Raw data is ~300 GB. Check available space:
```powershell
Get-Item c:\ | Select-Object @{Name="FreeGB";Expression={$_.AvailableFreeSpace/1GB}}
```

**Fix**: Use external SSD for `data/raw/`

### Problem: Pipeline Hangs During Stage 2
AMICA can take 2–3 hours per participant. Normal behavior.

**Monitor**:
```powershell
Get-Process matlab | Select-Object Handles, Memory
```

If no CPU usage, manually kill and restart stage 2.

---

## 📚 Documentation

| Doc | Purpose |
|-----|---------|
| [PILOT_TEST_GUIDE.md](PILOT_TEST_GUIDE.md) | Quick-start for pilot test |
| [VALIDATION_TOOLS.md](VALIDATION_TOOLS.md) | Detailed validation workflow |
| [scripts/validation/README.md](scripts/validation/README.md) | Validation infrastructure |
| [data/raw/README.md](data/raw/README.md) | Raw data structure |
| [scripts/preprocessing/subjective/README.md](scripts/preprocessing/subjective/README.md) | Subjective data pipeline |
| [scripts/README.md](scripts/README.md) | Script organization |

---

## 🎯 Success Criteria

### Pilot Test Pass ✓
- [ ] Health check: All items pass
- [ ] Pipeline completes stages 1–6 without fatal errors
- [ ] Output files exist (`final_data.csv`, EEG features, physio features)
- [ ] Validation: Correlation > 95% with old pipeline

### Full 48-Participant Run Pass ✓
- [ ] All 48 participants process successfully
- [ ] Final feature matrix: 48 rows × ~250 features
- [ ] Validation: Correlation maintained > 95%
- [ ] Results committed to git with timestamped log

---

## 📞 Support

**Questions?**
- Check logs: `output/logs/pipeline_*.log`
- Review configs: `config/`
- Check git history: `git log --oneline`

**Contact**: Joe Hartley <jh3968@bath.ac.uk>

---

**Last Updated**: January 2025  
**Status**: Production-ready for pilot test
