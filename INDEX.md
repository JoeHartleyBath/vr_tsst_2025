# VR-TSST Pipeline: Complete Setup & Validation Index

## 📍 You Are Here

Your pipeline is **fully prepared for P01–P03 pilot testing**. This document shows you what's ready, what to do next, and where to find everything.

---

## ✅ What's Complete

### Infrastructure
- [x] Python 3.9+ environment (venv) with all packages
- [x] R 4.5.2 with required packages (tidyverse, caret, etc.)
- [x] Raw data staged: 48 participants × (EEG + metadata + subjective)
- [x] Git repository configured with correct authorship
- [x] MATLAB/EEGLAB paths configured (awaiting final AMICA install)

### Code
- [x] Master pipeline orchestrator (`scripts/run_pipeline_master.py`)
- [x] Health check validator (`scripts/check_pipeline_health.py`)
- [x] Results comparison tool (`scripts/validation/compare_pipelines.py`)
- [x] Individual stage runners (stages 1–6)
- [x] Comprehensive logging and error handling

### Documentation
- [x] Quick-start guide (`PILOT_TEST_GUIDE.md`)
- [x] Validation workflow (`VALIDATION_TOOLS.md`)
- [x] Tools reference (`TOOLS_REFERENCE.md`)
- [x] Data management (`data/raw/README.md`)

---

## ⏳ What's Next (Your Action Items)

### 1. **Install MATLAB/EEGLAB/AMICA** (if not done)
   - **Status**: Manual (user responsibility)
   - **Documentation**: Check [NEW_PC_SETUP.md](NEW_PC_SETUP.md#matlab-setup)
   - **Estimated Time**: 1–2 hours

### 2. **Run Pre-Flight Health Check** (5 minutes)
   ```powershell
   python scripts/check_pipeline_health.py
   ```
   - Validates all dependencies
   - Must pass before proceeding

### 3. **Run P01–P03 Pilot Test** (3–4 hours)
   ```powershell
   venv\Scripts\Activate.ps1
   python scripts/run_pipeline_master.py
   ```
   - Processes 3 participants through all 6 stages
   - Creates output files in `output/`
   - Logs everything to `output/logs/pipeline_*.log`

### 4. **Validate Results** (5 minutes)
   ```powershell
   python scripts/validation/compare_pipelines.py
   ```
   - Compares new results vs. old pipeline
   - Reports correlations, RMSE, feature-by-feature
   - Success = r > 0.95 match rate

### 5. **Proceed to Full Run or Debug**
   - **If validation passes**: Scale to all 48 participants (16–24 hours)
   - **If validation fails**: Debug using individual stage runners

---

## 📚 Documentation Map

### Quick Guides (Start Here)
- [PILOT_TEST_GUIDE.md](PILOT_TEST_GUIDE.md) — Copy-paste commands for pilot
- [TOOLS_REFERENCE.md](TOOLS_REFERENCE.md) — All available commands & workflows
- [VALIDATION_TOOLS.md](VALIDATION_TOOLS.md) — Detailed validation process

### Detailed References
- [NEW_PC_SETUP.md](NEW_PC_SETUP.md) — Initial setup & dependency installation
- [scripts/validation/README.md](scripts/validation/README.md) — Validation infrastructure
- [data/raw/README.md](data/raw/README.md) — Raw data structure & staging
- [QUICK_START.md](QUICK_START.md) — Original quick-start (legacy, see above)

### Setup Notes
- [SESSION_NOTES.md](SESSION_NOTES.md) — Setup session log
- [EEG_CLEANING_STATUS.md](EEG_CLEANING_STATUS.md) — EEG pipeline status
- [pipeline_quality_report.md](pipeline_quality_report.md) — Legacy quality assessment

---

## 🚀 Three Simple Paths

### Path A: I Just Want to Run the Pilot
1. Open [PILOT_TEST_GUIDE.md](PILOT_TEST_GUIDE.md)
2. Copy the commands
3. Follow step-by-step

### Path B: I Want Full Control & Understanding
1. Read [VALIDATION_TOOLS.md](VALIDATION_TOOLS.md) (detailed workflow)
2. Use [TOOLS_REFERENCE.md](TOOLS_REFERENCE.md) (all available commands)
3. Follow the validation phases

### Path C: I Need to Debug Something
1. Check [TOOLS_REFERENCE.md](TOOLS_REFERENCE.md#troubleshooting) (troubleshooting)
2. Run individual stage: `python scripts/stages/run_stage_X_*.py`
3. View logs: `Get-Content output/logs/pipeline_*.log | Select-String ERROR`

---

## 🎯 Expected Timeline

| Phase | Time | Command |
|-------|------|---------|
| Pre-check | 5 min | `python scripts/check_pipeline_health.py` |
| Pilot run | 3–4 hrs | `python scripts/run_pipeline_master.py` |
| Validation | 5 min | `python scripts/validation/compare_pipelines.py` |
| **Subtotal** | **~4 hours** | - |
| Full run | 16–24 hrs | Edit script + re-run |
| **Total** | **~1 day** | - |

---

## 📁 Key Files

### You Will Run
- `scripts/run_pipeline_master.py` — Main orchestrator
- `scripts/check_pipeline_health.py` — Pre-flight check
- `scripts/validation/compare_pipelines.py` — Results comparison

### You May Edit
- `scripts/run_pipeline_master.py` line ~20 — Change participant list
- `config/*.yaml` — Adjust pipeline settings
- `scripts/preprocessing/raw_conversion/run/run_xdf_to_set_end2end.py` — Stage 1 settings

### You Will Review
- `output/logs/pipeline_*.log` — Execution log
- `output/final_data.csv` — Final feature matrix
- `output/comparison_report.json` — Validation report

---

## 💾 Outputs (What You'll Get)

After running the pipeline:

```
output/
├── final_data.csv                    # Master feature matrix
├── final_data.rds                    # R version
├── aggregated/
│   ├── eeg_features.csv             # EEG-only features
│   └── all_data_aggregated.csv      # With metadata
├── qc/
│   └── summary/
│       └── qc_failures_summary.csv  # Cleaning QC report
├── sets/
│   ├── P01.set, P01.fdt            # Raw EEGLAB format
│   └── P01_clean.set                # Cleaned format
├── eeg_features/
│   └── P01_features.csv             # Per-participant EEG
├── physio_features/
│   └── P01_physio.csv               # Per-participant physio
├── logs/
│   └── pipeline_20250120_153000.log # Full execution log
└── comparison_report.json            # Validation report
```

---

## 🔗 Git Commits

Your work is tracked in git. Recent commits:

```
f76c28e - Add complete tools reference guide
ece6884 - Add validation infrastructure (orchestrator, health check, validators, stage runners)
d3bc9e9 - Setup pipeline for new PC (installers, staging scripts, raw data)
```

View history: `git log --oneline`

---

## ⚠️ Critical Dependencies

| Dependency | Status | Action if Missing |
|-----------|--------|------------------|
| Python 3.9+ | ✓ Installed | N/A |
| R 4.5.2 | ✓ Installed | Install from r-project.org |
| MATLAB R2020a+ | ⏳ Manual install | Download + install MATLAB |
| EEGLAB | ⏳ Awaiting MATLAB | Download to c:/MATLAB/toolboxes/eeglab |
| AMICA | ⏳ **Automated installer ready** | `powershell -ExecutionPolicy Bypass install_amica.ps1` |
| Raw data (48 × 3 files) | ✓ Staged | Already in data/raw/ |

---

## 🎓 Learning Resources

### For New Users
- Start with [PILOT_TEST_GUIDE.md](PILOT_TEST_GUIDE.md)
- Run health check: `python scripts/check_pipeline_health.py`
- Ask: "What files will this create?" (Check `output/` structure above)

### For Advanced Users
- Review [VALIDATION_TOOLS.md](VALIDATION_TOOLS.md) for full workflow
- Customize `scripts/run_pipeline_master.py` to add participants
- Edit `config/*.yaml` to adjust feature extraction
- Run individual stages for debugging

### For Troubleshooting
- Check [TOOLS_REFERENCE.md#troubleshooting](TOOLS_REFERENCE.md#troubleshooting)
- View logs: `Get-Content output/logs/pipeline_*.log`
- Compare config: `git diff config/`

---

## ❓ FAQ

**Q: Do I need to install MATLAB?**  
A: Only if you're using the EEG cleaning (Stage 2). If you skip it, you can still run stages 1, 4, 5, 6.

**Q: How long does the pilot take?**  
A: ~3–4 hours total (mostly Stage 2 AMICA, which is slow).

**Q: What if validation fails?**  
A: Review [TOOLS_REFERENCE.md#troubleshooting](TOOLS_REFERENCE.md#troubleshooting). Most common: ICA is non-deterministic (expected r = 0.95–0.99).

**Q: Can I stop and resume?**  
A: Yes. Use `python scripts/run_pipeline_master.py --stages 4 5 6` to resume from Stage 4.

**Q: What if I run out of disk space?**  
A: Raw data = 300 GB, outputs = 50–100 GB. Total ~400 GB needed. Move `data/raw/` to external SSD if needed.

---

## 📞 Support

**Need help?**
1. Check [TOOLS_REFERENCE.md](TOOLS_REFERENCE.md#troubleshooting) for common issues
2. Review [VALIDATION_TOOLS.md](VALIDATION_TOOLS.md) for workflow details
3. Check git log: `git log --oneline` or `git show ece6884` (last orchestrator commit)

**Contact**: Joe Hartley <jh3968@bath.ac.uk>

---

## 🚦 Status

| Component | Status | Ready? |
|-----------|--------|--------|
| Python + packages | ✓ Complete | ✅ Yes |
| R + packages | ✓ Complete | ✅ Yes |
| MATLAB setup | ⏳ Pending | ⚠️ Manual |
| Raw data staging | ✓ Complete | ✅ Yes |
| Master orchestrator | ✓ Complete | ✅ Yes |
| Validation tools | ✓ Complete | ✅ Yes |
| Documentation | ✓ Complete | ✅ Yes |
| **Overall** | **Ready** | **✅ Start Pilot** |

---

## 🎬 Next Step

**If you haven't already:**
```powershell
cd c:\vr_tsst_2025
python scripts/check_pipeline_health.py
```

This takes 5 minutes and tells you exactly what's ready and what's missing.

---

**Created**: January 2025  
**Last Updated**: January 2025  
**Status**: Production-ready for P01–P03 pilot test  
**Contact**: Joe Hartley <jh3968@bath.ac.uk>
