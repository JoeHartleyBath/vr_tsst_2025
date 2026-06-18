# VR-TSST Analysis Pipeline

## 📍 Purpose
This repository contains the complete end-to-end processing pipeline for the VR-TSST experiment, from raw XDF/EEG data to statistical analysis and deep learning workload classification.

## 🚀 Quick Start (Start Here)

### Path A: Run the Pilot (Recommended)
1. Open [PILOT_TEST_GUIDE.md](PILOT_TEST_GUIDE.md)
2. Follow the step-by-step commands to process P01–P03 through **Stages 01–06** (preprocessing).

### Path B: Deep Learning Workload Models
- **Prep**: Run `pipelines/12_workload_deep_learning_prep/export_mne_epochs.py` to create training data.
- **Train**: Run `pipelines/13_workload_deep_learning_training/train_tcnet.py` to train models.

### Path C: Full Control
- Read [VALIDATION_TOOLS.md](VALIDATION_TOOLS.md) for the validation workflow.
- Use [TOOLS_REFERENCE.md](TOOLS_REFERENCE.md) for individual commands.

---

## 🗺️ Project Map

### Core Pipeline Structure
The pipeline is divided into distinct stages in `pipelines/`:
- **Stages 01–06**: Preprocessing (EEG Cleaning, Physio) & Feature Extraction.
- **Stages 07–11**: Classical Machine Learning (SVM, XGBoost) & Statistics.
- **Stage 12**: **Deep Learning Prep** ([`pipelines/12_...`](pipelines/12_workload_deep_learning_prep/))
  - *Contract*: Exports clean `.set` files → `.fif` MNE Epochs.
  - *Output*: `output/adaptive_workload/mne_epochs/`.
- **Stage 13**: **Deep Learning Training** ([`pipelines/13_...`](pipelines/13_workload_deep_learning_training/))
  - *Contract*: Consumes `.fif` files → Trains PyTorch models (TCNet, EEGNet).
  - *Scripts*: `train_tcnet.py`, `tune_tcnet_focused.py`.

### Support Directories
- **`scripts/`**: Master orchestrators (`run_pipeline_master.py`) and health checks.
- **`config/`**: YAML settings for all stages.
- **`data/`**: Raw input storage.
- **`output/`**: Generated results (staged by analysis type).

### Key Documentation
- [PILOT_TEST_GUIDE.md](PILOT_TEST_GUIDE.md) — **Primary entry point**.
- [TOOLS_REFERENCE.md](TOOLS_REFERENCE.md) — All commands & troubleshooting.
- [VALIDATION_TOOLS.md](VALIDATION_TOOLS.md) — Quality assurance guide.

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
**Last Updated**: January 2026  
**Status**: Core Pipeline Ready (Stages 1-6) | Deep Learning Active (Stages 12-13)  
**Contact**: Joe Hartley <jh3968@bath.ac.uk>
