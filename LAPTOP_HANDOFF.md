# Laptop Handoff Guide

## Status
- **Local Codebase**: Contains the latest scripts, data processing, and generated figures.
- **Overleaf Manuscript**: Out of sync with local changes. **DO NOT** try to sync via Git/Zip import as it may overwrite manual LaTeX polishing.

## Immediate Next Steps (On Laptop)
1. **Pull the latest code**:
   ```bash
   git pull origin feature/qc-eeg-only
   ```
2. **Review the Edit Ledger**:
   - Open `MANUAL_EDIT_LEDGER.md`.
   - This file contains a checklist of discrepancies found between the local analysis and the Overleaf text.
3. **Apply Edits to Overleaf**:
   - Manually update the Overleaf project to match the ledger items.
   - Focus on:
     - Removing ASR (Artifact Subspace Reconstruction) references (we are using selective filtering now).
     - Updating ANOVA F-statistics and p-values in the Results section.
     - Re-uploading Figures 3, 4, and 5 from `results/figures/` if they look different.

## Running the Pipeline
If you need to regenerate results on the laptop:
```powershell
.\run_48hr_manuscript_pipeline.ps1
```
*Note: This may take time. Ensure you have the `data/` folder synced or available.*
