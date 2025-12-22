#!/usr/bin/env powershell
# Professional Git Commit Script
# Usage: .\commit_changes.ps1

Write-Host "=== VR-TSST Pipeline Professional Commit ===" -ForegroundColor Cyan

# Check if git is available
$git = Get-Command git -ErrorAction SilentlyContinue
if ($null -eq $git) {
    Write-Host "ERROR: Git not found in PATH" -ForegroundColor Red
    exit 1
}

Write-Host "`n[1] Checking git status..." -ForegroundColor Yellow
git status --short

Write-Host "`n[2] Adding all changes..." -ForegroundColor Yellow
git add .

Write-Host "`n[3] Preparing commit message..." -ForegroundColor Yellow

$commitMessage = @"
fix(eeg-pipeline): add diagnostics, repair .set format, stabilize physio merge

BREAKING CHANGES:
- P10, P14, P23 flagged for exclusion due to memory errors in MATLAB cleaning

FEATURES:
- Add comprehensive try/catch diagnostics to run_clean_eeg_pipeline.m
- Auto-convert all .set files to two-file format (.set + .fdt) via fix_eeglab_set_files.m
- Auto-detect and convert EEG.data to single precision for memory efficiency
- Create setup_environment.ps1 for automated venv + dependency installation
- Pin all Python dependencies in requirements.txt (numpy 1.24.3, pandas 2.0.3, etc.)
- Professional .gitignore excluding data, outputs, and raw logs

FIXES:
- Resolve ModuleNotFoundError for numpy, yaml, pyxdf (all packages installed)
- Gracefully skip problematic participants in batch MATLAB processing
- Generate QC reports for all 48 participants
- Memory diagnostics: print EEG.data size and whos info before crash

TESTING:
- Verified XDF→SET conversion for P10, P17, P23, P14
- All 48 participants converted from raw .xdf to EEGLAB .set format
- MATLAB diagnostics confirm memory overflow at pop_loadset for P10, P14, P23
- Fixed .set files for all participants (no .fdt available → convert to two-file format)

REFS:
- Issue: Maximum variable size exceeded in MATLAB cleaning
- Solution: Memory diagnostics + format repair allows batch processing to continue
- Next: Physio extraction, DL dataset prep, SVM + XGBoost analysis

See COMMIT_SESSION_REPORT.md for detailed session notes.
"@

Write-Host "`n[4] Committing changes..." -ForegroundColor Yellow
git commit -m $commitMessage

if ($LASTEXITCODE -eq 0) {
    Write-Host "`n✓ Commit successful!" -ForegroundColor Green
    Write-Host "`n[5] Recent commits:" -ForegroundColor Yellow
    git log --oneline -5
    Write-Host "`nReady to push to remote:" -ForegroundColor Cyan
    Write-Host "  git push origin branch-name" -ForegroundColor Magenta
} else {
    Write-Host "`nERROR: Commit failed" -ForegroundColor Red
    git status
    exit 1
}
