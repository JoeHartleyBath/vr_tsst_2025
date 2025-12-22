# Comprehensive pre-pilot verification script

Write-Host "════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "PRE-PILOT VERIFICATION" -ForegroundColor Cyan
Write-Host "════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host ""

$allPass = $true

# Check 1: Python
Write-Host "[1] Python Environment..."
if (Test-Path "C:\vr_tsst_2025\venv\Scripts\python.exe") {
    Write-Host "  [OK] Python venv found" -ForegroundColor Green
} else {
    Write-Host "  [FAIL] Python venv not found" -ForegroundColor Red
    $allPass = $false
}

# Check 2: AMICA
Write-Host "[2] AMICA Installation..."
$amicaFiles = @(
    "C:\MATLAB\toolboxes\amica\runamica15.m",
    "C:\MATLAB\toolboxes\amica\amica15mkl.exe"
)
$amicaOk = $true
foreach ($file in $amicaFiles) {
    if (Test-Path $file) {
        Write-Host "  [OK] Found: $(Split-Path $file -Leaf)" -ForegroundColor Green
    } else {
        Write-Host "  [FAIL] Missing: $file" -ForegroundColor Red
        $amicaOk = $false
        $allPass = $false
    }
}

# Check 3: EEGLAB
Write-Host "[3] EEGLAB Installation..."
$eeglab = "C:\MATLAB\toolboxes\eeglab\eeglab.m"
if (Test-Path $eeglab) {
    Write-Host "  [OK] EEGLAB found at: C:\MATLAB\toolboxes\eeglab" -ForegroundColor Green
} else {
    Write-Host "  [FAIL] EEGLAB not found at: C:\MATLAB\toolboxes\eeglab" -ForegroundColor Red
    Write-Host "         Expected: $eeglab" -ForegroundColor Red
    $allPass = $false
}

# Check 4: R Installation
Write-Host "[4] R Installation..."
if (Test-Path "C:\Program Files\R\R-4.5.2\bin\Rscript.exe") {
    Write-Host "  [OK] R installation found" -ForegroundColor Green
    # Try to run Rscript
    try {
        $rVersion = & "C:\Program Files\R\R-4.5.2\bin\Rscript.exe" --version 2>&1 | Select-Object -First 1
        Write-Host "  [OK] $rVersion" -ForegroundColor Green
    } catch {
        Write-Host "  [WARN] R found but could not verify version" -ForegroundColor Yellow
    }
} else {
    Write-Host "  [FAIL] R not found at: C:\Program Files\R\R-4.5.2\bin" -ForegroundColor Red
    $allPass = $false
}

# Check 5: MATLAB
Write-Host "[5] MATLAB Installation..."
try {
    $matlabVersion = & matlab -v 2>&1 | Select-Object -First 1
    Write-Host "  [OK] MATLAB found: $matlabVersion" -ForegroundColor Green
} catch {
    Write-Host "  [WARN] MATLAB not found in PATH (may still work if installed)" -ForegroundColor Yellow
}

# Check 6: Raw Data
Write-Host "[6] Raw Data Staging..."
$eegFiles = @(Get-ChildItem "C:\vr_tsst_2025\data\raw\eeg" -Filter "*.xdf" -ErrorAction SilentlyContinue).Count
$metaFiles = @(Get-ChildItem "C:\vr_tsst_2025\data\raw\metadata" -Filter "*.csv" -ErrorAction SilentlyContinue).Count
$subjFiles = @(Get-ChildItem "C:\vr_tsst_2025\data\raw\subjective" -Filter "*.csv" -ErrorAction SilentlyContinue).Count

if ($eegFiles -gt 0) { Write-Host "  [OK] EEG files: $eegFiles" -ForegroundColor Green } else { Write-Host "  [FAIL] No EEG files found" -ForegroundColor Red; $allPass = $false }
if ($metaFiles -gt 0) { Write-Host "  [OK] Metadata files: $metaFiles" -ForegroundColor Green } else { Write-Host "  [FAIL] No metadata files" -ForegroundColor Red; $allPass = $false }
if ($subjFiles -gt 0) { Write-Host "  [OK] Subjective files: $subjFiles" -ForegroundColor Green } else { Write-Host "  [FAIL] No subjective files" -ForegroundColor Red; $allPass = $false }

# Check 7: Pipeline Scripts
Write-Host "[7] Pipeline Scripts..."
$scripts = @(
    "C:\vr_tsst_2025\scripts\run_pipeline_master.py",
    "C:\vr_tsst_2025\scripts\check_pipeline_health.py"
)
foreach ($script in $scripts) {
    if (Test-Path $script) {
        Write-Host "  [OK] $(Split-Path $script -Leaf)" -ForegroundColor Green
    } else {
        Write-Host "  [FAIL] Missing: $(Split-Path $script -Leaf)" -ForegroundColor Red
        $allPass = $false
    }
}

# Check 8: Config Files
Write-Host "[8] Configuration Files..."
$configs = Get-ChildItem "C:\vr_tsst_2025\config" -Filter "*.yaml" -ErrorAction SilentlyContinue
if ($configs.Count -gt 0) {
    Write-Host "  [OK] Config files found: $($configs.Count)" -ForegroundColor Green
} else {
    Write-Host "  [WARN] No YAML config files found" -ForegroundColor Yellow
}

# Final Summary
Write-Host ""
Write-Host "════════════════════════════════════════════════════════════" -ForegroundColor Cyan

if ($allPass) {
    Write-Host "[SUCCESS] ALL CHECKS PASSED - READY FOR PILOT!" -ForegroundColor Green
    Write-Host "════════════════════════════════════════════════════════════" -ForegroundColor Green
    Write-Host ""
    Write-Host "Next step: Run pilot" -ForegroundColor Cyan
    Write-Host "  cd c:\vr_tsst_2025" -ForegroundColor Cyan
    Write-Host "  venv\Scripts\Activate.ps1" -ForegroundColor Cyan
    Write-Host "  python scripts/run_pipeline_master.py" -ForegroundColor Cyan
    Write-Host ""
} else {
    Write-Host "[FAIL] SOME CHECKS FAILED - FIX BEFORE PILOT" -ForegroundColor Red
    Write-Host "════════════════════════════════════════════════════════════" -ForegroundColor Red
    Write-Host ""
    Write-Host "Issues found above. Please fix and re-run this verification." -ForegroundColor Yellow
    Write-Host ""
}
