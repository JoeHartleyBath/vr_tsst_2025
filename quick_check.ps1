#!/usr/bin/env powershell
# Quick verification before pilot

$checks = @{
    "AMICA" = "C:\MATLAB\toolboxes\amica\runamica15.m"
    "EEGLAB" = "C:\MATLAB\toolboxes\eeglab\eeglab.m"
    "Python" = "C:\vr_tsst_2025\venv\Scripts\python.exe"
    "Raw Data EEG" = "C:\vr_tsst_2025\data\RAW\eeg\P01.xdf"
    "Orchestrator" = "C:\vr_tsst_2025\scripts\run_pipeline_master.py"
}

Write-Host "Pre-Pilot Quick Check:" -ForegroundColor Cyan
Write-Host ""

$allGood = $true
foreach ($name in $checks.Keys) {
    $path = $checks[$name]
    if (Test-Path $path) {
        Write-Host "[OK] $name" -ForegroundColor Green
    } else {
        Write-Host "[MISSING] $name at: $path" -ForegroundColor Red
        $allGood = $false
    }
}

Write-Host ""
if ($allGood) {
    Write-Host "[READY] All systems GO for pilot!" -ForegroundColor Green
    Write-Host ""
    Write-Host "Run pilot with:" -ForegroundColor Cyan
    Write-Host "  cd c:\vr_tsst_2025" -ForegroundColor White
    Write-Host "  venv\Scripts\Activate.ps1" -ForegroundColor White
    Write-Host "  python scripts/run_pipeline_master.py" -ForegroundColor White
} else {
    Write-Host "[NOT READY] Fix missing items above" -ForegroundColor Red
}
