#!/usr/bin/env powershell
# Stage 2: Clean EEG data using ICA (MATLAB/EEGLAB)

$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$rootDir = Split-Path -Parent (Split-Path -Parent $scriptDir)

Write-Host "Running Stage 2: EEG Cleaning (ICA/AMICA)"
Write-Host "Script: $rootDir\scripts\preprocessing\eeg\cleaning\run_clean_eeg_pipeline.m"

# Check if MATLAB is available
$matlabPath = Get-Command matlab -ErrorAction SilentlyContinue
if (-not $matlabPath) {
    Write-Host "ERROR: MATLAB not found in PATH" -ForegroundColor Red
    Write-Host "Please install MATLAB and add it to your PATH"
    exit 1
}

# Run MATLAB script
& matlab -batch "cd('$rootDir'); run('scripts/preprocessing/eeg/cleaning/run_clean_eeg_pipeline.m'); exit"

exit $LASTEXITCODE
