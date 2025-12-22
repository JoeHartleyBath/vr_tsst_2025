#!/usr/bin/env powershell
# Stage 3: Extract EEG features (spectral, temporal)

$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$rootDir = Split-Path -Parent (Split-Path -Parent $scriptDir)

Write-Host "Running Stage 3: EEG Feature Extraction"
Write-Host "Script: $rootDir\scripts\preprocessing\eeg\feature_extraction\extract_eeg_features.m"

# Check if MATLAB is available
$matlabPath = Get-Command matlab -ErrorAction SilentlyContinue
if (-not $matlabPath) {
    Write-Host "ERROR: MATLAB not found in PATH" -ForegroundColor Red
    Write-Host "Please install MATLAB and add it to your PATH"
    exit 1
}

# Run MATLAB script
& matlab -batch "cd('$rootDir'); run('scripts/preprocessing/eeg/feature_extraction/extract_eeg_features.m'); exit"

exit $LASTEXITCODE
