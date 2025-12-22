#!/usr/bin/env powershell
# Stage 6: R preprocessing and analysis

$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$rootDir = Split-Path -Parent (Split-Path -Parent $scriptDir)

Write-Host "Running Stage 6: R Preprocessing & Analysis"

# Check if R is available
$rPath = Get-Command Rscript -ErrorAction SilentlyContinue
if (-not $rPath) {
    Write-Host "ERROR: R/Rscript not found in PATH" -ForegroundColor Red
    Write-Host "Please install R and add it to your PATH"
    exit 1
}

# Look for R preprocessing script
$rScript = "$rootDir\scripts\preprocessing\r_preprocessing.R"
if (-not (Test-Path $rScript)) {
    $rScript = "$rootDir\scripts\analysis\r_analysis.R"
}

if (-not (Test-Path $rScript)) {
    Write-Host "WARNING: R preprocessing script not found at:" -ForegroundColor Yellow
    Write-Host "  $rScript" -ForegroundColor Yellow
    Write-Host "Searching for R scripts in scripts/analysis/..."
    
    $rScripts = Get-ChildItem -Path "$rootDir\scripts\analysis\" -Filter "*.R" -ErrorAction SilentlyContinue
    if ($rScripts) {
        Write-Host "Found: $($rScripts -join ', ')"
    } else {
        Write-Host "No R scripts found. Skipping Stage 6."
        exit 1
    }
}

# Run R script via Rscript
& Rscript "$rScript"

exit $LASTEXITCODE
