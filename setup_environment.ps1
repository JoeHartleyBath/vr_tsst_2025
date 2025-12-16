# Environment Setup Script for VR-TSST Pipeline
# Usage: .\setup_environment.ps1

param(
    [switch]$SkipPython = $false,
    [switch]$SkipMATLAB = $false
)

Write-Host "=== VR-TSST Pipeline Environment Setup ===" -ForegroundColor Cyan

# Detect Python installation
if (-not $SkipPython) {
    Write-Host "`n[PYTHON] Checking Python installation..." -ForegroundColor Yellow
    $pythonExe = Get-Command python -ErrorAction SilentlyContinue
    
    if ($null -eq $pythonExe) {
        Write-Host "ERROR: Python not found in PATH" -ForegroundColor Red
        Write-Host "Please install Python 3.8+ from https://www.python.org/downloads/" -ForegroundColor Red
        exit 1
    }
    
    $pythonVersion = python --version
    Write-Host "✓ Found: $pythonVersion" -ForegroundColor Green
    
    # Create virtual environment if it doesn't exist
    if (-not (Test-Path "venv")) {
        Write-Host "`n[VENV] Creating virtual environment..." -ForegroundColor Yellow
        python -m venv venv
        Write-Host "✓ Virtual environment created" -ForegroundColor Green
    }
    
    # Activate venv
    Write-Host "`n[VENV] Activating virtual environment..." -ForegroundColor Yellow
    & "venv\Scripts\Activate.ps1"
    
    # Install requirements
    Write-Host "`n[PACKAGES] Installing Python dependencies..." -ForegroundColor Yellow
    pip install -r requirements.txt --upgrade
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✓ All packages installed successfully" -ForegroundColor Green
    } else {
        Write-Host "ERROR: Package installation failed" -ForegroundColor Red
        exit 1
    }
}

# Verify MATLAB/EEGLAB
if (-not $SkipMATLAB) {
    Write-Host "`n[MATLAB] Checking MATLAB installation..." -ForegroundColor Yellow
    $matlabExe = Get-Command matlab -ErrorAction SilentlyContinue
    
    if ($null -eq $matlabExe) {
        Write-Host "WARNING: MATLAB not found in PATH (optional)" -ForegroundColor Yellow
        Write-Host "Ensure MATLAB is installed for EEG cleaning pipeline" -ForegroundColor Yellow
    } else {
        Write-Host "✓ MATLAB found" -ForegroundColor Green
    }
}

Write-Host "`n=== Setup Complete ===" -ForegroundColor Cyan
Write-Host "`nTo activate the environment in future shells, run:" -ForegroundColor Cyan
Write-Host "  venv\Scripts\Activate.ps1" -ForegroundColor Magenta
Write-Host "`nTo start the pipeline, run:" -ForegroundColor Cyan
Write-Host "  python scripts/preprocessing/raw_conversion/run/run_xdf_to_set_end2end.py --participants all" -ForegroundColor Magenta
