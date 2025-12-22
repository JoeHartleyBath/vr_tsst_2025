# Installer script to set up Python venv, install Python and R packages
# Run in PowerShell (run as Administrator if needed)

Write-Host "Creating Python virtual environment 'venv'..." -ForegroundColor Cyan
python -m venv venv

Write-Host "Activating venv and upgrading pip..." -ForegroundColor Cyan
& .\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip

if (Test-Path requirements.txt) {
    Write-Host "Installing Python packages from requirements.txt..." -ForegroundColor Cyan
    pip install -r requirements.txt
} else {
    Write-Host "requirements.txt not found; skipping Python package install." -ForegroundColor Yellow
}

# Install R packages via Rscript if available
if (Get-Command Rscript -ErrorAction SilentlyContinue) {
    Write-Host "Installing R packages using install_R_packages.R..." -ForegroundColor Cyan
    Rscript install_R_packages.R
} else {
    Write-Host "Rscript not found. Please install R and ensure Rscript is on PATH to install R packages." -ForegroundColor Yellow
}

Write-Host "\nMATLAB is not automated by this script. Please install MATLAB and required toolboxes (EEGLAB, AMICA, yamlmatlab) manually." -ForegroundColor Yellow
Write-Host "Links: https://www.mathworks.com, https://sccn.ucsd.edu/eeglab/" -ForegroundColor Yellow

Write-Host "\nOptional: run the test script to validate environment:" -ForegroundColor Green
Write-Host ".\test_setup.ps1" -ForegroundColor Green
