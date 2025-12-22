# Test setup script generated from NEW_PC_SETUP.md
Write-Host "Testing Python..." -ForegroundColor Cyan
python -c "import pyxdf, mne, pandas; print('Python OK')"

Write-Host "`nTesting R..." -ForegroundColor Cyan
Rscript -e "library(tidyverse); library(e1071); cat('R OK\n')"

Write-Host "`nTesting MATLAB..." -ForegroundColor Cyan
matlab -batch "addpath('scripts'); disp('MATLAB OK'); exit"

Write-Host "`nChecking raw data..." -ForegroundColor Cyan
Get-ChildItem "data\raw\eeg\*.xdf" | Select-Object Name
Get-ChildItem "data\raw\metadata\*.csv" | Select-Object Name

Write-Host "`nSetup validation complete!" -ForegroundColor Green
