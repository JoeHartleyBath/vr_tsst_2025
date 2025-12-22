# Install EEGLAB and add R to PATH
# Run as Administrator

Write-Host "================================" -ForegroundColor Cyan
Write-Host "EEGLAB & R Setup" -ForegroundColor Cyan
Write-Host "================================" -ForegroundColor Cyan
Write-Host ""

# Step 1: Add R to PATH
Write-Host "[1/2] Adding R to system PATH..." -ForegroundColor Yellow
$rBinPath = "C:\Program Files\R\R-4.5.2\bin"
$currentPath = [Environment]::GetEnvironmentVariable("Path", "Machine")

if ($currentPath -notlike "*$rBinPath*") {
    [Environment]::SetEnvironmentVariable("Path", "$currentPath;$rBinPath", "Machine")
    Write-Host "  [OK] R added to PATH" -ForegroundColor Green
} else {
    Write-Host "  [OK] R already in PATH" -ForegroundColor Green
}

# Step 2: Download and install EEGLAB
Write-Host "[2/2] Downloading EEGLAB..." -ForegroundColor Yellow
$url = "https://sccn.ucsd.edu/eeglab/eeglab_current.zip"
$tempDir = "C:\temp_eeglab_download"
$zipFile = "$tempDir\eeglab.zip"
$targetDir = "C:\MATLAB\toolboxes\eeglab"

# Create temp directory
New-Item -ItemType Directory -Path $tempDir -Force | Out-Null

# Download EEGLAB
Write-Host "  Downloading from: $url"
try {
    $webClient = New-Object System.Net.WebClient
    $webClient.DownloadFile($url, $zipFile)
    $sizeMB = [math]::Round((Get-Item $zipFile).Length / 1MB, 1)
    Write-Host "  [OK] Downloaded: $sizeMB MB" -ForegroundColor Green
}
catch {
    Write-Host "  [ERROR] Download failed: $_" -ForegroundColor Red
    exit 1
}

# Extract
Write-Host "  Extracting..." -ForegroundColor Yellow
Expand-Archive -Path $zipFile -DestinationPath $tempDir -Force

# Find EEGLAB directory
$eeglab = Get-ChildItem $tempDir -Filter "eeglab*" -Directory | Select-Object -First 1
if (-not $eeglab) {
    Write-Host "  [ERROR] Could not find eeglab directory in extracted files" -ForegroundColor Red
    exit 1
}

# Remove existing and move new
if (Test-Path $targetDir) {
    Write-Host "  Removing existing EEGLAB directory..."
    Remove-Item -Recurse -Force $targetDir
}

Write-Host "  Installing to: $targetDir"
Move-Item $eeglab.FullName $targetDir -Force

# Verify
if (Test-Path "$targetDir\eeglab.m") {
    Write-Host "  [OK] EEGLAB installed successfully" -ForegroundColor Green
} else {
    Write-Host "  [ERROR] EEGLAB installation incomplete" -ForegroundColor Red
    exit 1
}

# Cleanup
Remove-Item $tempDir -Recurse -Force

Write-Host ""
Write-Host "================================" -ForegroundColor Green
Write-Host "[SUCCESS] Setup Complete!" -ForegroundColor Green
Write-Host "================================" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "  1. Restart terminal/PowerShell for R PATH to take effect"
Write-Host "  2. Restart MATLAB for EEGLAB to load"
Write-Host "  3. Run: python scripts/check_pipeline_health.py"
Write-Host ""
