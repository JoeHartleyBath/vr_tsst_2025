#!/usr/bin/env powershell
"""
AMICA Toolbox Installer
Downloads and installs AMICA from GitHub to MATLAB toolboxes directory
"""

$ErrorActionPreference = "Stop"

# Configuration
$TOOLBOX_BASE = "C:/MATLAB/toolboxes"
$AMICA_REPO = "https://github.com/sccn/amica.git"
$AMICA_DIR = "$TOOLBOX_BASE/amica"

Write-Host "════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "AMICA Toolbox Installer" -ForegroundColor Cyan
Write-Host "════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host ""

# Check if Git is installed
Write-Host "[1/4] Checking Git installation..."
try {
    $gitVersion = git --version
    Write-Host "  ✓ $gitVersion" -ForegroundColor Green
} catch {
    Write-Host "  ✗ Git not found in PATH" -ForegroundColor Red
    Write-Host "  Install Git: https://git-scm.com/download/win" -ForegroundColor Yellow
    exit 1
}

# Create toolbox directory if needed
Write-Host "[2/4] Creating toolbox directory..."
if (-not (Test-Path $TOOLBOX_BASE)) {
    Write-Host "  Creating: $TOOLBOX_BASE"
    New-Item -ItemType Directory -Path $TOOLBOX_BASE -Force | Out-Null
}
Write-Host "  ✓ Directory ready: $TOOLBOX_BASE" -ForegroundColor Green

# Clone AMICA repository
Write-Host "[3/4] Cloning AMICA from GitHub..."
if (Test-Path $AMICA_DIR) {
    Write-Host "  ⚠ AMICA directory already exists: $AMICA_DIR" -ForegroundColor Yellow
    $response = Read-Host "  Overwrite? (y/n)"
    if ($response -eq "y") {
        Write-Host "  Removing existing directory..."
        Remove-Item -Recurse -Force $AMICA_DIR
    } else {
        Write-Host "  Skipping clone." -ForegroundColor Yellow
        goto :validate
    }
}

Write-Host "  Cloning: $AMICA_REPO"
Write-Host "  Destination: $AMICA_DIR"
Write-Host "  This may take a minute..."

try {
    git clone $AMICA_REPO $AMICA_DIR 2>&1 | Write-Host
    Write-Host "  ✓ Clone complete" -ForegroundColor Green
} catch {
    Write-Host "  ✗ Failed to clone: $_" -ForegroundColor Red
    exit 1
}

# Validate installation
:validate
Write-Host "[4/4] Validating installation..."
$amicarunner = "$AMICA_DIR/amicarunner.m"
$doica = "$AMICA_DIR/doica.m"

if ((Test-Path $amicarunner) -and (Test-Path $doica)) {
    Write-Host "  ✓ amicarunner.m found" -ForegroundColor Green
    Write-Host "  ✓ doica.m found" -ForegroundColor Green
    Write-Host ""
    Write-Host "════════════════════════════════════════════════════════════" -ForegroundColor Green
    Write-Host "✓ AMICA installed successfully!" -ForegroundColor Green
    Write-Host "════════════════════════════════════════════════════════════" -ForegroundColor Green
    Write-Host ""
    Write-Host "Next steps:" -ForegroundColor Cyan
    Write-Host "  1. Add to MATLAB startup.m:"
    Write-Host "     addpath(genpath('$AMICA_DIR'))"
    Write-Host "  2. Or run health check:"
    Write-Host "     python scripts/check_pipeline_health.py"
    Write-Host ""
} else {
    Write-Host "  ✗ Installation incomplete" -ForegroundColor Red
    Write-Host "  Expected files not found:" -ForegroundColor Red
    if (-not (Test-Path $amicarunner)) {
        Write-Host "    - $amicarunner" -ForegroundColor Red
    }
    if (-not (Test-Path $doica)) {
        Write-Host "    - $doica" -ForegroundColor Red
    }
    exit 1
}
