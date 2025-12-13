# AMICA Toolbox Installer from GitHub
# Downloads and installs AMICA to MATLAB toolboxes directory

$ErrorActionPreference = "Stop"

$TOOLBOX_BASE = "C:/MATLAB/toolboxes"
$AMICA_REPO = "https://github.com/sccn/amica.git"
$AMICA_DIR = "$TOOLBOX_BASE/amica"

Write-Host "════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "AMICA Toolbox Installer" -ForegroundColor Cyan
Write-Host "════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host ""

# Step 1: Check Git
Write-Host "[1/4] Checking Git installation..."
try {
    $gitVersion = git --version
    Write-Host "  [OK] $gitVersion" -ForegroundColor Green
} catch {
    Write-Host "  [ERROR] Git not found in PATH" -ForegroundColor Red
    Write-Host "  Install from: https://git-scm.com/download/win" -ForegroundColor Yellow
    exit 1
}

# Step 2: Create toolbox directory
Write-Host "[2/4] Setting up toolbox directory..."
if (-not (Test-Path $TOOLBOX_BASE)) {
    New-Item -ItemType Directory -Path $TOOLBOX_BASE -Force | Out-Null
    Write-Host "  Created: $TOOLBOX_BASE" -ForegroundColor Green
} else {
    Write-Host "  [OK] Directory exists: $TOOLBOX_BASE" -ForegroundColor Green
}

# Step 3: Clone AMICA
Write-Host "[3/4] Cloning AMICA from GitHub..."

if (Test-Path $AMICA_DIR) {
    Write-Host "  [WARN] AMICA already exists at: $AMICA_DIR" -ForegroundColor Yellow
    $overwrite = Read-Host "  Overwrite (y/n)?"
    if ($overwrite -ne "y") {
        Write-Host "  Skipping clone step." -ForegroundColor Yellow
    } else {
        Remove-Item -Recurse -Force $AMICA_DIR
        Write-Host "  Cloning $AMICA_REPO..."
        & git clone $AMICA_REPO $AMICA_DIR
        Write-Host "  [OK] Clone complete" -ForegroundColor Green
    }
} else {
    Write-Host "  Cloning: $AMICA_REPO"
    Write-Host "  Destination: $AMICA_DIR"
    Write-Host "  (this may take a minute...)"
    & git clone $AMICA_REPO $AMICA_DIR
    Write-Host "  [OK] Clone complete" -ForegroundColor Green
}

# Step 4: Validate
Write-Host "[4/4] Validating installation..."

$amicarunner = Join-Path $AMICA_DIR "amicarunner.m"
$doica = Join-Path $AMICA_DIR "doica.m"

if ((Test-Path $amicarunner) -and (Test-Path $doica)) {
    Write-Host "  [OK] amicarunner.m found" -ForegroundColor Green
    Write-Host "  [OK] doica.m found" -ForegroundColor Green
    Write-Host ""
    Write-Host "════════════════════════════════════════════════════════════" -ForegroundColor Green
    Write-Host "[SUCCESS] AMICA installed successfully!" -ForegroundColor Green
    Write-Host "════════════════════════════════════════════════════════════" -ForegroundColor Green
    Write-Host ""
    Write-Host "Next step: Copy startup.m to MATLAB" -ForegroundColor Cyan
    Write-Host "  `$MatlabDir = `"$env:USERPROFILE\Documents\MATLAB`""
    Write-Host "  Copy-Item startup.m `$MatlabDir\startup.m -Force"
    Write-Host ""
    Write-Host "Then restart MATLAB and run:" -ForegroundColor Cyan
    Write-Host "  python scripts/check_pipeline_health.py"
    Write-Host ""
} else {
    Write-Host "  [ERROR] Installation incomplete" -ForegroundColor Red
    if (-not (Test-Path $amicarunner)) {
        Write-Host "    Missing: $amicarunner" -ForegroundColor Red
    }
    if (-not (Test-Path $doica)) {
        Write-Host "    Missing: $doica" -ForegroundColor Red
    }
    exit 1
}
