# AMICA Installation Guide

AMICA (Automatic ICA MATLAB Toolbox) is required for EEG artifact cleaning in the pipeline.

**GitHub Repository**: https://github.com/sccn/amica

---

## Quick Install (Recommended)

### Prerequisites
- Git for Windows installed: https://git-scm.com/download/win
- MATLAB R2020a or later
- ~500 MB disk space

### Step 1: Run Installer Script
```powershell
cd c:\vr_tsst_2025
powershell -ExecutionPolicy Bypass install_amica.ps1
```

**What this does**:
1. Checks Git installation
2. Creates `C:/MATLAB/toolboxes` directory
3. Clones AMICA from GitHub
4. Validates installation

### Step 2: Configure MATLAB Startup
Copy `startup.m` to your MATLAB user directory:
```powershell
# Default location (Windows)
$MatlabUserDir = "$env:USERPROFILE\Documents\MATLAB"
Copy-Item startup.m $MatlabUserDir\startup.m -Force
```

**Or manually**:
1. Open MATLAB
2. Create a file at: `%USERPROFILE%\Documents\MATLAB\startup.m`
3. Paste contents of `c:\vr_tsst_2025\startup.m`
4. Save and restart MATLAB

### Step 3: Verify Installation
In MATLAB console:
```matlab
% Check if AMICA is loaded
which amicarunner
% Output: C:\MATLAB\toolboxes\amica\amicarunner.m

% Check EEGLAB
which eeglab
% Output: C:\MATLAB\toolboxes\eeglab\eeglab.m
```

---

## Manual Installation (If Automated Fails)

### Option A: Using Git
```powershell
# Create toolbox directory
New-Item -ItemType Directory -Path C:\MATLAB\toolboxes -Force

# Clone AMICA
git clone https://github.com/sccn/amica.git C:\MATLAB\toolboxes\amica

# Verify
Get-ChildItem C:\MATLAB\toolboxes\amica\amicarunner.m
```

### Option B: Download ZIP
1. Visit: https://github.com/sccn/amica/releases
2. Download latest `amica-main.zip`
3. Extract to: `C:\MATLAB\toolboxes\amica`
4. Verify: `amicarunner.m` exists in the folder

### Add to MATLAB Path
```matlab
% In MATLAB console or startup.m:
addpath(genpath('c:/MATLAB/toolboxes/amica'));
savepath;
```

---

## AMICA Binaries

AMICA requires compiled binaries for your system:

### Windows (Included in GitHub)
- `amicarunner.exe` or `amica15c.exe`
- Included in the GitHub repository
- No additional download needed

### macOS/Linux
- Visit: https://github.com/sccn/amica/releases
- Download system-specific binary
- Place in AMICA folder alongside MATLAB scripts

---

## Troubleshooting

### Issue: "amicarunner not found"
```
Error: amicarunner.m not found in path
```

**Fix**:
1. Verify installation:
   ```powershell
   Test-Path C:\MATLAB\toolboxes\amica\amicarunner.m
   ```
2. If missing, re-run installer:
   ```powershell
   powershell -ExecutionPolicy Bypass install_amica.ps1
   ```

### Issue: "MATLAB not found" in installer
```
Git not found in PATH
```

**Fix**:
1. Install Git: https://git-scm.com/download/win
2. During installation, choose "Add Git to PATH"
3. Restart PowerShell and re-run installer

### Issue: Permission denied
```
Access Denied: C:\MATLAB\toolboxes
```

**Fix**:
1. Run PowerShell as Administrator
2. Re-run installer:
   ```powershell
   powershell -ExecutionPolicy Bypass install_amica.ps1
   ```

### Issue: Clone failed / Network error
```
fatal: unable to access 'https://github.com/sccn/amica.git'
```

**Fix**:
- Check internet connection
- Try manual download: https://github.com/sccn/amica/releases
- Or use the ZIP extraction method (Option B above)

---

## Verifying Full Setup

After installing AMICA, run the health check:
```powershell
cd c:\vr_tsst_2025
python scripts/check_pipeline_health.py
```

Expected output:
```
[5/5] Checking EEGLAB installation...
  ✓ Found at c:/MATLAB/toolboxes/eeglab
  ✓ AMICA found at c:/MATLAB/toolboxes/amica
```

---

## AMICA Documentation

- **GitHub**: https://github.com/sccn/amica
- **SCCN Page**: https://sccn.ucsd.edu/wiki/AMICA
- **Paper**: Delorme et al. (2012) - See GitHub releases for citation

---

## Next Steps

1. ✓ Install AMICA (this guide)
2. Run health check: `python scripts/check_pipeline_health.py`
3. Run pilot test: `python scripts/run_pipeline_master.py`
4. Validate results: `python scripts/validation/compare_pipelines.py`

---

**Need help?**  
Check [TOOLS_REFERENCE.md](../TOOLS_REFERENCE.md) or [PILOT_TEST_GUIDE.md](../PILOT_TEST_GUIDE.md)

**Contact**: Joe Hartley <jh3968@bath.ac.uk>
