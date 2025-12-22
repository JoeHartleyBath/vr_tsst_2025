# New PC Setup Guide - Full Pipeline

## Quick Start Checklist
- [ ] Clone repository
- [ ] Install Python dependencies
- [ ] Install MATLAB + toolboxes (EEGLAB, AMICA)
- [ ] Install R + packages
- [ ] Transfer raw data (≥3 participants)
- [ ] Create config file
- [ ] Test pipeline stage-by-stage

---

## 1. Repository Setup

```powershell
# Clone repository
cd C:\phd_projects
git clone <your-repo-url> vr_tsst_2025
cd vr_tsst_2025

# Check out your current branch
git branch -a
git checkout <your-branch-name>
```

---

## 2. Python Setup (Stages 1, 4, 5)

### Install Python (3.9+)
Download from: https://www.python.org/downloads/

### Install Dependencies
```powershell
pip install pyxdf mne numpy pandas pyyaml scipy neurokit2 matplotlib
```

**Required packages:**
- `pyxdf` - XDF file reading
- `mne` - EEG processing
- `numpy`, `pandas` - Data manipulation
- `pyyaml` - YAML config parsing
- `scipy` - Signal processing
- `neurokit2` - Physiological data cleaning
- `matplotlib` - Visualization

---

## 3. MATLAB Setup (Stages 2, 3)

### Install MATLAB (R2020a+)
Ensure you have:
- Signal Processing Toolbox
- Statistics and Machine Learning Toolbox

### Install EEGLAB
```matlab
% Download EEGLAB from: https://sccn.ucsd.edu/eeglab/
% Extract to: C:\MATLAB\toolboxes\eeglab
addpath('C:\MATLAB\toolboxes\eeglab');
eeglab;  % Initialize
```

### Install EEGLAB Plugins
```matlab
% In EEGLAB GUI:
% File > Manage EEGLAB Extensions > Data Import/Export
% - ICLabel (for artifact detection)
% - clean_rawdata (for ASR)
% - Firfilt (for filtering)
```

### Install AMICA (Automated via GitHub)
**Option 1: Automatic Installation (Recommended)**
```powershell
cd c:\vr_tsst_2025
powershell -ExecutionPolicy Bypass install_amica.ps1
```
This script clones AMICA from GitHub and configures it automatically.

**Option 2: Manual Installation**
1. Clone from GitHub:
   ```bash
   git clone https://github.com/sccn/amica.git C:\MATLAB\toolboxes\amica
   ```
   Or download from: https://github.com/sccn/amica/releases

2. Extract to: `C:\MATLAB\toolboxes\amica`

3. Verify: Check that `amicarunner.m` exists in the folder

4. Add to MATLAB path:
   ```matlab
   addpath(genpath('C:\MATLAB\toolboxes\amica'));
   ```

### Install yamlmatlab
```matlab
% Download from: https://github.com/ewiger/yamlmatlab
% Extract to: C:\MATLAB\toolboxes\yamlmatlab
addpath(genpath('C:\MATLAB\toolboxes\yamlmatlab'));
```

### Test MATLAB Setup
```matlab
% Add project to path
cd('C:\phd_projects\vr_tsst_2025');
addpath(genpath('scripts'));
addpath(genpath('config'));

% Test YAML loading
config = yaml.loadFile('config/general.yaml');
disp(config);
```

---

## 4. R Setup (Stages 6, 7)

### Install R (4.2+)
Download from: https://cran.r-project.org/

### Install RStudio (Optional)
Download from: https://posit.co/download/rstudio-desktop/

### Install R Packages
```r
# Set CRAN mirror
options(repos = c(CRAN = "https://cran.r-project.org/"))

# Core packages
install.packages(c(
  "tidyverse",    # Data manipulation
  "yaml",         # Config parsing
  "readxl",       # Excel reading
  "e1071",        # SVM
  "caret",        # ML utilities
  "yardstick",    # Performance metrics
  "doParallel",   # Parallel processing
  "rstatix",      # Statistics
  "rmcorr"        # Repeated measures correlation
))
```

### Test R Setup
```r
setwd("C:/phd_projects/vr_tsst_2025")
config <- yaml::read_yaml("scripts/utils/config.yaml")
print(config)
```

---

## 5. Raw Data Transfer

### Required Data Structure
```
data/
├── raw/
│   ├── eeg/
│   │   ├── P01.xdf
│   │   ├── P02.xdf
│   │   └── P03.xdf (minimum 3 participants)
│   └── metadata/
│       ├── P01.csv
│       ├── P02.csv
│       └── P03.csv
├── experimental_counterbalance.xlsx
```

### Copy from Old PC
Transfer these folders from your current PC:
```powershell
# On OLD PC - Create archive
Compress-Archive -Path "C:\phd_projects\vr_tsst_2025\data\raw" -DestinationPath "C:\raw_data_backup.zip"

# Transfer raw_data_backup.zip to NEW PC, then:
Expand-Archive -Path "C:\raw_data_backup.zip" -DestinationPath "C:\phd_projects\vr_tsst_2025\data\"
```

### Minimum 3 Participants Required
- **P01, P02, P03** (or any 3 participants with complete data)
- Each needs: `.xdf` file + corresponding `.csv` metadata
- Counterbalance sheet must include all 3 participants

---

## 6. Configuration File Setup

### Create Missing Config File
Create: `scripts/utils/config.yaml`

```yaml
paths:
  eeg_data: "data/raw/eeg"
  events: "data/raw/events"
  metadata: "data/raw/metadata"
  physio: "data/raw/metadata"
  subjective: "data/raw/subjective"
  raw_data: "data/raw/metadata"
  output: "output"
  logs: "output/logs"
  results: "results"
  failed_qc: "output/qc/summary"
  counterbalance: "data"
  xgb_results: "results/xgb/results"
  xgb_bayes: "results/xgb/bayes_opt"
  init_pruned: "results/xgb/init_pruned"
```

### Create Empty QC Failures File
```powershell
# Create directory
New-Item -Path "output\qc\summary" -ItemType Directory -Force

# Create empty QC failures CSV
"Participant_ID" | Out-File -FilePath "output\qc\summary\qc_failures_summary.csv" -Encoding UTF8
```

---

## 7. Pipeline Execution Order

### Stage 1: XDF → SET (Python)
```powershell
cd C:\phd_projects\vr_tsst_2025
python scripts/preprocessing/raw_conversion/run/run_xdf_to_set_end2end.py
```
**Edit script first** to loop through P01-P03 instead of just P01.

### Stage 2: EEG Cleaning (MATLAB)
```matlab
cd('C:\phd_projects\vr_tsst_2025');
addpath(genpath('scripts'));

% Edit participant list in run_clean_eeg_pipeline.m line 23:
% participant_numbers = [1, 2, 3];

run('scripts/preprocessing/eeg/cleaning/run_clean_eeg_pipeline.m');
```

### Stage 3: EEG Feature Extraction (MATLAB)
```matlab
% Edit extract_eeg_features.m to specify participants
run('scripts/preprocessing/eeg/feature_extraction/extract_eeg_features.m');
```

### Stage 4: Physio Feature Extraction (Python)
```powershell
python scripts/preprocessing/physio/feature_extraction/extract_physio_features.py
```

### Stage 5: Merge Features (Python)
```powershell
python scripts/preprocessing/physio/feature_extraction/mvp_merge_pipeline.py
```
**Output:** `output/aggregated/all_data_aggregated.csv`

### Stage 6: R Preprocessing (R)
```r
source("scripts/preproccess_for_xgb.R")
```
**Output:** `output/final_data.rds`

### Stage 7: SVM Classification (R)
```r
source("scripts/modeling/svm/svm.R")
```
**Output:** `results/svm/svm_progress_*.csv` with accuracy scores

---

## 8. Quick Test Script

Save as `test_setup.ps1`:

```powershell
# Test Python
Write-Host "Testing Python..." -ForegroundColor Cyan
python -c "import pyxdf, mne, pandas; print('Python OK')"

# Test R
Write-Host "`nTesting R..." -ForegroundColor Cyan
Rscript -e "library(tidyverse); library(e1071); cat('R OK\n')"

# Test MATLAB (requires MATLAB in PATH)
Write-Host "`nTesting MATLAB..." -ForegroundColor Cyan
matlab -batch "addpath('scripts'); disp('MATLAB OK'); exit"

# Check raw data
Write-Host "`nChecking raw data..." -ForegroundColor Cyan
Get-ChildItem "data\raw\eeg\*.xdf" | Select-Object Name
Get-ChildItem "data\raw\metadata\*.csv" | Select-Object Name

Write-Host "`nSetup validation complete!" -ForegroundColor Green
```

Run with:
```powershell
.\test_setup.ps1
```

---

## 9. Troubleshooting

### MATLAB Path Issues
```matlab
% Save startup.m in MATLAB userpath
userpath
% Create startup.m with:
addpath(genpath('C:\MATLAB\toolboxes\eeglab'));
addpath(genpath('C:\MATLAB\toolboxes\amica'));
addpath(genpath('C:\MATLAB\toolboxes\yamlmatlab'));
addpath(genpath('C:\phd_projects\vr_tsst_2025\scripts'));
```

### AMICA Not Found
- Ensure binary (`amica15c.exe` on Windows) is in AMICA folder
- Check 64-bit vs 32-bit compatibility
- May need to compile from source

### Python Import Errors
```powershell
# Create virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install pyxdf mne numpy pandas pyyaml scipy neurokit2
```

### R Package Installation Fails
```r
# Install Rtools (for compiling packages)
# Download from: https://cran.r-project.org/bin/windows/Rtools/

# Set library path
.libPaths("C:/Users/<YourUser>/R/win-library/4.2")
```

---

## 10. Expected Runtime (3 Participants)

| Stage | Time | Bottleneck |
|-------|------|------------|
| XDF→SET | 5-10 min | Disk I/O |
| EEG Cleaning | **2-3 hours** | **AMICA** |
| EEG Features | 15-30 min | Power computation |
| Physio Features | 10-20 min | Neurokit cleaning |
| Merge | <1 min | - |
| R Preprocessing | 5-10 min | Transform functions |
| SVM LOSO | 30-60 min | Nested CV |

**Total: ~3-5 hours** for 3 participants end-to-end.

---

## 11. Data Backup Recommendation

On new PC, set up automatic backups:
```powershell
# Backup outputs after each stage
robocopy "C:\phd_projects\vr_tsst_2025\output" "D:\backups\vr_tsst_output" /MIR /Z
```

---

## Contact/Notes
- AMICA is the slowest step - consider testing with `runica` first
- SVM requires ≥3 participants minimum
- All paths in configs use forward slashes (works on Windows)
