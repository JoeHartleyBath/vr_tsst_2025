# Quick Start - Run Full Pipeline (3 Participants)

## Prerequisites ✓
- [ ] All dependencies installed (see NEW_PC_SETUP.md)
- [ ] Raw data for P01, P02, P03 in `data/raw/`
- [ ] MATLAB, Python, R all working

---

## Step-by-Step Execution

### 1. Initialize Empty QC File (One-time)
```powershell
New-Item -Path "output\qc\summary" -ItemType Directory -Force
"Participant_ID" | Out-File "output\qc\summary\qc_failures_summary.csv"
```

### 2. XDF → SET Conversion (Python)
Edit `scripts/preprocessing/raw_conversion/run/run_xdf_to_set_end2end.py`:

```python
if __name__ == "__main__":
    participants = [1, 2, 3]  # Add P02, P03
    
    for p_num in participants:
        xdf = Path(rf"C:\phd_projects\vr_tsst_2025\data\raw\eeg\P{p_num:02d}.xdf")
        out = Path(rf"C:\phd_projects\vr_tsst_2025\output\processed\P{p_num:02d}.set")
        
        try:
            summary = xdf_to_set(xdf, out)
            print(f"P{p_num:02d} conversion complete")
        except Exception as e:
            print(f"P{p_num:02d} failed:", e)
```

Run:
```powershell
python scripts\preprocessing\raw_conversion\run\run_xdf_to_set_end2end.py
```

---

### 3. EEG Cleaning (MATLAB - SLOW!)
Edit `scripts/preprocessing/eeg/cleaning/run_clean_eeg_pipeline.m` line 23:

```matlab
participant_numbers = [1, 2, 3];  % Change from [1]
```

Run in MATLAB:
```matlab
cd('C:\phd_projects\vr_tsst_2025');
addpath(genpath('scripts'));
run('scripts/preprocessing/eeg/cleaning/run_clean_eeg_pipeline.m');
```

⏱️ **Expected: 2-3 hours** (AMICA is slow)

---

### 4. EEG Feature Extraction (MATLAB)
Check participants list in `scripts/preprocessing/eeg/feature_extraction/extract_eeg_features.m` 
(Should auto-detect cleaned files)

Run in MATLAB:
```matlab
run('scripts/preprocessing/eeg/feature_extraction/extract_eeg_features.m');
```

**Output:** `output/aggregated/eeg_features.csv`

---

### 5. Physiological Feature Extraction (Python)
```powershell
python scripts\preprocessing\physio\feature_extraction\extract_physio_features.py
```

---

### 6. Merge All Features (Python)
```powershell
python scripts\preprocessing\physio\feature_extraction\mvp_merge_pipeline.py
```

**Output:** `output/aggregated/all_data_aggregated.csv`

---

### 7. R Preprocessing & Transformation (R)
```r
setwd("C:/phd_projects/vr_tsst_2025")
source("scripts/preproccess_for_xgb.R")
```

**Output:** `output/final_data.rds`

---

### 8. SVM Classification (R)
```r
source("scripts/modeling/svm/svm.R")
```

**Output:** `results/svm/svm_progress_stress_label.csv` with accuracy scores!

---

## Expected Outputs

After full pipeline completion:

```
output/
├── processed/
│   ├── P01.set, P02.set, P03.set
├── cleaned_eeg/
│   ├── P01_cleaned.set, P02_cleaned.set, P03_cleaned.set
├── aggregated/
│   ├── eeg_features.csv
│   └── all_data_aggregated.csv
├── final_data.csv
└── final_data.rds

results/
└── svm/
    ├── svm_progress_stress_label.csv       ← ACCURACY SCORES
    ├── svm_progress_workload_label.csv     ← ACCURACY SCORES
    ├── svm_progress_stress_label.csv_tuning.csv
    └── svm_progress_workload_label.csv_tuning.csv
```

---

## Validation Checks

### After Stage 2 (Cleaning):
```powershell
Get-ChildItem output\cleaned_eeg\*.set
# Should see: P01_cleaned.set, P02_cleaned.set, P03_cleaned.set
```

### After Stage 3 (EEG Features):
```powershell
Get-Content output\aggregated\eeg_features.csv -Head 5
# Should see: Participant,Condition,<127 feature columns>
```

### After Stage 6 (R Preprocessing):
```r
df <- readRDS("output/final_data.rds")
dim(df)  # Should be ~12 rows (3 participants × 4 conditions)
```

### After Stage 7 (SVM):
```r
results <- read.csv("results/svm/svm_progress_stress_label.csv")
print(results$final_acc)  # Check accuracy scores
```

---

## Troubleshooting

**Pipeline fails at Stage X?**
- Check logs in `output/logs/`
- Verify previous stage outputs exist
- Check `output/qc/` for QC failures

**MATLAB runs out of memory?**
- Close other applications
- Reduce AMICA `max_iter` to 200 in `clean_eeg.m`

**R can't find final_data.rds?**
- Check `output/aggregated/all_data_aggregated.csv` exists
- Re-run Stage 6

**SVM accuracy = 50%?**
- This is chance level (binary classification)
- May indicate data quality issues
- Check feature distributions in `final_data.rds`
