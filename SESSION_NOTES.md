# Session Notes - December 12, 2025

## Work Completed Today

### MATLAB Compatibility Fixes & Pipeline Validation
**Goal: Validate refactored pipeline end-to-end on real data**

#### Fixed Issues:
1. ✅ **Data type compatibility**
   - Python int/float types incompatible with MATLAB operations
   - Added defensive type conversion in MATLAB: `double(EEG.xmin)`, `double(EEG.srate)`, etc.
   - Added explicit casting in Python: `float()`, `int()` for numeric fields

2. ✅ **Event/channel structure format**
   - scipy.savemat doesn't auto-convert Python lists to MATLAB struct arrays
   - Fixed: Convert to numpy structured arrays before saving
   - Events: `dtype=[('latency','f8'),('type','O'),('duration','f8')]`
   - Chanlocs: `dtype=[('labels','O')]`

3. ✅ **Removed external dependencies**
   - CleanLine plugin requires Statistics and Machine Learning Toolbox (not available)
   - Replaced with standard EEGLAB notch filter: `pop_eegfiltnew` (49-51 Hz, revfilt=1)

4. ✅ **Channel locations**
   - Added ANT Neuro 128-channel equidistant layout: `config/chanlocs/NA-271.elc`
   - Updated scripts to use relative path from project root

5. ✅ **Test infrastructure**
   - Created 5-minute test subset from P01.set for faster validation
   - Automated test script: `test_each_step_auto.m`
   - Validates steps 0-6: Load, chanlocs, resample, bandpass, notch (50Hz + 25Hz), ASR

#### Known Issues / TODO:
- ⚠️ **ICA rank problem**: Data rank reports 0 after ASR, preventing ICA
  - May need PCA dimension reduction before ICA
  - AMICA configuration pending (binary path issues)
  - Test script stops after ASR (step 6) until resolved
  
- ⚠️ **Interpolation timing**: 
  - For standard ICA: Should interpolate before ICA
  - For AMICA: Can interpolate after (as in legacy)
  - Current implementation matches legacy (after artifact removal)

#### Commits:
- `7b5edfe`: "Fix MATLAB compatibility and remove external dependencies"

---

### EEG Feature Extraction Refactoring
**Goal: Simplify and streamline feature extraction pipeline**

#### Analysis of Legacy Script:
- **Original:** `extract_eeg_feats.m` (1003 lines)
- **Identified issues:**
  - Hardcoded paths, participant lists, frequency bands, regions
  - Label mapping duplicated from `config/conditions.yaml`
  - 107-line canonicalization function with hardcoded logic
  - Complex LITE mode toggle with conditional features
  - Rolling window features (64 columns, not needed)
  - Connectivity features (wPLI, not needed)
  - Aperiodic slope fitting (not needed)
  - Statistics (Mean/SD/Slope, not needed)

#### Refactored Solution:
**Created `config/eeg_feature_extraction.yaml`:**
- Frequency bands (8 bands: Delta, Theta, Alpha variants, Beta variants)
- Channel regions (11 regions with channel lists)
- Feature flags (simplified to 3: band_power, ratios, entropy)
- Notch exclusion participants (P01-P07)
- Parallel processing settings (8 workers)
- Toolbox paths

**Created `scripts/eeg_features/extract_eeg_features.m` (530 lines):**
- Loads all config from YAML (no hardcoded values)
- Computes only essential features:
  * Band power per region × band: 88 features
  * Power ratios: 5 features (frontal asymmetry, alpha/beta, theta/beta, etc.)
  * Entropy per region: 22 features (sample + spectral)
- **Total:** ~115 columns (vs 281 in legacy)
- **Removed:** Rolling windows, statistics, connectivity, aperiodic (~600 lines of code)

#### Benefits:
- ✅ 47% code reduction (530 vs 1003 lines)
- ✅ Configuration-driven (change YAML, not code)
- ✅ No hardcoded paths, conditions, or channel definitions
- ✅ Faster execution (fewer features)
- ✅ Easier to maintain and test
- ✅ Parallel processing retained (8 workers)

#### File Organization:
```
scripts/eeg_features/
├── extract_eeg_features.m          # New streamlined version (530 lines)
├── extract_eeg_features_legacy.m   # Original (1003 lines)
└── legacy/
    ├── step03_extract_eeg_features_fixed_lite.m
    └── step03_extract_eeg_features_ratios_only.m
```

#### Commits:
- `44bf711`: "Organize EEG feature extraction scripts"
- `90824d2`: "Create streamlined EEG feature extraction pipeline"

---

# Session Notes - December 11, 2025

## Work Completed Today

### 1. Python Pipeline Refactoring (xdf_to_set.py)
**Major architectural changes to separate data loading from event handling:**

- ✅ **Removed hardcoded condition mappings** from Python code
  - Previously: 29 hardcoded condition names in multiple locations
  - Now: All conditions loaded from `config/conditions.yaml`
  
- ✅ **Refactored event marker system**
  - Old: Mixed condition onsets + response markers in single pipeline
  - New: Separate functions for condition events vs response transitions
  - `extract_event_timestamps()` - condition onset times only
  - `extract_response_timestamps()` - response state transitions only
  - `build_eeg_event_list()` - combines both into unified event list

- ✅ **Improved timestamp alignment**
  - Fixed datetime64 conversion for proper EEG sample alignment
  - Added `align_timestamps()` to convert physio timestamps → EEG sample indices
  - Handles missing events gracefully with `ignore_missing=True`

- ✅ **Enhanced exposure type handling**
  - `add_exposure_type()` now properly labels task conditions
  - Maps raw CSV labels → standardized exposure types from config
  - Handles edge cases (Pre/Post Exposure baselines, calibrations, relaxation)

- ✅ **Config-driven event export**
  - Event labels loaded from `config/conditions.yaml`
  - Easy to modify exported event names without code changes
  - Maintains backward compatibility with existing analysis scripts

### 2. MATLAB Pipeline Refactoring
- ✅ Analyzed legacy MATLAB script (`step02_clean_eeg_data.m`) - identified ~300 lines of redundant code
- ✅ Created streamlined `clean_eeg.m` (~450 lines vs 745 in legacy)
  - Removed event loading, trigger channel management, hardcoded condition mappings
  - Python handles all data prep → MATLAB focuses purely on signal processing
  - Focused on: filtering, ASR, AMICA, ICLabel, interpolation, re-referencing
  
### 2. Config Externalization
- ✅ Created `config/eeglab_template.yaml` with all 40+ required EEGLAB fields
- ✅ Refactored `xdf_to_set.py` to load EEGLAB structure from YAML instead of hardcoded dict
- ✅ Fixed pandas `FutureWarning` by replacing `groupby().apply()` with simple loop
- ✅ All condition mappings now in `config/conditions.yaml` (removed from code)

### 3. Testing Infrastructure
- ✅ Created end-to-end test: `scripts/run/run_xdf_to_set_end2end.py`
- ✅ Created `test_clean_eeg.m` - synthetic data test for MATLAB pipeline
- ✅ Created `run_clean_eeg_pipeline.m` - batch processor for multiple participants
- ✅ Created `test_each_step.m` - interactive step-by-step tester with visualizations
- ✅ Created `test_each_step_auto.m` - automated version for CI/batch testing
- ✅ Created `create_test_subset.m` - extract 5-minute segments for rapid testing

### 4. Code Cleanup
- ✅ Removed obsolete test scripts (regenerate_p01_set.py, test_pipeline.py, quick_test_clean.m)
- ✅ Fixed import path in `run_xdf_to_set_end2end.py`
- ✅ Renamed legacy script to `clean_eeg_legacy.m` for reference

### 5. Git Commits
- ✅ All changes committed to `eeg-refactor` branch
- ✅ 17 commits ahead of origin (not yet pushed)
- Key commits:
  - Python pipeline refactor (event handling separation)
  - MATLAB pipeline streamlining
  - Config externalization
  - Pandas compatibility fixes

## Architecture Changes

### Before: Monolithic Pipeline
```
Python (xdf_to_set.py):
- Load XDF
- Hardcoded conditions
- Create events
- Add trigger channel
- Save .set

MATLAB (step02_clean_eeg_data.m):
- Load .set
- Parse events from CSV again
- Create trigger channel again
- Apply cleaning
- Re-map events again
- Save
```

### After: Separation of Concerns
```
Python (xdf_to_set.py):
- Load XDF
- Align streams
- Load conditions from config
- Extract condition onsets
- Extract response transitions
- Build unified event list
- Create complete EEGLAB structure
- Save .set (with all events embedded)

MATLAB (clean_eeg.m):
- Load .set (events already embedded)
- Apply signal processing only:
  * Filtering (1-49 Hz, CleanLine, notch)
  * ASR (artifact rejection)
  * AMICA (ICA)
  * ICLabel (component classification)
  * Artifact removal
  * Interpolation
  * Re-referencing
- Save cleaned data + QC metrics
```

### Benefits:
1. **No code duplication** - event handling done once in Python
2. **Config-driven** - easy to modify conditions without code changes
3. **MATLAB focuses on signal processing** - what it does best
4. **Easier testing** - can test event handling separately from cleaning
5. **Better maintainability** - clear responsibilities for each language

## Current Status

### Blocked - Awaiting P01.set Regeneration
The Python pipeline needs to regenerate `P01.set` with the complete EEGLAB structure before MATLAB testing can proceed.

**Issue:** Previous P01.set was missing required fields (setname, filename, etc.), causing MATLAB's `eeg_checkset()` to fail.

**Fix Applied:** Updated `build_eeglab_struct()` to include all required fields from `config/eeglab_template.yaml`.

## Next Steps (Tomorrow)

### Priority 1: Complete Pipeline Validation
1. **Regenerate P01.set**
   ```bash
   python scripts\run\run_xdf_to_set_end2end.py
   ```
   - Takes 2-5 minutes for 1-hour recording
   - Output: `output/processed/P01.set` with complete EEGLAB structure

2. **Verify EEGLAB Compatibility**
   - Open MATLAB, load `P01.set`, verify `eeg_checkset()` passes
   - Check for all required fields present

3. **Create Test Subset**
   ```matlab
   run scripts/clean_eeg/create_test_subset.m
   ```
   - Output: `P01_subset_5min.set` (~80-100 MB vs 982 MB)
   - Faster iteration for testing (5-10 min vs 30-60 min)

### Priority 2: Test MATLAB Cleaning Pipeline
4. **Run Automated Step-by-Step Test**
   ```matlab
   run scripts/clean_eeg/test_each_step_auto.m
   ```
   - Tests all 11 cleaning steps on subset
   - Validates: load, chanlocs, resample, bandpass, cleanline, notch, ASR, AMICA, ICLabel, artifact removal, interpolation, re-reference

5. **Run Full Pipeline on Complete Dataset**
   ```matlab
   [EEG, qc] = clean_eeg('output/processed/P01.set', ...
                         'output/cleaned_eeg', 1, ...
                         'output/vis/P01', 'output/qc', []);
   ```
   - Takes 30-60 minutes for full hour of data
   - Generates P01_cleaned.mat, P01_cleaned.set, QC metrics

### Priority 3: Process Remaining Participants
6. **Batch Processing**
   - Update `run_clean_eeg_pipeline.m` participant list
   - Process all participants (P02-P40)
   - Monitor QC metrics for consistency

## Known Issues

### Resolved
- ✅ Missing EEGLAB fields in .set files → Fixed with YAML template
- ✅ Pandas FutureWarning → Fixed with simple loop
- ✅ Import path error in end2end script → Fixed with sys.path.insert()

### Open
- ⏳ Need to push commits to remote (17 commits ahead)
- ⏳ P01.set regeneration not yet complete

## File Structure

```
config/
├── conditions.yaml              # NEW: Condition definitions + event export labels
└── eeglab_template.yaml         # NEW: EEGLAB structure definition

scripts/
├── xdf_to_set/
│   └── xdf_to_set.py            # REFACTORED: Event handling + config-driven
├── clean_eeg/
│   ├── clean_eeg.m              # NEW: Streamlined pipeline (450 lines)
│   ├── clean_eeg_legacy.m       # RENAMED: Original for reference (745 lines)
│   ├── test_clean_eeg.m         # NEW: Synthetic data test
│   ├── run_clean_eeg_pipeline.m # NEW: Batch processor
│   ├── test_each_step.m         # NEW: Interactive tester
│   ├── test_each_step_auto.m    # NEW: Automated tester
│   └── create_test_subset.m     # NEW: 5-minute subset extractor
└── run/
    └── run_xdf_to_set_end2end.py # MODIFIED: Fixed import path

data/raw/
└── metadata/
    └── P01.csv                  # Now includes Response column for transitions

output/processed/
└── P01.set                      # Will include both condition events + response transitions
```

## Key Functions Refactored

### xdf_to_set.py
- `load_condition_config()` - Load conditions from YAML
- `add_exposure_type()` - Map CSV labels → exposure types
- `extract_event_timestamps()` - Get condition onset times
- `extract_response_timestamps()` - Get response state transitions
- `build_eeg_event_list()` - Combine events with latencies
- `build_eeglab_struct()` - Load structure from YAML template
- `xdf_to_set()` - Main pipeline orchestrator

### clean_eeg.m
- `clean_eeg()` - Main orchestrator (no event handling)
- `run_amica_pipeline()` - ICA decomposition with logging
- `flag_and_remove_artifacts()` - ICLabel-based removal
- `compute_qc_metrics()` - Comprehensive quality metrics
- `compute_eventwise_retention()` - Per-event data quality
- `write_qc_report()` - Human-readable QC summary

## Time Estimates

- P01.set regeneration: 2-5 minutes
- Subset creation: 30 seconds
- Test on subset: 5-10 minutes
- Full pipeline on 1 hour: 30-60 minutes
- Batch processing (P02-P40): ~20-40 hours total

## Notes for Tomorrow

1. **Start fresh terminal session** - current sessions may have stale imports
2. **Monitor memory usage** - MATLAB AMICA is memory-intensive
3. **Check QC metrics** - flag any participants with >20% data rejection
4. **Consider parallelization** - batch script can run multiple participants if memory allows

## Branch Status

**Branch:** `eeg-refactor`  
**Status:** Ready for testing once P01.set regenerates  
**Commits ahead:** 17 (not pushed)  
**Action needed:** Push after successful validation
