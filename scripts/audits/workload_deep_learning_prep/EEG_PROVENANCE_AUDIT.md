# EEG PIPELINE PROVENANCE AUDIT
**Generated**: 2026-01-16  
**Audited Pipeline**: XDF → SET → MATLAB Cleaning → MNE Epochs  
**Issue**: Data scale appears extremely small (std ~ 7.7e-09 in MNE units)

---

## 1. SCRIPTS THAT WRITE CLEANED .SET FILES

### Primary Cleaning Pipeline
**File**: [pipelines/02_matlab_cleaning/clean_eeg.m](pipelines/02_matlab_cleaning/clean_eeg.m)  
**Function**: `clean_eeg()`  
**Line 329**: `pop_saveset(EEG, 'filename', sprintf('P%02d_cleaned.set', participant_num), 'filepath', output_folder)`  
**Output**: `output/cleaned_eeg/P{XX}_cleaned.set`

### Wrapper Scripts
- [pipelines/02_matlab_cleaning/run_clean_eeg_pipeline_parallel.m](pipelines/02_matlab_cleaning/run_clean_eeg_pipeline_parallel.m) - Parallel execution
- [pipelines/02_matlab_cleaning/run_clean_eeg_pipeline.m](pipelines/02_matlab_cleaning/run_clean_eeg_pipeline.m) - Sequential execution

---

## 2. COMPLETE TRANSFORMATION PIPELINE (IN ORDER)

### STAGE 0: Raw XDF Input
**File**: [pipelines/01_xdf_to_set/xdf_to_set.py](pipelines/01_xdf_to_set/xdf_to_set.py)  
**Lines 161-163**:
```python
# Raw EEG samples: shape (samples, channels)
data = np.asarray(stream["time_series"], dtype=float)

# ⚠️ CRITICAL UNIT CONVERSION
# Scale data: ANT eego streams appear to be in millivolts; EEGLAB expects microvolts
data = data * 1000.0
```

**Scaling**: **mV → µV** (multiply by 1000)  
**Input**: ANT Neuro eego raw stream in millivolts  
**Output**: Raw .set file in microvolts (EEGLAB convention)

---

### STAGE 1: Load Raw Data
**File**: [pipelines/02_matlab_cleaning/clean_eeg.m](pipelines/02_matlab_cleaning/clean_eeg.m#L66-L87)  
**Lines 66-87**:
```matlab
EEG = pop_loadset('filename', [filename, ext], 'filepath', filepath);

% Fix data types for MATLAB compatibility
EEG.xmin = double(EEG.xmin);
EEG.xmax = double(EEG.xmax);
EEG.srate = double(EEG.srate);
% ... (no amplitude scaling)
```

**Transformations**: Type casting only, **NO scaling**  
**Units**: Microvolts (µV) - preserved from raw .set  
**Logged Stats** (Line 87): `min, max, mean, std` in µV

---

### STAGE 2: Assign Channel Locations
**Lines 92-106**:
```matlab
chanlocs_file = fullfile(projectRoot, 'config', 'chanlocs', 'NA-271.elc');
EEG = pop_chanedit(EEG, 'lookup', chanlocs_file);
EEG.etc.orig_chanlocs = EEG.chanlocs;
```

**Transformations**: Channel location assignment only  
**Units**: µV (unchanged)

---

### STAGE 3: Basic Cleaning (Filtering)
**Lines 111-145**:
```matlab
% Resample to 125 Hz if needed
if EEG.srate ~= 125
    EEG = pop_resample(EEG, 125);
end

% Band-pass filter 1-49 Hz
EEG = pop_eegfiltnew(EEG, 'locutoff', 1, 'hicutoff', 49);

% Remove 50 Hz line noise with notch filter
EEG = pop_eegfiltnew(EEG, 'locutoff', 49, 'hicutoff', 51, 'revfilt', 1);

% Apply 25 Hz notch for participants 1-7
if ismember(participant_num, 1:7)
    EEG = pop_eegfiltnew(EEG, 'locutoff', 24.5, 'hicutoff', 25.5, 'revfilt', 1);
end
```

**Transformations**:
1. **Resampling**: 500 Hz → 125 Hz
2. **Bandpass**: 1-49 Hz (FIR filter, zero-phase)
3. **Notch 50 Hz**: Line noise removal
4. **Notch 25 Hz**: Participants 1-7 only (artifact-specific)

**Units**: µV (filtering does NOT change amplitude units)  
**Logged Stats** (Line 143): `min, max, mean, std` in µV

---

### STAGE 4: Bad Channel Detection (No Data Removal)
**Lines 150-177**:
```matlab
[EEG, ~] = clean_artifacts(EEG, ...
    'FlatlineCriterion', 5, ...
    'ChannelCriterion', 0.60, ...
    'LineNoiseCriterion', 4, ...
    'BurstCriterion', 'off', ...      % No burst detection/repair
    'WindowCriterion', 'off');         % No window rejection

% Ensure all samples are marked as kept
if ~isfield(EEG.etc, 'clean_sample_mask') || isempty(EEG.etc.clean_sample_mask)
    EEG.etc.clean_sample_mask = true(1, EEG.pnts);
end
```

**Transformations**: Flagging only - **NO data removal**, **NO amplitude scaling**  
**Purpose**: Identify bad channels for later interpolation  
**Units**: µV (unchanged)  
**Logged Stats** (Line 173): `min, max, mean, std` in µV

---

### STAGE 5: Run AMICA (ICA)
**Lines 182-218** (+ helper function Lines 379-423):
```matlab
[EEG, LL_trace] = run_amica_pipeline(EEG, participant_num, logfile, max_threads_override);

% Inside run_amica_pipeline:
[weights, sphere, mods] = runamica15(EEG.data, ...
    'num_models', 1, ...
    'max_iter', 200, ...
    'max_threads', max_threads);

EEG.icaweights = weights;
EEG.icasphere  = sphere;
EEG = eeg_checkset(EEG);
```

**Transformations**: 
- ICA decomposition (Adaptive Mixture ICA)
- Weights/sphere computed from data
- **NO explicit amplitude scaling**

**Units**: µV (ICA preserves units - unmixing matrix is dimensionless)  
**Note**: AMICA computes `icaweights` and `icasphere` - reconstruction via `weights * sphere * data` preserves original scale  
**Logged Stats** (Line 226): `min, max, mean, std` in µV

---

### STAGE 6: ICLabel and Artifact Removal
**Lines 231-261** (+ helper function Lines 426-471):
```matlab
EEG = iclabel(EEG);
EEG = flag_and_remove_artifacts(EEG, logfile);

% Inside flag_and_remove_artifacts:
probs = EEG.etc.ic_classification.ICLabel.classifications;
toRemove = find(eyeProb >= 0.8 | muscleProb >= 0.8 | channelNoiseProb >= 0.8);
EEG = pop_subcomp(EEG, toRemove, 0);  % Remove components
```

**Transformations**:
- ICLabel classification (Brain, Muscle, Eye, Heart, Line Noise, Channel Noise, Other)
- Remove components with Eye/Muscle/Channel Noise ≥ 80% probability
- Reconstruct data WITHOUT removed components
- **NO explicit amplitude scaling**

**Units**: µV (component removal preserves units)  
**Logged Stats** (Line 257 + inside helper Line 464): `min, max, mean, std` in µV

---

### STAGE 7: Interpolate and Re-reference
**Lines 266-280**:
```matlab
EEG = pop_interp(EEG, EEG.etc.orig_chanlocs, 'spherical');
EEG = pop_reref(EEG, []);  % Average reference
```

**Transformations**:
1. **Interpolation**: Spherical spline interpolation of bad channels
2. **Re-reference**: Average reference (subtract grand mean)

**Units**: µV (interpolation and re-referencing preserve units)  
**Logged Stats** (Line 278): `min, max, mean, std` in µV  
**Note**: Average referencing makes `mean(EEG.data(:))` ≈ 0

---

### STAGE 8: Compute QC Metrics
**Lines 285-300** (helper function Lines 474-531):
```matlab
qc = compute_qc_metrics(EEG, badLabels, logfile);
```

**Transformations**: None - metadata only  
**Units**: µV (unchanged)

---

### STAGE 9: Save Cleaned Data
**Lines 305-340**:
```matlab
% Save as .mat (data matrix only)
cleaned_mat_path = fullfile(output_folder, sprintf('P%02d_cleaned.mat', participant_num));
cleaned_EEG = double(EEG.data);
stats = [min(cleaned_EEG(:)), max(cleaned_EEG(:)), mean(cleaned_EEG(:)), std(cleaned_EEG(:))];
log_message(logfile, sprintf('Stats at save: min=%.6f max=%.6f mean=%.6f std=%.6f', stats));
save(cleaned_mat_path, 'cleaned_EEG', '-v7.3');

% Save as .set (full EEGLAB structure)
cleaned_set_path = fullfile(output_folder, sprintf('P%02d_cleaned.set', participant_num));
EEG.setname = sprintf('P%02d_cleaned', participant_num);
pop_saveset(EEG, 'filename', sprintf('P%02d_cleaned.set', participant_num), ...
    'filepath', output_folder);
```

**Transformations**: None - saving only  
**Units**: µV (EEGLAB convention preserved)  
**Output Files**:
- `P{XX}_cleaned.mat` - Data matrix only (128 channels × N samples)
- `P{XX}_cleaned.set` - Full EEGLAB structure (includes events, channel locations, metadata)

---

## 3. UNIT CONVERSION LOCATIONS

### ✅ ONLY ONE EXPLICIT SCALING IN ENTIRE PIPELINE

**Location**: [pipelines/01_xdf_to_set/xdf_to_set.py](pipelines/01_xdf_to_set/xdf_to_set.py#L163)  
**Line 163**: `data = data * 1000.0`  
**Conversion**: mV → µV (multiply by 1000)  
**Reason**: ANT Neuro eego streams are in millivolts, EEGLAB expects microvolts

### ❌ NO OTHER SCALING OPERATIONS

Verified by searching:
- `1e6` - Not found in cleaning pipeline
- `1e-6` - Not found in cleaning pipeline
- `scale.*volt` - Not found
- `*1000` or `/1000` - Only in xdf_to_set.py (Line 163, documented above)

**Conclusion**: All MATLAB cleaning operations preserve the µV scale from the raw .set files.

---

## 4. RESULTING DATA STRUCTURE

### File Format
**Type**: EEGLAB `.set` file (MATLAB structure)  
**Format**: Continuous raw EEG (NOT epoched)

### Data Properties
```
Channels: 128 (ANT Neuro equidistant montage)
Sampling rate: 125 Hz (resampled from 500 Hz)
Duration: ~3700 seconds (~62 minutes per participant)
Reference: Average reference (grand mean subtracted)
Filtering: 1-49 Hz bandpass + 50 Hz notch (+ 25 Hz notch for P01-P07)
```

### Units
**EEGLAB/MATLAB**: Microvolts (µV)  
**MNE Python**: Volts (V) - **MNE automatically converts µV → V when loading .set files**

### Annotations/Events
**Total events per file**: ~196 events  
**Event types**: 
- `101-104`: Task condition markers (HighStress_HighCog, etc.)
- `20, 21, 23, 24`: Forest baseline markers
- `10-13`: Calibration/other markers
- `200, 201`: Additional metadata markers

**Purpose**: Mark temporal onsets of experimental conditions

### Data Structure in Python/MNE
```python
raw = mne.io.read_raw_eeglab('P01_cleaned.set')
# raw.get_data() returns shape (128 channels, N_samples)
# Units: VOLTS (not µV) - this is MNE convention
# To convert to µV: data_uv = raw.get_data() * 1e6
```

---

## 5. DIAGNOSTIC CODE SNIPPETS

### Snippet 1: Print Peak-to-Peak in µV (10 channels, 10 seconds)
```python
import mne
import numpy as np

raw = mne.io.read_raw_eeglab('output/cleaned_eeg/P01_cleaned.set', preload=True, verbose=False)

# MNE loads in VOLTS, convert to µV
data_10ch_10s = raw.get_data(picks=range(10), start=0, stop=int(10*raw.info['sfreq'])) * 1e6

print("Peak-to-Peak Amplitudes (µV) for first 10 channels over 10 seconds:")
for i, ch_name in enumerate(raw.ch_names[:10]):
    p2p = data_10ch_10s[i].max() - data_10ch_10s[i].min()
    print(f"{ch_name}: {p2p:.2f} µV")
```

### Snippet 2: Print Per-Channel Std in µV
```python
import mne

raw = mne.io.read_raw_eeglab('output/cleaned_eeg/P01_cleaned.set', preload=True, verbose=False)

# Convert to µV
data_uv = raw.get_data() * 1e6

print("Standard Deviation (µV) per channel:")
for i, ch_name in enumerate(raw.ch_names):
    std_uv = data_uv[i].std()
    print(f"{ch_name}: {std_uv:.2f} µV")
```

### Snippet 3: Print Data Structure and Event Counts
```python
import mne

raw = mne.io.read_raw_eeglab('output/cleaned_eeg/P01_cleaned.set', preload=True, verbose=False)

print(f"Data Type: Continuous Raw EEG (not epoched)")
print(f"Channels: {len(raw.ch_names)}")
print(f"Sampling Rate: {raw.info['sfreq']} Hz")
print(f"Duration: {raw.times[-1]:.1f} seconds")
print(f"Total Samples: {raw.n_times}")

# Extract events
events, event_id = mne.events_from_annotations(raw)
print(f"\nEvent Types: {list(event_id.keys())}")
print(f"Total Events: {len(events)}")
print(f"\nEvent Counts:")
for event_name, event_code in event_id.items():
    count = sum(events[:, 2] == event_code)
    print(f"  {event_name}: {count} events")
```

---

## 6. ROOT CAUSE ANALYSIS

### Why is std ~ 7.7e-09 in "MNE units"?

**Answer**: Because MNE uses **VOLTS** as its native unit, not microvolts.

### Conversion Chain:
```
ANT Neuro eego stream → millivolts (mV)
         ↓ [xdf_to_set.py: × 1000]
Raw .set file → microvolts (µV)
         ↓ [MATLAB cleaning: NO scaling]
Cleaned .set file → microvolts (µV)
         ↓ [MNE load: automatic µV → V conversion]
MNE raw.get_data() → VOLTS (V)
```

### Verification:
```python
# From diagnostic output:
MNE native (V):  std = 4.866251e-09 V
Convert to µV:   std = 0.0049 µV  (4.866251e-09 × 1e6)
```

**Expected EEG std**: ~7-8 µV for cleaned, filtered, average-referenced data  
**Your data std**: ~0.005 µV = **1000× too small**

---

## 7. CRITICAL FINDING: DATA IS TOO SMALL BY 1000×

### Expected vs Actual (P01 sample):
```
Expected (typical cleaned EEG):  std ~ 7-8 µV
Your data (in µV):               std ~ 0.005 µV
Ratio:                           1000× smaller
```

### Hypothesis: Double Scaling Error

**Possible cause**: The raw XDF data from ANT Neuro was **already in microvolts**, not millivolts, but `xdf_to_set.py` Line 163 assumed millivolts and multiplied by 1000.

**Result**: Data is in "milli-microvolts" (nanovolts) instead of microvolts.

### To Verify:
1. Check the ANT Neuro eego documentation for stream units
2. Inspect a raw .xdf file before conversion
3. Compare amplitude statistics from raw XDF vs raw .set

### To Fix (if hypothesis is correct):
**Option 1**: Remove the `* 1000.0` scaling in [xdf_to_set.py Line 163](pipelines/01_xdf_to_set/xdf_to_set.py#L163)  
**Option 2**: Re-run entire pipeline with corrected scaling  
**Option 3**: Apply 1000× correction when loading in Python:
```python
raw = mne.io.read_raw_eeglab('P01_cleaned.set', preload=True)
raw._data *= 1000  # Multiply by 1000 to correct units
```

---

## 8. RECOMMENDATIONS

### Immediate Actions:
1. ✅ **Verify raw XDF units** - Check ANT Neuro documentation
2. ✅ **Inspect raw .xdf amplitude** - Load one XDF file with pyxdf and check `stream['time_series']` statistics
3. ✅ **Compare raw vs cleaned** - Load raw .set (before MATLAB cleaning) and check if amplitude is already 1000× too small

### If Units Are Confirmed Wrong:
1. **Re-generate raw .set files** - Fix xdf_to_set.py scaling and re-run Stage 01
2. **Re-run MATLAB cleaning** - Process corrected raw .set files through clean_eeg.m
3. **Re-export epochs** - Run export_mne_epochs.py again with corrected data

### If Units Are Actually Correct:
1. Document why ANT Neuro streams are in nanovolts (unusual but possible)
2. Update all downstream code to expect this scale
3. Add explicit unit conversion when comparing to literature values

---

## PROVENANCE SUMMARY

| Stage | Script | Operation | Units In | Units Out | Scaling |
|-------|--------|-----------|----------|-----------|---------|
| 0 | xdf_to_set.py | Load XDF | mV? | µV | ×1000 ⚠️ |
| 1 | clean_eeg.m | Load .set | µV | µV | None |
| 2 | clean_eeg.m | Assign chanlocs | µV | µV | None |
| 3 | clean_eeg.m | Filter (1-49 Hz, notch) | µV | µV | None |
| 4 | clean_eeg.m | Detect bad channels | µV | µV | None |
| 5 | clean_eeg.m | AMICA (ICA) | µV | µV | None |
| 6 | clean_eeg.m | Remove ICs | µV | µV | None |
| 7 | clean_eeg.m | Interpolate + reref | µV | µV | None |
| 8 | clean_eeg.m | Save .set | µV | µV | None |
| 9 | MNE (Python) | Load .set | µV | **V** | ×1e-6 (automatic) |

**Total Pipeline Scaling**: ×1000 (Stage 0) → ×1e-6 (Stage 9) = **×0.001 net**  
**Expected**: ×1e-6 only (µV → V in MNE)  
**Discrepancy**: 1000× too small suggests Stage 0 scaling is incorrect

---

**Audit Complete**  
**Next Step**: Verify ANT Neuro eego stream units in raw .xdf files
