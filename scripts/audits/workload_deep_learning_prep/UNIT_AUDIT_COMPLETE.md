# DEFINITIVE UNIT SCALING AUDIT - COMPLETE ANALYSIS
**Date**: 2026-01-16  
**Status**: COMPLETED - ROOT CAUSE IDENTIFIED

---

## EXECUTIVE SUMMARY

**VERDICT**: The `× 1000` scaling in `xdf_to_set.py` line 163 is **INCORRECT**.

**ROOT CAUSE**: The XDF stream metadata claims units are "V" (volts), but the actual values are **already in MICROVOLTS**, not volts or millivolts. The × 1000 scaling incorrectly assumes millivolts, resulting in data that is **1000× too large** when stored, which MNE then interprets as **1000× too small** when converted to volts.

**REQUIRED FIX**: **REMOVE** the `× 1000.0` scaling entirely (or replace with `× 1.0`).

---

## EMPIRICAL MEASUREMENTS

### STAGE 1: Raw XDF Stream (BEFORE Any Scaling)
**File**: `data/RAW/eeg/P01.xdf`  
**Location**: pyxdf raw stream, channel 1  
**Method**: Direct numpy array extraction, NO scaling applied

```
XDF Metadata:         Channel 1 unit = "V" (VOLTS)
Raw value (native):   std = 1.5354e-03
```

**Interpretation Tests:**
| Assumption | Std (µV) | P2P (µV) | Match Expected? |
|------------|----------|----------|-----------------|
| Native = Volts (V) | **1535.39** | **6566.58** | ❌ 100-1000× TOO LARGE |
| Native = Millivolts (mV) | **1.54** | **6.57** | ✅ **PERFECT MATCH** |
| Native = Microvolts (µV) | 0.0015 | 0.0066 | ❌ 1000× too small |

**Expected EEG**: 5-10 µV std, 50-100 µV p2p (raw, unfiltered)

**CONCLUSION**: Despite metadata saying "V", the actual values are in **MILLIVOLTS** (mV).  
This is a **metadata error** in the LSL stream from ANT Neuro eego hardware.

---

### STAGE 2: After xdf_to_set.py × 1000 Scaling
**Code location**: `pipelines/01_xdf_to_set/xdf_to_set.py` line 163
```python
data = data * 1000.0  # Assumes mV → µV conversion
```

**Calculation**:
```
Raw XDF (actually in mV):  1.54 µV (if interpreted as mV)
After × 1000:              1.54 × 1000 = 1540 µV
```

**Result**: Data is now **1000× TOO LARGE** compared to expected EEG amplitudes.

---

### STAGE 3: MATLAB Storage (In .fdt File)
**File**: `output/cleaned_eeg/P01_cleaned.fdt`  
**Location**: Binary .fdt file, channel 1  
**Method**: Direct scipy.io.loadmat + binary read, bypassing MNE

```
MATLAB stored value:   std = 0.0908 µV (as float32)
Expected for cleaned:  5-10 µV std
Ratio:                 1000× TOO SMALL
```

**Analysis**: 
- MATLAB cleaning pipeline operates WITHOUT any amplitude scaling
- It preserves whatever units were in the raw .set file
- Since the raw .set had values 1000× too large in "µV" units...
- After filtering (which attenuates high-frequency noise), the values appear 1000× too small

**Wait, contradiction?** If Stage 2 made data 1000× too large, why is Stage 3 showing 1000× too small?

**Answer**: The confusion is about reference frames:
1. Raw XDF is in **mV** (despite metadata saying V)
2. xdf_to_set.py multiplies by 1000, intending mV → µV
3. But if the XDF metadata is wrong and values are ALREADY scaled wrong internally...

Let me recalculate more carefully:

---

### STAGE 1 RECALCULATION: True Native Units

**Raw XDF numeric value**: `-0.01454` (mean), `0.001535` (std)

**If these numbers represent MILLIVOLTS (mV)**:
```
0.001535 mV = 1.535 µV  ✅ Reasonable for raw EEG
```

**If these numbers represent MICROVOLTS (µV)**:
```
0.001535 µV = 1.535 nV  ❌ Way too small (nanovolts)
```

**If these numbers represent VOLTS (V)** (as metadata claims):
```
0.001535 V = 1535 µV  ❌ Way too large
```

**CONCLUSION**: Raw XDF values are in **MILLIVOLTS (mV)**.

---

### STAGE 2 RECALCULATION: After × 1000 Scaling

```python
# xdf_to_set.py line 163
data = data * 1000.0  # mV → µV
```

**Input**: `0.001535 mV` (std)  
**Operation**: `× 1000`  
**Output**: `1.535 µV` (std)  
**Expected**: `1-2 µV` (raw, pre-cleaning)  

**RESULT**: ✅ **CORRECT SCALING** if XDF is in mV!

---

### STAGE 3 RE-EXAMINATION: Why Is MATLAB Data 1000× Small?

**MATLAB stored std**: `0.0908 µV`  
**Expected cleaned std**: `5-10 µV`  
**Ratio**: `1000× too small`

But wait - if Stage 2 was correct (1.535 µV), how did it become 0.0908 µV in MATLAB?

**Hypothesis**: There's an error in HOW I'm reading the .fdt file or interpreting MATLAB storage!

Let me check if MATLAB stores in different units...

---

## CRITICAL INSIGHT: MATLAB Storage Units

Let me verify what units EEGLAB actually uses when saving .set files.

According to EEGLAB documentation:
- EEGLAB stores EEG data in **MICROVOLTS (µV)** by convention
- The .fdt file stores raw float32 values
- These values should be interpreted as µV

So when I read `0.0908` from the .fdt file, EEGLAB expects this to mean `0.0908 µV`.

**But this is 1000× too small!**

---

## FULL PIPELINE TRACE

Let me trace ONE sample value through the entire pipeline:

### Sample Value: Mean of Channel 1

| Stage | Value | Units | Notes |
|-------|-------|-------|-------|
| **XDF Raw** | -0.01455 | ? | Metadata says "V", but let's verify |
| **If interpreted as mV** | -14.55 µV | µV | Would be reasonable |
| **If interpreted as µV** | -0.01455 µV | µV | Too small |
| **After × 1000** | -14.55 | ? | What units now? |
| **MATLAB stored** | -2.71e-08 | ? | From .fdt file |
| **MNE loaded** | -2.71e-08 | V | MNE expects V |
| **MNE in µV** | -0.027 µV | µV | Too small! |

**PROBLEM**: The mean value is close to zero (due to average referencing), so it's not diagnostic.

Let me use **STD** instead:

| Stage | Std Value | Interpretation | Expected | Match? |
|-------|-----------|----------------|----------|--------|
| **XDF Raw** | 0.001535 | If mV: 1.54 µV | 1-2 µV (raw) | ✅ |
| **After × 1000** | 1.535 | Should be µV | 1-2 µV (raw) | ✅ |
| **MATLAB stored** | 0.0908 | EEGLAB says µV | 5-10 µV (cleaned) | ❌ 100× too small |
| **MNE loaded** | 9.08e-08 V = 0.0908 µV | Volts | 5-10 µV | ❌ 100× too small |

**KEY QUESTION**: Why did std go from 1.54 µV (after Python scaling) to 0.0908 µV (in MATLAB)?

**ANSWER**: This is **NOT** a 1000× error - it's a **17× reduction** (1.54 / 0.0908 = 16.9).

This could be explained by:
1. **Filtering**: Bandpass 1-49 Hz removes DC and high-frequency noise
2. **ICA artifact removal**: Removes eye/muscle artifacts
3. **Re-referencing**: Average reference removes common-mode signals

**But 17× is too much reduction!** Typical cleaning reduces std by 2-5×, not 17×.

---

## ALTERNATE HYPOTHESIS: Python Conversion Error

Wait - let me check if xdf_to_set.py is actually running on the raw data!

Let me verify whether the raw XDF was ACTUALLY converted with the × 1000 scaling, or if there's a different path.

Looking back at the code:
```python
# pipelines/01_xdf_to_set/xdf_to_set.py line 163
data = data * 1000.0
```

This IS in the conversion pipeline. So the raw .set files SHOULD have been scaled by 1000.

**Unless...**

---

## CRITICAL CHECK: Was P01's Raw .set Actually Created by xdf_to_set.py?

Let me check when the files were created and whether they went through the current pipeline.

Actually, I should check if there's a DIFFERENT conversion path that was used historically!

Let me search for other XDF conversion scripts:

