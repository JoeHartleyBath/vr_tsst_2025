"""
SIMPLIFIED CROSS-STAGE AUDIT - FINAL VERDICT

Compares the SAME 10-second segment across:
1. XDF raw stream (native)
2. Raw .set (MATLAB)  
3. Cleaned .set (MATLAB)

Computes exact ratios to determine where scaling occurred.
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import pyxdf
import scipy.io
from pathlib import Path

participant = "P01"

xdf_path = Path(f"c:/vr_tsst_2025/data/RAW/eeg/{participant}.xdf")
raw_set_path = Path(f"c:/vr_tsst_2025/output/sets/{participant}.set")
cleaned_set_path = Path(f"c:/vr_tsst_2025/output/cleaned_eeg/{participant}_cleaned.set")

print("="*80)
print(f"CROSS-STAGE AMPLITUDE AUDIT: {participant}")
print("="*80)

# ============================================================================
# STAGE 1: XDF
# ============================================================================
print("\n[1] Loading XDF raw stream...")
streams, _ = pyxdf.load_xdf(str(xdf_path))
eeg_stream = [s for s in streams if s["info"]["type"][0] == "EEG"][0]
data_xdf = np.asarray(eeg_stream["time_series"], dtype=float)[:, :64]  # First 64 EEG channels

seg_start, seg_end = 5000, 10000
segment_xdf = data_xdf[seg_start:seg_end, :]
median_std_xdf = np.median(segment_xdf.std(axis=0))

print(f"    Segment: samples {seg_start}-{seg_end} (10 sec @ 500 Hz)")
print(f"    Median std across channels: {median_std_xdf:.10e}")

# ============================================================================
# STAGE 2: Raw .set
# ============================================================================
print("\n[2] Loading raw .set (XDF -> SET conversion)...")
mat = scipy.io.loadmat(raw_set_path, squeeze_me=False, struct_as_record=False)

if 'EEG' in mat:
    EEG = mat['EEG'][0, 0]
    data_raw = EEG.data
else:
    data_raw = mat['data']

if data_raw.size <= 1:
    # Load from .fdt
    fdt_path = raw_set_path.with_suffix('.fdt')
    n_ch = int(mat.get('nbchan', EEG.nbchan if 'EEG' in mat else 128)[0][0])
    n_samp = int(mat.get('pnts', EEG.pnts if 'EEG' in mat else 0)[0][0])
    data_flat = np.fromfile(fdt_path, dtype=np.float32)
    data_raw = data_flat.reshape(n_ch, n_samp)

segment_raw = data_raw[:, seg_start:seg_end]
median_std_raw = np.median(segment_raw.std(axis=1))

print(f"    Segment: samples {seg_start}-{seg_end} (10 sec @ 500 Hz)")
print(f"    Median std across channels: {median_std_raw:.10e}")

# ============================================================================
# STAGE 3: Cleaned .set
# ============================================================================
print("\n[3] Loading cleaned .set (after MATLAB pipeline)...")
mat_clean = scipy.io.loadmat(cleaned_set_path, squeeze_me=False, struct_as_record=False)

fdt_path_clean = cleaned_set_path.with_suffix('.fdt')
n_ch_clean = int(mat_clean['nbchan'][0][0])
n_samp_clean = int(mat_clean['pnts'][0][0])
data_flat_clean = np.fromfile(fdt_path_clean, dtype=np.float32)
data_clean = data_flat_clean.reshape(n_ch_clean, n_samp_clean)

# Account for resampling: 500 Hz -> 125 Hz
seg_start_clean = int(seg_start * 125 / 500)
seg_end_clean = int(seg_end * 125 / 500)

segment_clean = data_clean[:, seg_start_clean:seg_end_clean]
median_std_clean = np.median(segment_clean.std(axis=1))

print(f"    Segment: samples {seg_start_clean}-{seg_end_clean} (10 sec @ 125 Hz)")
print(f"    Median std across channels: {median_std_clean:.10e}")

# ============================================================================
# RATIO ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("RATIO ANALYSIS")
print("="*80)

R1 = median_std_raw / median_std_xdf
R2 = median_std_clean / median_std_raw

print(f"\nR1: Raw .set / XDF = {R1:.2f}")
print(f"    Expected if x1000 applied: ~1000")
print(f"    Expected if NO scaling:    ~1")

if 900 < R1 < 1100:
    verdict_r1 = "YES - x1000 scaling WAS APPLIED"
else:
    verdict_r1 = "NO - x1000 scaling WAS NOT APPLIED"

print(f"    VERDICT: {verdict_r1}")

print(f"\nR2: Cleaned .set / Raw .set = {R2:.4f}")
print(f"    Expected: 0.3-0.7 (cleaning reduces noise)")

if 0.2 < R2 < 0.8:
    verdict_r2 = "YES - MATLAB preserved scale appropriately"
else:
    verdict_r2 = f"UNEXPECTED - Scale changed by {R2:.2f}x"

print(f"    VERDICT: {verdict_r2}")

# ============================================================================
# SUMMARY TABLE
# ============================================================================
print("\n" + "="*80)
print("SUMMARY TABLE")
print("="*80)

print(f"\n{'Stage':<25} {'Median Std':<20} {'Notes'}")
print("-"*70)
print(f"{'XDF raw':<25} {median_std_xdf:.6e} {'Native (metadata says V)'}")
print(f"{'Raw .set (MATLAB)':<25} {median_std_raw:.6e} {'After xdf_to_set.py'}")
print(f"{'Cleaned .set (MATLAB)':<25} {median_std_clean:.6e} {'After MATLAB pipeline'}")

# ============================================================================
# FINAL VERDICT
# ============================================================================
print("\n" + "="*80)
print("FINAL VERDICT")
print("="*80)

if 900 < R1 < 1100:
    print("\nQ: Was x1000 scaling applied in XDF -> RAW .SET?")
    print(f"A: YES (ratio = {R1:.2f})")
else:
    print("\nQ: Was x1000 scaling applied in XDF -> RAW .SET?")
    print(f"A: NO (ratio = {R1:.2f}, expected ~1000)")

if 0.2 < R2 < 0.8:
    print("\nQ: Did MATLAB cleaning preserve amplitude scale?")
    print(f"A: YES (ratio = {R2:.4f}, expected 0.3-0.7)")
else:
    print("\nQ: Did MATLAB cleaning preserve amplitude scale?")
    print(f"A: UNEXPECTED CHANGE (ratio = {R2:.4f})")

# Compute what the cleaned value SHOULD be if pipeline was correct
expected_clean_if_correct = median_std_xdf * 1000 * 0.5  # x1000 scaling, ~50% reduction from cleaning
actual_clean = median_std_clean

ratio_actual_to_expected = actual_clean / expected_clean_if_correct

print(f"\nQ: Are the cleaned file amplitudes correct?")
print(f"A: Cleaned std = {median_std_clean:.6e}")
print(f"   Expected std = {expected_clean_if_correct:.6e} (XDF x1000 x0.5 cleaning)")
print(f"   Ratio = {ratio_actual_to_expected:.2f}")

if 0.5 < ratio_actual_to_expected < 2.0:
    print("   VERDICT: Amplitudes are CORRECT")
else:
    print(f"   VERDICT: Amplitudes are {ratio_actual_to_expected:.2f}x away from expected")

# ============================================================================
# REQUIRED ACTION
# ============================================================================
print("\n" + "="*80)
print("REQUIRED ACTION")
print("="*80)

if 900 < R1 < 1100 and 0.5 < ratio_actual_to_expected < 2.0:
    print("\nNO ACTION NEEDED")
    print("- x1000 scaling was correctly applied")
    print("- MATLAB cleaning preserved scale")
    print("- Final amplitudes are in expected range")
else:
    if R1 < 900:
        print("\nPROBLEM: XDF -> SET conversion did NOT apply x1000 scaling")
        print("FIX: Re-run XDF -> SET conversion + MATLAB cleaning")
    elif ratio_actual_to_expected < 0.5:
        print("\nPROBLEM: Final amplitudes are too small despite correct pipeline")
        print("FIX: Investigate additional scaling in analysis code")
    else:
        print("\nPROBLEM: Unexpected scale changes detected")
        print("FIX: Manual investigation required")

print("\n" + "="*80)
