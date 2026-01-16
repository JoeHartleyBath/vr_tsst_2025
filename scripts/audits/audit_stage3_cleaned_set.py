"""
UNIT AUDIT - STAGE 3: Measure CLEANED .set file values.

Purpose:
- Load the CLEANED .set file (after MATLAB processing)
- Measure same statistics
- Compare to Stages 1 & 2

This shows what MNE receives after MATLAB cleaning.
"""

import mne
import numpy as np

# Use P01's cleaned .set file (after MATLAB cleaning)
set_path = "c:/vr_tsst_2025/output/cleaned_eeg/P01_cleaned.set"

print("="*80)
print("STAGE 3: CLEANED .SET FILE (AFTER MATLAB PROCESSING)")
print("="*80)

# Load with MNE
print(f"\nLoading: {set_path}")
raw = mne.io.read_raw_eeglab(set_path, preload=True, verbose=False)

print(f"Channels: {len(raw.ch_names)}")
print(f"Sampling rate: {raw.info['sfreq']} Hz")
print(f"Duration: {raw.times[-1]:.1f} seconds")

# MNE loads in VOLTS (V) - this is MNE's convention
# Get channel 1 data (first EEG channel)
ch1_name = raw.ch_names[0]
ch1_data_V = raw.get_data(picks=[ch1_name])[0]

print(f"\n--- CHANNEL 1 ({ch1_name}) STATISTICS ---")

print("\nIn MNE's native units (VOLTS):")
print(f"Mean:   {ch1_data_V.mean():.10e} V")
print(f"Std:    {ch1_data_V.std():.10e} V")
print(f"Min:    {ch1_data_V.min():.10e} V")
print(f"Max:    {ch1_data_V.max():.10e} V")
print(f"P2P:    {(ch1_data_V.max() - ch1_data_V.min()):.10e} V")

# Convert to µV for interpretability
ch1_data_uV = ch1_data_V * 1e6

print("\nConverted to microvolts (µV):")
print(f"Mean:   {ch1_data_uV.mean():.6f} µV")
print(f"Std:    {ch1_data_uV.std():.6f} µV")
print(f"Min:    {ch1_data_uV.min():.6f} µV")
print(f"Max:    {ch1_data_uV.max():.6f} µV")
print(f"P2P:    {ch1_data_uV.max() - ch1_data_uV.min():.6f} µV")

print("\nExpected values for CLEANED EEG:")
print("  Mean: ~0 µV (average referenced)")
print("  Std:  ~5-10 µV (cleaned, filtered)")
print("  P2P:  ~30-60 µV")

# Sample values
print(f"\nFirst 10 samples (in µV):")
print(ch1_data_uV[:10])

print("\n" + "="*80)
print("KEY COMPARISON:")
print("="*80)
print("\nStage 1 (Raw XDF in mV):      Std = 1.54 µV (if interpreted as mV)")
print("Stage 3 (Cleaned .set in MNE): Std = {:.6f} µV".format(ch1_data_uV.std()))
print("\nIf Stage 3 std is ~0.001-0.002 µV → 1000× TOO SMALL")
print("If Stage 3 std is ~5-10 µV → CORRECT")
print("="*80)
