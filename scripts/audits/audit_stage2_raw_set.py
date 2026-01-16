"""
UNIT AUDIT - STAGE 2: Measure RAW .set file values (after xdf_to_set.py conversion).

Purpose:
- Load the RAW .set file created by xdf_to_set.py
- This file has already been scaled by × 1000 in Python
- Measure same statistics as Stage 1
- Compare to Stage 1 to verify scaling effect

This shows what EEGLAB receives before any MATLAB processing.
"""

import mne
import numpy as np

# Use P01's raw .set file (created by xdf_to_set.py, BEFORE MATLAB cleaning)
set_path = "c:/vr_tsst_2025/output/sets/P01.set"

print("="*80)
print("STAGE 2: RAW .SET FILE (AFTER xdf_to_set.py × 1000 SCALING)")
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
print(f"Mean:   {ch1_data_uV.mean():.4f} µV")
print(f"Std:    {ch1_data_uV.std():.4f} µV")
print(f"Min:    {ch1_data_uV.min():.4f} µV")
print(f"Max:    {ch1_data_uV.max():.4f} µV")
print(f"P2P:    {ch1_data_uV.max() - ch1_data_uV.min():.4f} µV")

print("\nExpected values for raw EEG (before cleaning):")
print("  Std:  ~10-30 µV (depends on reference and artifacts)")
print("  P2P:  ~100-300 µV")

# Sample values
print(f"\nFirst 10 samples (in µV):")
print(ch1_data_uV[:10])

print("\n" + "="*80)
print("CRITICAL CHECK:")
print("="*80)
print("\nMNE automatically converts EEGLAB .set files from µV to V.")
print("So if the .set file stores values as X µV,")
print("MNE returns X × 1e-6 V (which converts back to X µV).")
print("\nThe question: Are the values stored in the .set file")
print("actually in µV units as EEGLAB expects?")
print("="*80)
