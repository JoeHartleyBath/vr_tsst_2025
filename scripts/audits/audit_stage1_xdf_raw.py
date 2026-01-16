"""
UNIT AUDIT - STAGE 1: Measure RAW XDF stream values BEFORE any scaling.

Purpose:
- Load raw XDF file using pyxdf
- Extract FIRST EEG stream WITHOUT applying any scaling
- Measure statistics on channel 1 (first EEG channel)
- Report values in native units AND converted to µV for comparison

This establishes the GROUND TRUTH for what units the hardware actually outputs.
"""

import numpy as np
import pyxdf
from pathlib import Path

# Use P01's raw XDF file
xdf_path = Path("c:/vr_tsst_2025/data/RAW/eeg/P01.xdf")

print("="*80)
print("STAGE 1: RAW XDF STREAM (NO SCALING)")
print("="*80)

# Load XDF
print(f"\nLoading: {xdf_path}")
streams, header = pyxdf.load_xdf(str(xdf_path))

# Find EEG streams
eeg_streams = [s for s in streams if s["info"]["type"][0] == "EEG"]
print(f"Found {len(eeg_streams)} EEG stream(s)")

# Take first EEG stream
stream = eeg_streams[0]
stream_name = stream["info"]["name"][0]
print(f"Stream name: {stream_name}")

# Get raw data (NO SCALING)
data_raw = np.asarray(stream["time_series"], dtype=float)
print(f"Data shape: {data_raw.shape} (samples × channels)")

# Check for channel units in metadata
print("\n--- Checking XDF metadata for channel units ---")
try:
    channels = stream["info"]["desc"][0]["channels"][0]["channel"]
    if isinstance(channels, list):
        # Multiple channels
        ch1_info = channels[0]
    else:
        # Single channel info
        ch1_info = channels
    
    if "unit" in ch1_info:
        print(f"Channel 1 unit from metadata: {ch1_info['unit'][0]}")
    else:
        print("No 'unit' field found in channel metadata")
except Exception as e:
    print(f"Could not parse channel metadata: {e}")

# Measure statistics on channel 1 (index 0)
ch1_data = data_raw[:, 0]

print("\n--- CHANNEL 1 STATISTICS (NATIVE XDF UNITS) ---")
print(f"Mean:   {ch1_data.mean():.10e}")
print(f"Std:    {ch1_data.std():.10e}")
print(f"Min:    {ch1_data.min():.10e}")
print(f"Max:    {ch1_data.max():.10e}")
print(f"Range:  {ch1_data.max() - ch1_data.min():.10e}")

# Calculate peak-to-peak
p2p = ch1_data.max() - ch1_data.min()
print(f"Peak-to-peak: {p2p:.10e}")

# Sample values (first 10 samples)
print(f"\nFirst 10 samples (native units):")
print(ch1_data[:10])

print("\n" + "="*80)
print("INTERPRETATION TESTS")
print("="*80)

# Test different unit assumptions
print("\n--- If native units are VOLTS (V) ---")
print(f"  Mean:   {ch1_data.mean() * 1e6:.4f} µV")
print(f"  Std:    {ch1_data.std() * 1e6:.4f} µV")
print(f"  P2P:    {p2p * 1e6:.4f} µV")
print(f"  Expected: 5-10 µV std, 50-100 µV p2p")

print("\n--- If native units are MILLIVOLTS (mV) ---")
print(f"  Mean:   {ch1_data.mean() * 1e3:.4f} µV")
print(f"  Std:    {ch1_data.std() * 1e3:.4f} µV")
print(f"  P2P:    {p2p * 1e3:.4f} µV")
print(f"  Expected: 5-10 µV std, 50-100 µV p2p")

print("\n--- If native units are MICROVOLTS (µV) ---")
print(f"  Mean:   {ch1_data.mean():.4f} µV")
print(f"  Std:    {ch1_data.std():.4f} µV")
print(f"  P2P:    {p2p:.4f} µV")
print(f"  Expected: 5-10 µV std, 50-100 µV p2p")

print("\n" + "="*80)
print("CONCLUSION:")
print("Compare the three interpretations above.")
print("Whichever gives ~5-10 µV std and ~50-100 µV p2p is the TRUE native unit.")
print("="*80)
