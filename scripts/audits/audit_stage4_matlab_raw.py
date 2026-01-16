"""
UNIT AUDIT - STAGE 4: Check what MATLAB actually stored in the .set file.

Purpose:
- Load the .set file using scipy.io.loadmat to read the raw MATLAB structure
- Check the actual numeric values stored in EEG.data
- This bypasses MNE's conversion and shows MATLAB's raw storage

This tells us what EEGLAB thinks the units are.
"""

import scipy.io
import numpy as np

# Load P01 cleaned .set file
set_path = "c:/vr_tsst_2025/output/cleaned_eeg/P01_cleaned.set"

print("="*80)
print("STAGE 4: READ .SET FILE RAW MATLAB STRUCTURE (NO MNE)")
print("="*80)

# Load the .mat structure directly
print(f"\nLoading MATLAB structure from: {set_path}")
mat = scipy.io.loadmat(set_path, squeeze_me=False, struct_as_record=False)

# EEGLAB .set files have a flat structure with 'data' field
print(f"Fields in .set file: nbchan={mat['nbchan'][0][0]}, pnts={mat['pnts'][0][0]}, srate={mat['srate'][0][0]}")

# Check if data is stored in the .set or separate .fdt file
data_field = mat['data']
print(f"data field shape: {data_field.shape}, size: {data_field.size}")

if data_field.size > 1:
    data = data_field
    print(f"Data stored inline in .set file")
    print(f"Data shape: {data.shape}")
else:
    # Data is in separate .fdt file (data field contains just filename reference)
    if 'datfile' in mat and mat['datfile'].size > 0:
        datfile = str(mat['datfile'][0])
        print(f"Data stored in separate .fdt file: {datfile}")
    else:
        print(f"Data in separate .fdt file (no datfile field)")
    
    # Load the .fdt file
    fdt_path = set_path.replace('.set', '.fdt')
    print(f"Loading: {fdt_path}")
    try:
        with open(fdt_path, 'rb') as f:
            # EEGLAB stores as float32
            n_channels = int(mat['nbchan'][0][0])
            n_samples = int(mat['pnts'][0][0])
            
            data_flat = np.fromfile(f, dtype=np.float32)
            data = data_flat.reshape(n_channels, n_samples)
            print(f"Loaded .fdt data shape: {data.shape} (channels x samples)")
    except FileNotFoundError:
        print(f"ERROR: .fdt file not found at {fdt_path}")
        exit(1)

# Get channel 1 (first EEG channel)
ch1_data = data[0, :]

print("\n--- CHANNEL 1 STATISTICS (RAW MATLAB VALUES) ---")
print(f"Mean:   {ch1_data.mean():.10e}")
print(f"Std:    {ch1_data.std():.10e}")
print(f"Min:    {ch1_data.min():.10e}")
print(f"Max:    {ch1_data.max():.10e}")
print(f"P2P:    {ch1_data.max() - ch1_data.min():.10e}")

print(f"\nFirst 10 samples:")
print(ch1_data[:10])

print("\n" + "="*80)
print("UNIT INTERPRETATION")
print("="*80)

print("\nIf MATLAB stored in microvolts (EEGLAB convention):")
print(f"  Std: {ch1_data.std():.6f} uV")
print(f"  Expected: 5-10 uV")

print("\nIf MATLAB stored in millivolts:")
print(f"  Std: {ch1_data.std() * 1000:.6f} uV")
print(f"  Expected: 5-10 uV")

print("\nIf MATLAB stored in volts:")
print(f"  Std: {ch1_data.std() * 1e6:.6f} uV")
print(f"  Expected: 5-10 uV")

print("\n" + "="*80)
print("MNE CONVERSION CHECK")
print("="*80)

print("\nMNE reads EEGLAB .set files and assumes they are in microvolts.")
print("MNE then divides by 1e6 to convert to volts (MNE's native unit).")
print(f"\nSo if MATLAB stored: {ch1_data.std():.10e}")
print(f"MNE returns:        {ch1_data.std() / 1e6:.10e} V")
print(f"Which converts to:  {ch1_data.std():.6f} uV")

print("\n" + "="*80)
