"""
FINAL VERIFICATION: Check MNE interpretation of MATLAB-stored values
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')

import mne
import numpy as np

# Load cleaned .set with MNE
cleaned_path = "c:/vr_tsst_2025/output/cleaned_eeg/P01_cleaned.set"
raw_mne = mne.io.read_raw_eeglab(cleaned_path, preload=True, verbose=False)

# Get 10-second segment at 125 Hz
seg_start, seg_end = 1250, 2500
data_mne = raw_mne.get_data()[:, seg_start:seg_end]
median_std_mne_V = np.median(data_mne.std(axis=1))
median_std_mne_uV = median_std_mne_V * 1e6

print("MNE Loading of Cleaned .set:")
print(f"  Median std (V):  {median_std_mne_V:.10e}")
print(f"  Median std (µV): {median_std_mne_uV:.10e}")

# From MATLAB direct read
matlab_std = 5.1626e-03

# Compute MNE/MATLAB ratio
ratio = median_std_mne_V / matlab_std

print(f"\nMNE / MATLAB ratio: {ratio:.10e}")
print(f"Expected ratio: 1e-6 (MNE converts µV → V)")

if 0.9e-6 < ratio < 1.1e-6:
    print("VERDICT: MNE correctly interprets MATLAB values as µV and converts to V")
elif 0.9e-3 < ratio < 1.1e-3:
    print("VERDICT: MNE is dividing by 1e3 instead of 1e6 (interprets as mV?)")
elif 0.9 < ratio < 1.1:
    print("VERDICT: MNE is NOT converting units (keeping same scale)")
else:
    print(f"VERDICT: Unexpected conversion factor ({ratio:.2e})")
