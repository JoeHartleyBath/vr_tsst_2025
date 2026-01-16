"""
Verify MNE interpretation of RAW .set files
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')

import mne
import numpy as np
import scipy.io

# Check if raw .set exists and load with scipy first
raw_set_path = "c:/vr_tsst_2025/output/sets/P01.set"

try:
    # Load with scipy (MATLAB native)
    mat = scipy.io.loadmat(raw_set_path)
    if 'EEG' in mat and hasattr(mat['EEG'][0,0], 'data'):
        data_matlab = mat['EEG'][0,0].data
    else:
        data_matlab = mat['data']
    
    seg_start, seg_end = 5000, 10000
    segment_matlab = data_matlab[:, seg_start:seg_end]
    median_std_matlab = np.median(segment_matlab.std(axis=1))
    
    print("MATLAB Loading of Raw .set:")
    print(f"  Median std: {median_std_matlab:.10e}")
    
except Exception as e:
    print(f"MATLAB loading failed: {e}")
    median_std_matlab = None

# Try MNE
try:
    raw_mne = mne.io.read_raw_eeglab(raw_set_path, preload=True, verbose=False)
    data_mne = raw_mne.get_data()[:, seg_start:seg_end]
    median_std_mne_V = np.median(data_mne.std(axis=1))
    median_std_mne_uV = median_std_mne_V * 1e6
    
    print("\nMNE Loading of Raw .set:")
    print(f"  Median std (V):  {median_std_mne_V:.10e}")
    print(f"  Median std (µV): {median_std_mne_uV:.10e}")
    
    if median_std_matlab is not None:
        ratio = median_std_mne_V / median_std_matlab
        print(f"\nMNE / MATLAB ratio: {ratio:.10e}")
        print(f"Expected: 1e-6 if MATLAB stores µV")
        
        if 0.9e-6 < ratio < 1.1e-6:
            print("VERDICT: MNE correctly interprets raw .set (µV → V)")
        else:
            print(f"VERDICT: MNE misinterprets by factor of {ratio:.2e}")
            
except Exception as e:
    print(f"\nMNE loading failed: {e}")
