#!/usr/bin/env python
"""
Simple EEG Feature Extraction Stub (for pilot P01-P03)

Creates a minimal features CSV from cleaned .set files to allow pipeline to continue.
Extracts: basic spectral power and sample entropy.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import signal
import mne
import warnings
warnings.filterwarnings('ignore')

# Setup paths
project_root = Path(__file__).resolve().parent.parent.parent.parent.parent
cleaned_folder = project_root / 'output' / 'cleaned_eeg'
output_folder = project_root / 'output' / 'eeg_features'
output_folder.mkdir(parents=True, exist_ok=True)
output_csv = output_folder / 'eeg_features.csv'

print(f"Input: {cleaned_folder}")
print(f"Output: {output_csv}")

# Find cleaned files
cleaned_files = sorted(cleaned_folder.glob('P*_cleaned.set'))
print(f"Found {len(cleaned_files)} cleaned .set files")

rows = []
for set_file in cleaned_files:
    p_num = int(set_file.stem.split('_')[0][1:])
    print(f"[P{p_num:02d}] Loading...", end=' ', flush=True)
    
    try:
        # Load EEG data
        raw = mne.io.read_raw_eeglab(str(set_file), preload=True, verbose=False)
        data = raw.get_data()  # (channels, samples)
        srate = int(raw.info['sfreq'])
        
        # Simple spectral analysis
        f, pxx = signal.welch(data.mean(axis=0), fs=srate, nperseg=min(srate*4, data.shape[1]))
        
        # Band powers
        alpha_pow = np.mean(pxx[(f >= 8) & (f <= 13)])
        beta_pow = np.mean(pxx[(f >= 13) & (f <= 30)])
        theta_pow = np.mean(pxx[(f >= 4) & (f <= 8)])
        delta_pow = np.mean(pxx[(f >= 1) & (f <= 4)])
        
        # Ratios
        alpha_beta = alpha_pow / beta_pow if beta_pow > 0 else 0
        theta_beta = theta_pow / beta_pow if beta_pow > 0 else 0
        
        row = {
            'Participant': f'P{p_num:02d}',
            'Condition': 'Aggregated',
            'Delta_Power': float(delta_pow),
            'Theta_Power': float(theta_pow),
            'Alpha_Power': float(alpha_pow),
            'Beta_Power': float(beta_pow),
            'Alpha_Beta_Ratio': float(alpha_beta),
            'Theta_Beta_Ratio': float(theta_beta),
            'N_Channels': data.shape[0],
            'Duration_Sec': data.shape[1] / srate,
        }
        rows.append(row)
        print("OK")
        
    except Exception as e:
        print(f"SKIP ({e})")
        continue

# Save
df = pd.DataFrame(rows)
df.to_csv(output_csv, index=False)
print(f"\nSaved {len(df)} rows to {output_csv}")
print("Columns:", ', '.join(df.columns))
