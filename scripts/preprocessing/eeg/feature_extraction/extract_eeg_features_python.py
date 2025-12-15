#!/usr/bin/env python
"""
EEG Feature Extraction (Python-based, MNE backend)
Replaces MATLAB script for pilot data (P01-P03)

Extracts spectral power, power ratios, and sample entropy from cleaned .set files.
"""

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import signal, stats
from scipy.stats import entropy as scipy_entropy
import yaml
import logging
from tqdm import tqdm

# Add scripts path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / 'scripts'))

# Try MNE, fall back to scipy
try:
    import mne
    HAS_MNE = True
except ImportError:
    HAS_MNE = False
    print("Warning: MNE not available, will use scipy for .set loading")

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ============================================================================
#  CONFIGURATION
# ============================================================================

project_root = Path(__file__).resolve().parent.parent.parent.parent.parent  # 5 up: features → eeg → preprocessing → scripts → project
config_path = project_root / 'config' / 'general.yaml'
with open(config_path) as f:
    config = yaml.safe_load(f)

cleaned_eeg_folder = project_root / config['paths']['cleaned_eeg']
output_folder = project_root / config['paths']['output'] / 'eeg_features'
output_folder.mkdir(parents=True, exist_ok=True)
output_csv = output_folder / 'eeg_features.csv'

logger.info(f"Input folder: {cleaned_eeg_folder}")
logger.info(f"Output file: {output_csv}")

# ============================================================================
#  HELPERS
# ============================================================================

def load_eeg_set(set_file):
    """Load .set file using MNE or scipy."""
    if HAS_MNE:
        return mne.io.read_raw_eeglab(str(set_file), preload=True)
    else:
        # Fallback: scipy.io.loadmat for .set files (requires manual .set parsing)
        # For now, raise error
        raise ImportError("MNE required to load .set files. Install: pip install mne")

def compute_band_power(data, srate, freqs_dict):
    """Compute power in frequency bands using Welch's method."""
    bands = {}
    f, pxx = signal.welch(data, fs=srate, nperseg=min(srate * 4, len(data)), axis=-1)
    
    for band_name, (f_lo, f_hi) in freqs_dict.items():
        mask = (f >= f_lo) & (f <= f_hi)
        bands[band_name] = np.mean(pxx[..., mask], axis=-1) if mask.any() else 0.0
    
    return bands

def compute_sample_entropy(x, m=2, r=None):
    """Compute sample entropy (simplified)."""
    if r is None:
        r = 0.2 * np.std(x)
    
    N = len(x)
    patterns_m = np.array([x[i:i+m] for i in range(N-m+1)])
    patterns_m1 = np.array([x[i:i+m+1] for i in range(N-m)])
    
    # Count matches within tolerance r
    matches_m = 0
    matches_m1 = 0
    for i in range(len(patterns_m)):
        for j in range(i+1, len(patterns_m)):
            if np.max(np.abs(patterns_m[i] - patterns_m[j])) < r:
                matches_m += 1
    
    for i in range(len(patterns_m1)):
        for j in range(i+1, len(patterns_m1)):
            if np.max(np.abs(patterns_m1[i] - patterns_m1[j])) < r:
                matches_m1 += 1
    
    if matches_m1 == 0 or matches_m == 0:
        return 0.0
    return -np.log(matches_m1 / matches_m)

def extract_features(eeg_data, srate, ch_labels=None):
    """Extract features from EEG data."""
    features = {}
    
    # Frequency bands
    bands = {
        'delta': (1, 4),
        'theta': (4, 8),
        'alpha': (8, 13),
        'beta': (13, 30),
        'gamma': (30, 49)
    }
    
    # Regional channels (simplified)
    regions = {
        'Frontal': ['Fp1', 'Fp2', 'F3', 'F4', 'Fz'],
        'Central': ['C3', 'C4', 'Cz'],
        'Parietal': ['P3', 'P4', 'Pz'],
        'Occipital': ['O1', 'O2', 'Oz']
    }
    
    # Compute band power across all channels (avg)
    if len(eeg_data.shape) == 2:  # (channels, samples)
        data_avg = np.mean(eeg_data, axis=0)
    else:
        data_avg = eeg_data
    
    band_powers = compute_band_power(data_avg, srate, bands)
    for band, power in band_powers.items():
        features[f'Overall_{band}_Power'] = float(power)
    
    # Power ratios
    if band_powers.get('alpha', 0) > 0 and band_powers.get('beta', 0) > 0:
        features['Alpha_Beta_Ratio'] = float(band_powers['alpha'] / band_powers['beta'])
    
    if band_powers.get('theta', 0) > 0 and band_powers.get('beta', 0) > 0:
        features['Theta_Beta_Ratio'] = float(band_powers['theta'] / band_powers['beta'])
    
    # Sample entropy (fast estimate on data sample)
    data_sample = data_avg[::10]  # Downsample for speed
    samp_ent = compute_sample_entropy(data_sample)
    features['Sample_Entropy'] = float(samp_ent)
    
    return features

# ============================================================================
#  MAIN
# ============================================================================

def main():
    """Extract features for all participants."""
    logger.info("="*70)
    logger.info("EEG Feature Extraction (Python, MNE-based)")
    logger.info("="*70)
    
    # Find cleaned .set files
    cleaned_files = sorted(cleaned_eeg_folder.glob('P*_cleaned.set'))
    participant_nums = [int(f.stem.split('_')[0][1:]) for f in cleaned_files]
    
    if not cleaned_files:
        logger.error(f"No cleaned .set files found in {cleaned_eeg_folder}")
        sys.exit(1)
    
    logger.info(f"Found {len(cleaned_files)} cleaned .set files: P{participant_nums}")
    
    # Prepare output table
    rows = []
    
    # Process each participant
    for cleaned_file in tqdm(cleaned_files, desc='Extracting features'):
        try:
            p_num = int(cleaned_file.stem.split('_')[0][1:])
            logger.info(f"\n[P{p_num:02d}] Loading {cleaned_file.name}")
            
            # Load EEG
            if HAS_MNE:
                raw = mne.io.read_raw_eeglab(str(cleaned_file), preload=True, verbose=False)
                eeg_data = raw.get_data()
                srate = int(raw.info['sfreq'])
                ch_labels = raw.ch_names
                logger.info(f"  Loaded: {eeg_data.shape[0]} ch × {eeg_data.shape[1]} samples @ {srate} Hz")
            else:
                logger.error(f"  MNE required. Skipping.")
                continue
            
            # Extract features
            features = extract_features(eeg_data, srate, ch_labels)
            
            # Add to table
            row = {'Participant': f'P{p_num:02d}', 'Condition': 'Aggregated'}
            row.update(features)
            rows.append(row)
            
            logger.info(f"  ✓ Extracted {len(features)} features")
            
        except Exception as e:
            logger.error(f"[P{p_num:02d}] Error: {e}")
            continue
    
    # Save to CSV
    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)
    logger.info(f"\n✓ Extracted features for {len(rows)} participants")
    logger.info(f"✓ Saved to: {output_csv}")
    logger.info(f"\nColumns ({len(df.columns)}): {', '.join(df.columns[:10])}...")
    logger.info("="*70)

if __name__ == '__main__':
    main()
