import mne
import os
import pandas as pd
import numpy as np

# ==============================================================================
# CONFIGURATION
# ==============================================================================
INPUT_MANIFEST = r'C:\vr_tsst_2025\output\aggregated\eeg_features_rolling_windows.csv'
CLEANED_DIR = r'C:\vr_tsst_2025\output\cleaned_eeg'
CB_FILE = r'C:\vr_tsst_2025\data\experimental_counterbalance.xlsx'
OUTPUT_DIR = r'C:\vr_tsst_2025\output\adaptive_workload\mne_epochs'

# All 44 EEG-valid participants
ALL_PIDS = [1, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 15, 16, 17, 18, 19, 20, 21, 22, 23, 
            24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 
            42, 43, 44, 45, 47, 48]
SUBSET_PIDS = ALL_PIDS  # Use all 44 participants

# All 4 conditions (LowStress + HighStress)
TARGET_CONDITIONS = {
    'LowWorkload': ['LowStress_LowCog_Task', 'HighStress_LowCog_Task'],
    'HighWorkload': ['LowStress_HighCog1022_Task', 'LowStress_HighCog2043_Task',
                     'HighStress_HighCog1022_Task', 'HighStress_HighCog2043_Task']
}

# Mapping for Round matching (all 4 condition types)
CB_CONDITION_MAP = {
    'Calm Addition': ['LowStress_LowCog_Task'],
    'Calm Subtraction': ['LowStress_HighCog1022_Task', 'LowStress_HighCog2043_Task'],
    'Stress Addition': ['HighStress_LowCog_Task'],
    'Stress Subtraction': ['HighStress_HighCog1022_Task', 'HighStress_HighCog2043_Task']
}

os.makedirs(OUTPUT_DIR, exist_ok=True)

def export_subject_epochs(pid, manifest_df, cb_df):
    set_path = os.path.join(CLEANED_DIR, f'P{pid:02d}_cleaned.set')
    if not os.path.exists(set_path):
        # Try without leading zero if P01 fails
        set_path = os.path.join(CLEANED_DIR, f'P{pid}_cleaned.set')
        if not os.path.exists(set_path):
            print(f"Skipping P{pid}: File not found ({set_path})")
            return

    print(f"\nProcessing P{pid}...")
    # Load cleaned EEG. eeglab files often have .fdt in same dir.
    raw = mne.io.read_raw_eeglab(set_path, preload=True)
    
    # Resample for DL (250Hz is standard for these models)
    if raw.info['sfreq'] > 250:
        print(f"  Resampling from {raw.info['sfreq']}Hz to 250Hz")
        raw.resample(250.0)
    
    # Get manifest for this subject
    sub_df = manifest_df[manifest_df['pid'] == pid].copy()
    
    # Identify Task and Forest windows
    all_target_labels = [lbl for sublist in TARGET_CONDITIONS.values() for lbl in sublist]
    task_windows = sub_df[sub_df['event_label'].isin(all_target_labels)].copy()
    
    # Map workload classes
    label_to_class = {lbl: cls for cls, lbls in TARGET_CONDITIONS.items() for lbl in lbls}
    task_windows['workload_class'] = task_windows['event_label'].map(label_to_class)
    
    # Identify which Forest corresponds to each Task
    # (Using counterbalance logic to match tasks back to their round)
    p_cb = cb_df[cb_df['Participant'] == pid]
    if p_cb.empty:
        print(f"  WARNING: No counterbalance data for P{pid}")
        return
        
    round_map = {} # label -> ForestID
    for r in [1, 2, 3, 4]:
        round_text = p_cb[f'Round {r}'].values[0]
        event_labels = CB_CONDITION_MAP.get(round_text, [])
        for lbl in event_labels:
            round_map[lbl] = f'Forest{r}'

    all_epoch_data = []
    all_metadata = []

    for idx, row in task_windows.iterrows():
        tmin = row['window_start']
        tmax = row['window_end']
        
        # Ensure we don't exceed duration (small epsilon for floating point)
        if tmax > raw.times[-1] + 1e-3:
            continue
        
        # Clip tmax to duration if just barely over
        tmax = min(tmax, raw.times[-1])
            
        data = raw.get_data(tmin=tmin, tmax=tmax)
        
        # Metadata
        meta = {
            'pid': pid,
            'window_idx': row['window_idx'],
            'event_label': row['event_label'],
            'workload_class': row['workload_class'],
            'baseline_id': round_map.get(row['event_label'], 'None'),
            'tmin': tmin,
            'tmax': tmax
        }
        
        all_epoch_data.append(data)
        all_metadata.append(meta)

    if not all_epoch_data:
        print(f"  No valid windows found for P{pid}")
        return

    # Check sample consistency
    n_samples = [d.shape[1] for d in all_epoch_data]
    min_samples = min(n_samples)
    print(f"  Exporting {len(all_epoch_data)} epochs, aligned to {min_samples} samples per window")
    all_epoch_data_fixed = [d[:, :min_samples] for d in all_epoch_data]
    
    epochs_data = np.stack(all_epoch_data_fixed)
    
    # Create info object
    info = raw.info.copy()
    
    # Create EpochsArray
    epochs = mne.EpochsArray(epochs_data, info)
    epochs.metadata = pd.DataFrame(all_metadata)
    
    # Event IDs: 0 for Low, 1 for High
    epochs.event_id = {'LowWorkload': 0, 'HighWorkload': 1}
    # Create events array
    events = np.zeros((len(all_metadata), 3), dtype=int)
    events[:, 0] = np.arange(len(all_metadata)) # Arbitrary indices
    events[:, 2] = [0 if m['workload_class'] == 'LowWorkload' else 1 for m in all_metadata]
    epochs.events = events

    # Save
    out_path = os.path.join(OUTPUT_DIR, f'P{pid:02d}_workload-epo.fif')
    epochs.save(out_path, overwrite=True)
    print(f"  Saved to {out_path}")

# ==============================================================================
# MAIN
# ==============================================================================
if __name__ == "__main__":
    print(f"Loading manifest: {INPUT_MANIFEST}")
    manifest = pd.read_csv(INPUT_MANIFEST)
    
    print(f"Loading counterbalance: {CB_FILE}")
    cb = pd.read_excel(CB_FILE)
    
    for pid in SUBSET_PIDS:
        try:
            export_subject_epochs(pid, manifest, cb)
        except Exception as e:
            print(f"Error processing P{pid}: {e}")
            import traceback
            traceback.print_exc()
