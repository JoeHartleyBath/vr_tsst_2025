"""
Fast Window Experiments using Existing Pre-windowed Data

The existing MNE epochs are 10s @ 125Hz with 50% overlap.
We test CONDITION-LEVEL trimming (exclude windows from start/end of 180s conditions)
NOT within-window edge trimming (which is meaningless with overlapping windows).

Experiments:
- Baseline: all windows (1-35)
- Condition trimming: exclude first/last 10-20s of each condition  
- Late windows: only use stabilized cognitive state portion

Usage:
    python fast_window_experiments.py          # Full run (all 43 participants)
    python fast_window_experiments.py --fast   # Fast run (~15 participants)
"""

import sys
import os
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, f1_score
import mne
import pandas as pd
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))
from train_tcnet import EEGTCNet

# ==============================================================================
# CONFIGURATION
# ==============================================================================
DATA_DIR = r'C:\vr_tsst_2025\output\adaptive_workload\mne_epochs'
RESULTS_DIR = r'C:\vr_tsst_2025\results\window_sweep'

# All participants vs fast subset
ALL_PIDS = [1, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 15, 16, 17, 18, 19, 20, 21, 22, 23, 
            24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 
            42, 43, 45, 47, 48]
FAST_PIDS = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 3, 12, 23, 33, 43]  # ~15 participants

# Training config
N_FOLDS = 3
EPOCHS = 50
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
PATIENCE = 10
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Configurations to test
# Condition-level trimming: exclude windows from start/end of 180s conditions
# window_idx 1-35 (1=0-10s, 35=170-180s with 50% overlap)
CONFIGS = [
    {'name': 'all_windows', 'window_range': (1, 35)},              # Baseline - all windows
    {'name': 'trim_first_10s', 'window_range': (3, 35)},           # Skip first ~10s
    {'name': 'trim_last_10s', 'window_range': (1, 33)},            # Skip last ~10s
    {'name': 'trim_both_10s', 'window_range': (3, 33)},            # Skip first & last ~10s
    {'name': 'trim_both_20s', 'window_range': (5, 31)},            # Skip first & last ~20s (BEST)
    {'name': 'trim_both_30s', 'window_range': (7, 29)},            # Skip first & last ~30s
    {'name': 'late_only_60s', 'window_range': (25, 35)},           # Late 60s only
    {'name': 'middle_60s', 'window_range': (13, 23)},              # Middle 60s only
]


def load_epochs(subset_pids=None):
    """Load pre-windowed epochs with normalization.
    
    Args:
        subset_pids: Optional list of participant IDs to include. If None, load all.
    """
    data_path = Path(DATA_DIR)
    epoch_files = sorted(data_path.glob('*_workload-epo.fif'))
    
    # Filter to subset if specified
    if subset_pids is not None:
        subset_set = set(subset_pids)
        epoch_files = [f for f in epoch_files 
                       if int(f.stem.split('_')[0][1:]) in subset_set]
    
    all_data = []
    all_labels = []
    all_pids = []
    all_window_idx = []
    
    for fpath in epoch_files:
        epochs = mne.read_epochs(fpath, preload=True, verbose=False)
        data = epochs.get_data()
        labels = (epochs.metadata['workload_class'] == 'HighWorkload').astype(int).values
        pids = epochs.metadata['pid'].values
        window_idx = epochs.metadata['window_idx'].values
        
        all_data.append(data)
        all_labels.append(labels)
        all_pids.append(pids)
        all_window_idx.append(window_idx)
    
    # Align sample counts
    min_samples = min(d.shape[2] for d in all_data)
    all_data = [d[:, :, :min_samples] for d in all_data]
    
    X = np.concatenate(all_data, axis=0).astype(np.float32)
    y = np.concatenate(all_labels, axis=0)
    pids = np.concatenate(all_pids, axis=0)
    window_idx = np.concatenate(all_window_idx, axis=0)
    
    # CRITICAL: Per-epoch z-score normalization (same as MNEEpochsDataset)
    # Z-score across time for each channel within each epoch
    mean = X.mean(axis=2, keepdims=True)
    std = X.std(axis=2, keepdims=True) + 1e-6
    X = (X - mean) / std
    
    return X, y, pids, window_idx


def apply_config(X, y, pids, window_idx, config):
    """Apply condition-level window filtering."""
    # Filter by window range (condition-level trimming)
    win_min, win_max = config['window_range']
    mask = (window_idx >= win_min) & (window_idx <= win_max)
    
    X_out = X[mask]
    y_out = y[mask]
    pids_out = pids[mask]
    
    return X_out, y_out, pids_out


class EpochsDataset(Dataset):
    def __init__(self, data, labels, pids):
        self.data = torch.FloatTensor(data).unsqueeze(1)
        self.labels = torch.LongTensor(labels)
        self.pids = pids
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx], self.pids[idx]


def train_fold(model, train_loader, val_loader, device, class_weights):
    """Train for one fold with early stopping and class weights."""
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    
    best_val_acc = 0
    best_val_f1 = 0
    patience_counter = 0
    
    for epoch in range(EPOCHS):
        model.train()
        for batch_x, batch_y, _ in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
        
        model.eval()
        val_preds, val_true = [], []
        with torch.no_grad():
            for batch_x, batch_y, _ in val_loader:
                batch_x = batch_x.to(device)
                outputs = model(batch_x)
                preds = outputs.argmax(dim=1).cpu().numpy()
                val_preds.extend(preds)
                val_true.extend(batch_y.numpy())
        
        val_acc = accuracy_score(val_true, val_preds)
        val_f1 = f1_score(val_true, val_preds, average='macro')
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_f1 = val_f1
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                break
    
    return best_val_acc, best_val_f1


def evaluate_config(X, y, pids, config_name):
    """Evaluate using group k-fold CV."""
    n_channels = X.shape[1]
    n_timepoints = X.shape[2]
    
    # Compute class weights
    unique, counts = np.unique(y, return_counts=True)
    weights = len(y) / (len(unique) * counts)
    class_weights = torch.FloatTensor(weights).to(DEVICE)
    print(f"  Class weights: {weights}")
    
    unique_pids = np.unique(pids)
    gkf = GroupKFold(n_splits=min(N_FOLDS, len(unique_pids)))
    
    fold_accs = []
    fold_f1s = []
    
    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, pids)):
        train_ds = EpochsDataset(X[train_idx], y[train_idx], pids[train_idx])
        val_ds = EpochsDataset(X[val_idx], y[val_idx], pids[val_idx])
        
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
        
        model = EEGTCNet(
            chans=n_channels,
            classes=2,
            time_points=n_timepoints,
            F1=16, D=2, kernLength=32, tcn_depth=3,
            use_se_attention=False, use_temporal_attention=False
        ).to(DEVICE)
        
        acc, f1 = train_fold(model, train_loader, val_loader, DEVICE, class_weights)
        fold_accs.append(acc)
        fold_f1s.append(f1)
        print(f"    Fold {fold+1}: Acc={acc:.3f}, F1={f1:.3f}")
    
    return np.mean(fold_accs), np.std(fold_accs), np.mean(fold_f1s)


def parse_args():
    parser = argparse.ArgumentParser(description='Window sweep experiments')
    parser.add_argument('--fast', action='store_true', 
                        help='Use subset of ~15 participants for faster sweep')
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Select participants
    subset_pids = FAST_PIDS if args.fast else None
    mode_str = f"FAST ({len(FAST_PIDS)} participants)" if args.fast else f"FULL ({len(ALL_PIDS)} participants)"
    
    print(f"\n{'='*70}")
    print(f"Window Experiments: Condition-Level Trimming")
    print(f"Mode: {mode_str}")
    print(f"Device: {DEVICE}")
    print(f"{'='*70}\n")
    
    # Load base data once
    print("Loading base 10s epochs...")
    X_base, y_base, pids_base, window_idx_base = load_epochs(subset_pids=subset_pids)
    print(f"  Base data: {X_base.shape[0]} windows, {X_base.shape[1]} channels, {X_base.shape[2]} samples")
    print(f"  Participants: {len(np.unique(pids_base))}")
    print(f"  Window idx range: {window_idx_base.min()}-{window_idx_base.max()}")
    print(f"  Class balance: {np.mean(y_base):.1%} HighWorkload\n")
    
    results = []
    
    for config in CONFIGS:
        print(f"\n{'='*70}")
        print(f"Testing: {config['name']}")
        win_min, win_max = config['window_range']
        print(f"  Window range: {win_min}-{win_max}")
        print(f"{'='*70}")
        
        # Apply configuration
        X, y, pids = apply_config(
            X_base.copy(), y_base.copy(), pids_base.copy(), 
            window_idx_base.copy(), config
        )
        print(f"  Data shape: ({X.shape[0]}, {X.shape[1]}, {X.shape[2]})")
        print(f"  Class balance: {np.mean(y):.1%} HighWorkload")
        
        # Evaluate
        mean_acc, std_acc, mean_f1 = evaluate_config(X, y, pids, config['name'])
        
        results.append({
            'config': config['name'],
            'window_range': f"{win_min}-{win_max}",
            'n_windows': X.shape[0],
            'time_points': X.shape[2],
            'mean_acc': mean_acc,
            'std_acc': std_acc,
            'mean_f1': mean_f1
        })
        print(f"\n  Result: Acc={mean_acc:.3f} ± {std_acc:.3f}, F1={mean_f1:.3f}")
    
    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('mean_acc', ascending=False)
    print(results_df.to_string(index=False))
    
    # Save
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    mode_suffix = '_fast' if args.fast else ''
    results_path = os.path.join(RESULTS_DIR, f'condition_trim_experiments{mode_suffix}_{timestamp}.csv')
    results_df.to_csv(results_path, index=False)
    print(f"\nResults saved to: {results_path}")
    
    if not results_df.empty:
        best = results_df.iloc[0]
        print(f"\nBest: {best['config']} with {best['mean_acc']:.1%} accuracy")


if __name__ == '__main__':
    main()
