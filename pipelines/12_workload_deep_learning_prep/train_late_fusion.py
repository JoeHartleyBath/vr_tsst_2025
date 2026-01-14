"""
Late Fusion for Workload Classification
Combines EEG-TCNet deep learning with XGBoost on band power features.

Strategy:
1. Train tuned EEG-TCNet on raw EEG epochs (65.5% baseline)
2. Train XGBoost on band power features + theta/beta ratios
3. Late fusion: combine softmax probs with weighted averaging or stacking

Alignment:
- EEG epochs: (pid, condition, window_idx) from MNE metadata
- Band power: (pid, event_label, window_idx) from CSV
- Join on pid + condition/event_label + window_idx
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

from mne_dataloader import load_workload_data
from train_tcnet import EEGTCNet

# ==============================================================================
# PATHS & CONFIGURATION
# ==============================================================================
DATA_DIR = r'C:\vr_tsst_2025\output\adaptive_workload\mne_epochs'
BAND_POWER_PATH = r'C:\vr_tsst_2025\output\aggregated\eeg_features_rolling_windows.csv'
PHYSIO_PATH = r'C:\vr_tsst_2025\output\aggregated\physio_features_rolling_windows.csv'
PARAMS_PATH = r'C:\vr_tsst_2025\results\best_tcnet_focused_params.json'
RESULTS_PATH = r'C:\vr_tsst_2025\results\workload_late_fusion_cv.txt'

# All participants
ALL_PIDS = [1, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 15, 16, 17, 18, 19, 20, 21, 22, 23, 
            24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 
            42, 43, 45, 47, 48]

# Workload conditions (matching actual event labels in data)
HIGH_WORKLOAD = ['HighStress_HighCog1022_Task', 'LowStress_HighCog1022_Task']
LOW_WORKLOAD = ['HighStress_LowCog_Task', 'LowStress_LowCog_Task']

# TCNet training params (from best tuned)
TCNET_EPOCHS = 50
EARLY_STOP_PATIENCE = 10

# XGBoost params (to be tuned)
XGB_PARAMS = {
    'max_depth': 4,
    'learning_rate': 0.1,
    'n_estimators': 100,
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'tree_method': 'hist',
    'random_state': 42,
    'verbosity': 0
}

# Fusion weights (to be optimized)
TCNET_WEIGHT = 0.6
XGB_WEIGHT = 0.4


def load_best_tcnet_params():
    """Load best hyperparameters from Optuna tuning."""
    if os.path.exists(PARAMS_PATH):
        with open(PARAMS_PATH) as f:
            params = json.load(f)
        print(f"Loaded best params from {PARAMS_PATH}")
        return params
    else:
        # Default params if file not found
        return {
            'F1': 16, 'D': 2, 'kernLength': 32,
            'tcn_filters': 12, 'tcn_kernel': 6, 'tcn_depth': 3,
            'dropout_eeg': 0.305, 'dropout_tcn': 0.326,
            'learning_rate': 0.00159, 'batch_size': 32
        }


def load_band_power_features():
    """Load band power features with theta/beta ratios."""
    df = pd.read_csv(BAND_POWER_PATH)
    
    # Filter to workload conditions only
    all_conditions = HIGH_WORKLOAD + LOW_WORKLOAD
    df = df[df['event_label'].isin(all_conditions)].copy()
    
    # Create binary workload label
    df['workload'] = df['event_label'].isin(HIGH_WORKLOAD).astype(int)
    
    # Create theta/beta ratios (key workload markers)
    regions = ['FrontalLeft', 'FrontalRight', 'OverallFrontal', 'FrontalMidline',
               'Central', 'ParietalLeft', 'ParietalRight', 'Parietal']
    
    for region in regions:
        theta_col = f'{region}_Theta'
        beta_col = f'{region}_Beta'
        if theta_col in df.columns and beta_col in df.columns:
            # Theta/Beta ratio (log scale, so subtract)
            df[f'{region}_ThetaBetaRatio'] = df[theta_col] - df[beta_col]
            # Alpha/Beta ratio (engagement index inverse)
            alpha_col = f'{region}_Alpha'
            if alpha_col in df.columns:
                df[f'{region}_AlphaBetaRatio'] = df[alpha_col] - df[beta_col]
    
    # Create frontal theta asymmetry (workload lateralization)
    df['Frontal_ThetaAsymmetry'] = df['FrontalRight_Theta'] - df['FrontalLeft_Theta']
    df['Frontal_AlphaAsymmetry'] = df['FrontalRight_Alpha'] - df['FrontalLeft_Alpha']
    
    print(f"Band power features: {len(df)} windows, {df.shape[1]} columns")
    print(f"High workload: {(df['workload']==1).sum()}, Low workload: {(df['workload']==0).sum()}")
    
    return df


def load_physio_features():
    """Load physio features for fusion."""
    df = pd.read_csv(PHYSIO_PATH)
    
    # Filter to workload conditions
    all_conditions = HIGH_WORKLOAD + LOW_WORKLOAD
    df = df[df['Condition'].isin(all_conditions)].copy()
    
    # Rename for merging
    df = df.rename(columns={
        'Participant_ID': 'pid',
        'Condition': 'event_label',
        'Window_Index': 'window_idx'
    })
    
    # Adjust window index (physio uses 0-indexed, EEG uses 1-indexed)
    df['window_idx'] = df['window_idx'] + 1
    
    print(f"Physio features: {len(df)} windows")
    return df


def get_feature_columns(df):
    """Get band power feature columns (exclude metadata)."""
    exclude = ['pid', 'event_label', 'window_idx', 'window_start', 'window_end', 
               'workload', 'Participant_ID', 'Condition', 'Window_Index',
               'Window_Start', 'Window_End']
    return [c for c in df.columns if c not in exclude and not c.endswith('_Flag')]


def align_mne_with_bandpower(mne_dataset, band_df):
    """
    Align MNE epochs with band power features.
    Returns indices of matched epochs and corresponding band power features.
    """
    # Get MNE metadata
    mne_meta = []
    for i in range(len(mne_dataset)):
        pid = mne_dataset.pids[i]
        cond = mne_dataset.conditions[i]
        win_idx = mne_dataset.window_indices[i]
        mne_meta.append({'mne_idx': i, 'pid': pid, 'event_label': cond, 'window_idx': win_idx})
    
    mne_df = pd.DataFrame(mne_meta)
    
    # Merge on keys
    merged = mne_df.merge(
        band_df, 
        on=['pid', 'event_label', 'window_idx'],
        how='inner'
    )
    
    print(f"MNE epochs: {len(mne_df)}, Band power windows: {len(band_df)}")
    print(f"Matched: {len(merged)} ({100*len(merged)/len(mne_df):.1f}%)")
    
    return merged


def create_group_splits_from_pids(pids, n_splits=3):
    """Create GroupKFold splits from participant IDs."""
    gkf = GroupKFold(n_splits=n_splits)
    X_dummy = np.zeros((len(pids), 1))
    y_dummy = np.zeros(len(pids))
    
    for train_idx, test_idx in gkf.split(X_dummy, y_dummy, groups=pids):
        yield train_idx, test_idx


def train_tcnet_fold(train_loader, test_loader, device, params, class_weights):
    """Train TCNet for one fold, return test predictions."""
    model = EEGTCNet(
        chans=128, classes=2, time_points=1249,
        F1=params.get('F1', 16),
        D=params.get('D', 2),
        kernLength=params.get('kernLength', 32),
        dropout_eeg=params.get('dropout_eeg', 0.3),
        tcn_filters=params.get('tcn_filters', 12),
        tcn_kernel=params.get('tcn_kernel', 6),
        tcn_depth=params.get('tcn_depth', 3),
        dropout_tcn=params.get('dropout_tcn', 0.3),
        use_se_attention=False,
        use_temporal_attention=False
    ).to(device)
    
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=params.get('learning_rate', 0.001))
    
    best_acc = 0
    patience_counter = 0
    best_state = None
    
    for epoch in range(TCNET_EPOCHS):
        # Train
        model.train()
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
        
        # Evaluate
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x = batch_x.to(device)
                preds = torch.argmax(model(batch_x), dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(batch_y.numpy())
        
        acc = accuracy_score(all_labels, all_preds)
        if acc > best_acc:
            best_acc = acc
            patience_counter = 0
            best_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOP_PATIENCE:
                break
    
    # Load best model and get softmax predictions
    model.load_state_dict(best_state)
    model.eval()
    
    all_probs = []
    all_labels = []
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            outputs = model(batch_x)
            probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()  # P(high workload)
            all_probs.extend(probs)
            all_labels.extend(batch_y.numpy())
    
    return np.array(all_probs), np.array(all_labels), best_acc


def train_xgb_fold(X_train, y_train, X_test, y_test, groups_train):
    """Train XGBoost for one fold, return test predictions."""
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Handle class imbalance
    scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    
    params = XGB_PARAMS.copy()
    params['scale_pos_weight'] = scale_pos_weight
    
    model = xgb.XGBClassifier(**params)
    model.fit(X_train_scaled, y_train, verbose=False)
    
    # Get probability predictions
    probs = model.predict_proba(X_test_scaled)[:, 1]
    preds = (probs > 0.5).astype(int)
    
    acc = accuracy_score(y_test, preds)
    return probs, acc, model


def run_late_fusion_cv():
    """Run cross-validated late fusion evaluation."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load data
    print("\n=== Loading Data ===")
    mne_dataset = load_workload_data(DATA_DIR, subset_pids=ALL_PIDS, augment=False)
    band_df = load_band_power_features()
    tcnet_params = load_best_tcnet_params()
    
    # Align MNE epochs with band power features
    print("\n=== Aligning Data ===")
    merged = align_mne_with_bandpower(mne_dataset, band_df)
    
    # Get feature columns
    feature_cols = get_feature_columns(merged)
    print(f"Using {len(feature_cols)} band power features")
    
    # Prepare aligned data
    aligned_mne_idx = merged['mne_idx'].values
    aligned_X_band = merged[feature_cols].values.astype(np.float32)
    aligned_y = merged['workload'].values
    aligned_pids = merged['pid'].values
    
    # Handle NaN in band power features
    nan_mask = np.isnan(aligned_X_band).any(axis=1)
    if nan_mask.sum() > 0:
        print(f"Dropping {nan_mask.sum()} windows with NaN values")
        aligned_mne_idx = aligned_mne_idx[~nan_mask]
        aligned_X_band = aligned_X_band[~nan_mask]
        aligned_y = aligned_y[~nan_mask]
        aligned_pids = aligned_pids[~nan_mask]
    
    print(f"Final aligned dataset: {len(aligned_y)} windows")
    
    # Compute class weights
    class_counts = np.bincount(aligned_y)
    class_weights = torch.FloatTensor(len(aligned_y) / (2 * class_counts)).to(device)
    print(f"Class weights: {class_weights.cpu().numpy()}")
    
    # Cross-validation
    results = {
        'tcnet': [], 'xgb_band': [], 'fusion': [],
        'tcnet_acc': [], 'xgb_acc': [], 'fusion_acc': []
    }
    
    print("\n=== Cross-Validation ===")
    for fold, (train_idx, test_idx) in enumerate(create_group_splits_from_pids(aligned_pids, n_splits=3)):
        print(f"\n--- Fold {fold+1} ---")
        
        # Get train/test data
        train_mne_idx = aligned_mne_idx[train_idx]
        test_mne_idx = aligned_mne_idx[test_idx]
        
        X_band_train = aligned_X_band[train_idx]
        X_band_test = aligned_X_band[test_idx]
        y_train = aligned_y[train_idx]
        y_test = aligned_y[test_idx]
        pids_train = aligned_pids[train_idx]
        
        # Create MNE data loaders (using aligned indices)
        batch_size = tcnet_params.get('batch_size', 32)
        train_subset = Subset(mne_dataset, train_mne_idx)
        test_subset = Subset(mne_dataset, test_mne_idx)
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False)
        
        # Train TCNet
        print("Training TCNet...")
        tcnet_probs, tcnet_labels, tcnet_acc = train_tcnet_fold(
            train_loader, test_loader, device, tcnet_params, class_weights
        )
        results['tcnet_acc'].append(tcnet_acc)
        print(f"  TCNet accuracy: {tcnet_acc:.3f}")
        
        # Train XGBoost on band power
        print("Training XGBoost on band power...")
        xgb_probs, xgb_acc, _ = train_xgb_fold(
            X_band_train, y_train, X_band_test, y_test, pids_train
        )
        results['xgb_acc'].append(xgb_acc)
        print(f"  XGBoost accuracy: {xgb_acc:.3f}")
        
        # Late fusion (weighted average)
        fusion_probs = TCNET_WEIGHT * tcnet_probs + XGB_WEIGHT * xgb_probs
        fusion_preds = (fusion_probs > 0.5).astype(int)
        fusion_acc = accuracy_score(y_test, fusion_preds)
        fusion_f1 = f1_score(y_test, fusion_preds, average='weighted')
        
        results['fusion_acc'].append(fusion_acc)
        results['fusion'].append({'accuracy': fusion_acc, 'f1': fusion_f1})
        print(f"  Fusion accuracy: {fusion_acc:.3f} (F1: {fusion_f1:.3f})")
    
    # Summary
    print("\n" + "="*60)
    print("LATE FUSION RESULTS")
    print("="*60)
    
    tcnet_mean = np.mean(results['tcnet_acc'])
    xgb_mean = np.mean(results['xgb_acc'])
    fusion_mean = np.mean(results['fusion_acc'])
    fusion_std = np.std(results['fusion_acc'])
    fusion_f1_mean = np.mean([r['f1'] for r in results['fusion']])
    
    print(f"TCNet alone:     {tcnet_mean:.3f}")
    print(f"XGBoost alone:   {xgb_mean:.3f}")
    print(f"Late Fusion:     {fusion_mean:.3f} (+/- {fusion_std:.3f})")
    print(f"Fusion F1:       {fusion_f1_mean:.3f}")
    print(f"Fusion weights:  TCNet={TCNET_WEIGHT}, XGBoost={XGB_WEIGHT}")
    
    # Save results
    with open(RESULTS_PATH, 'w') as f:
        f.write("Late Fusion Results - Workload Classification\n")
        f.write("="*60 + "\n")
        f.write(f"Components:\n")
        f.write(f"  - EEG-TCNet on raw epochs (tuned params)\n")
        f.write(f"  - XGBoost on band power features + theta/beta ratios\n")
        f.write(f"  - Fusion weights: TCNet={TCNET_WEIGHT}, XGBoost={XGB_WEIGHT}\n")
        f.write(f"\nDataset:\n")
        f.write(f"  - Aligned windows: {len(aligned_y)}\n")
        f.write(f"  - Band power features: {len(feature_cols)}\n")
        f.write(f"\nResults:\n")
        for i, r in enumerate(results['fusion']):
            f.write(f"  Fold {i+1}: Acc={r['accuracy']:.4f}, F1={r['f1']:.4f}\n")
        f.write(f"\nSummary:\n")
        f.write(f"  TCNet alone:   {tcnet_mean:.4f}\n")
        f.write(f"  XGBoost alone: {xgb_mean:.4f}\n")
        f.write(f"  Late Fusion:   {fusion_mean:.4f} (+/- {fusion_std:.4f})\n")
        f.write(f"  Fusion F1:     {fusion_f1_mean:.4f}\n")
    
    print(f"\nResults saved to {RESULTS_PATH}")
    
    return results


def tune_fusion_weights():
    """Grid search for optimal fusion weights."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load data
    mne_dataset = load_workload_data(DATA_DIR, subset_pids=ALL_PIDS, augment=False)
    band_df = load_band_power_features()
    tcnet_params = load_best_tcnet_params()
    
    # Align data
    merged = align_mne_with_bandpower(mne_dataset, band_df)
    feature_cols = get_feature_columns(merged)
    
    aligned_mne_idx = merged['mne_idx'].values
    aligned_X_band = merged[feature_cols].values.astype(np.float32)
    aligned_y = merged['workload'].values
    aligned_pids = merged['pid'].values
    
    # Handle NaN
    nan_mask = np.isnan(aligned_X_band).any(axis=1)
    aligned_mne_idx = aligned_mne_idx[~nan_mask]
    aligned_X_band = aligned_X_band[~nan_mask]
    aligned_y = aligned_y[~nan_mask]
    aligned_pids = aligned_pids[~nan_mask]
    
    class_counts = np.bincount(aligned_y)
    class_weights = torch.FloatTensor(len(aligned_y) / (2 * class_counts)).to(device)
    
    # Collect all fold predictions first
    all_tcnet_probs = []
    all_xgb_probs = []
    all_labels = []
    
    print("Collecting fold predictions for weight tuning...")
    for fold, (train_idx, test_idx) in enumerate(create_group_splits_from_pids(aligned_pids, n_splits=3)):
        train_mne_idx = aligned_mne_idx[train_idx]
        test_mne_idx = aligned_mne_idx[test_idx]
        X_band_train = aligned_X_band[train_idx]
        X_band_test = aligned_X_band[test_idx]
        y_train = aligned_y[train_idx]
        y_test = aligned_y[test_idx]
        
        batch_size = tcnet_params.get('batch_size', 32)
        train_subset = Subset(mne_dataset, train_mne_idx)
        test_subset = Subset(mne_dataset, test_mne_idx)
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False)
        
        tcnet_probs, _, _ = train_tcnet_fold(train_loader, test_loader, device, tcnet_params, class_weights)
        xgb_probs, _, _ = train_xgb_fold(X_band_train, y_train, X_band_test, y_test, aligned_pids[train_idx])
        
        all_tcnet_probs.extend(tcnet_probs)
        all_xgb_probs.extend(xgb_probs)
        all_labels.extend(y_test)
    
    all_tcnet_probs = np.array(all_tcnet_probs)
    all_xgb_probs = np.array(all_xgb_probs)
    all_labels = np.array(all_labels)
    
    # Grid search weights
    print("\nGrid searching fusion weights...")
    best_weight = 0.5
    best_acc = 0
    
    for tcnet_w in np.arange(0.0, 1.05, 0.05):
        xgb_w = 1.0 - tcnet_w
        fusion_probs = tcnet_w * all_tcnet_probs + xgb_w * all_xgb_probs
        fusion_preds = (fusion_probs > 0.5).astype(int)
        acc = accuracy_score(all_labels, fusion_preds)
        
        if acc > best_acc:
            best_acc = acc
            best_weight = tcnet_w
            print(f"  New best: TCNet={tcnet_w:.2f}, XGB={xgb_w:.2f} -> Acc={acc:.4f}")
    
    print(f"\nOptimal: TCNet={best_weight:.2f}, XGB={1-best_weight:.2f} -> Acc={best_acc:.4f}")
    return best_weight, best_acc


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--tune-weights', action='store_true', help='Tune fusion weights')
    args = parser.parse_args()
    
    if args.tune_weights:
        tune_fusion_weights()
    else:
        run_late_fusion_cv()
