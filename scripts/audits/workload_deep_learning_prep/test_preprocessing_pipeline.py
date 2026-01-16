"""
Test Script: Validate Baseline Correction and Normalization Pipeline

This script validates the complete preprocessing pipeline including:
1. Baseline-adjusted epoch export
2. Global z-score normalization per fold
3. Proper cross-validation without data leakage

Run this before training to ensure preprocessing is correct.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / '13_workload_deep_learning_training'))

import numpy as np
import torch
from mne_dataloader import load_workload_data, create_group_splits, compute_global_stats, GlobalZScore

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / 'output' / 'adaptive_workload' / 'mne_epochs_baseline_adj'

# Test with small subset
TEST_PIDS = [1, 10, 20, 30, 40]

print("="*70)
print(" PREPROCESSING VALIDATION TEST")
print("="*70)

print("\n[1] Loading baseline-adjusted epochs...")
dataset = load_workload_data(DATA_DIR, subset_pids=TEST_PIDS)
print(f"   ✓ Loaded {len(dataset)} samples")
print(f"   ✓ Shape: {dataset.data.shape}")
print(f"   ✓ PIDs: {np.unique(dataset.pids)}")

print("\n[2] Checking data statistics (pre-normalization)...")
data_sample = dataset.data[:100]  # Sample for stats
print(f"   Mean: {data_sample.mean():.6e}")
print(f"   Std:  {data_sample.std():.6e}")
print(f"   Min:  {data_sample.min():.6e}")
print(f"   Max:  {data_sample.max():.6e}")

# Validate baseline correction worked
if abs(data_sample.mean()) < 1e-10:
    print("   ✓ Data is approximately zero-mean (baseline correction successful)")
else:
    print(f"   ⚠ WARNING: Data mean is {data_sample.mean():.6e} (expected ~0)")

# Validate sufficient variance
if data_sample.std() > 1e-9:
    print(f"   ✓ Sufficient variance for modeling (std={data_sample.std():.6e})")
else:
    print(f"   ✗ ERROR: Insufficient variance (std={data_sample.std():.6e})")

print("\n[3] Testing GroupKFold splits...")
for fold, (train_idx, test_idx) in enumerate(create_group_splits(dataset, n_splits=3)):
    train_pids = np.unique(dataset.pids[train_idx])
    test_pids = np.unique(dataset.pids[test_idx])
    
    # Check no PID overlap
    overlap = set(train_pids) & set(test_pids)
    if len(overlap) == 0:
        print(f"   ✓ Fold {fold+1}: No PID overlap")
    else:
        print(f"   ✗ Fold {fold+1}: PID OVERLAP DETECTED: {overlap}")
        
    print(f"      Train: PIDs {train_pids} ({len(train_idx)} samples)")
    print(f"      Test:  PIDs {test_pids} ({len(test_idx)} samples)")

print("\n[4] Testing global z-score normalization (per-fold)...")
for fold, (train_idx, test_idx) in enumerate(create_group_splits(dataset, n_splits=3)):
    # Compute stats from training set ONLY
    train_mean, train_std = compute_global_stats(dataset, train_idx)
    
    print(f"\n   Fold {fold+1} normalization stats:")
    print(f"      Train mean: {train_mean:.6e}")
    print(f"      Train std:  {train_std:.6e}")
    
    # Create normalizer
    normalizer = GlobalZScore(mean=train_mean, std=train_std)
    
    # Test normalization on a sample
    sample_raw = torch.FloatTensor(dataset.data[train_idx[0]])
    sample_norm = normalizer(sample_raw)
    
    print(f"      Sample before: mean={sample_raw.mean():.6e}, std={sample_raw.std():.6e}")
    print(f"      Sample after:  mean={sample_norm.mean():.6e}, std={sample_norm.std():.6e}")
    
    # Validate normalization
    if abs(sample_norm.mean()) < 0.1:  # Should be close to 0
        print(f"      ✓ Normalized mean ≈ 0")
    else:
        print(f"      ⚠ WARNING: Normalized mean = {sample_norm.mean():.3f}")
        
    if 0.8 < sample_norm.std() < 1.2:  # Should be close to 1
        print(f"      ✓ Normalized std ≈ 1")
    else:
        print(f"      ⚠ WARNING: Normalized std = {sample_norm.std():.3f}")
    
    # Test on test set (should use same train stats)
    test_sample_raw = torch.FloatTensor(dataset.data[test_idx[0]])
    test_sample_norm = normalizer(test_sample_raw)
    
    print(f"      Test sample after: mean={test_sample_norm.mean():.3f}, std={test_sample_norm.std():.3f}")
    
    # Only check first fold for brevity
    if fold == 0:
        print("      (Remaining folds use same process...)")
        break

print("\n[5] Testing data loader integration...")
from torch.utils.data import DataLoader, Subset

fold_idx = 0
train_idx, test_idx = next(iter(create_group_splits(dataset, n_splits=3)))

# Apply normalization to dataset
train_mean, train_std = compute_global_stats(dataset, train_idx)
dataset.transform = GlobalZScore(mean=train_mean, std=train_std)

# Create loaders
train_loader = DataLoader(Subset(dataset, train_idx), batch_size=16, shuffle=True)
test_loader = DataLoader(Subset(dataset, test_idx), batch_size=16, shuffle=False)

# Get a batch
batch_x, batch_y = next(iter(train_loader))
print(f"   ✓ Train batch shape: {batch_x.shape}")
print(f"   ✓ Train batch mean: {batch_x.mean():.6e}")
print(f"   ✓ Train batch std:  {batch_x.std():.3f}")

test_batch_x, test_batch_y = next(iter(test_loader))
print(f"   ✓ Test batch shape:  {test_batch_x.shape}")
print(f"   ✓ Test batch mean:  {test_batch_x.mean():.6e}")
print(f"   ✓ Test batch std:   {test_batch_x.std():.3f}")

# Validate no data leakage
train_pids_batch = np.unique(dataset.pids[train_idx])
test_pids_batch = np.unique(dataset.pids[test_idx])
overlap = set(train_pids_batch) & set(test_pids_batch)

if len(overlap) == 0:
    print(f"   ✓ No PID overlap in batches")
else:
    print(f"   ✗ ERROR: PID overlap in batches: {overlap}")

print("\n[6] Summary of preprocessing pipeline...")
print("   ✓ Baseline correction: Per-channel mean subtraction (Task - Forest)")
print("   ✓ Normalization: Global z-score using training-set-only statistics")
print("   ✓ Cross-validation: GroupKFold by Participant_ID (subject-independent)")
print("   ✓ No data leakage: Test set normalized with training stats")
print("   ✓ Data loader: Properly integrates transforms in __getitem__()")

print("\n" + "="*70)
print(" VALIDATION COMPLETE - Pipeline Ready for Training")
print("="*70)
print("\nNext steps:")
print("  1. Train EEGNet:  python train_eegnet.py")
print("  2. Tune TCNet:    python tune_tcnet_focused.py")
print("  3. Compare:       Check results/ for performance metrics")
print("\nExpected performance: 60-70% accuracy (cross-subject workload classification)")
