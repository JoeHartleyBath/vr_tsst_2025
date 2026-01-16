"""Focused EEG-TCNet Hyperparameter Tuning (VALID safe-CV)

This script is the Optuna tuner for EEG-TCNet workload classification.

Key properties (must stay aligned with train_tcnet.py safe-CV):
- NO AUGMENTATION
- NO ATTENTION
- Fold-safe normalization: compute per-channel mean/std using TRAIN indices only,
  across all epochs and timepoints; apply that normalization to BOTH train and test.
- Deterministic folds: GroupKFold/StratifiedGroupKFold splits precomputed once and reused.
- Optimize mean macro-F1 across folds (primary). Accuracy logged as secondary.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
from typing import Any

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import time
import json
from datetime import datetime
import math
import torch.nn.functional as F

from mne_dataloader import load_workload_data, create_group_splits
from train_tcnet import EEGTCNet, apply_baseline_adjustment_inplace, BASELINE_CACHE_PATH

# ==============================================================================
# CONFIGURATION
# ==============================================================================
DATA_DIR = r'C:\vr_tsst_2025\output\adaptive_workload\mne_epochs'
RESULTS_DIR = r'C:\vr_tsst_2025\results'

DEFAULT_STUDY_NAME_PREFIX = 'eeg_tcnet_staged_tuning'
DEFAULT_DB_FILE = os.path.join(RESULTS_DIR, 'optuna_tcnet_staged.db')

# SMOKE mode: set env var SMOKE=1 for a quick execution+logging validation.
SMOKE = os.getenv('SMOKE', '0').strip() == '1'

# All 44 EEG-valid participants (P44 excluded - no valid windows)
ALL_PIDS = [1, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
            24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 
            42, 43, 45, 47, 48]

ACTIVE_PIDS = [1, 3] if SMOKE else ALL_PIDS

# Fixed parameters
CHANS = 128
TIME_POINTS = 1250
CLASSES = 2

# Baseline adjustment (match train_tcnet.py baseline sweep winner)
BASELINE_ADJUST = 'zscore'  # 'none'|'mean'|'divstd'|'zscore'

# Safety / determinism
SEED = 1337

# CLI-controlled toggles (defaults; overridden in main())
LABEL_SMOOTHING_ENABLED = True
LOCK_BATCH_SIZE = False

# Tuning settings
TIME_LIMIT_HOURS = 12
TIME_LIMIT_SECONDS = TIME_LIMIT_HOURS * 3600
MAX_EPOCHS = 60  # Slightly more epochs
EARLY_STOP_PATIENCE = 12
N_FOLDS = 3

if SMOKE:
    MAX_EPOCHS = 5
    N_FOLDS = 2

# FIXED: These hurt performance - don't tune them
USE_SE_ATTENTION = False
USE_TEMPORAL_ATTENTION = False
USE_AUGMENTATION = False

# ==============================================================================
# GLOBAL DATA LOADING (load once, reuse across trials)
# ==============================================================================
DATASET = None
SPLITS = None
DATASET_LOAD_COUNT = 0

def get_dataset():
    global DATASET
    global DATASET_LOAD_COUNT
    if DATASET is None:
        DATASET_LOAD_COUNT += 1
        DATASET = load_workload_data(
            DATA_DIR,
            subset_pids=ACTIVE_PIDS,
            augment=False,
            target_chans=CHANS,
            target_time_points=TIME_POINTS,
        )

        # Apply the same baseline adjustment used by train_tcnet.py (in-place).
        # This is separate from fold-safe CV normalization.
        _ = apply_baseline_adjustment_inplace(
            DATASET,
            baseline_adjust=str(BASELINE_ADJUST),
            baseline_cache_path=str(BASELINE_CACHE_PATH),
            debug_baseline=False,
        )
        # Hard shape assertion: dataset stores (N,1,C,T)
        assert hasattr(DATASET, 'data'), "Dataset must expose .data"
        assert DATASET.data.ndim == 4, f"Expected dataset.data to be 4D (N,1,C,T), got {DATASET.data.ndim}D"
        assert DATASET.data.shape[1] == 1, f"Expected singleton conv dim: got {DATASET.data.shape}"
        assert DATASET.data.shape[2] == CHANS, f"Expected CHANS={CHANS}, got {DATASET.data.shape[2]}"
        assert DATASET.data.shape[3] == TIME_POINTS, f"Expected TIME_POINTS={TIME_POINTS}, got {DATASET.data.shape[3]}"
        print(
            f"Loaded {len(DATASET)} samples | shape={tuple(DATASET.data.shape)} | "
            f"SMOKE={SMOKE} | DATASET_LOAD_COUNT={DATASET_LOAD_COUNT}"
        )
    return DATASET


def get_splits():
    """Precompute folds once and reuse the exact same splits for every trial."""
    global SPLITS
    if SPLITS is None:
        dataset = get_dataset()
        SPLITS = list(create_group_splits(dataset, n_splits=N_FOLDS, seed=SEED))
        if len(SPLITS) != N_FOLDS:
            raise RuntimeError(f"Expected {N_FOLDS} splits, got {len(SPLITS)}")
        # Minimal sanity: disjoint within each fold.
        for fold, (tr, te) in enumerate(SPLITS):
            tr_set = set(map(int, tr))
            te_set = set(map(int, te))
            if tr_set & te_set:
                raise RuntimeError(f"Fold {fold} train/test overlap")
        print(f"Precomputed {len(SPLITS)} deterministic folds (seed={SEED})")
    return SPLITS


# ==============================================================================
# TRAINING HELPERS (copied from train_tcnet.py to ensure identical safe-CV)
# ==============================================================================


def set_reproducibility(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True)
    except Exception as e:
        print(f"WARNING: torch.use_deterministic_algorithms(True) failed: {e}")


def compute_train_channel_stats(dataset, train_indices, batch_size=64):
    """Compute per-channel mean/std on TRAIN only across epochs and timepoints.

    Returns tensors shaped (1, CHANS, 1) on CPU.
    """
    loader = DataLoader(Subset(dataset, train_indices), batch_size=batch_size, shuffle=False)

    total_count = 0
    sum_c = None
    sumsq_c = None
    nan_found = False

    for batch_x, _ in loader:
        # batch_x: (B, 1, C, T)
        if torch.isnan(batch_x).any() or torch.isinf(batch_x).any():
            nan_found = True
        x = batch_x.squeeze(1)  # (B, C, T)
        b, c, t = x.shape
        x = x.reshape(b, c, t)
        if sum_c is None:
            sum_c = x.sum(dim=(0, 2), dtype=torch.float64)  # (C,)
            sumsq_c = (x.double() ** 2).sum(dim=(0, 2))
        else:
            sum_c += x.sum(dim=(0, 2), dtype=torch.float64)
            sumsq_c += (x.double() ** 2).sum(dim=(0, 2))
        total_count += b * t

    mean = (sum_c / float(total_count)).float().view(1, -1, 1)
    var = (sumsq_c / float(total_count) - (mean.view(-1) ** 2)).clamp_min(0.0)
    std = torch.sqrt(var).float().view(1, -1, 1)
    std = std.clamp_min(1e-12)
    return mean, std, nan_found


def normalize_batch(batch_x, mean_c, std_c):
    # batch_x: (B, 1, C, T); mean_c/std_c: (1, C, 1)
    assert batch_x.ndim == 4, f"batch_x must be 4D (B,1,C,T), got {tuple(batch_x.shape)}"
    assert batch_x.shape[1] == 1, f"batch_x must have singleton dim=1, got {tuple(batch_x.shape)}"
    assert batch_x.shape[2] == CHANS, f"batch_x channels mismatch: got {batch_x.shape[2]}, expected {CHANS}"
    assert batch_x.shape[3] == TIME_POINTS, f"batch_x time mismatch: got {batch_x.shape[3]}, expected {TIME_POINTS}"
    assert tuple(mean_c.shape) == (1, CHANS, 1), f"mean_c shape must be (1,{CHANS},1), got {tuple(mean_c.shape)}"
    assert tuple(std_c.shape) == (1, CHANS, 1), f"std_c shape must be (1,{CHANS},1), got {tuple(std_c.shape)}"
    mean = mean_c.unsqueeze(1)  # (1, 1, C, 1)
    std = std_c.unsqueeze(1)    # (1, 1, C, 1)
    return (batch_x - mean) / std


def evaluate_normalized(model, loader, criterion, device, mean_c, std_c, max_batches=None):
    """Evaluate model with fold-safe normalization applied."""
    model.eval()
    total_loss = 0.0
    n_batches = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for bi, (batch_x, batch_y) in enumerate(loader):
            if max_batches is not None and bi >= max_batches:
                break
            assert batch_y.dtype == torch.long, f"batch_y dtype must be torch.long, got {batch_y.dtype}"
            batch_x = batch_x.to(device)
            batch_x = normalize_batch(batch_x, mean_c, std_c)
            batch_y = batch_y.to(device)
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            total_loss += float(loss.item())
            n_batches += 1
            preds = torch.argmax(outputs, dim=1).detach().cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(batch_y.detach().cpu().numpy())

    mean_loss = total_loss / max(1, n_batches)
    acc = float(accuracy_score(all_labels, all_preds)) if len(all_labels) else 0.0
    f1_macro = float(f1_score(all_labels, all_preds, average='macro', zero_division=0)) if len(all_labels) else 0.0
    return mean_loss, acc, f1_macro


def _assert_first_batch(batch_x: torch.Tensor, batch_y: torch.Tensor, *, fold: int, split: str):
    assert isinstance(batch_x, torch.Tensor) and isinstance(batch_y, torch.Tensor)
    assert batch_x.ndim == 4, f"Fold {fold} {split}: batch_x must be (B,1,C,T); got {tuple(batch_x.shape)}"
    assert batch_x.shape[1] == 1, f"Fold {fold} {split}: expected batch_x[:,1]==1; got {tuple(batch_x.shape)}"
    assert batch_x.shape[2] == CHANS, f"Fold {fold} {split}: expected CHANS={CHANS}; got {batch_x.shape[2]}"
    assert batch_x.shape[3] == TIME_POINTS, f"Fold {fold} {split}: expected TIME_POINTS={TIME_POINTS}; got {batch_x.shape[3]}"
    assert batch_y.dtype == torch.long, f"Fold {fold} {split}: labels must be torch.long; got {batch_y.dtype}"


def train_epoch_normalized(model, loader, criterion, optimizer, device, mean_c, std_c, *, fold: int):
    model.train()
    total_loss = 0.0
    n_batches = 0
    all_preds = []
    all_labels = []
    asserted = False

    for batch_x, batch_y in loader:
        if not asserted:
            _assert_first_batch(batch_x, batch_y, fold=fold, split='train')
            asserted = True

        batch_x = batch_x.to(device)
        batch_x = normalize_batch(batch_x, mean_c, std_c)
        batch_y = batch_y.to(device)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item())
        n_batches += 1
        all_preds.extend(torch.argmax(outputs, dim=1).detach().cpu().numpy())
        all_labels.extend(batch_y.detach().cpu().numpy())

    mean_loss = total_loss / max(1, n_batches)
    acc = float(accuracy_score(all_labels, all_preds)) if len(all_labels) else 0.0
    return mean_loss, acc


class SmoothedCrossEntropy(nn.Module):
    def __init__(self, *, weight: torch.Tensor | None, label_smoothing: float, num_classes: int):
        super().__init__()
        self.register_buffer('weight', weight if weight is not None else None)
        self.label_smoothing = float(label_smoothing)
        self.num_classes = int(num_classes)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.label_smoothing <= 0.0:
            return F.cross_entropy(logits, target, weight=self.weight)
        log_probs = F.log_softmax(logits, dim=1)
        n = int(self.num_classes)
        if n <= 1:
            return (-log_probs.gather(1, target.unsqueeze(1))).mean()
        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs)
            true_dist.fill_(self.label_smoothing / float(n - 1))
            true_dist.scatter_(1, target.unsqueeze(1), 1.0 - self.label_smoothing)
        if self.weight is None:
            return (-true_dist * log_probs).sum(dim=1).mean()
        # NOTE: This is a reasonable fallback; PyTorch's built-in label_smoothing is preferred.
        w = self.weight.unsqueeze(0)  # (1, C)
        return ((-true_dist * log_probs) * w).sum(dim=1).mean()


def make_criterion(*, class_weights: torch.Tensor, label_smoothing: float | None) -> nn.Module:
    ls = float(label_smoothing) if label_smoothing is not None else 0.0
    if ls <= 0.0:
        return nn.CrossEntropyLoss(weight=class_weights)
    try:
        return nn.CrossEntropyLoss(weight=class_weights, label_smoothing=ls)
    except TypeError:
        return SmoothedCrossEntropy(weight=class_weights, label_smoothing=ls, num_classes=CLASSES)


def train_one_fold(
    *,
    trial: optuna.Trial,
    fold: int,
    dataset,
    train_idx,
    test_idx,
    device: torch.device,
    hparams: dict[str, Any],
    trial_seed: int,
) -> dict[str, Any]:
    """Train a single fold with fold-safe normalization.

    Early stopping is based on macro-F1 (primary), accuracy tie-break (secondary).
    Pruning reports fold-0 best macro-F1 per epoch.
    """
    batch_size = int(hparams['batch_size'])
    learning_rate = float(hparams['learning_rate'])
    weight_decay = float(hparams.get('weight_decay', 0.0))
    label_smoothing = float(hparams.get('label_smoothing', 0.0))

    train_subset = Subset(dataset, train_idx)
    test_subset = Subset(dataset, test_idx)

    mean_c, std_c, nan_found = compute_train_channel_stats(dataset, train_idx, batch_size=64)
    mean_c = mean_c.to(device)
    std_c = std_c.to(device)
    if nan_found:
        print(f"  [WARN] Fold {fold}: NaN/Inf found while computing train stats")

    generator = torch.Generator()
    generator.manual_seed(trial_seed)

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        generator=generator,
    )
    test_loader = DataLoader(
        test_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    # Model params
    model = EEGTCNet(
        chans=CHANS,
        classes=CLASSES,
        time_points=TIME_POINTS,
        F1=int(hparams['F1']),
        D=int(hparams['D']),
        kernLength=int(hparams['kernLength']),
        dropout_eeg=float(hparams['dropout_eeg']),
        tcn_filters=int(hparams['tcn_filters']),
        tcn_kernel=int(hparams['tcn_kernel']),
        tcn_depth=int(hparams['tcn_depth']),
        dropout_tcn=float(hparams['dropout_tcn']),
        use_se_attention=USE_SE_ATTENTION,
        use_temporal_attention=USE_TEMPORAL_ATTENTION,
    ).to(device)

    class_weights = dataset.get_class_weights().to(device)
    criterion = make_criterion(class_weights=class_weights, label_smoothing=label_smoothing)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    best_f1 = 0.0
    best_acc = 0.0
    best_epoch = -1
    best_state = None
    patience_counter = 0
    _test_asserted = False

    for epoch in range(MAX_EPOCHS):
        train_loss, train_acc = train_epoch_normalized(
            model, train_loader, criterion, optimizer, device, mean_c, std_c, fold=fold
        )

        if not _test_asserted:
            bx, by = next(iter(test_loader))
            _assert_first_batch(bx, by, fold=fold, split='test')
            _test_asserted = True

        val_loss, val_acc, val_f1 = evaluate_normalized(model, test_loader, criterion, device, mean_c, std_c)

        improved = (val_f1 > best_f1) or (abs(val_f1 - best_f1) < 1e-10 and val_acc > best_acc)
        if improved:
            best_f1 = float(val_f1)
            best_acc = float(val_acc)
            best_epoch = int(epoch)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOP_PATIENCE:
                break

        if fold == 0:
            # Pruning signal should reflect current epoch performance, not best-so-far.
            trial.report(float(val_f1), epoch)
            if trial.should_prune():
                trial.set_user_attr('pruned_at_epoch', int(epoch))
                trial.set_user_attr('fold0_best_f1', float(best_f1))
                trial.set_user_attr('fold0_best_acc', float(best_acc))
                # Compact partial fold summaries (avoid DB bloat)
                trial.set_user_attr('fold_best_f1', [float(best_f1)])
                trial.set_user_attr('fold_best_acc', [float(best_acc)])
                trial.set_user_attr('fold_best_epoch', [int(best_epoch)])
                raise optuna.TrialPruned()

    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    final_loss, final_acc, final_f1 = evaluate_normalized(model, test_loader, criterion, device, mean_c, std_c)

    out = {
        'fold': int(fold),
        'best_epoch': int(best_epoch),
        'best_macro_f1': float(best_f1),
        'best_acc': float(best_acc),
        'final_macro_f1': float(final_f1),
        'final_acc': float(final_acc),
    }

    del model
    torch.cuda.empty_cache()
    return out


def _clip(x: float, lo: float, hi: float) -> float:
    return float(min(max(float(x), float(lo)), float(hi)))


def _log_clip_band(center: float, *, div: float, mul: float, lo: float, hi: float) -> tuple[float, float]:
    c = float(center)
    low = _clip(c / float(div), lo, hi)
    high = _clip(c * float(mul), lo, hi)
    return (min(low, high), max(low, high))


def objective_stage1(trial: optuna.Trial) -> float:
    """Stage 1: broad-but-sane search (architecture + regularisation + optimiser)."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    trial_seed = int(SEED + trial.number)
    set_reproducibility(trial_seed)

    hparams: dict[str, Any] = {
        # Architecture
        'F1': trial.suggest_categorical('F1', [4, 8, 16]),
        'D': trial.suggest_categorical('D', [1, 2]),
        'kernLength': trial.suggest_categorical('kernLength', [16, 32, 64, 128]),
        'dropout_eeg': trial.suggest_float('dropout_eeg', 0.05, 0.45),
        'tcn_filters': trial.suggest_categorical('tcn_filters', [8, 12, 16, 24]),
        'tcn_kernel': trial.suggest_categorical('tcn_kernel', [3, 4, 6]),
        'tcn_depth': trial.suggest_int('tcn_depth', 1, 3),
        'dropout_tcn': trial.suggest_float('dropout_tcn', 0.10, 0.50),
        # Optimiser / regularisation
        'learning_rate': trial.suggest_float('learning_rate', 1e-4, 3e-3, log=True),
        'weight_decay': trial.suggest_float('weight_decay', 1e-8, 5e-2, log=True),
        'batch_size': trial.suggest_categorical('batch_size', [16, 32]),
    }
    # Optional label smoothing (can be disabled via CLI to avoid extra param/bloat)
    if LABEL_SMOOTHING_ENABLED:
        hparams['label_smoothing'] = trial.suggest_float('label_smoothing', 0.0, 0.1)
    else:
        hparams['label_smoothing'] = 0.0

    dataset = get_dataset()
    splits = get_splits()

    trial.set_user_attr('stage', 'stage1')
    trial.set_user_attr('seed', trial_seed)
    trial.set_user_attr('smoke', bool(SMOKE))
    trial.set_user_attr('time_points', int(TIME_POINTS))

    fold_results: list[dict[str, Any]] = []
    fold_best_f1: list[float] = []
    fold_best_acc: list[float] = []
    fold_best_epoch: list[int] = []
    for fold, (train_idx, test_idx) in enumerate(splits):
        fr = train_one_fold(
            trial=trial,
            fold=fold,
            dataset=dataset,
            train_idx=train_idx,
            test_idx=test_idx,
            device=device,
            hparams=hparams,
            trial_seed=trial_seed,
        )
        fold_results.append(fr)
        fold_best_f1.append(float(fr['best_macro_f1']))
        fold_best_acc.append(float(fr['best_acc']))
        fold_best_epoch.append(int(fr['best_epoch']))

    # Compact fold summaries only (avoid DB bloat)
    trial.set_user_attr('fold_best_f1', fold_best_f1)
    trial.set_user_attr('fold_best_acc', fold_best_acc)
    trial.set_user_attr('fold_best_epoch', fold_best_epoch)

    mean_macro_f1 = float(np.mean([fr['best_macro_f1'] for fr in fold_results])) if fold_results else 0.0
    mean_acc = float(np.mean([fr['best_acc'] for fr in fold_results])) if fold_results else 0.0
    trial.set_user_attr('mean_macro_f1', mean_macro_f1)
    trial.set_user_attr('mean_acc', mean_acc)
    return mean_macro_f1


def objective_stage2(trial: optuna.Trial, *, best_arch: dict[str, Any], center: dict[str, Any]) -> float:
    """Stage 2: narrowed search around best regularisation/optimiser for a fixed architecture."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    trial_seed = int(SEED + 100000 + trial.number)
    set_reproducibility(trial_seed)

    # Locked architecture
    hparams: dict[str, Any] = dict(best_arch)

    # Narrow bands around stage1 best
    de_center = float(center.get('dropout_eeg', 0.25))
    dt_center = float(center.get('dropout_tcn', 0.25))
    lr_center = float(center.get('learning_rate', 1e-3))
    wd_center = float(center.get('weight_decay', 1e-4))
    ls_center = float(center.get('label_smoothing', 0.0))

    hparams['dropout_eeg'] = trial.suggest_float(
        'dropout_eeg',
        _clip(de_center - 0.08, 0.05, 0.50),
        _clip(de_center + 0.08, 0.05, 0.50),
    )
    hparams['dropout_tcn'] = trial.suggest_float(
        'dropout_tcn',
        _clip(dt_center - 0.08, 0.05, 0.60),
        _clip(dt_center + 0.08, 0.05, 0.60),
    )

    lr_lo, lr_hi = _log_clip_band(lr_center, div=5.0, mul=5.0, lo=1e-5, hi=5e-3)
    wd_lo, wd_hi = _log_clip_band(wd_center, div=20.0, mul=20.0, lo=1e-10, hi=5e-2)
    hparams['learning_rate'] = trial.suggest_float('learning_rate', lr_lo, lr_hi, log=True)
    hparams['weight_decay'] = trial.suggest_float('weight_decay', wd_lo, wd_hi, log=True)

    # Batch size: keep flexible (or lock if you prefer)
    if LOCK_BATCH_SIZE:
        hparams['batch_size'] = int(center.get('batch_size', 32))
    else:
        hparams['batch_size'] = trial.suggest_categorical('batch_size', [16, 32])

    # Optional label smoothing narrow band
    if LABEL_SMOOTHING_ENABLED:
        hparams['label_smoothing'] = trial.suggest_float(
            'label_smoothing',
            _clip(ls_center - 0.05, 0.0, 0.2),
            _clip(ls_center + 0.05, 0.0, 0.2),
        )
    else:
        hparams['label_smoothing'] = 0.0

    dataset = get_dataset()
    splits = get_splits()

    trial.set_user_attr('stage', 'stage2')
    trial.set_user_attr('seed', trial_seed)
    trial.set_user_attr('smoke', bool(SMOKE))
    trial.set_user_attr('time_points', int(TIME_POINTS))
    trial.set_user_attr('best_arch', dict(best_arch))
    trial.set_user_attr('center', dict(center))

    fold_results: list[dict[str, Any]] = []
    fold_best_f1: list[float] = []
    fold_best_acc: list[float] = []
    fold_best_epoch: list[int] = []
    for fold, (train_idx, test_idx) in enumerate(splits):
        fr = train_one_fold(
            trial=trial,
            fold=fold,
            dataset=dataset,
            train_idx=train_idx,
            test_idx=test_idx,
            device=device,
            hparams=hparams,
            trial_seed=trial_seed,
        )
        fold_results.append(fr)
        fold_best_f1.append(float(fr['best_macro_f1']))
        fold_best_acc.append(float(fr['best_acc']))
        fold_best_epoch.append(int(fr['best_epoch']))

    # Compact fold summaries only (avoid DB bloat)
    trial.set_user_attr('fold_best_f1', fold_best_f1)
    trial.set_user_attr('fold_best_acc', fold_best_acc)
    trial.set_user_attr('fold_best_epoch', fold_best_epoch)

    mean_macro_f1 = float(np.mean([fr['best_macro_f1'] for fr in fold_results])) if fold_results else 0.0
    mean_acc = float(np.mean([fr['best_acc'] for fr in fold_results])) if fold_results else 0.0
    trial.set_user_attr('mean_macro_f1', mean_macro_f1)
    trial.set_user_attr('mean_acc', mean_acc)
    return mean_macro_f1


class TimeoutCallback:
    """Stop study after time limit."""
    
    def __init__(self, time_limit_seconds):
        self.time_limit = time_limit_seconds
        self.start_time = time.time()
        
    def __call__(self, study, trial):
        elapsed = time.time() - self.start_time
        remaining = self.time_limit - elapsed
        
        if remaining < 0:
            print(f"\nTime limit reached ({self.time_limit/3600:.1f} hours)")
            study.stop()
        elif trial.number % 10 == 0:
            print(f"  Time remaining: {remaining/3600:.1f} hours")


def _normalize_storage(db_path: str) -> str:
    s = str(db_path).strip()
    if s.startswith('sqlite:///'):
        return s
    # Interpret as a filesystem path
    p = os.path.abspath(s)
    p_norm = p.replace('\\', '/')
    return f"sqlite:///{p_norm}"


def _trial_row(trial: optuna.trial.FrozenTrial) -> dict[str, Any]:
    return {
        'number': int(trial.number),
        'value': float(trial.value) if trial.value is not None else None,
        'state': str(trial.state),
        'params': dict(trial.params),
    }


def _select_top_k(study: optuna.Study, top_k: int) -> list[optuna.trial.FrozenTrial]:
    completed = [t for t in study.trials if t.value is not None and str(t.state) == 'TrialState.COMPLETE']
    completed.sort(key=lambda t: float(t.value), reverse=True)
    return completed[: int(top_k)]


def _best_arch_from_trial(trial: optuna.trial.FrozenTrial) -> dict[str, Any]:
    return {
        'F1': int(trial.params['F1']),
        'D': int(trial.params['D']),
        'kernLength': int(trial.params['kernLength']),
        'tcn_filters': int(trial.params['tcn_filters']),
        'tcn_kernel': int(trial.params['tcn_kernel']),
        'tcn_depth': int(trial.params['tcn_depth']),
    }


def _center_from_trial(trial: optuna.trial.FrozenTrial) -> dict[str, Any]:
    return {
        'dropout_eeg': float(trial.params.get('dropout_eeg', 0.25)),
        'dropout_tcn': float(trial.params.get('dropout_tcn', 0.25)),
        'learning_rate': float(trial.params.get('learning_rate', 1e-3)),
        'weight_decay': float(trial.params.get('weight_decay', 1e-4)),
        'label_smoothing': float(trial.params.get('label_smoothing', 0.0)),
        'batch_size': int(trial.params.get('batch_size', 32)),
    }


def save_stage_artifacts(
    *,
    stage: str,
    study: optuna.Study,
    prefix: str,
    elapsed_s: float,
    top_k: int,
    best_arch: dict[str, Any] | None = None,
    center: dict[str, Any] | None = None,
):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    top_trials = _select_top_k(study, top_k)
    top_rows = []
    for t in top_trials:
        row = {
            'trial_number': int(t.number),
            'value_macro_f1': float(t.value),
            'params': {k: t.params.get(k) for k in sorted(t.params.keys())},
        }
        top_rows.append(row)

    summary = {
        'stage': stage,
        'study_name': str(study.study_name),
        'storage': str(study._storage),
        'best_value_macro_f1': float(study.best_value) if study.best_value is not None else None,
        'best_params': dict(study.best_params) if study.best_params is not None else None,
        'best_arch': dict(best_arch) if best_arch is not None else None,
        'center': dict(center) if center is not None else None,
        'n_trials': int(len(study.trials)),
        'elapsed_s': float(elapsed_s),
        'folds': int(N_FOLDS),
        'time_points': int(TIME_POINTS),
        'smoke': bool(SMOKE),
        'top_k': int(top_k),
        'top_k_trials': top_rows,
    }

    out_path = os.path.join(RESULTS_DIR, f"{prefix}_{stage}_summary.json")
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    print(f"Saved {stage} summary -> {out_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Staged Optuna tuning for EEG-TCNet (safe-CV, macro-F1 objective).')
    p.add_argument('--stage', choices=['1', '2', 'both'], default='both')
    p.add_argument('--stage1_trials', type=int, default=100)
    p.add_argument('--stage2_trials', type=int, default=50)
    p.add_argument('--top_k', type=int, default=10)
    p.add_argument('--seed', type=int, default=1337)
    p.add_argument('--db_path', type=str, default=DEFAULT_DB_FILE)
    p.add_argument('--study_name_prefix', type=str, default=DEFAULT_STUDY_NAME_PREFIX)
    p.add_argument('--stage1_hours', type=float, default=8.0)
    p.add_argument('--stage2_hours', type=float, default=4.0)
    p.add_argument('--disable_label_smoothing', action='store_true')
    p.add_argument('--lock_batch_size', action='store_true')
    p.add_argument('--stage2_arch_k', type=int, default=3)
    p.add_argument('--stage2_trials_per_arch', type=int, default=25)
    return p.parse_args()


def run_stage1(*, storage: str, prefix: str, n_trials: int, top_k: int, seed: int, stage1_hours: float) -> optuna.Study:
    sampler = TPESampler(seed=int(seed), n_startup_trials=15)
    pruner = MedianPruner(n_startup_trials=15, n_warmup_steps=15, interval_steps=5)

    study_name = f"{prefix}_stage1"
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction='maximize',
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True,
    )

    trials_jsonl_path = os.path.join(RESULTS_DIR, f"optuna_trials_tcnet_{prefix}_stage1.jsonl")

    class TrialJSONLLogger:
        def __init__(self, path: str):
            self.path = path

        def __call__(self, study, trial):
            record = {
                'ts': datetime.utcnow().isoformat() + 'Z',
                'study_name': str(study.study_name),
                'trial_number': int(trial.number),
                'state': str(trial.state),
                'value_macro_f1': float(trial.value) if trial.value is not None else None,
                'params': dict(trial.params),
                'stage': trial.user_attrs.get('stage', None),
                'fold_best_f1': trial.user_attrs.get('fold_best_f1', None),
                'fold_best_acc': trial.user_attrs.get('fold_best_acc', None),
                'fold_best_epoch': trial.user_attrs.get('fold_best_epoch', None),
            }
            with open(self.path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(record) + "\n")

    start = time.time()
    timeout_s = None
    if stage1_hours is not None and float(stage1_hours) > 0:
        timeout_s = float(stage1_hours) * 3600.0

    study.optimize(
        objective_stage1,
        n_trials=int(n_trials),
        timeout=timeout_s,
        callbacks=[TrialJSONLLogger(trials_jsonl_path)],
        show_progress_bar=True,
        gc_after_trial=True,
    )
    elapsed = time.time() - start

    top_trials = _select_top_k(study, top_k)
    best_arch = _best_arch_from_trial(study.best_trial)
    center = _center_from_trial(study.best_trial)
    save_stage_artifacts(stage='stage1', study=study, prefix=prefix, elapsed_s=elapsed, top_k=top_k, best_arch=best_arch, center=center)
    print(f"Stage1 JSONL log -> {trials_jsonl_path}")
    if top_trials:
        print(f"Stage1 top-{len(top_trials)} best macro-F1: {[round(float(t.value), 4) for t in top_trials]}")
    return study


def load_stage1_best(*, storage: str, prefix: str, top_k: int) -> tuple[dict[str, Any], dict[str, Any]]:
    study_name = f"{prefix}_stage1"
    study = optuna.load_study(study_name=study_name, storage=storage)
    _ = _select_top_k(study, top_k)
    return _best_arch_from_trial(study.best_trial), _center_from_trial(study.best_trial)


def _unique_architectures_from_top_trials(study: optuna.Study, top_k: int) -> list[dict[str, Any]]:
    """Return unique architectures from the top-K completed trials.

    Also capture per-architecture center values from the source trial.
    """
    top_trials = _select_top_k(study, top_k)
    seen = set()
    items: list[dict[str, Any]] = []
    for t in top_trials:
        arch = _best_arch_from_trial(t)
        key = (
            int(arch['F1']),
            int(arch['D']),
            int(arch['kernLength']),
            int(arch['tcn_filters']),
            int(arch['tcn_kernel']),
            int(arch['tcn_depth']),
        )
        if key in seen:
            continue
        seen.add(key)
        items.append(
            {
                'arch': dict(arch),
                'center': _center_from_trial(t),
                'source_trial_number': int(t.number),
                'source_value': float(t.value) if t.value is not None else None,
            }
        )
    return items


def run_stage2(
    *,
    storage: str,
    prefix: str,
    n_trials: int,
    top_k: int,
    seed: int,
    best_arch: dict[str, Any],
    center: dict[str, Any],
    stage2_hours: float,
) -> optuna.Study:
    sampler = TPESampler(seed=int(seed), n_startup_trials=15)
    pruner = MedianPruner(n_startup_trials=15, n_warmup_steps=15, interval_steps=5)

    study_name = f"{prefix}_stage2"
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction='maximize',
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True,
    )

    trials_jsonl_path = os.path.join(RESULTS_DIR, f"optuna_trials_tcnet_{prefix}_stage2.jsonl")

    class TrialJSONLLogger:
        def __init__(self, path: str):
            self.path = path

        def __call__(self, study, trial):
            record = {
                'ts': datetime.utcnow().isoformat() + 'Z',
                'study_name': str(study.study_name),
                'trial_number': int(trial.number),
                'state': str(trial.state),
                'value_macro_f1': float(trial.value) if trial.value is not None else None,
                'params': dict(trial.params),
                'stage': trial.user_attrs.get('stage', None),
                'fold_best_f1': trial.user_attrs.get('fold_best_f1', None),
                'fold_best_acc': trial.user_attrs.get('fold_best_acc', None),
                'fold_best_epoch': trial.user_attrs.get('fold_best_epoch', None),
            }
            with open(self.path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(record) + "\n")

    def obj(trial: optuna.Trial) -> float:
        return objective_stage2(trial, best_arch=best_arch, center=center)

    start = time.time()
    timeout_s = None
    if stage2_hours is not None and float(stage2_hours) > 0:
        timeout_s = float(stage2_hours) * 3600.0

    study.optimize(
        obj,
        n_trials=int(n_trials),
        timeout=timeout_s,
        callbacks=[TrialJSONLLogger(trials_jsonl_path)],
        show_progress_bar=True,
        gc_after_trial=True,
    )
    elapsed = time.time() - start

    save_stage_artifacts(stage='stage2', study=study, prefix=prefix, elapsed_s=elapsed, top_k=top_k, best_arch=best_arch, center=center)
    print(f"Stage2 JSONL log -> {trials_jsonl_path}")
    return study


def main() -> None:
    global SEED
    global SPLITS
    global LABEL_SMOOTHING_ENABLED
    global LOCK_BATCH_SIZE

    args = parse_args()
    SEED = int(args.seed)
    SPLITS = None  # re-derive folds for this seed
    LABEL_SMOOTHING_ENABLED = not bool(args.disable_label_smoothing)
    LOCK_BATCH_SIZE = bool(args.lock_batch_size)

    storage = _normalize_storage(args.db_path)
    prefix = str(args.study_name_prefix).strip()
    stage = str(args.stage)

    print("=" * 60)
    print("EEG-TCNet Staged Optuna Tuning (safe-CV)")
    print("=" * 60)
    print(f"Stage: {stage}")
    print(f"Seed: {SEED}")
    print(f"Folds: {N_FOLDS}")
    print(f"TIME_POINTS: {TIME_POINTS}")
    print(f"Fixed: augmentation={USE_AUGMENTATION}, attention={USE_SE_ATTENTION or USE_TEMPORAL_ATTENTION}")
    print(f"Storage: {storage}")
    print(f"Study prefix: {prefix}")
    print(f"SMOKE: {SMOKE} | ACTIVE_PIDS={ACTIVE_PIDS}")
    print(f"Label smoothing enabled: {LABEL_SMOOTHING_ENABLED}")
    print(f"Lock batch size (stage2): {LOCK_BATCH_SIZE}")

    print("Loading dataset and precomputing folds (once)...")
    get_dataset()
    get_splits()

    stage1_study = None
    if stage in {'1', 'both'}:
        print("\n--- STAGE 1 ---")
        stage1_study = run_stage1(
            storage=storage,
            prefix=prefix,
            n_trials=int(args.stage1_trials),
            top_k=int(args.top_k),
            seed=SEED,
            stage1_hours=float(args.stage1_hours),
        )

    if stage in {'2', 'both'}:
        print("\n--- STAGE 2 ---")
        # Load stage1 study (to extract top-K architectures) if needed.
        if stage1_study is None:
            stage1_study = optuna.load_study(study_name=f"{prefix}_stage1", storage=storage)

        # Extract up to K unique architectures (and their per-arch centers) from top-K Stage 1 trials.
        all_items = _unique_architectures_from_top_trials(stage1_study, int(args.top_k))
        arch_k = min(int(args.stage2_arch_k), len(all_items))
        items = all_items[:arch_k]
        if not items:
            raise RuntimeError("No completed Stage 1 trials found; cannot run Stage 2")

        total_stage2_hours = float(args.stage2_hours)
        per_arch_hours = None
        if total_stage2_hours is not None and total_stage2_hours > 0:
            per_arch_hours = total_stage2_hours / float(len(items))

        best_overall = {
            'study_name': None,
            'best_value_macro_f1': None,
            'best_params': None,
            'best_arch': None,
        }

        for i, item in enumerate(items, start=1):
            best_arch = dict(item['arch'])
            center = dict(item['center'])
            arch_suffix = f"stage2_arch{i}"
            arch_prefix = f"{prefix}_{arch_suffix}"
            # Each architecture gets its own study.
            _ = run_stage2(
                storage=storage,
                prefix=arch_prefix,
                n_trials=int(args.stage2_trials_per_arch),
                top_k=int(args.top_k),
                seed=SEED,
                best_arch=best_arch,
                center=center,
                stage2_hours=per_arch_hours,
            )

            stage2_study = optuna.load_study(study_name=f"{arch_prefix}_stage2", storage=storage)
            val = float(stage2_study.best_value) if stage2_study.best_value is not None else None
            if val is None:
                continue
            if best_overall['best_value_macro_f1'] is None or val > float(best_overall['best_value_macro_f1']):
                best_overall = {
                    'study_name': str(stage2_study.study_name),
                    'best_value_macro_f1': float(val),
                    'best_params': dict(stage2_study.best_params),
                    'best_arch': dict(best_arch),
                }

        os.makedirs(RESULTS_DIR, exist_ok=True)
        overall_path = os.path.join(RESULTS_DIR, 'stage2_overall_summary.json')
        with open(overall_path, 'w', encoding='utf-8') as f:
            json.dump(
                {
                    **best_overall,
                    'seed': int(SEED),
                    'top_k_stage1': int(args.top_k),
                    'stage2_arch_k': int(args.stage2_arch_k),
                    'stage2_trials_per_arch': int(args.stage2_trials_per_arch),
                    'stage2_hours_total': float(args.stage2_hours),
                    'stage2_hours_per_arch': float(per_arch_hours) if per_arch_hours is not None else None,
                },
                f,
                indent=2,
            )
        print(f"Saved Stage 2 overall summary -> {overall_path}")


if __name__ == '__main__':
    main()
