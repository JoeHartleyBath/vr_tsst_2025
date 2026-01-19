"""
Focused EEG-TCNet Hyperparameter Tuning - 12 Hour Budget

VERSION 2 DEBUG CHECK

Key fixes from previous failed run:
1. NO AUGMENTATION - it was hurting performance badly
2. NO ATTENTION - both SE and temporal attention hurt performance
3. Focused search on core architecture params only
4. 12 hour timeout with convergence stopping

Target: Beat 60.2% baseline from train_tcnet.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

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

from mne_dataloader import load_workload_data, create_group_splits
from train_tcnet import EEGTCNet

# ==============================================================================
# CONFIGURATION
# ==============================================================================
DATA_DIR = r'C:\vr_tsst_2025\output\adaptive_workload\mne_epochs'
RESULTS_DIR = r'C:\vr_tsst_2025\results'
STUDY_NAME = 'eeg_tcnet_focused_tuning_phase2'
DB_PATH = f'sqlite:///{RESULTS_DIR.replace("\\\\", "/")}/optuna_focused.db'

# All 43 EEG-valid participants
ALL_PIDS = [1, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 15, 16, 17, 18, 19, 20, 21, 22, 23, 
            24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 
            42, 43, 45, 47, 48]

# Fixed parameters
CHANS = 128
TIME_POINTS = 1249
CLASSES = 2

# Tuning settings
TIME_LIMIT_HOURS = 12
TIME_LIMIT_SECONDS = TIME_LIMIT_HOURS * 3600
MAX_TRIALS = 150  # More trials since each is faster without augmentation
CONVERGENCE_TRIALS = 25  # Stop if no improvement for 25 trials
CONVERGENCE_THRESHOLD = 0.63  # Updated since baseline is 62.3%
MAX_EPOCHS = 50  # Slightly fewer epochs, relying on scheduler
EARLY_STOP_PATIENCE = 10
N_FOLDS = 3

# FIXED: These hurt performance - don't tune them
USE_SE_ATTENTION = False
USE_TEMPORAL_ATTENTION = False
USE_AUGMENTATION = False
# ==============================================================================
# GLOBAL DATA LOADING (load once, reuse across trials)
# ==============================================================================
print("Loading data once for all trials...")
DATASET = None

def get_dataset():
    global DATASET
    if DATASET is None:
        DATASET = load_workload_data(DATA_DIR, subset_pids=ALL_PIDS, augment=False)
        print(f"Loaded {len(DATASET)} samples")
    return DATASET


# ==============================================================================
# TRAINING FUNCTIONS
# ==============================================================================

def train_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    all_preds, all_labels = [], []
    
    for batch_x, batch_y, _ in loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        all_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
        all_labels.extend(batch_y.cpu().numpy())
    
    return total_loss / len(loader), accuracy_score(all_labels, all_preds)


def evaluate(model, loader, criterion, device):
    """Evaluate model."""
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for batch_x, batch_y, _ in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            
            total_loss += loss.item()
            all_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())
    
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    return total_loss / len(loader), acc, f1


def objective(trial):
    """
    Phase 2 objective - Fine-tuning regularization only.
    Architecture fixed to best values from phase 1.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # ==== FIXED ARCHITECTURE (from Phase 1 Best Params) ====
    # Refinement of the best performing architecture from Phase 1
    F1 = 24
    D = 3
    kernLength = 64
    tcn_filters = 24
    tcn_kernel = 3
    tcn_depth = 2
    batch_size = 64
    
    # ==== TUNED HYPERPARAMETERS (Regularization & Learning Rate) ====
    # Tuning only regularization and optimization parameters
    dropout_eeg = trial.suggest_float('dropout_eeg', 0.05, 0.25)
    dropout_tcn = trial.suggest_float('dropout_tcn', 0.15, 0.5)
    
    learning_rate = trial.suggest_float('learning_rate', 3e-4, 3e-3, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)
    
    # Get cached dataset (no reload each trial)
    dataset = get_dataset()
    
    fold_accuracies = []
    
    for fold, (train_idx, test_idx) in enumerate(create_group_splits(dataset, n_splits=N_FOLDS)):
        # Create data loaders - NO augmentation
        train_subset = Subset(dataset, train_idx)
        test_subset = Subset(dataset, test_idx)
        
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, 
                                  num_workers=0, pin_memory=True)
        test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False,
                                num_workers=0, pin_memory=True)
        
        # Compute class weights from TRAINING data only (prevent leakage)
        train_labels = dataset.labels[train_idx]
        unique, counts = np.unique(train_labels, return_counts=True)
        weights = len(train_labels) / (len(unique) * counts)
        class_weights = torch.FloatTensor(weights).to(device)
            tcn_filters=tcn_filters, tcn_kernel=tcn_kernel, tcn_depth=tcn_depth, 
            dropout_tcn=dropout_tcn,
            use_se_attention=USE_SE_ATTENTION,
            use_temporal_attention=USE_TEMPORAL_ATTENTION
        ).to(device)
        
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
        
        # Training loop with early stopping
        best_acc = 0
        patience_counter = 0
        
        for epoch in range(MAX_EPOCHS):
            train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
            test_loss, test_acc, test_f1 = evaluate(model, test_loader, criterion, device)
            
            scheduler.step(test_acc)
            
            if test_acc > best_acc:
                best_acc = test_acc
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= EARLY_STOP_PATIENCE:
                    break
            
            # Pruning: report after fold 0 only
            if fold == 0:
                trial.report(test_acc, epoch)
                if trial.should_prune():
                    raise optuna.TrialPruned()
        
        fold_accuracies.append(best_acc)
        
        # Clean up GPU memory
        del model
        torch.cuda.empty_cache()
    
    mean_acc = np.mean(fold_accuracies)
    return mean_acc


class ConvergenceCallback:
    """Stop study if no improvement for N trials above threshold."""
    
    def __init__(self, threshold=CONVERGENCE_THRESHOLD, patience=CONVERGENCE_TRIALS):
        self.threshold = threshold
        self.patience = patience
        self.best_value = 0
        self.trials_without_improvement = 0
        
    def __call__(self, study, trial):
        if trial.value is not None and trial.value > self.best_value:
            self.best_value = trial.value
            self.trials_without_improvement = 0
            print(f"  New best: {self.best_value:.4f}")
        else:
            self.trials_without_improvement += 1
            
        if self.best_value >= self.threshold and self.trials_without_improvement >= self.patience:
            print(f"\nConvergence: No improvement for {self.patience} trials above {self.threshold}")
            study.stop()


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


def run_tuning():
    """Run focused hyperparameter tuning."""
    print("=" * 60)
    print("EEG-TCNet FOCUSED Hyperparameter Tuning")
    print("=" * 60)
    print(f"Time limit: {TIME_LIMIT_HOURS} hours")
    print(f"Max trials: {MAX_TRIALS}")
    print(f"Convergence: {CONVERGENCE_TRIALS} trials without improvement at {CONVERGENCE_THRESHOLD}")
    print(f"FIXED: attention=False, augmentation=False")
    print()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Pre-load dataset
    get_dataset()
    
    # Create study with aggressive pruning
    sampler = TPESampler(seed=42, n_startup_trials=15)  # 15 random, then TPE
    pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=10)
    
    print(f"DEBUG: Loading/creating study '{STUDY_NAME}' in '{DB_PATH}'", flush=True)
    
    # Simply use load_if_exists=True which handles everything
    study = optuna.create_study(
        study_name=STUDY_NAME,
        storage=DB_PATH,
        direction='maximize',
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True
    )
    print(f"DEBUG: Study ready with {len(study.trials)} existing trials.", flush=True)
    
    # Callbacks
    callbacks = [
        ConvergenceCallback(threshold=CONVERGENCE_THRESHOLD, patience=CONVERGENCE_TRIALS),
        TimeoutCallback(TIME_LIMIT_SECONDS)
    ]
    
    # Run optimization
    study.optimize(
        objective,
        n_trials=MAX_TRIALS,
        callbacks=callbacks,
        show_progress_bar=True,
        gc_after_trial=True
    )
    
    # Results
    print("\n" + "=" * 60)
    print("TUNING COMPLETE")
    print("=" * 60)
    print(f"Total trials: {len(study.trials)}")
    print(f"Best accuracy: {study.best_value:.4f}")
    print(f"Best parameters:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")
    
    # Save best params
    best_params_path = os.path.join(RESULTS_DIR, 'best_tcnet_focused_params.json')
    with open(best_params_path, 'w') as f:
        json.dump({
            'best_accuracy': study.best_value,
            'best_params': study.best_params,
            'fixed_params': {
                'use_se_attention': USE_SE_ATTENTION,
                'use_temporal_attention': USE_TEMPORAL_ATTENTION,
                'use_augmentation': USE_AUGMENTATION
            },
            'total_trials': len(study.trials)
        }, f, indent=2)
    print(f"\nBest params saved to {best_params_path}")
    
    # Comparison
    print("\n--- COMPARISON TO BASELINES ---")
    print(f"SVM Baseline:       56.4%")
    print(f"EEGNet:             56.9%")
    print(f"EEG-TCNet (base):   60.2%")
    print(f"EEG-TCNet (tuned):  {study.best_value*100:.1f}%")
    
    return study


if __name__ == "__main__":
    study = run_tuning()
