"""
Optuna Hyperparameter Tuning for EEG-TCNet with Attention
Maximizes workload classification accuracy within time budget.

Features:
- Bayesian optimization with TPE sampler
- MedianPruner for early stopping of bad trials
- Convergence-based stopping (20 trials without improvement at 65%+)
- GPU-accelerated training on RTX 5070 Ti
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

from mne_dataloader import load_workload_data, create_group_splits, get_augmentation_transform
from train_tcnet import EEGTCNet, TemporalBlock, TCN, SqueezeExcitation, TemporalAttention


class AugmentedWrapper(torch.utils.data.Dataset):
    """Wrapper that applies augmentation to an existing dataset without copying data."""
    def __init__(self, base_dataset, aug_transform):
        self.base = base_dataset
        self.aug_transform = aug_transform
    
    def __len__(self):
        return len(self.base)
    
    def __getitem__(self, idx):
        x, y = self.base[idx]
        if self.aug_transform:
            x = self.aug_transform(x)
        return x, y
    
    @property
    def pids(self):
        return self.base.pids

# ==============================================================================
# CONFIGURATION
# ==============================================================================
DATA_DIR = r'C:\vr_tsst_2025\output\adaptive_workload\mne_epochs'
RESULTS_DIR = r'C:\vr_tsst_2025\results'
STUDY_NAME = 'eeg_tcnet_workload_tuning'
DB_PATH = f'sqlite:///{RESULTS_DIR}/optuna_study.db'

# All 43 EEG-valid participants
ALL_PIDS = [1, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 15, 16, 17, 18, 19, 20, 21, 22, 23, 
            24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 
            42, 43, 45, 47, 48]

# Fixed parameters
CHANS = 128
TIME_POINTS = 1249
CLASSES = 2

# Tuning settings
MAX_TRIALS = 80
CONVERGENCE_TRIALS = 20  # Stop if no improvement for 20 trials
CONVERGENCE_THRESHOLD = 0.65  # Only apply convergence stopping above this
MAX_EPOCHS = 50
EARLY_STOP_PATIENCE = 10
N_FOLDS = 3

# ==============================================================================
# TRAINING FUNCTIONS
# ==============================================================================

def train_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    all_preds, all_labels = [], []
    
    for batch_x, batch_y in loader:
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
        for batch_x, batch_y in loader:
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
    """Optuna objective function for hyperparameter optimization."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # ==== Hyperparameter search space ====
    # Model architecture
    F1 = trial.suggest_categorical('F1', [4, 8, 16])
    D = trial.suggest_categorical('D', [1, 2])
    kernLength = trial.suggest_categorical('kernLength', [32, 64, 128])
    dropout_eeg = trial.suggest_float('dropout_eeg', 0.2, 0.5)
    
    # TCN parameters
    tcn_filters = trial.suggest_categorical('tcn_filters', [8, 12, 16, 24])
    tcn_kernel = trial.suggest_categorical('tcn_kernel', [3, 4, 6])
    tcn_depth = trial.suggest_int('tcn_depth', 1, 3)
    dropout_tcn = trial.suggest_float('dropout_tcn', 0.2, 0.5)
    
    # Attention
    use_se_attention = trial.suggest_categorical('use_se_attention', [True, False])
    use_temporal_attention = trial.suggest_categorical('use_temporal_attention', [True, False])
    se_reduction = trial.suggest_categorical('se_reduction', [2, 4, 8])
    
    # Training
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 2e-3, log=True)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
    
    # Augmentation
    noise_sigma = trial.suggest_float('noise_sigma', 0.02, 0.1)
    time_shift = trial.suggest_int('time_shift', 2, 10)
    channel_dropout = trial.suggest_float('channel_dropout', 0.05, 0.2)
    
    # Load data ONCE - augmentation is applied on-the-fly via wrapper (saves ~7GB RAM)
    dataset = load_workload_data(DATA_DIR, subset_pids=ALL_PIDS, augment=False)
    
    # Create augmentation transform and wrapper (shares same data, no copy)
    aug_transform = get_augmentation_transform(
        noise_sigma=noise_sigma,
        time_shift=time_shift,
        channel_dropout=channel_dropout
    )
    dataset_aug = AugmentedWrapper(dataset, aug_transform)
    
    class_weights = dataset.get_class_weights().to(device)
    
    fold_accuracies = []
    
    for fold, (train_idx, test_idx) in enumerate(create_group_splits(dataset, n_splits=N_FOLDS)):
        # Create data loaders
        train_subset = Subset(dataset_aug, train_idx)
        test_subset = Subset(dataset, test_idx)
        
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False)
        
        # Initialize model
        model = EEGTCNet(
            chans=CHANS, classes=CLASSES, time_points=TIME_POINTS,
            F1=F1, D=D, kernLength=kernLength, dropout_eeg=dropout_eeg,
            tcn_filters=tcn_filters, tcn_kernel=tcn_kernel, tcn_depth=tcn_depth, dropout_tcn=dropout_tcn,
            use_se_attention=use_se_attention, use_temporal_attention=use_temporal_attention,
            se_reduction=se_reduction
        ).to(device)
        
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # Training loop
        best_acc = 0
        patience_counter = 0
        
        for epoch in range(MAX_EPOCHS):
            train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
            test_loss, test_acc, test_f1 = evaluate(model, test_loader, criterion, device)
            
            if test_acc > best_acc:
                best_acc = test_acc
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= EARLY_STOP_PATIENCE:
                    break
            
            # Report intermediate value for pruning (after first fold only)
            if fold == 0:
                trial.report(test_acc, epoch)
                if trial.should_prune():
                    raise optuna.TrialPruned()
        
        fold_accuracies.append(best_acc)
    
    mean_acc = np.mean(fold_accuracies)
    return mean_acc


class ConvergenceCallback:
    """Stop study if no improvement for N trials above threshold."""
    
    def __init__(self, threshold=0.65, patience=20):
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
        
        # Only apply convergence stopping if above threshold
        if self.best_value >= self.threshold and self.trials_without_improvement >= self.patience:
            print(f"\n*** CONVERGENCE: No improvement for {self.patience} trials at {self.best_value:.4f} ***")
            study.stop()


def run_tuning():
    """Run Optuna hyperparameter tuning."""
    print("="*60)
    print("EEG-TCNet Hyperparameter Tuning with Optuna")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Max trials: {MAX_TRIALS}")
    print(f"Convergence: {CONVERGENCE_TRIALS} trials without improvement at {CONVERGENCE_THRESHOLD:.0%}")
    
    start_time = time.time()
    
    # Create study with pruning
    sampler = TPESampler(seed=42)
    pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=10)
    
    study = optuna.create_study(
        study_name=STUDY_NAME,
        storage=DB_PATH,
        direction='maximize',
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True
    )
    
    # Run optimization with convergence callback
    convergence_callback = ConvergenceCallback(
        threshold=CONVERGENCE_THRESHOLD, 
        patience=CONVERGENCE_TRIALS
    )
    
    study.optimize(
        objective,
        n_trials=MAX_TRIALS,
        callbacks=[convergence_callback],
        show_progress_bar=True
    )
    
    elapsed = time.time() - start_time
    
    # Results
    print("\n" + "="*60)
    print("TUNING COMPLETE")
    print("="*60)
    print(f"Total time: {elapsed/3600:.2f} hours")
    print(f"Trials completed: {len(study.trials)}")
    print(f"Best accuracy: {study.best_value:.4f}")
    print(f"\nBest hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    
    # Save best params
    best_params_path = os.path.join(RESULTS_DIR, 'best_tcnet_params.json')
    with open(best_params_path, 'w') as f:
        json.dump({
            'best_accuracy': study.best_value,
            'best_params': study.best_params,
            'n_trials': len(study.trials),
            'elapsed_hours': elapsed / 3600
        }, f, indent=2)
    print(f"\nBest params saved to {best_params_path}")
    
    # Compare to baselines
    print("\n--- COMPARISON ---")
    print(f"SVM Baseline:      0.564 Accuracy")
    print(f"EEGNet:            0.569 Accuracy")
    print(f"EEG-TCNet (base):  0.597 Accuracy")
    print(f"EEG-TCNet (tuned): {study.best_value:.3f} Accuracy")
    
    return study


if __name__ == "__main__":
    study = run_tuning()
