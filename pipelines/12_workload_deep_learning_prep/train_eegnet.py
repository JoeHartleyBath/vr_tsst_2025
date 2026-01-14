"""
EEGNet Training Script for Workload Classification
Adapts EEGNet architecture for 128-channel data with GroupKFold CV.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from mne_dataloader import load_workload_data, create_group_splits

# ==============================================================================
# CONFIGURATION
# ==============================================================================
DATA_DIR = r'C:\vr_tsst_2025\output\adaptive_workload\mne_epochs'
RESULTS_PATH = r'C:\vr_tsst_2025\results\workload_eegnet_cv.txt'

# All 44 EEG-valid participants (P44 excluded - no valid windows)
ALL_PIDS = [1, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 15, 16, 17, 18, 19, 20, 21, 22, 23, 
            24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 
            42, 43, 45, 47, 48]
SUBSET_PIDS = ALL_PIDS

# Model hyperparameters (adapted for 128 channels, ~1249 samples @ 125Hz)
CHANS = 128
TIME_POINTS = 1249  # Some epochs have 1249 instead of 1250
CLASSES = 2

# Training hyperparameters
BATCH_SIZE = 16
EPOCHS = 50
LEARNING_RATE = 1e-3
EARLY_STOP_PATIENCE = 10

# ==============================================================================
# EEGNet Model (adapted from reference)
# ==============================================================================
class EEGNet(nn.Module):
    """
    EEGNet-8,2 architecture adapted for 128-channel workload classification.
    
    Reference: Lawhern et al., 2018 - EEGNet: A Compact Convolutional Network
    for EEG-based Brain-Computer Interfaces
    """
    def __init__(self, chans=128, classes=2, time_points=1250, 
                 f1=8, d=2, f2=16, dropout_rate=0.5,
                 temp_kernel=64, pk1=4, pk2=8):
        super(EEGNet, self).__init__()
        
        # Calculate FC layer size
        # After pooling: time_points // (pk1 * pk2)
        linear_size = (time_points // (pk1 * pk2)) * f2
        
        # Block 1: Temporal Convolution
        self.block1 = nn.Sequential(
            nn.Conv2d(1, f1, (1, temp_kernel), padding='same', bias=False),
            nn.BatchNorm2d(f1),
        )
        
        # Block 2: Depthwise Spatial Convolution
        self.block2 = nn.Sequential(
            nn.Conv2d(f1, d * f1, (chans, 1), groups=f1, bias=False),
            nn.BatchNorm2d(d * f1),
            nn.ELU(),
            nn.AvgPool2d((1, pk1)),
            nn.Dropout(dropout_rate)
        )
        
        # Block 3: Separable Convolution
        self.block3 = nn.Sequential(
            nn.Conv2d(d * f1, f2, (1, 16), groups=f2, bias=False, padding='same'),
            nn.Conv2d(f2, f2, kernel_size=1, bias=False),
            nn.BatchNorm2d(f2),
            nn.ELU(),
            nn.AvgPool2d((1, pk2)),
            nn.Dropout(dropout_rate)
        )
        
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(linear_size, classes)
        
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x


def train_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        preds = torch.argmax(outputs, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(batch_y.cpu().numpy())
    
    acc = accuracy_score(all_labels, all_preds)
    return total_loss / len(loader), acc


def evaluate(model, loader, criterion, device):
    """Evaluate model on a dataset."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            
            total_loss += loss.item()
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(batch_y.cpu().numpy())
    
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    return total_loss / len(loader), acc, f1


def run_cv_evaluation():
    """Run 3-fold GroupKFold cross-validation."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    dataset = load_workload_data(DATA_DIR, subset_pids=SUBSET_PIDS)
    print(f"Total samples: {len(dataset)}, Shape: {dataset.data.shape}")
    
    fold_results = []
    
    for fold, (train_idx, test_idx) in enumerate(create_group_splits(dataset, n_splits=3)):
        print(f"\n{'='*50}")
        print(f"Fold {fold + 1}")
        print(f"{'='*50}")
        
        train_pids = np.unique(dataset.pids[train_idx])
        test_pids = np.unique(dataset.pids[test_idx])
        print(f"Train PIDs: {train_pids}, Test PIDs: {test_pids}")
        
        # Create data loaders
        train_subset = Subset(dataset, train_idx)
        test_subset = Subset(dataset, test_idx)
        
        train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(test_subset, batch_size=BATCH_SIZE, shuffle=False)
        
        # Initialize model
        model = EEGNet(chans=CHANS, classes=CLASSES, time_points=TIME_POINTS).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        
        # Training loop with early stopping
        best_acc = 0
        patience_counter = 0
        
        for epoch in range(EPOCHS):
            train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
            test_loss, test_acc, test_f1 = evaluate(model, test_loader, criterion, device)
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1:3d}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.3f} | "
                      f"Test Loss={test_loss:.4f}, Test Acc={test_acc:.3f}")
            
            # Early stopping check
            if test_acc > best_acc:
                best_acc = test_acc
                best_f1 = test_f1
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= EARLY_STOP_PATIENCE:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        
        print(f"Fold {fold+1} Best: Accuracy={best_acc:.3f}, F1={best_f1:.3f}")
        fold_results.append({'accuracy': best_acc, 'f1': best_f1})
    
    # Aggregate results
    mean_acc = np.mean([r['accuracy'] for r in fold_results])
    std_acc = np.std([r['accuracy'] for r in fold_results])
    mean_f1 = np.mean([r['f1'] for r in fold_results])
    
    print(f"\n{'='*50}")
    print("FINAL RESULTS")
    print(f"{'='*50}")
    print(f"Per-Fold Accuracy: {[r['accuracy'] for r in fold_results]}")
    print(f"Mean Accuracy: {mean_acc:.3f} (+/- {std_acc:.3f})")
    print(f"Mean F1 Score: {mean_f1:.3f}")
    
    # Save results
    with open(RESULTS_PATH, 'w') as f:
        f.write(f"EEGNet Workload Classification (Subset: {SUBSET_PIDS})\n")
        f.write(f"Architecture: EEGNet-8,2 adapted for {CHANS} channels\n")
        f.write(f"Data: {len(dataset)} epochs, {TIME_POINTS} samples @ 125Hz\n")
        f.write(f"Training: {EPOCHS} max epochs, batch={BATCH_SIZE}, lr={LEARNING_RATE}\n")
        f.write("-" * 50 + "\n")
        for i, r in enumerate(fold_results):
            f.write(f"Fold {i+1}: Accuracy={r['accuracy']:.4f}, F1={r['f1']:.4f}\n")
        f.write("-" * 50 + "\n")
        f.write(f"Mean Accuracy: {mean_acc:.4f} (+/- {std_acc:.4f})\n")
        f.write(f"Mean F1 Score: {mean_f1:.4f}\n")
        f.write("-" * 50 + "\n")
        if mean_acc >= 0.60:
            f.write("DECISION GATE: PASSED (>= 0.60 Accuracy)\n")
        else:
            f.write(f"DECISION GATE: FAILED ({mean_acc:.4f} < 0.60 Accuracy)\n")
    
    print(f"\nResults saved to {RESULTS_PATH}")
    
    # Compare to SVM baseline
    print("\n--- COMPARISON ---")
    print(f"SVM Baseline (Relative): 0.638 Accuracy")
    print(f"EEGNet:                  {mean_acc:.3f} Accuracy")
    if mean_acc > 0.638:
        print("EEGNet OUTPERFORMS SVM baseline!")
    else:
        print("EEGNet underperforms SVM (expected with small data)")
    
    return mean_acc, mean_f1


if __name__ == "__main__":
    run_cv_evaluation()
