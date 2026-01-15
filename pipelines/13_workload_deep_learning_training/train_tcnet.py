"""
EEG-TCNet Training Script for Workload Classification (PyTorch Port)
Combines EEGNet with Temporal Convolutional Network + Attention Mechanisms.

Reference: Ingolfsson et al., 2020 - EEG-TCNet: An Accurate Temporal Convolutional
Network for Embedded Motor-Imagery Brain-Machine Interfaces

Enhancements:
- Squeeze-and-Excitation channel attention
- Temporal attention before classification
- Class-weighted loss for imbalanced data
- Data augmentation support
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

from mne_dataloader import load_workload_data, create_group_splits, MNEEpochsDataset

# ==============================================================================
# CONFIGURATION
# ==============================================================================
DATA_DIR = r'C:\vr_tsst_2025\output\adaptive_workload\mne_epochs'
RESULTS_PATH = r'C:\vr_tsst_2025\results\workload_tcnet_attention_cv.txt'

# All 44 EEG-valid participants (P44 excluded - no valid windows)
ALL_PIDS = [1, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 15, 16, 17, 18, 19, 20, 21, 22, 23, 
            24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 
            42, 43, 45, 47, 48]
SUBSET_PIDS = ALL_PIDS

# Model hyperparameters
CHANS = 128
TIME_POINTS = 1249
CLASSES = 2

# Training hyperparameters
BATCH_SIZE = 16
EPOCHS = 50
LEARNING_RATE = 1e-3
EARLY_STOP_PATIENCE = 10

# Augmentation settings
USE_AUGMENTATION = False  # Disabled for baseline test
NOISE_SIGMA = 0.02  # Reduced from 0.05
TIME_SHIFT = 3       # Reduced from 5
CHANNEL_DROPOUT = 0.05  # Reduced from 0.1

# ==============================================================================
# ATTENTION MODULES
# ==============================================================================

class SqueezeExcitation(nn.Module):
    """Squeeze-and-Excitation block for channel attention."""
    
    def __init__(self, channels, reduction=4):
        super(SqueezeExcitation, self).__init__()
        reduced_channels = max(channels // reduction, 1)
        self.fc1 = nn.Linear(channels, reduced_channels)
        self.fc2 = nn.Linear(reduced_channels, channels)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # x: (batch, channels, height, width) or (batch, channels, time)
        # Global average pooling
        if x.dim() == 4:
            squeeze = x.mean(dim=(2, 3))  # (batch, channels)
        else:
            squeeze = x.mean(dim=2)  # (batch, channels)
        
        # Excitation
        excitation = self.fc1(squeeze)
        excitation = self.relu(excitation)
        excitation = self.fc2(excitation)
        excitation = self.sigmoid(excitation)
        
        # Scale
        if x.dim() == 4:
            return x * excitation.unsqueeze(2).unsqueeze(3)
        else:
            return x * excitation.unsqueeze(2)


class TemporalAttention(nn.Module):
    """Lightweight 1D convolutional attention for temporal features."""
    
    def __init__(self, channels, kernel_size=7):
        super(TemporalAttention, self).__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv1d(channels, 1, kernel_size, padding=padding)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # x: (batch, channels, time)
        attn = self.conv(x)  # (batch, 1, time)
        attn = self.sigmoid(attn)
        return x * attn


# ==============================================================================
# EEG-TCNet Model with Attention (PyTorch Port)
# ==============================================================================

class TemporalBlock(nn.Module):
    """Single TCN block with dilated causal convolutions and residual connection."""
    
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout=0.3):
        super(TemporalBlock, self).__init__()
        
        padding = (kernel_size - 1) * dilation  # Causal padding
        
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, 
                               dilation=dilation, padding=padding)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                               dilation=dilation, padding=padding)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        
        # Residual connection
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.relu_out = nn.ReLU()
        
    def forward(self, x):
        # x: (batch, channels, time)
        out = self.conv1(x)
        out = out[:, :, :x.size(2)]  # Causal: trim to original length
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.dropout1(out)
        
        out = self.conv2(out)
        out = out[:, :, :x.size(2)]  # Causal: trim to original length
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.dropout2(out)
        
        # Residual
        res = x if self.downsample is None else self.downsample(x)
        return self.relu_out(out + res)


class TCN(nn.Module):
    """Temporal Convolutional Network with exponentially increasing dilation."""
    
    def __init__(self, in_channels, out_channels, kernel_size=4, depth=2, dropout=0.3):
        super(TCN, self).__init__()
        
        layers = []
        for i in range(depth):
            dilation = 2 ** i
            in_ch = in_channels if i == 0 else out_channels
            layers.append(TemporalBlock(in_ch, out_channels, kernel_size, dilation, dropout))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)


class EEGTCNet(nn.Module):
    """
    EEG-TCNet with Attention: EEGNet + SE Attention + TCN + Temporal Attention.
    
    Architecture:
    1. EEGNet feature extractor (temporal + spatial convolutions)
    2. Squeeze-and-Excitation channel attention after spatial conv
    3. TCN for temporal pattern learning
    4. Temporal attention before classification
    5. Dense classification head
    """
    
    def __init__(self, chans=128, classes=2, time_points=1249,
                 F1=8, D=2, kernLength=64, dropout_eeg=0.2,
                 tcn_filters=12, tcn_kernel=4, tcn_depth=2, dropout_tcn=0.3,
                 use_se_attention=True, use_temporal_attention=True, se_reduction=4):
        super(EEGTCNet, self).__init__()
        
        self.use_se_attention = use_se_attention
        self.use_temporal_attention = use_temporal_attention
        
        F2 = F1 * D
        
        # ===== EEGNet Block =====
        # Block 1: Temporal convolution
        self.conv1 = nn.Conv2d(1, F1, (1, kernLength), padding='same', bias=False)
        self.bn1 = nn.BatchNorm2d(F1)
        
        # Block 2: Depthwise spatial convolution
        self.depthwise = nn.Conv2d(F1, F1 * D, (chans, 1), groups=F1, bias=False)
        self.bn2 = nn.BatchNorm2d(F2)
        self.elu1 = nn.ELU()
        self.pool1 = nn.AvgPool2d((1, 8))
        self.drop1 = nn.Dropout(dropout_eeg)
        
        # Squeeze-and-Excitation after depthwise conv
        if use_se_attention:
            self.se_block = SqueezeExcitation(F2, reduction=se_reduction)
        
        # Block 3: Separable convolution
        self.separable_depth = nn.Conv2d(F2, F2, (1, 16), padding='same', groups=F2, bias=False)
        self.separable_point = nn.Conv2d(F2, F2, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(F2)
        self.elu2 = nn.ELU()
        self.pool2 = nn.AvgPool2d((1, 8))
        self.drop2 = nn.Dropout(dropout_eeg)
        
        # ===== TCN Block =====
        self.tcn = TCN(F2, tcn_filters, kernel_size=tcn_kernel, depth=tcn_depth, dropout=dropout_tcn)
        
        # Temporal attention after TCN
        if use_temporal_attention:
            self.temporal_attention = TemporalAttention(tcn_filters, kernel_size=7)
        
        # ===== Classification Head =====
        self.fc = nn.Linear(tcn_filters, classes)
        
    def forward(self, x):
        # x: (batch, 1, chans, time)
        
        # EEGNet Block 1
        x = self.conv1(x)
        x = self.bn1(x)
        
        # EEGNet Block 2 with SE attention
        x = self.depthwise(x)
        x = self.bn2(x)
        x = self.elu1(x)
        x = self.pool1(x)
        x = self.drop1(x)
        
        if self.use_se_attention:
            x = self.se_block(x)
        
        # EEGNet Block 3
        x = self.separable_depth(x)
        x = self.separable_point(x)
        x = self.bn3(x)
        x = self.elu2(x)
        x = self.pool2(x)
        x = self.drop2(x)
        
        # Reshape for TCN: (batch, F2, 1, time') -> (batch, F2, time')
        x = x.squeeze(2)
        
        # TCN
        x = self.tcn(x)
        
        # Temporal attention
        if self.use_temporal_attention:
            x = self.temporal_attention(x)
        
        # Take last time step for classification
        x = x[:, :, -1]
        
        # Classification
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
    """Run 3-fold GroupKFold cross-validation with augmentation and class weighting."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Augmentation: {USE_AUGMENTATION}")
    
    # Load data WITHOUT augmentation first (for class weights and splits)
    dataset = load_workload_data(DATA_DIR, subset_pids=SUBSET_PIDS, augment=False)
    print(f"Total samples: {len(dataset)}, Shape: {dataset.data.shape}")
    
    # Compute class weights for imbalanced data
    class_weights = dataset.get_class_weights().to(device)
    print(f"Class weights: {class_weights.cpu().numpy()}")
    
    # Load augmented version for training
    if USE_AUGMENTATION:
        dataset_aug = load_workload_data(
            DATA_DIR, subset_pids=SUBSET_PIDS, augment=True,
            noise_sigma=NOISE_SIGMA, time_shift=TIME_SHIFT, channel_dropout=CHANNEL_DROPOUT
        )
    else:
        dataset_aug = dataset
    
    fold_results = []
    
    for fold, (train_idx, test_idx) in enumerate(create_group_splits(dataset, n_splits=3)):
        print(f"\n{'='*50}")
        print(f"Fold {fold + 1}")
        print(f"{'='*50}")
        
        train_pids = np.unique(dataset.pids[train_idx])
        test_pids = np.unique(dataset.pids[test_idx])
        print(f"Train PIDs: {len(train_pids)} subjects, Test PIDs: {len(test_pids)} subjects")
        
        # Create data loaders - use augmented for train, non-augmented for test
        train_subset = Subset(dataset_aug, train_idx)  # Augmented
        test_subset = Subset(dataset, test_idx)        # Not augmented
        
        train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(test_subset, batch_size=BATCH_SIZE, shuffle=False)
        
        # Initialize model (attention configurable via Optuna tuning)
        model = EEGTCNet(
            chans=CHANS, classes=CLASSES, time_points=TIME_POINTS,
            use_se_attention=False, use_temporal_attention=False
        ).to(device)
        
        # Use weighted loss for class imbalance
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        
        # Training loop with early stopping
        best_acc = 0
        best_f1 = 0
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
    print("FINAL RESULTS - EEG-TCNet with Attention")
    print(f"{'='*50}")
    print(f"Per-Fold Accuracy: {[r['accuracy'] for r in fold_results]}")
    print(f"Mean Accuracy: {mean_acc:.3f} (+/- {std_acc:.3f})")
    print(f"Mean F1 Score: {mean_f1:.3f}")
    
    # Save results
    with open(RESULTS_PATH, 'w') as f:
        f.write(f"EEG-TCNet with Attention - Workload Classification (N={len(SUBSET_PIDS)} subjects)\n")
        f.write(f"Architecture: EEGNet + SE Attention + TCN + Temporal Attention\n")
        f.write(f"Data: {len(dataset)} epochs, {TIME_POINTS} samples @ 125Hz\n")
        f.write(f"Augmentation: noise={NOISE_SIGMA}, shift={TIME_SHIFT}, ch_drop={CHANNEL_DROPOUT}\n")
        f.write(f"Training: {EPOCHS} max epochs, batch={BATCH_SIZE}, lr={LEARNING_RATE}\n")
        f.write(f"Class weights: {class_weights.cpu().numpy()}\n")
        f.write("-" * 50 + "\n")
        for i, r in enumerate(fold_results):
            f.write(f"Fold {i+1}: Accuracy={r['accuracy']:.4f}, F1={r['f1']:.4f}\n")
        f.write("-" * 50 + "\n")
        f.write(f"Mean Accuracy: {mean_acc:.4f} (+/- {std_acc:.4f})\n")
        f.write(f"Mean F1 Score: {mean_f1:.4f}\n")
    
    print(f"\nResults saved to {RESULTS_PATH}")
    
    # Compare to baselines
    print("\n--- COMPARISON ---")
    print(f"SVM Baseline:      0.564 Accuracy")
    print(f"EEGNet:            0.569 Accuracy")
    print(f"EEG-TCNet (base):  0.597 Accuracy")
    print(f"EEG-TCNet+Attn:    {mean_acc:.3f} Accuracy")
    
    return mean_acc, mean_f1


if __name__ == "__main__":
    run_cv_evaluation()
