"""
MNE Epochs DataLoader for PyTorch
Loads .fif epoch files and provides PyTorch-compatible datasets for training.
Includes data augmentation for improved generalization.
"""
import torch
from torch.utils.data import Dataset, DataLoader
import mne
import numpy as np
import os
from pathlib import Path


# ==============================================================================
# DATA AUGMENTATION TRANSFORMS
# ==============================================================================

class GaussianNoise:
    """Add Gaussian noise to EEG signal."""
    def __init__(self, sigma=0.05):
        self.sigma = sigma
    
    def __call__(self, x):
        noise = torch.randn_like(x) * self.sigma
        return x + noise


class TimeShift:
    """Randomly shift signal in time (circular)."""
    def __init__(self, max_shift=5):
        self.max_shift = max_shift
    
    def __call__(self, x):
        shift = np.random.randint(-self.max_shift, self.max_shift + 1)
        return torch.roll(x, shifts=shift, dims=-1)


class ChannelDropout:
    """Randomly zero out entire channels."""
    def __init__(self, p=0.1):
        self.p = p
    
    def __call__(self, x):
        # x shape: (1, C, T)
        mask = torch.rand(x.shape[1]) > self.p
        mask = mask.unsqueeze(0).unsqueeze(-1)  # (1, C, 1)
        return x * mask.float()


class RandomScale:
    """Randomly scale amplitude."""
    def __init__(self, scale_range=(0.8, 1.2)):
        self.scale_range = scale_range
    
    def __call__(self, x):
        scale = np.random.uniform(*self.scale_range)
        return x * scale


class Compose:
    """Compose multiple transforms."""
    def __init__(self, transforms):
        self.transforms = transforms
    
    def __call__(self, x):
        for t in self.transforms:
            x = t(x)
        return x


def get_augmentation_transform(noise_sigma=0.05, time_shift=5, channel_dropout=0.1):
    """Get standard augmentation pipeline for training."""
    return Compose([
        GaussianNoise(sigma=noise_sigma),
        TimeShift(max_shift=time_shift),
        ChannelDropout(p=channel_dropout),
    ])


class MNEEpochsDataset(Dataset):
    """PyTorch Dataset for MNE Epochs files with optional augmentation."""
    
    def __init__(
        self,
        epoch_files,
        transform=None,
        normalize=False,
        augment=False,
        noise_sigma=0.05,
        time_shift=5,
        channel_dropout=0.1,
        target_chans=128,
        target_time_points=1250,
    ):
        """
        Args:
            epoch_files: List of paths to .fif epoch files
            transform: Optional additional transform to apply
            normalize: If True, apply per-epoch z-scoring
            augment: If True, apply data augmentation (for training only)
            noise_sigma: Gaussian noise standard deviation
            time_shift: Max time shift in samples
            channel_dropout: Probability of dropping a channel
        """
        self.transform = transform
        self.normalize = normalize
        self.augment = augment
        self.target_chans = int(target_chans)
        self.target_time_points = int(target_time_points)

        self.shape_fixes = {
            'cropped_time': 0,
            'padded_time': 0,
            'fixed_from_time_lengths': {},
            'nan_found': False,
        }
        
        if augment:
            self.aug_transform = get_augmentation_transform(
                noise_sigma=noise_sigma,
                time_shift=time_shift,
                channel_dropout=channel_dropout
            )
        else:
            self.aug_transform = None
        
        all_data = []
        all_labels = []
        all_pids = []
        all_conditions = []
        all_window_indices = []
        
        for fpath in epoch_files:
            epochs = mne.read_epochs(fpath, preload=True, verbose=False)
            
            # Get data: shape (n_epochs, n_channels, n_times)
            data = epochs.get_data()

            if data.shape[1] != self.target_chans:
                raise ValueError(
                    f"Unexpected channel count in {fpath}: {data.shape[1]} (expected {self.target_chans})"
                )

            orig_t = int(data.shape[2])
            self.shape_fixes['fixed_from_time_lengths'][orig_t] = (
                self.shape_fixes['fixed_from_time_lengths'].get(orig_t, 0) + int(data.shape[0])
            )

            # Enforce fixed number of samples deterministically.
            # Prefer crop: x[..., :target_time_points]. If too short, pad zeros at end.
            if orig_t > self.target_time_points:
                data = data[:, :, : self.target_time_points]
                self.shape_fixes['cropped_time'] += int(data.shape[0])
            elif orig_t < self.target_time_points:
                pad = self.target_time_points - orig_t
                data = np.pad(data, ((0, 0), (0, 0), (0, pad)), mode='constant', constant_values=0.0)
                self.shape_fixes['padded_time'] += int(data.shape[0])
            
            # Get labels and metadata
            labels = (epochs.metadata['workload_class'] == 'HighWorkload').astype(int).values
            pids = epochs.metadata['pid'].values
            conditions = epochs.metadata['event_label'].values
            window_indices = epochs.metadata['window_idx'].values
            
            all_data.append(data)
            all_labels.append(labels)
            all_pids.append(pids)
            all_conditions.append(conditions)
            all_window_indices.append(window_indices)
        
        # Concatenate all subjects - USE FLOAT32 to save memory!
        self.data = np.concatenate(all_data, axis=0).astype(np.float32)  # (N, C, T)
        self.labels = np.concatenate(all_labels, axis=0)
        self.pids = np.concatenate(all_pids, axis=0)
        self.conditions = np.concatenate(all_conditions, axis=0)
        self.window_indices = np.concatenate(all_window_indices, axis=0)
        
        # Normalize per-epoch if requested (NOT recommended for CV; prefer fold-safe train-only normalization).
        if self.normalize:
            # Z-score across time for each channel within each epoch
            mean = self.data.mean(axis=2, keepdims=True)
            std = self.data.std(axis=2, keepdims=True) + 1e-6
            self.data = (self.data - mean) / std

        if np.isnan(self.data).any() or np.isinf(self.data).any():
            self.shape_fixes['nan_found'] = True
        
        # Add channel dimension for Conv2d: (N, 1, C, T)
        self.data = self.data[:, np.newaxis, :, :]
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        x = torch.FloatTensor(self.data[idx])
        y = torch.LongTensor([self.labels[idx]])[0]

        # Hard shape assertion for safety.
        if x.ndim != 3 or x.shape[0] != 1 or x.shape[1] != self.target_chans or x.shape[2] != self.target_time_points:
            raise RuntimeError(
                f"Epoch shape mismatch at idx={idx}: got {tuple(x.shape)}, expected (1,{self.target_chans},{self.target_time_points})"
            )
        
        # Apply augmentation if enabled (for training)
        if self.aug_transform is not None:
            x = self.aug_transform(x)
        
        if self.transform:
            x = self.transform(x)
            
        return x, y
    
    def get_class_weights(self):
        """Compute class weights for imbalanced data."""
        unique, counts = np.unique(self.labels, return_counts=True)
        weights = len(self.labels) / (len(unique) * counts)
        return torch.FloatTensor(weights)
    
    def get_pids(self):
        """Return participant IDs for GroupKFold splitting."""
        return self.pids


def load_workload_data(data_dir, subset_pids=None, augment=False, **aug_kwargs):
    """
    Load all workload epoch files from a directory.
    
    Args:
        data_dir: Path to directory containing *_workload-epo.fif files
        subset_pids: Optional list of participant IDs to include
        augment: Whether to apply data augmentation
        **aug_kwargs: Additional augmentation parameters (noise_sigma, time_shift, channel_dropout)
        
    Returns:
        MNEEpochsDataset instance
    """
    data_path = Path(data_dir)
    epoch_files = sorted(data_path.glob('*_workload-epo.fif'))
    
    if subset_pids:
        # Filter files by PID
        filtered = []
        for f in epoch_files:
            # Extract PID from filename (e.g., P01_workload-epo.fif -> 1)
            pid_str = f.stem.split('_')[0]  # 'P01'
            pid = int(pid_str[1:])  # 1
            if pid in subset_pids:
                filtered.append(f)
        epoch_files = filtered
    
    if not epoch_files:
        raise ValueError(f"No epoch files found in {data_dir}")
    
    print(f"Loading {len(epoch_files)} epoch files...")
    return MNEEpochsDataset(epoch_files, augment=augment, **aug_kwargs)


def create_group_splits(dataset, n_splits=3, seed=1337):
    """
    Create train/test indices for GroupKFold by participant.
    
    Args:
        dataset: MNEEpochsDataset instance
        n_splits: Number of CV folds
        
    Yields:
        (train_indices, test_indices) for each fold
    """
    try:
        from sklearn.model_selection import StratifiedGroupKFold
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    except Exception:
        from sklearn.model_selection import GroupKFold
        splitter = GroupKFold(n_splits=n_splits)
    
    pids = dataset.get_pids()
    X = np.zeros(len(pids))  # Dummy X for sklearn API
    y = dataset.labels
    
    for train_idx, test_idx in splitter.split(X, y, groups=pids):
        yield train_idx, test_idx


if __name__ == "__main__":
    # Quick test
    DATA_DIR = r'C:\vr_tsst_2025\output\adaptive_workload\mne_epochs'
    SUBSET = [1, 10, 20]
    
    dataset = load_workload_data(DATA_DIR, subset_pids=SUBSET)
    print(f"Dataset size: {len(dataset)}")
    print(f"Data shape: {dataset.data.shape}")  # (N, 1, 128, 1250)
    print(f"Unique PIDs: {np.unique(dataset.pids)}")
    
    # Test DataLoader
    loader = DataLoader(dataset, batch_size=16, shuffle=True)
    batch_x, batch_y = next(iter(loader))
    print(f"Batch X shape: {batch_x.shape}")  # (16, 1, 128, 1250)
    print(f"Batch Y shape: {batch_y.shape}")  # (16,)
    
    # Test GroupKFold splits
    print("\nGroupKFold splits:")
    for fold, (train_idx, test_idx) in enumerate(create_group_splits(dataset)):
        train_pids = np.unique(dataset.pids[train_idx])
        test_pids = np.unique(dataset.pids[test_idx])
        print(f"  Fold {fold+1}: Train PIDs {train_pids}, Test PIDs {test_pids}")
