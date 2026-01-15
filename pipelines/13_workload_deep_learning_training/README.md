# Pipeline 13: Workload Deep Learning Training

## Purpose
This pipeline focuses on training, tuning, and evaluating deep learning models for cognitive workload classification. It consumes the standardized MNE Epochs prepared in Stage 12.

## Scripts
### Data Loading
- `mne_dataloader.py`: Shared utility to load `.fif` files, serving as the interface between Stage 12 and 13.

### Training & Tuning
- `train_tcnet.py` / `tune_tcnet.py`: Temporal Convolutional Network (TCN) implementation and hyperparameter tuning.
- `train_eegnet.py`: EEGNet implementation.
- `train_ensemble.py`: Ensemble methods combining multiple models.
- `train_late_fusion.py` / `train_physio.py`: Multimodal fusion experiments.
- `tune_xgb_bandpower.py`: Gradient boosting baselines on spectral features.

## Input
- `output/adaptive_workload/mne_epochs/*.fif` (from Stage 12)

## Output
- `results/optuna_*.db`: Optimization study databases.
- `models/*.pth`: Trained PyTorch model checkpoints.
- `logs/`: Training logs.

## Usage
To tune a model (e.g., TCNet):
```powershell
python pipelines/13_workload_deep_learning_training/tune_tcnet_focused.py
```

To Train a specific configuration:
```powershell
python pipelines/13_workload_deep_learning_training/train_tcnet.py
```
