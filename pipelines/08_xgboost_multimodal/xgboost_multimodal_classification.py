"""
XGBoost Multimodal Condition Classification with Preprocessing Strategy Comparison

Implements binary classification for:
1. Stress: High vs Low Stress
2. Workload: High vs Low Cognitive Load

Compares multiple preprocessing strategies:
- Baseline adjustment: none, subtract, zscore, percent
- Normalization: none, within_subject, global
- Baseline duration: last_60s, last_90s, last_120s, full

Uses multimodal features (EEG + physiological) from rolling windows.
Removes only near-zero variance features (no correlation pruning).
Outputs comprehensive results, plots, and strategy comparison heatmaps.

Usage:
    python xgboost_multimodal_classification.py
    python xgboost_multimodal_classification.py --quick  # Test subset of strategies

Author: VR-TSST Project
Date: December 2025
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import GroupKFold
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns
import json
from datetime import datetime
import itertools
import time
from typing import Dict, List, Tuple
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
import multiprocessing as mp
warnings.filterwarnings('ignore')


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_DATA = 'output/aggregated/multimodal_features_rolling_windows.csv'
COUNTERBALANCE = 'data/experimental_counterbalance.xlsx'
OUTPUT_DIR = 'output/xgboost_results'
PLOT_DIR = 'output/plots'

# Global runtime flags (set from CLI in main())
TUNE = False
OPTIMIZE_THRESHOLD = False
CALIBRATE = False

# Classification tasks - using only conditions that exist in data
CLASSIFICATION_TASKS = {
    'stress_classification': {
        'HighStress_LowCog_Task': 1,
        'HighStress_HighCog1022_Task': 1,  # HighCog with version 1022
        'LowStress_LowCog_Task': 0,
        'LowStress_HighCog1022_Task': 0,   # HighCog with version 1022
    },
    'workload_classification': {
        'HighStress_HighCog1022_Task': 1,  # HighCog
        'LowStress_HighCog1022_Task': 1,   # HighCog
        'HighStress_LowCog_Task': 0,       # LowCog
        'LowStress_LowCog_Task': 0         # LowCog
    }
}

# Preprocessing strategies
# Locked settings (Dec 2025): based on current results and to avoid
# configuration drift across runs, we standardize on z-score baseline
# adjustment, within-subject normalization, and 90s baseline duration.
BASELINE_ADJUSTMENTS = ['zscore']
NORMALIZATIONS = ['within_subject']  # Only within-subject to avoid data leakage
BASELINE_DURATIONS = ['last_60s', 'last_90s', 'last_120s', 'full']

# Model configurations
MODELS = {
    'xgboost': {
        'name': 'XGBoost',
        'params': {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'learning_rate': 0.1,
            'max_depth': 6,
            'n_estimators': 100,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'use_label_encoder': False
        }
    },
    'random_forest': {
        'name': 'Random Forest',
        'params': {
            'n_estimators': 100,
            'max_depth': 10,
            'random_state': 42,
            'n_jobs': 1
        }
    },
    'logistic_regression': {
        'name': 'Logistic Regression',
        'params': {
            'max_iter': 1000,
            'random_state': 42,
            'n_jobs': 1
        }
    },
    'svm': {
        'name': 'SVM',
        'params': {
            'kernel': 'rbf',
            'C': 1.0,
            'gamma': 'scale',
            'random_state': 42,
            'probability': True
        }
    },
    'mlp': {
        'name': 'Neural Network (MLP)',
        'params': {
            'hidden_layer_sizes': (100, 50),
            'activation': 'relu',
            'solver': 'adam',
            'alpha': 0.0001,
            'batch_size': 32,
            'learning_rate': 'adaptive',
            'max_iter': 500,
            'random_state': 42,
            'early_stopping': True,
            'validation_fraction': 0.1
        }
    },
    'lightgbm': {
        'name': 'LightGBM',
        'params': {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.1,
            'n_estimators': 100,
            'random_state': 42,
            'verbose': -1
        }
    },
    'knn': {
        'name': 'K-Nearest Neighbors',
        'params': {
            'n_neighbors': 5,
            'weights': 'distance',
            'metric': 'minkowski',
            'n_jobs': 1
        }
    },
    'lda': {
        'name': 'Linear Discriminant Analysis',
        'params': {
            'solver': 'svd',
            'shrinkage': None
        }
    }
}

# Small hyperparameter grids for inner GroupKFold tuning
PARAM_GRIDS = {
    'logistic_regression': [
        {'C': c, 'penalty': pen, 'solver': 'saga', 'max_iter': 5000, 'class_weight': 'balanced'}
        for c in [0.1, 0.5, 1.0, 2.0, 5.0]
        for pen in ['l2', 'l1']
    ],
    'svm': [
        {'C': c, 'gamma': g, 'probability': True, 'class_weight': 'balanced'}
        for c in [0.5, 1.0, 2.0, 5.0]
        for g in ['scale', 0.1, 0.01]
    ],
    'random_forest': [
        {'n_estimators': n, 'max_depth': d, 'min_samples_leaf': m, 'class_weight': 'balanced_subsample'}
        for n in [200, 500]
        for d in [None, 10, 20]
        for m in [1, 5]
    ],
    'xgboost': [
        {'n_estimators': n, 'max_depth': d, 'learning_rate': eta, 'subsample': ss, 'colsample_bytree': cs}
        for n in [200, 400]
        for d in [4, 6, 8]
        for eta in [0.05, 0.1]
        for ss in [0.7, 0.9]
        for cs in [0.7, 0.9]
    ],
    'lda': [
        {'solver': s} for s in ['svd', 'lsqr']
    ],
    'knn': [
        {'n_neighbors': k, 'weights': w}
        for k in [5, 11, 21]
        for w in ['uniform', 'distance']
    ],
    'mlp': [
        {'hidden_layer_sizes': h, 'alpha': a}
        for h in [(50,), (100,), (50, 50)]
        for a in [0.0001, 0.001]
    ]
}

# Feature modalities to test
FEATURE_MODALITIES = ['all', 'eeg', 'physio']


# ============================================================
# DATA LOADING AND PREPARATION
# ============================================================

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="XGBoost multimodal classification with strategy comparison"
    )
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Quick test mode: run only a subset of strategies'
    )
    parser.add_argument(
        '--input',
        type=str,
        default=INPUT_DATA,
        help='Path to multimodal rolling window features'
    )
    parser.add_argument(
        '--n-jobs',
        type=int,
        default=max(1, mp.cpu_count() - 2),
        help='Number of parallel jobs (default: CPU cores - 2)'
    )
    parser.add_argument(
        '--tune', action='store_true',
        help='Enable small hyperparameter tuning with inner GroupKFold (default: off)'
    )
    parser.add_argument(
        '--optimize-threshold', action='store_true',
        help='Optimize decision threshold using inner OOF predictions to maximize balanced accuracy (default: off)'
    )
    parser.add_argument(
        '--calibrate', action='store_true',
        help='Calibrate probabilities on training folds (sigmoid) (default: off)'
    )
    return parser.parse_args()


def load_data(input_path: str) -> pd.DataFrame:
    """Load multimodal features."""
    print(f"[INFO] Loading data from: {input_path}")
    df = pd.read_csv(input_path)
    print(f"[INFO] Loaded {df.shape[0]} windows, {df.shape[1]} columns")
    print(f"[INFO] Participants: {df['Participant_ID'].nunique()}")
    print(f"[INFO] Conditions: {df['Condition'].unique()}")
    return df


def identify_feature_columns(df: pd.DataFrame) -> Dict[str, List[str]]:
    """Identify EEG and physiological feature columns."""
    
    # Metadata columns to exclude (including SCR event markers)
    meta_cols = ['Participant_ID', 'Condition', 'Window_Index', 'Window_Start', 
                 'Window_End', 'class_label', 'meaningful_scrs', 'low_var_flag', 
                 'total_scrs', 'participant_id', 'condition', 'pid', 'event_label', 'window_idx']
    
    all_cols = [col for col in df.columns if col not in meta_cols]
    
    # EEG features: band power features by region (exclude Delta, Gamma, Occipital)
    eeg_features = [col for col in all_cols if any(
        x in col for x in ['Theta', 'Alpha', 'Beta',
                           'Frontal', 'Parietal', 'Temporal', 'Central']
    )]
    
    # Physio features: HR, GSR, pupil (exclude Blink, HRV, Peak, Tonic)
    # Include patterns capture Polar/HeartRate/RR Interval/BPM plus GSR/EDA/Pupil/Shimmer
    physio_include_patterns = [
        'HeartRate', 'RR_Interval', 'BPM', 'Polar',  # heart metrics
        'GSR', 'EDA', 'Pupil', 'Shimmer'            # gsr/pupil/shimmer
    ]
    physio_features = [col for col in all_cols if any(
        x in col for x in physio_include_patterns
    )]
    
    # Exclude unwanted patterns
    exclude_patterns = [
        'Delta', 'Gamma', 'Occipital',
        'Blink', 'HRV', 'Peak', 'Tonic', 'peak', 'tonic',
        # Exclude SCR event count/proportion features
        'TotalSCRs', 'MeaningfulSCRs', 'ProportionMeaningful', 'SCRs'
    ]
    eeg_features = [f for f in eeg_features if not any(pat in f for pat in exclude_patterns)]
    physio_features = [f for f in physio_features if not any(pat in f for pat in exclude_patterns)]
    
    # Remove any overlap
    physio_features = [f for f in physio_features if f not in eeg_features]
    
    # Filter to only numeric columns
    eeg_features = [f for f in eeg_features if df[f].dtype in ['float64', 'float32', 'int64', 'int32']]
    physio_features = [f for f in physio_features if df[f].dtype in ['float64', 'float32', 'int64', 'int32']]
    
    print(f"[INFO] Feature groups:")
    print(f"  EEG features: {len(eeg_features)}")
    print(f"  Physio features: {len(physio_features)}")
    print(f"  Total features: {len(eeg_features) + len(physio_features)}")
    
    return {
        'eeg': eeg_features,
        'physio': physio_features,
        'all': eeg_features + physio_features
    }


# ============================================================
# BASELINE ADJUSTMENT
# ============================================================

def get_baseline(
    df: pd.DataFrame,
    participant_id: int,
    task_condition: str,
    counterbalance_data: pd.DataFrame,
    baseline_duration: str
) -> pd.DataFrame:
    """Get baseline for a specific task."""
    
    # Get counterbalance info
    p_cb = counterbalance_data[counterbalance_data['Participant'] == participant_id]
    if len(p_cb) == 0:
        return None
    
    # Find round number for this task
    condition_map = {
        'Calm Addition': 'LowStress_LowCog_Task',
        'Calm Subtraction': 'LowStress_HighCog_Task',
        'Stress Addition': 'HighStress_LowCog_Task',
        'Stress Subtraction': 'HighStress_HighCog_Task'
    }
    
    # Strip version suffixes (1022, 2043) for matching  
    # e.g., 'HighStress_HighCog1022_Task' -> 'HighStress_HighCog_Task'
    task_base = task_condition
    task_base = task_base.replace('HighCog1022', 'HighCog')
    task_base = task_base.replace('HighCog2043', 'HighCog')
    
    round_num = None
    for r in [1, 2, 3, 4]:
        round_cond = p_cb[f'Round {r}'].values[0]
        mapped_cond = condition_map.get(round_cond, round_cond)
        if mapped_cond == task_base:  # Compare against base version
            round_num = r
            break
    
    if round_num is None:
        return None
    
    # Get forest data
    forest_cond = f'Forest{round_num}'
    if forest_cond not in df['Condition'].values:
        forest_cond = f'Relaxation{round_num}'
    
    forest_data = df[
        (df['Participant_ID'] == participant_id) &
        (df['Condition'] == forest_cond)
    ].copy()
    
    if len(forest_data) == 0:
        return None
    
    # Apply duration windowing
    if baseline_duration != 'full':
        time_col = 'Window_Start' if 'Window_Start' in forest_data.columns else 'Adjusted_Time'
        if time_col in forest_data.columns:
            max_time = forest_data[time_col].max()
            
            duration_seconds = {
                'last_60s': 60,
                'last_90s': 90,
                'last_120s': 120
            }
            
            cutoff = max_time - duration_seconds.get(baseline_duration, 0)
            forest_data = forest_data[forest_data[time_col] >= cutoff]
    
    return forest_data


def apply_baseline_adjustment(
    task_df: pd.DataFrame,
    full_df: pd.DataFrame,
    method: str,
    duration: str,
    feature_cols: List[str],
    counterbalance_data: pd.DataFrame
) -> pd.DataFrame:
    """Apply baseline adjustment strategy.
    
    Args:
        task_df: DataFrame with only task conditions
        full_df: Full DataFrame including Forest/Relaxation baseline conditions
        method: Adjustment method (subtract, zscore, percent)
        duration: Baseline duration to use
        feature_cols: Feature columns to adjust
        counterbalance_data: Counterbalance sheet
    
    Returns:
        Adjusted task DataFrame
    """
    
    if method == 'none':
        return task_df  # No adjustment
    
    print(f"[INFO] Applying baseline adjustment: {method}, duration: {duration}")
    
    adjusted_windows = []
    failed_participants = []
    total_participants = len(task_df['Participant_ID'].unique())
    
    for pid in task_df['Participant_ID'].unique():
        # Get task data for this participant
        p_task_data = task_df[task_df['Participant_ID'] == pid].copy()
        # Get FULL data for this participant (including Forest)
        p_full_data = full_df[full_df['Participant_ID'] == pid].copy()
        task_conditions = [c for c in p_task_data['Condition'].unique() if 'Task' in c]
        
        if len(task_conditions) == 0:
            failed_participants.append(pid)
            continue
        
        participant_had_baseline = False
        
        for task_cond in task_conditions:
            task_windows = p_task_data[p_task_data['Condition'] == task_cond].copy()
            
            if len(task_windows) == 0:
                continue
            
            # Get baseline from FULL data (which includes Forest conditions)
            baseline_data = get_baseline(
                df=p_full_data,
                participant_id=pid,
                task_condition=task_cond,
                counterbalance_data=counterbalance_data,
                baseline_duration=duration
            )
            
            if baseline_data is None or len(baseline_data) == 0:
                continue
            
            participant_had_baseline = True
            baseline_mean = baseline_data[feature_cols].mean()
            baseline_sd = baseline_data[feature_cols].std()
            
            # Apply method
            if method == 'subtract':
                task_windows[feature_cols] = task_windows[feature_cols].values - baseline_mean.values
            
            elif method == 'percent':
                for col in feature_cols:
                    if baseline_mean[col] != 0:
                        task_windows[col] = (
                            (task_windows[col] - baseline_mean[col]) / 
                            abs(baseline_mean[col]) * 100
                        )
            
            elif method == 'zscore':
                for col in feature_cols:
                    if baseline_sd[col] > 0:
                        task_windows[col] = (
                            (task_windows[col] - baseline_mean[col]) / 
                            baseline_sd[col]
                        )
            
            adjusted_windows.append(task_windows)
        
        if not participant_had_baseline:
            failed_participants.append(pid)
    
    # Handle empty results gracefully with detailed error message
    if len(adjusted_windows) == 0:
        print(f"[ERROR] Baseline adjustment failed for ALL {total_participants} participants!")
        print(f"[ERROR] No Forest/Relaxation baseline data found for any participant")
        if failed_participants:
            print(f"[ERROR] Failed participants: {failed_participants[:10]}")
        print(f"[ERROR] Check that:")
        print(f"  1. Data file contains Forest1-4 conditions")
        print(f"  2. Participant IDs match between data and counterbalance sheet")
        print(f"  3. Counterbalance sheet path is correct: {COUNTERBALANCE}")
        return pd.DataFrame()  # Return empty instead of crashing
    
    if failed_participants:
        print(f"[WARNING] {len(failed_participants)}/{total_participants} participants had no matching baseline")
        print(f"[WARNING] First 5 failed: {failed_participants[:5]}")
    
    return pd.concat(adjusted_windows, ignore_index=True)


# ============================================================
# NORMALIZATION
# ============================================================

def apply_normalization(
    df: pd.DataFrame,
    method: str,
    feature_cols: List[str]
) -> pd.DataFrame:
    """Apply normalization strategy.
    
    Only within-subject z-scoring is used to avoid data leakage.
    Each participant's data is normalized independently using only their own
    mean and std, preventing test set information from leaking into training.
    """
    
    print(f"[INFO] Applying normalization: {method}")
    
    df_norm = df.copy()
    
    if method == 'within_subject':
        # Z-score within each participant (no data leakage)
        # Each participant normalized independently before train/test split
        for pid in df['Participant_ID'].unique():
            mask = df['Participant_ID'] == pid
            for col in feature_cols:
                data = df.loc[mask, col]
                mean = data.mean()
                std = data.std()
                if std > 0:
                    df_norm.loc[mask, col] = (data - mean) / std
                else:
                    df_norm.loc[mask, col] = 0
    
    return df_norm


# ============================================================
# FEATURE PRUNING
# ============================================================

def prune_features(df: pd.DataFrame, feature_cols: List[str]) -> Tuple[pd.DataFrame, List[str]]:
    """Remove near-zero variance features only."""
    
    print(f"[INFO] Pruning features (variance threshold only)...")
    print(f"  Input features: {len(feature_cols)}")
    
    # Near-zero variance removal
    vt = VarianceThreshold(threshold=1e-6)
    X_pruned = vt.fit_transform(df[feature_cols])
    kept_indices = vt.get_support(indices=True)
    kept_features = [feature_cols[i] for i in kept_indices]
    
    print(f"  After variance pruning: {len(kept_features)}")
    print(f"  Removed: {len(feature_cols) - len(kept_features)} features")
    
    df_pruned = df.copy()
    df_pruned[kept_features] = X_pruned
    
    return df_pruned, kept_features


# ============================================================
# MODEL TRAINING AND EVALUATION
# ============================================================

def train_evaluate_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_type: str,
    model_params: Dict
) -> Dict:
    """Train model and evaluate."""
    
    # Create model based on type
    if model_type == 'xgboost':
        model = xgb.XGBClassifier(**model_params)
    elif model_type == 'random_forest':
        model = RandomForestClassifier(**model_params)
    elif model_type == 'logistic_regression':
        model = LogisticRegression(**model_params)
    elif model_type == 'svm':
        model = SVC(**model_params)
    elif model_type == 'mlp':
        model = MLPClassifier(**model_params)
    elif model_type == 'lightgbm':
        if not LIGHTGBM_AVAILABLE:
            raise ValueError("LightGBM not installed. Install with: pip install lightgbm")
        model = lgb.LGBMClassifier(**model_params)
    elif model_type == 'knn':
        model = KNeighborsClassifier(**model_params)
    elif model_type == 'lda':
        model = LinearDiscriminantAnalysis(**model_params)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
    
    # Metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0),
        'auc': roc_auc_score(y_test, y_pred_proba) if (y_pred_proba is not None and len(np.unique(y_test)) > 1) else 0,
        'y_pred': y_pred,
        'y_test': y_test,
        'model': model
    }
    
    return metrics


def inner_group_oof(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    model_type: str,
    base_params: Dict,
    params: Dict,
    calibrate: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute inner OOF probabilities via 3-fold GroupKFold for given params."""
    n_splits = min(3, len(np.unique(groups)))
    gkf = GroupKFold(n_splits=n_splits)
    oof = np.zeros_like(y, dtype=float)
    for train_idx, val_idx in gkf.split(X, y, groups):
        X_tr, X_va = X[train_idx], X[val_idx]
        y_tr = y[train_idx]
        # Merge base and candidate params
        use_params = dict(base_params)
        use_params.update(params)
        # Model factory
        if model_type == 'xgboost':
            model = xgb.XGBClassifier(**use_params)
        elif model_type == 'random_forest':
            model = RandomForestClassifier(**use_params)
        elif model_type == 'logistic_regression':
            model = LogisticRegression(**use_params)
        elif model_type == 'svm':
            model = SVC(**use_params)
        elif model_type == 'mlp':
            model = MLPClassifier(**use_params)
        elif model_type == 'lightgbm':
            if not LIGHTGBM_AVAILABLE:
                raise ValueError("LightGBM not installed. Install with: pip install lightgbm")
            model = lgb.LGBMClassifier(**use_params)
        elif model_type == 'knn':
            model = KNeighborsClassifier(**use_params)
        elif model_type == 'lda':
            model = LinearDiscriminantAnalysis(**use_params)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        if calibrate:
            from sklearn.calibration import CalibratedClassifierCV
            model = CalibratedClassifierCV(model, method='sigmoid', cv=2)
        model.fit(X_tr, y_tr)
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(X_va)[:, 1]
        else:
            # fallback: decision_function scaled via sigmoid
            if hasattr(model, 'decision_function'):
                from scipy.special import expit
                proba = expit(model.decision_function(X_va))
            else:
                # no proba; use predictions (0/1)
                proba = model.predict(X_va)
        oof[val_idx] = proba
    return oof, y


def tune_params_and_threshold(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    model_type: str,
    base_params: Dict,
    enable_tune: bool,
    optimize_threshold: bool,
    calibrate: bool
) -> Tuple[Dict, float]:
    """Return best params (if tuned) and decision threshold (if optimized)."""
    best_params = {}
    best_threshold = 0.5
    if enable_tune and model_type in PARAM_GRIDS:
        best_auc = -1
        for params in PARAM_GRIDS[model_type]:
            oof, y_true = inner_group_oof(X, y, groups, model_type, base_params, params, calibrate)
            try:
                auc = roc_auc_score(y_true, oof)
            except Exception:
                auc = 0
            if auc > best_auc:
                best_auc = auc
                best_params = params
        # After selecting best params, optionally get threshold on OOF
        if optimize_threshold:
            from sklearn.metrics import balanced_accuracy_score
            oof, y_true = inner_group_oof(X, y, groups, model_type, base_params, best_params, calibrate)
            # Scan thresholds
            thresholds = np.linspace(0.2, 0.8, 25)
            scores = []
            for t in thresholds:
                y_hat = (oof >= t).astype(int)
                scores.append(balanced_accuracy_score(y_true, y_hat))
            best_threshold = float(thresholds[int(np.argmax(scores))])
    return best_params, best_threshold


def run_cross_validation(
    df: pd.DataFrame,
    feature_cols: List[str],
    task_name: str,
    model_type: str,
    model_params: Dict
) -> Dict:
    """Run GroupKFold cross-validation."""
    
    X = df[feature_cols].values
    y = df['class_label'].values
    groups = df['Participant_ID'].values
    
    # Handle NaN/Inf
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    n_splits = min(5, len(np.unique(groups)))
    gkf = GroupKFold(n_splits=n_splits)
    
    fold_results = []
    all_y_true = []
    all_y_pred = []
    window_acc = {}
    window_counts = {}
    feature_importances = []
    fold_thresholds = []
    fold_best_params = []
    
    print(f"[INFO] Running {n_splits}-fold cross-validation...")
    
    for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        groups_train = groups[train_idx]
        
        # Tune hyperparameters and threshold on training groups only (if enabled)
        best_params, best_threshold = tune_params_and_threshold(
            X_train, y_train, groups_train,
            model_type, MODELS[model_type]['params'],
            enable_tune=TUNE, optimize_threshold=OPTIMIZE_THRESHOLD, calibrate=CALIBRATE
        )
        fold_best_params.append(best_params)
        # Merge base params with tuned params
        cur_params = dict(MODELS[model_type]['params'])
        cur_params.update(best_params)
        # Class weighting for imbalance (if not in tuned params)
        if model_type in ['logistic_regression', 'svm'] and 'class_weight' not in cur_params:
            cur_params['class_weight'] = 'balanced'
        # XGBoost scale_pos_weight (pos/neg ratio)
        if model_type == 'xgboost':
            pos = float((y_train == 1).sum())
            neg = float((y_train == 0).sum())
            if pos > 0:
                cur_params['scale_pos_weight'] = max(1.0, neg / pos)
        
        # Train and evaluate
        fold_metrics = train_evaluate_model(X_train, y_train, X_test, y_test, model_type, cur_params)
        
        fold_results.append({
            'fold': fold_idx,
            'accuracy': fold_metrics['accuracy'],
            'precision': fold_metrics['precision'],
            'recall': fold_metrics['recall'],
            'f1': fold_metrics['f1'],
            'auc': fold_metrics['auc'],
            'n_train': len(X_train),
            'n_test': len(X_test)
        })
        
        # Optional threshold optimization using tuned threshold
        if OPTIMIZE_THRESHOLD and hasattr(fold_metrics['model'], 'predict_proba'):
            y_proba = fold_metrics['model'].predict_proba(X_test)[:, 1]
            y_pred_thresh = (y_proba >= best_threshold).astype(int)
            # Recompute metrics with thresholded predictions
            fold_results[-1].update({
                'accuracy': accuracy_score(y_test, y_pred_thresh),
                'precision': precision_score(y_test, y_pred_thresh, zero_division=0),
                'recall': recall_score(y_test, y_pred_thresh, zero_division=0),
                'f1': f1_score(y_test, y_pred_thresh, zero_division=0)
            })
            # Replace in-memory predictions for downstream summaries
            fold_metrics['y_pred'] = y_pred_thresh
            fold_thresholds.append(best_threshold)
        
        all_y_true.extend(fold_metrics['y_test'])
        all_y_pred.extend(fold_metrics['y_pred'])
        
        # Feature importance (if available)
        if hasattr(fold_metrics['model'], 'feature_importances_'):
            feature_importances.append(fold_metrics['model'].feature_importances_)
        elif hasattr(fold_metrics['model'], 'coef_'):
            feature_importances.append(np.abs(fold_metrics['model'].coef_[0]))
        
        # Window-level accuracy tracking
        if 'Window_Index' in df.columns:
            test_df = df.iloc[test_idx]
            for idx, (win_idx, true_label, pred_label) in enumerate(
                zip(test_df['Window_Index'], fold_metrics['y_test'], fold_metrics['y_pred'])
            ):
                if win_idx not in window_acc:
                    window_acc[win_idx] = 0
                    window_counts[win_idx] = 0
                window_acc[win_idx] += int(true_label == pred_label)
                window_counts[win_idx] += 1
    
    # Aggregate results
    metrics_df = pd.DataFrame(fold_results)
    avg_feature_importance = np.mean(feature_importances, axis=0)
    
    # Aggregate tuned params and threshold across folds
    tuned_params_agg = {}
    if fold_best_params:
        try:
            from collections import Counter
            keys = [json.dumps(p, sort_keys=True) for p in fold_best_params]
            common = Counter(keys).most_common(1)[0][0]
            tuned_params_agg = json.loads(common)
        except Exception:
            tuned_params_agg = fold_best_params[0] if fold_best_params else {}
    threshold_agg = float(np.median(fold_thresholds)) if fold_thresholds else None

    results = {
        'fold_metrics': metrics_df,
        'accuracy_mean': metrics_df['accuracy'].mean(),
        'accuracy_std': metrics_df['accuracy'].std(),
        'precision_mean': metrics_df['precision'].mean(),
        'recall_mean': metrics_df['recall'].mean(),
        'f1_mean': metrics_df['f1'].mean(),
        'auc_mean': metrics_df['auc'].mean(),
        'all_y_true': all_y_true,
        'all_y_pred': all_y_pred,
        'window_acc': window_acc,
        'window_counts': window_counts,
        'feature_importance': avg_feature_importance,
        'feature_names': feature_cols,
        'n_train_mean': metrics_df['n_train'].mean(),
        'n_test_mean': metrics_df['n_test'].mean(),
        'tuned_params': tuned_params_agg,
        'threshold': threshold_agg
    }
    
    # Classification report
    print(f"\n[RESULTS] {task_name}")
    print(f"  Accuracy: {results['accuracy_mean']:.3f} ± {results['accuracy_std']:.3f}")
    print(f"  Precision: {results['precision_mean']:.3f}")
    print(f"  Recall: {results['recall_mean']:.3f}")
    print(f"  F1: {results['f1_mean']:.3f}")
    print(f"  AUC: {results['auc_mean']:.3f}")
    
    return results


# ============================================================
# PLOTTING
# ============================================================

def plot_window_accuracy(
    window_acc: Dict,
    window_counts: Dict,
    task_name: str,
    strategy_name: str,
    output_dir: str
):
    """Plot accuracy by window index (reuse SVM logic)."""
    
    if not window_acc:
        return
    
    window_idxs = sorted(window_acc.keys())
    accs = [window_acc[w] / window_counts[w] for w in window_idxs]
    
    plt.figure(figsize=(10, 4))
    plt.plot(window_idxs, accs, marker='o', linestyle='-', color='b', linewidth=2)
    
    # Annotate max
    max_idx = np.argmax(accs)
    max_win = window_idxs[max_idx]
    max_acc = accs[max_idx]
    plt.annotate(
        f'Max: {max_acc:.2f}',
        xy=(max_win, max_acc),
        xytext=(max_win, max_acc + 0.05),
        arrowprops=dict(facecolor='red', shrink=0.05),
        ha='center',
        color='red',
        fontsize=10
    )
    
    plt.title(f'Window Accuracy: {task_name.replace("_", " ").title()}\n{strategy_name}')
    plt.xlabel('Window Index')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    
    # Improved naming: task_strategy_window_accuracy.png
    plot_path = os.path.join(output_dir, f'{task_name}_{strategy_name}_window_acc.png')
    plt.savefig(plot_path, dpi=150)
    plt.close()


def plot_strategy_heatmap(results_df: pd.DataFrame, task_name: str, output_dir: str):
    """Plot heatmap of strategies."""
    
    # Pivot for heatmap: baseline_adjustment x normalization (aggregate over durations)
    pivot = results_df.groupby(['baseline_adjustment', 'normalization'])['accuracy_mean'].mean().unstack()
    
    plt.figure(figsize=(10, 6))
    sns.heatmap(
        pivot,
        annot=True,
        fmt='.3f',
        cmap='RdYlGn',
        vmin=0.5,
        vmax=1.0,
        cbar_kws={'label': 'Mean Accuracy'}
    )
    plt.title(f'Strategy Comparison: {task_name.replace("_", " ").title()}')
    plt.xlabel('Normalization Method')
    plt.ylabel('Baseline Adjustment Method')
    plt.tight_layout()
    
    plot_path = os.path.join(output_dir, f'{task_name}_strategy_heatmap.png')
    plt.savefig(plot_path, dpi=150)
    plt.close()
    
    print(f"[INFO] Saved heatmap: {plot_path}")


def plot_feature_importance(
    feature_importance: np.ndarray,
    feature_names: List[str],
    task_name: str,
    output_dir: str,
    top_n: int = 20
):
    """Plot top N feature importances."""
    
    # Handle cases where feature importance is not available
    if feature_importance is None:
        print(f"[INFO] No feature importance available for {task_name} (model doesn't support it)")
        return
    
    # Convert to array and handle 0-d arrays
    feature_importance = np.atleast_1d(np.asarray(feature_importance))
    
    if len(feature_importance) == 0 or (feature_importance.ndim == 1 and feature_importance.shape[0] == 0):
        print(f"[INFO] No feature importance available for {task_name} (empty array)")
        return
    
    # Sort features by importance
    try:
        importance_pairs = sorted(
            zip(feature_names, feature_importance),
            key=lambda x: abs(x[1]),  # Use abs in case of negative values
            reverse=True
        )[:top_n]
    except (TypeError, ValueError) as e:
        print(f"[INFO] Could not plot feature importance for {task_name}: {e}")
        return
    
    if len(importance_pairs) == 0:
        print(f"[INFO] No features to plot for {task_name}")
        return
    
    features, importances = zip(*importance_pairs)
    
    plt.figure(figsize=(10, 8))
    plt.barh(range(len(features)), importances, color='steelblue')
    plt.yticks(range(len(features)), features, fontsize=8)
    plt.xlabel('Importance')
    plt.title(f'Top {top_n} Features: {task_name.replace("_", " ").title()}')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    
    plot_path = os.path.join(output_dir, f'{task_name}_top_features.png')
    plt.savefig(plot_path, dpi=150)
    plt.close()
    
    print(f"[INFO] Saved feature importance plot: {plot_path}")


# ============================================================
# MAIN PIPELINE
# ============================================================

def run_single_strategy(
    df: pd.DataFrame,
    task_name: str,
    task_map: Dict,
    baseline_adj: str,
    norm: str,
    duration: str,
    counterbalance_data: pd.DataFrame,
    feature_groups: Dict,
    model_type: str = 'xgboost',
    modality: str = 'all'
) -> Dict:
    """Run one preprocessing strategy."""
    
    start_time = time.time()
    
    strategy_name = f"{baseline_adj}_{norm}_{duration}_{model_type}_{modality}"
    print(f"\n{'='*70}")
    print(f"Strategy: {strategy_name}")
    print(f"{'='*70}")
    
    try:
        # Filter to task conditions
        task_df = df[df['Condition'].isin(task_map.keys())].copy()
        task_df['class_label'] = task_df['Condition'].map(task_map)
        
        if len(task_df) == 0:
            print(f"[WARNING] No data for task {task_name}")
            return None
        
        print(f"[INFO] Task data: {len(task_df)} windows")
        print(f"[INFO] Model: {MODELS[model_type]['name']}")
        print(f"[INFO] Modality: {modality}")
        
        # Select features based on modality
        selected_features = feature_groups[modality]
        if len(selected_features) == 0:
            print(f"[WARNING] No features for modality {modality}")
            return None
        
        print(f"[INFO] Using {len(selected_features)} {modality} features")
        
        # Apply baseline adjustment
        if baseline_adj != 'none':
            task_df = apply_baseline_adjustment(
                task_df=task_df,
                full_df=df,  # Pass full dataframe with Forest conditions
                method=baseline_adj,
                duration=duration,
                feature_cols=selected_features,
                counterbalance_data=counterbalance_data
            )
            
            if len(task_df) == 0:
                print(f"[WARNING] No data after baseline adjustment")
                return None
        
        # Apply normalization
        task_df = apply_normalization(
            df=task_df,
            method=norm,
            feature_cols=selected_features
        )
        
        # Prune features
        task_df, kept_features = prune_features(task_df, selected_features)
        
        if len(kept_features) == 0:
            print(f"[WARNING] No features remaining after pruning")
            return None

        # Log final feature list used for this run (for transparency)
        try:
            feature_list_dir = os.path.join(OUTPUT_DIR, 'feature_lists')
            os.makedirs(feature_list_dir, exist_ok=True)
            safe_model = MODELS[model_type]['name'].replace(' ', '')
            feature_file = os.path.join(
                feature_list_dir,
                f"{task_name}_{modality}_{baseline_adj}_{norm}_{duration}_{safe_model}_features.txt"
            )
            with open(feature_file, 'w', encoding='utf-8') as f:
                f.write(f"Task: {task_name}\nModel: {MODELS[model_type]['name']}\nModality: {modality}\n")
                f.write(f"Baseline: {baseline_adj}\nNormalization: {norm}\nDuration: {duration}\n")
                f.write(f"Feature count: {len(kept_features)}\n\n")
                for feat in kept_features:
                    f.write(feat + "\n")
            print(f"[INFO] Wrote feature list: {feature_file}")
        except Exception as _log_err:
            print(f"[WARNING] Could not write feature list: {_log_err}")
        
        # Cross-validation
        cv_results = run_cross_validation(
            task_df, kept_features, task_name, 
            model_type, MODELS[model_type]['params']
        )
        
        # Plotting
        plot_window_accuracy(
            cv_results['window_acc'],
            cv_results['window_counts'],
            task_name,
            strategy_name,
            PLOT_DIR
        )
        
        # Best window accuracy
        if cv_results['window_acc']:
            window_accs = [
                cv_results['window_acc'][w] / cv_results['window_counts'][w]
                for w in cv_results['window_acc'].keys()
            ]
            best_window_acc = max(window_accs)
        else:
            best_window_acc = 0
        
        elapsed_time = time.time() - start_time
        
        # Compile results
        result = {
            'task': task_name,
            'model': MODELS[model_type]['name'],
            'modality': modality,
            'baseline_adjustment': baseline_adj,
            'normalization': norm,
            'baseline_duration': duration,
            'n_features': len(kept_features),
            'n_windows_train': cv_results['n_train_mean'],
            'n_windows_test': cv_results['n_test_mean'],
            'accuracy_mean': cv_results['accuracy_mean'],
            'accuracy_std': cv_results['accuracy_std'],
            'precision_mean': cv_results['precision_mean'],
            'recall_mean': cv_results['recall_mean'],
            'f1_mean': cv_results['f1_mean'],
            'auc_mean': cv_results['auc_mean'],
            'best_window_accuracy': best_window_acc,
            'training_time_seconds': elapsed_time,
            'feature_importance': cv_results['feature_importance'].tolist(),
            'feature_names': cv_results['feature_names'],
            'tuned_params': cv_results.get('tuned_params', {}),
            'threshold': cv_results.get('threshold', None)
        }
        
        return result
        
    except Exception as e:
        print(f"[ERROR] Strategy failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def run_strategy_wrapper(args_tuple):
    """
    Wrapper function for parallel execution.
    Unpacks arguments and runs a single strategy.
    """
    (
        task_name, task_map, baseline_adj, norm, duration,
        model_type, modality, df, counterbalance, feature_groups,
        tune_flag, optimize_threshold_flag, calibrate_flag
    ) = args_tuple
    
    # Set global flags for this worker process
    global TUNE, OPTIMIZE_THRESHOLD, CALIBRATE
    TUNE = tune_flag
    OPTIMIZE_THRESHOLD = optimize_threshold_flag
    CALIBRATE = calibrate_flag
    
    return run_single_strategy(
        df=df,
        task_name=task_name,
        task_map=task_map,
        baseline_adj=baseline_adj,
        norm=norm,
        duration=duration,
        counterbalance_data=counterbalance,
        feature_groups=feature_groups,
        model_type=model_type,
        modality=modality
    )


def main():
    """Main pipeline."""
    
    args = parse_arguments()
    # Set global flags
    global TUNE, OPTIMIZE_THRESHOLD, CALIBRATE
    TUNE = bool(args.tune)
    OPTIMIZE_THRESHOLD = bool(args.optimize_threshold)
    CALIBRATE = bool(args.calibrate)
    
    # Create output directories
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(PLOT_DIR, exist_ok=True)
    
    print("="*70)
    print("XGBOOST MULTIMODAL CONDITION CLASSIFICATION")
    print("="*70)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Load data
    df = load_data(args.input)
    feature_groups = identify_feature_columns(df)
    
    # Load counterbalance
    counterbalance = pd.read_excel(COUNTERBALANCE)
    print(f"[INFO] Loaded counterbalance data: {len(counterbalance)} participants\n")
    
    # Define strategies
    if args.quick:
        print("[INFO] Quick mode: locked settings (zscore, within_subject, last_90s)")
        strategies = [('zscore', 'within_subject', 'last_90s')]
        test_models = ['xgboost']
        test_modalities = ['all']
    else:
        # Locked configuration for future runs (Dec 2025):
        # baseline_adjustment='zscore', normalization='within_subject', baseline_duration='last_90s'
        # Rationale: strong and stable across tasks; avoids leakage and reduces runtime.
        strategies = [('zscore', 'within_subject', 'last_90s')]
        # Filter out unavailable models
        available_models = list(MODELS.keys())
        if not LIGHTGBM_AVAILABLE:
            available_models = [m for m in available_models if m != 'lightgbm']
            print("[INFO] LightGBM not available, skipping")
        test_models = available_models
        test_modalities = FEATURE_MODALITIES
    
    total_combinations = len(strategies) * len(CLASSIFICATION_TASKS) * len(test_models) * len(test_modalities)
    print(f"[INFO] Total combinations to test: {total_combinations}")
    print(f"[INFO]   - {len(strategies)} preprocessing strategies")
    print(f"[INFO]   - {len(test_models)} models: {test_models}")
    print(f"[INFO]   - {len(test_modalities)} modalities: {test_modalities}")
    print(f"[INFO]   - {len(CLASSIFICATION_TASKS)} tasks")
    print(f"[INFO] Parallel jobs: {args.n_jobs}\n")
    if TUNE:
        print("[INFO] Hyperparameter tuning: enabled (small grids, inner GroupKFold)")
    if OPTIMIZE_THRESHOLD:
        print("[INFO] Threshold optimization: enabled (balanced accuracy on inner OOF)")
    if CALIBRATE:
        print("[INFO] Probability calibration: enabled (sigmoid)")
    
    all_results = []
    best_models = {}
    
    # Checkpoint file for incremental saving
    checkpoint_path = os.path.join(OUTPUT_DIR, 'checkpoint_results.json')
    checkpoint_csv_path = os.path.join(OUTPUT_DIR, 'checkpoint_results.csv')
    
    # Prepare all strategy combinations as arguments
    strategy_args = []
    for task_name, task_map in CLASSIFICATION_TASKS.items():
        for baseline_adj, norm, duration in strategies:
            for model_type in test_models:
                for modality in test_modalities:
                    strategy_args.append((
                        task_name, task_map, baseline_adj, norm, duration,
                        model_type, modality, df, counterbalance, feature_groups,
                        TUNE, OPTIMIZE_THRESHOLD, CALIBRATE
                    ))
    
    # Execute strategies in parallel
    print(f"[INFO] Running {len(strategy_args)} strategies in parallel...\n")
    
    if args.n_jobs == 1:
        # Sequential execution (for debugging)
        print("[INFO] Sequential mode (n_jobs=1)")
        results = []
        for i, args_tuple in enumerate(strategy_args, 1):
            print(f"\n[{i}/{len(strategy_args)}] Processing...")
            result = run_strategy_wrapper(args_tuple)
            results.append(result)
    else:
        # Parallel execution
        results = []
        with ProcessPoolExecutor(max_workers=args.n_jobs) as executor:
            # Submit all tasks
            future_to_strategy = {
                executor.submit(run_strategy_wrapper, args_tuple): args_tuple
                for args_tuple in strategy_args
            }
            
            # Collect results as they complete
            completed = 0
            for future in as_completed(future_to_strategy):
                completed += 1
                try:
                    result = future.result()
                    results.append(result)
                    if result is not None:
                        all_results.append(result)
                        print(f"[{completed}/{len(strategy_args)}] Completed: {result['task']} | "
                              f"{result['model']} | {result['modality']} | "
                              f"{result['baseline_adjustment']}_{result['normalization']} | "
                              f"Acc: {result['accuracy_mean']:.3f}")
                    else:
                        print(f"[{completed}/{len(strategy_args)}] Failed (returned None)")
                    
                    # Incremental save every 50 results
                    if completed % 50 == 0 and all_results:
                        try:
                            # Save JSON checkpoint
                            with open(checkpoint_path, 'w') as f:
                                json.dump(all_results, f, indent=2)
                            # Save CSV checkpoint
                            checkpoint_df = pd.DataFrame(all_results)
                            csv_df = checkpoint_df.drop(columns=['feature_importance', 'feature_names'], errors='ignore')
                            csv_df.to_csv(checkpoint_csv_path, index=False)
                            print(f"  [CHECKPOINT] Saved {len(all_results)} results to {checkpoint_path}")
                        except Exception as save_error:
                            print(f"  [WARNING] Checkpoint save failed: {save_error}")
                except Exception as e:
                    print(f"[{completed}/{len(strategy_args)}] Error: {e}")
                    results.append(None)
    
    # Organize results by task
    task_results_dict = {task_name: [] for task_name in CLASSIFICATION_TASKS.keys()}
    for result in results:
        if result is not None:
            task_results_dict[result['task']].append(result)
    
    # Process results for each task
    for task_name, task_results in task_results_dict.items():
        
        print("\n" + "="*70)
        print(f"TASK SUMMARY: {task_name.upper()}")
        print("="*70)
        
        
        # Find best strategy for this task
        if task_results:
            best_result = max(task_results, key=lambda x: x['accuracy_mean'])
            best_models[task_name] = best_result
            
            print(f"\n[BEST] {task_name}:")
            print(f"  Strategy: {best_result['baseline_adjustment']}_{best_result['normalization']}_{best_result['baseline_duration']}")
            print(f"  Accuracy: {best_result['accuracy_mean']:.3f} ± {best_result['accuracy_std']:.3f}")
            print(f"  Strategies tested: {len(task_results)}")
            
            # Plot feature importance for best model
            plot_feature_importance(
                np.array(best_result['feature_importance']),
                best_result['feature_names'],
                task_name,
                PLOT_DIR
            )
            
            # Plot strategy comparison heatmap
            task_df = pd.DataFrame(task_results)
            plot_strategy_heatmap(task_df, task_name, PLOT_DIR)
        else:
            print(f"  No successful results for {task_name}")
    
    # Save summary
    if all_results:
        summary_df = pd.DataFrame(all_results)
        
        # Drop complex columns for CSV
        csv_df = summary_df.drop(columns=['feature_importance', 'feature_names'], errors='ignore')
        csv_path = os.path.join(OUTPUT_DIR, 'multimodal_strategy_comparison_summary.csv')
        csv_df.to_csv(csv_path, index=False)
        print(f"\n[INFO] Saved summary CSV: {csv_path}")
        
        # Save detailed results as JSON
        json_path = os.path.join(OUTPUT_DIR, 'multimodal_detailed_results.json')
        with open(json_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"[INFO] Saved detailed JSON: {json_path}")
        
        # Save best models
        best_path = os.path.join(OUTPUT_DIR, 'best_models_summary.json')
        with open(best_path, 'w') as f:
            json.dump(best_models, f, indent=2)
        print(f"[INFO] Saved best models: {best_path}")
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total strategies tested: {len(all_results)}")
    print(f"Results saved to: {OUTPUT_DIR}")
    print(f"Plots saved to: {PLOT_DIR}")
    print("="*70)


if __name__ == "__main__":
    main()
