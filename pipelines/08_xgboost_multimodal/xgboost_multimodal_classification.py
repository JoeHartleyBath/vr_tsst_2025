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
COUNTERBALANCE = 'data/processed/VR-TSST Counterbalance sheet.xlsx'
OUTPUT_DIR = 'output/xgboost_results'
PLOT_DIR = 'output/plots'

# Classification tasks
CLASSIFICATION_TASKS = {
    'stress_classification': {
        'HighStress_LowCog_Task': 1,
        'HighStress_HighCog_Task': 1,
        'HighStress_HighCog1022_Task': 1,  # Version 1022
        'HighStress_HighCog2043_Task': 1,  # Version 2043
        'LowStress_LowCog_Task': 0,
        'LowStress_HighCog_Task': 0,
        'LowStress_HighCog1022_Task': 0,   # Version 1022
        'LowStress_HighCog2043_Task': 0    # Version 2043
    },
    'workload_classification': {
        'HighStress_HighCog_Task': 1,      # HighCog
        'HighStress_HighCog1022_Task': 1,  # HighCog - Version 1022
        'HighStress_HighCog2043_Task': 1,  # HighCog - Version 2043
        'LowStress_HighCog_Task': 1,       # HighCog
        'LowStress_HighCog1022_Task': 1,   # HighCog - Version 1022
        'LowStress_HighCog2043_Task': 1,   # HighCog - Version 2043
        'HighStress_LowCog_Task': 0,       # LowCog
        'LowStress_LowCog_Task': 0         # LowCog
    }
}

# Preprocessing strategies
BASELINE_ADJUSTMENTS = ['none', 'subtract', 'zscore', 'percent']
NORMALIZATIONS = ['none', 'within_subject', 'global']
BASELINE_DURATIONS = ['last_60s', 'last_90s', 'last_120s', 'full']

# XGBoost parameters
XGBOOST_PARAMS = {
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
    
    # Metadata columns to exclude
    meta_cols = ['Participant_ID', 'Condition', 'Window_Index', 'Window_Start', 
                 'Window_End', 'class_label']
    
    all_cols = [col for col in df.columns if col not in meta_cols]
    
    eeg_features = [col for col in all_cols if any(
        x in col for x in ['Theta', 'Alpha', 'Beta',
                           'Frontal', 'Parietal', 'Temporal', 'Central']
    )]
    
    physio_features = [col for col in all_cols if any(
        x in col for x in ['HR',  'GSR', 'EDA', 'Pupil','Shimmer']
    )]
    
    # Remove any overlap
    physio_features = [f for f in physio_features if f not in eeg_features]
    
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
    
    round_num = None
    for r in [1, 2, 3, 4]:
        round_cond = p_cb[f'Round {r}'].values[0]
        mapped_cond = condition_map.get(round_cond, round_cond)
        if mapped_cond == task_condition:
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
    df: pd.DataFrame,
    method: str,
    duration: str,
    feature_cols: List[str],
    counterbalance_data: pd.DataFrame
) -> pd.DataFrame:
    """Apply baseline adjustment strategy."""
    
    if method == 'none':
        return df  # No adjustment
    
    print(f"[INFO] Applying baseline adjustment: {method}, duration: {duration}")
    
    adjusted_windows = []
    
    for pid in df['Participant_ID'].unique():
        p_data = df[df['Participant_ID'] == pid].copy()
        task_conditions = [c for c in p_data['Condition'].unique() if 'Task' in c]
        
        for task_cond in task_conditions:
            task_windows = p_data[p_data['Condition'] == task_cond].copy()
            
            if len(task_windows) == 0:
                continue
            
            # Get baseline
            baseline_data = get_baseline(
                df=p_data,
                participant_id=pid,
                task_condition=task_cond,
                counterbalance_data=counterbalance_data,
                baseline_duration=duration
            )
            
            if baseline_data is None or len(baseline_data) == 0:
                continue
            
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
    
    return pd.concat(adjusted_windows, ignore_index=True)


# ============================================================
# NORMALIZATION
# ============================================================

def apply_normalization(
    df: pd.DataFrame,
    method: str,
    feature_cols: List[str]
) -> pd.DataFrame:
    """Apply normalization strategy."""
    
    if method == 'none':
        return df
    
    print(f"[INFO] Applying normalization: {method}")
    
    df_norm = df.copy()
    
    if method == 'within_subject':
        # Z-score within each participant
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
    
    elif method == 'global':
        # Global StandardScaler
        scaler = StandardScaler()
        df_norm[feature_cols] = scaler.fit_transform(df[feature_cols])
    
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

def train_evaluate_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray
) -> Dict:
    """Train XGBoost and evaluate."""
    
    model = xgb.XGBClassifier(**XGBOOST_PARAMS)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0),
        'auc': roc_auc_score(y_test, y_pred_proba) if len(np.unique(y_test)) > 1 else 0,
        'y_pred': y_pred,
        'y_test': y_test,
        'model': model
    }
    
    return metrics


def run_cross_validation(
    df: pd.DataFrame,
    feature_cols: List[str],
    task_name: str
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
    
    print(f"[INFO] Running {n_splits}-fold cross-validation...")
    
    for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Train and evaluate
        fold_metrics = train_evaluate_xgboost(X_train, y_train, X_test, y_test)
        
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
        
        all_y_true.extend(fold_metrics['y_test'])
        all_y_pred.extend(fold_metrics['y_pred'])
        
        # Feature importance
        feature_importances.append(fold_metrics['model'].feature_importances_)
        
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
        'n_test_mean': metrics_df['n_test'].mean()
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
    
    plt.title(f'XGBoost Accuracy: {task_name.replace("_", " ").title()}\n{strategy_name}')
    plt.xlabel('Window Index')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    
    plot_path = os.path.join(output_dir, f'xgboost_accuracy_{task_name}_{strategy_name}.png')
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
    
    plot_path = os.path.join(output_dir, f'xgboost_strategy_heatmap_{task_name}.png')
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
    
    # Sort features by importance
    importance_pairs = sorted(
        zip(feature_names, feature_importance),
        key=lambda x: x[1],
        reverse=True
    )[:top_n]
    
    features, importances = zip(*importance_pairs)
    
    plt.figure(figsize=(10, 8))
    plt.barh(range(len(features)), importances, color='steelblue')
    plt.yticks(range(len(features)), features, fontsize=8)
    plt.xlabel('Importance')
    plt.title(f'Top {top_n} Features: {task_name.replace("_", " ").title()}')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    
    plot_path = os.path.join(output_dir, f'xgboost_feature_importance_{task_name}.png')
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
    feature_groups: Dict
) -> Dict:
    """Run one preprocessing strategy."""
    
    start_time = time.time()
    
    strategy_name = f"{baseline_adj}_{norm}_{duration}"
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
        
        # Apply baseline adjustment
        if baseline_adj != 'none':
            task_df = apply_baseline_adjustment(
                df=task_df,
                method=baseline_adj,
                duration=duration,
                feature_cols=feature_groups['all'],
                counterbalance_data=counterbalance_data
            )
            
            if len(task_df) == 0:
                print(f"[WARNING] No data after baseline adjustment")
                return None
        
        # Apply normalization
        task_df = apply_normalization(
            df=task_df,
            method=norm,
            feature_cols=feature_groups['all']
        )
        
        # Prune features
        task_df, kept_features = prune_features(task_df, feature_groups['all'])
        
        if len(kept_features) == 0:
            print(f"[WARNING] No features remaining after pruning")
            return None
        
        # Cross-validation
        cv_results = run_cross_validation(task_df, kept_features, task_name)
        
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
            'feature_names': cv_results['feature_names']
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
        df, counterbalance, feature_groups
    ) = args_tuple
    
    return run_single_strategy(
        df=df,
        task_name=task_name,
        task_map=task_map,
        baseline_adj=baseline_adj,
        norm=norm,
        duration=duration,
        counterbalance_data=counterbalance,
        feature_groups=feature_groups
    )


def main():
    """Main pipeline."""
    
    args = parse_arguments()
    
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
        print("[INFO] Quick mode: testing subset of strategies")
        strategies = [
            ('none', 'none', 'last_90s'),
            ('zscore', 'within_subject', 'last_90s'),
        ]
    else:
        strategies = list(itertools.product(
            BASELINE_ADJUSTMENTS,
            NORMALIZATIONS,
            BASELINE_DURATIONS
        ))
    
    print(f"[INFO] Total strategies to test: {len(strategies) * len(CLASSIFICATION_TASKS)}")
    print(f"[INFO] Parallel jobs: {args.n_jobs}\n")
    
    all_results = []
    best_models = {}
    
    # Prepare all strategy combinations as arguments
    strategy_args = []
    for task_name, task_map in CLASSIFICATION_TASKS.items():
        for baseline_adj, norm, duration in strategies:
            strategy_args.append((
                task_name, task_map, baseline_adj, norm, duration,
                df, counterbalance, feature_groups
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
                        print(f"[{completed}/{len(strategy_args)}] Completed: {result['task']} - "
                              f"{result['baseline_adjustment']}_{result['normalization']}_{result['baseline_duration']} - "
                              f"Acc: {result['accuracy_mean']:.3f}")
                    else:
                        print(f"[{completed}/{len(strategy_args)}] Failed (returned None)")
                except Exception as e:
                    print(f"[{completed}/{len(strategy_args)}] Error: {e}")
                    results.append(None)
    
    # Organize results by task
    task_results_dict = {task_name: [] for task_name in CLASSIFICATION_TASKS.keys()}
    for result in results:
        if result is not None:
            all_results.append(result)
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
