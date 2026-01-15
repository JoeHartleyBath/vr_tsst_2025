"""
XGBoost Tuning for Band Power Features
Optuna hyperparameter optimization for the band power classifier component.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import json
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import optuna
from optuna.samplers import TPESampler
import warnings
warnings.filterwarnings('ignore')

# ==============================================================================
# PATHS & CONFIGURATION
# ==============================================================================
BAND_POWER_PATH = r'C:\vr_tsst_2025\output\aggregated\eeg_features_rolling_windows.csv'
RESULTS_PATH = r'C:\vr_tsst_2025\results\best_xgb_bandpower_params.json'
STUDY_DB = r'C:\vr_tsst_2025\results\optuna_xgb_bandpower.db'

# Workload conditions
HIGH_WORKLOAD = ['HighStress_HighCog1022_Task', 'LowStress_HighCog1022_Task']
LOW_WORKLOAD = ['HighStress_LowCog_Task', 'LowStress_LowCog_Task']

# Tuning settings
N_TRIALS = 100
TIMEOUT_HOURS = 2


def load_band_power_features():
    """Load band power features with theta/beta ratios."""
    df = pd.read_csv(BAND_POWER_PATH)
    
    # Filter to workload conditions only
    all_conditions = HIGH_WORKLOAD + LOW_WORKLOAD
    df = df[df['event_label'].isin(all_conditions)].copy()
    
    # Create binary workload label
    df['workload'] = df['event_label'].isin(HIGH_WORKLOAD).astype(int)
    
    # Create theta/beta ratios (key workload markers)
    regions = ['FrontalLeft', 'FrontalRight', 'OverallFrontal', 'FrontalMidline',
               'Central', 'ParietalLeft', 'ParietalRight', 'Parietal', 
               'TemporalLeft', 'TemporalRight', 'Temporal', 'Occipital']
    
    for region in regions:
        theta_col = f'{region}_Theta'
        beta_col = f'{region}_Beta'
        if theta_col in df.columns and beta_col in df.columns:
            # Theta/Beta ratio (log scale, so subtract)
            df[f'{region}_ThetaBetaRatio'] = df[theta_col] - df[beta_col]
            # Frontal theta / parietal alpha (engagement)
            alpha_col = f'{region}_Alpha'
            if alpha_col in df.columns:
                df[f'{region}_AlphaBetaRatio'] = df[alpha_col] - df[beta_col]
    
    # Frontal asymmetry features
    df['Frontal_ThetaAsymmetry'] = df['FrontalRight_Theta'] - df['FrontalLeft_Theta']
    df['Frontal_AlphaAsymmetry'] = df['FrontalRight_Alpha'] - df['FrontalLeft_Alpha']
    df['Frontal_BetaAsymmetry'] = df['FrontalRight_Beta'] - df['FrontalLeft_Beta']
    
    # Parietal asymmetry
    df['Parietal_AlphaAsymmetry'] = df['ParietalRight_Alpha'] - df['ParietalLeft_Alpha']
    
    # Frontal midline theta (key workload marker)
    df['FMTheta_Relative'] = df['FrontalMidline_Theta'] - df['Parietal_Theta']
    
    print(f"Band power features: {len(df)} windows, {df.shape[1]} columns")
    print(f"High workload: {(df['workload']==1).sum()}, Low workload: {(df['workload']==0).sum()}")
    
    return df


def get_feature_columns(df):
    """Get band power feature columns (exclude metadata)."""
    exclude = ['pid', 'event_label', 'window_idx', 'window_start', 'window_end', 
               'workload', 'Participant_ID', 'Condition', 'Window_Index',
               'Window_Start', 'Window_End']
    return [c for c in df.columns if c not in exclude and not c.endswith('_Flag')]


def objective(trial, X, y, groups):
    """Optuna objective for XGBoost hyperparameter tuning."""
    params = {
        'max_depth': trial.suggest_int('max_depth', 2, 8),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'gamma': trial.suggest_float('gamma', 0, 5),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10, log=True),
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'tree_method': 'hist',
        'random_state': 42,
        'verbosity': 0
    }
    
    # 3-fold GroupKFold CV
    gkf = GroupKFold(n_splits=3)
    fold_accs = []
    
    for train_idx, test_idx in gkf.split(X, y, groups=groups):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Handle class imbalance
        scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
        params['scale_pos_weight'] = scale_pos_weight
        
        model = xgb.XGBClassifier(**params)
        model.fit(X_train_scaled, y_train, verbose=False)
        
        preds = model.predict(X_test_scaled)
        acc = accuracy_score(y_test, preds)
        fold_accs.append(acc)
    
    return np.mean(fold_accs)


def run_tuning():
    """Run Optuna hyperparameter tuning."""
    print("Loading data...")
    df = load_band_power_features()
    feature_cols = get_feature_columns(df)
    
    X = df[feature_cols].values.astype(np.float32)
    y = df['workload'].values
    groups = df['pid'].values
    
    # Handle NaN
    nan_mask = np.isnan(X).any(axis=1)
    if nan_mask.sum() > 0:
        print(f"Dropping {nan_mask.sum()} windows with NaN values")
        X = X[~nan_mask]
        y = y[~nan_mask]
        groups = groups[~nan_mask]
    
    print(f"Dataset: {len(y)} samples, {X.shape[1]} features")
    
    # Create study
    sampler = TPESampler(seed=42)
    study = optuna.create_study(
        study_name='xgb_bandpower_tuning',
        storage=f'sqlite:///{STUDY_DB}',
        direction='maximize',
        sampler=sampler,
        load_if_exists=True
    )
    
    print(f"\nStarting Optuna tuning ({N_TRIALS} trials, {TIMEOUT_HOURS}h timeout)...")
    study.optimize(
        lambda trial: objective(trial, X, y, groups),
        n_trials=N_TRIALS,
        timeout=TIMEOUT_HOURS * 3600,
        show_progress_bar=True
    )
    
    # Results
    print("\n" + "="*60)
    print("TUNING COMPLETE")
    print("="*60)
    print(f"Best accuracy: {study.best_value:.4f}")
    print(f"Best params: {study.best_params}")
    
    # Save best params
    with open(RESULTS_PATH, 'w') as f:
        json.dump({
            'accuracy': study.best_value,
            'params': study.best_params
        }, f, indent=2)
    
    print(f"\nResults saved to {RESULTS_PATH}")
    
    return study.best_params, study.best_value


if __name__ == "__main__":
    run_tuning()
