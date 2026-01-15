"""
Physio Late Fusion Model for Workload Classification
Uses 10s-window-appropriate physiological features (pupil, EDA, HR) aligned with EEG epochs.

Excludes: RMSSD, HRV frequency metrics (require >60s windows)
Includes: Pupil dilation, EDA tonic/phasic, basic HR
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score
import xgboost as xgb
import json
from pathlib import Path

# ==============================================================================
# CONFIGURATION
# ==============================================================================
PHYSIO_PATH = r'C:\vr_tsst_2025\output\aggregated\physio_features_rolling_windows.csv'
RESULTS_PATH = r'C:\vr_tsst_2025\results\workload_physio_cv.txt'

# All 43 EEG-valid participants
ALL_PIDS = [1, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 15, 16, 17, 18, 19, 20, 21, 22, 23, 
            24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 
            42, 43, 45, 47, 48]

# Task conditions for workload classification
TASK_CONDITIONS = [
    'LowStress_LowCog_Task',
    'HighStress_LowCog_Task', 
    'LowStress_HighCog1022_Task',
    'LowStress_HighCog2043_Task',
    'HighStress_HighCog1022_Task',
    'HighStress_HighCog2043_Task'
]

# 10s-window appropriate physio features (excluding HRV which requires longer windows)
PHYSIO_FEATURES = [
    # Pupil dilation - best workload indicator for short windows
    'Full_Pupil_Dilation_Mean',
    'Full_Pupil_Dilation_Median', 
    'Full_Pupil_Dilation_SD',
    'Full_Pupil_Dilation_Asymmetry',
    'Foveal_Corrected_Dilation_Left_CLEANED_ABS_Mean',
    'Foveal_Corrected_Dilation_Right_CLEANED_ABS_Mean',
    
    # EDA - fast responding, valid for 10s
    'Shimmer_D36A_GSR_Skin_Conductance_uS_CLEANED_ABS_CLEANED_NK_EDA_Tonic_Mean',
    'Shimmer_D36A_GSR_Skin_Conductance_uS_CLEANED_ABS_CLEANED_NK_EDA_PeakRate',
    'Shimmer_D36A_GSR_Skin_Conductance_uS_CLEANED_ABS_CLEANED_NK_EDA_PeakHeight_Mean',
    'Shimmer_D36A_GSR_Skin_Conductance_uS_CLEANED_ABS_CLEANED_NK_EDA_TotalSCRs',
    
    # Basic HR (not HRV) - valid for short windows
    'Polar_HeartRate_BPM_CLEANED_ABS_Mean',
    'Polar_HeartRate_BPM_CLEANED_ABS_SD',
    
    # Blink features - if enough events
    'Inter_Blink_Interval_CLEANED_ABS_Mean',
    'Current_Blink_Duration_CLEANED_ABS_Mean',
]

N_FOLDS = 3

# ==============================================================================
# DATA LOADING
# ==============================================================================

def load_physio_data():
    """Load and preprocess physio rolling windows data."""
    print(f"Loading physio data from {PHYSIO_PATH}")
    df = pd.read_csv(PHYSIO_PATH, low_memory=False)
    
    # Filter to task conditions only
    df = df[df['Condition'].isin(TASK_CONDITIONS)]
    
    # Filter to valid participants (Participant_ID is numeric in this file)
    df = df[df['Participant_ID'].isin(ALL_PIDS)]
    
    # Create workload label
    def get_workload_label(condition):
        if 'LowCog' in condition:
            return 0  # LowWorkload
        else:
            return 1  # HighWorkload
    
    df['workload'] = df['Condition'].apply(get_workload_label)
    
    # PID for grouping (already numeric)
    df['pid'] = df['Participant_ID'].astype(int)
    
    # Select features
    available_features = [f for f in PHYSIO_FEATURES if f in df.columns]
    print(f"Using {len(available_features)}/{len(PHYSIO_FEATURES)} physio features")
    
    # Drop rows with all NaN features
    df_features = df[available_features].copy()
    valid_mask = df_features.notna().any(axis=1)
    df = df[valid_mask].copy()
    
    print(f"Data shape after filtering: {df.shape}")
    print(f"Class distribution: {df['workload'].value_counts().to_dict()}")
    
    return df, available_features


def run_physio_cv():
    """Run cross-validation on physio-only model."""
    df, feature_cols = load_physio_data()
    
    X = df[feature_cols].values
    y = df['workload'].values
    groups = df['pid'].values
    
    # Handle missing values with median imputation
    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(strategy='median')
    X = imputer.fit_transform(X)
    
    print(f"\nFeature matrix shape: {X.shape}")
    print(f"Class balance: {np.bincount(y)}")
    
    fold_results = []
    gkf = GroupKFold(n_splits=N_FOLDS)
    
    for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups)):
        print(f"\n{'='*50}")
        print(f"Fold {fold + 1}")
        print(f"{'='*50}")
        
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        train_pids = np.unique(groups[train_idx])
        test_pids = np.unique(groups[test_idx])
        print(f"Train PIDs: {len(train_pids)}, Test PIDs: {len(test_pids)}")
        
        # Scale features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        # Train XGBoost
        model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss'
        )
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        print(f"Fold {fold+1}: Accuracy={acc:.3f}, F1={f1:.3f}")
        fold_results.append({
            'accuracy': acc,
            'f1': f1,
            'y_test': y_test,
            'y_prob': y_prob
        })
    
    # Aggregate results
    mean_acc = np.mean([r['accuracy'] for r in fold_results])
    std_acc = np.std([r['accuracy'] for r in fold_results])
    mean_f1 = np.mean([r['f1'] for r in fold_results])
    
    print(f"\n{'='*50}")
    print("FINAL RESULTS - Physio-Only Model (XGBoost)")
    print(f"{'='*50}")
    print(f"Features used: {len(feature_cols)}")
    print(f"Per-Fold Accuracy: {[r['accuracy'] for r in fold_results]}")
    print(f"Mean Accuracy: {mean_acc:.3f} (+/- {std_acc:.3f})")
    print(f"Mean F1 Score: {mean_f1:.3f}")
    
    # Save results
    with open(RESULTS_PATH, 'w') as f:
        f.write("Physio-Only Workload Classification (10s windows)\n")
        f.write(f"Features: {feature_cols}\n")
        f.write(f"Data: {len(df)} samples, {N_FOLDS}-fold GroupKFold\n")
        f.write("-" * 50 + "\n")
        for i, r in enumerate(fold_results):
            f.write(f"Fold {i+1}: Accuracy={r['accuracy']:.4f}, F1={r['f1']:.4f}\n")
        f.write("-" * 50 + "\n")
        f.write(f"Mean Accuracy: {mean_acc:.4f} (+/- {std_acc:.4f})\n")
        f.write(f"Mean F1 Score: {mean_f1:.4f}\n")
    
    print(f"\nResults saved to {RESULTS_PATH}")
    
    # Feature importance
    print("\n--- Feature Importance (last fold) ---")
    importance = model.feature_importances_
    for feat, imp in sorted(zip(feature_cols, importance), key=lambda x: -x[1])[:10]:
        print(f"  {feat}: {imp:.3f}")
    
    return mean_acc, mean_f1, fold_results


if __name__ == "__main__":
    run_physio_cv()
