import pandas as pd
import numpy as np
import os
from sklearn.svm import SVC
from sklearn.model_selection import GroupKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score

# ==============================================================================
# CONFIGURATION
# ==============================================================================
INPUT_FILE = r'C:\vr_tsst_2025\output\aggregated\eeg_features_rolling_windows.csv'
CB_FILE = r'C:\vr_tsst_2025\data\experimental_counterbalance.xlsx'

# All 44 EEG-valid participants
ALL_PIDS = [1, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 15, 16, 17, 18, 19, 20, 21, 22, 23, 
            24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 
            42, 43, 44, 45, 47, 48]
SUBSET_PIDS = ALL_PIDS

# Mapping from Counterbalance names to event_labels in rolling windows
CB_CONDITION_MAP = {
    'Calm Addition': ['LowStress_LowCog_Task'],
    'Calm Subtraction': ['LowStress_HighCog1022_Task', 'LowStress_HighCog2043_Task'],
    'Stress Addition': ['HighStress_LowCog_Task'],
    'Stress Subtraction': ['HighStress_HighCog1022_Task', 'HighStress_HighCog2043_Task']
}

# All 4 conditions for binary workload classification (LowCog vs HighCog)
TARGET_CONDITIONS = {
    'LowWorkload': ['LowStress_LowCog_Task', 'HighStress_LowCog_Task'],
    'HighWorkload': ['LowStress_HighCog1022_Task', 'LowStress_HighCog2043_Task',
                     'HighStress_HighCog1022_Task', 'HighStress_HighCog2043_Task']
}

def get_base_features(df):
    """Filter columns to get only band power features."""
    all_cols = df.columns.tolist()
    # Features start after window_end (index 5)
    feature_cols = all_cols[5:]
    
    selected = []
    for col in feature_cols:
        # Only want Band Power (no entropy or ratios yet for this baseline)
        if '_' in col:
             # Check band
             if any(band in col for band in EXCLUDE_BANDS):
                 continue
             # Check region
             if any(region in col for region in EXCLUDE_REGIONS):
                 continue
             selected.append(col)
             
    return selected

def run_svm_evaluation(X_df, y, groups, title, features, apply_per_subject_scaling=True):
    """Run SVM with GroupKFold CV and optional per-subject scaling."""
    X_processed = X_df.copy()
    
    if apply_per_subject_scaling:
        # Reinstating per-subject Z-scoring to remove trait variance
        for pid in np.unique(groups):
            mask = (groups == pid)
            # Need to handle case where std is 0
            subj_data = X_processed.loc[mask, features]
            X_processed.loc[mask, features] = (subj_data - subj_data.mean()) / (subj_data.std() + 1e-6)

    X_vals = X_processed[features].values
    
    # Standard GroupKFold
    cv = GroupKFold(n_splits=3)
    
    accuracies = []
    f1s = []
    
    for train_idx, test_idx in cv.split(X_vals, y, groups=groups):
        X_train, X_test = X_vals[train_idx], X_vals[test_idx]
        y_train, y_test = y.values[train_idx], y.values[test_idx]
        
        # Additional global scaling based on training set (for the SVM classifier)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        clf = SVC(kernel='linear', C=1.0)
        clf.fit(X_train_scaled, y_train)
        
        y_pred = clf.predict(X_test_scaled)
        accuracies.append(accuracy_score(y_test, y_pred))
        f1s.append(f1_score(y_test, y_pred, average='weighted'))
    
    print(f"\n--- {title} ---")
    print(f"Per-Fold Accuracy: {np.array(accuracies)}")
    print(f"Mean Accuracy: {np.mean(accuracies):.3f} (+/- {np.std(accuracies):.3f})")
    print(f"Mean F1 Score: {np.mean(f1s):.3f}")
    return np.mean(accuracies)

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
if __name__ == "__main__":
    print(f"Loading data from {INPUT_FILE}...")
    df_raw = pd.read_csv(INPUT_FILE)
    
    print(f"Loading counterbalancing from {CB_FILE}...")
    cb = pd.read_excel(CB_FILE)
    
    # 1. Filter for Subset Participants
    df = df_raw[df_raw['pid'].isin(SUBSET_PIDS)].copy()
    
    # 2. Get Feature List
    features = get_base_features(df)
    print(f"Using {len(features)} EEG features (after excluding {EXCLUDE_BANDS} and {EXCLUDE_REGIONS})")
    
    # 3. Calculate Subject-Level Baselines using R-style Logic (Round matching)
    # R Logic: Round X Task is matched with Forest X (Relaxation_level X)
    baseline_stats = {}
    for pid in SUBSET_PIDS:
        baseline_stats[pid] = {}
        
        # Get counterbalance for this participant
        p_cb = cb[cb['Participant'] == pid]
        if p_cb.empty:
            print(f"WARNING: No counterbalance data for P{pid}")
            continue
            
        for r in [1, 2, 3, 4]:
            round_condition_text = p_cb[f'Round {r}'].values[0]
            event_labels = CB_CONDITION_MAP.get(round_condition_text)
            
            if event_labels is None:
                continue
                
            # Corresponding forest is Forest{r}
            forest_label = f'Forest{r}'
            
            # Compute mean of forest windows
            bl_df = df[(df['pid'] == pid) & (df['event_label'] == forest_label)]
            if bl_df.empty:
                print(f"WARNING: No {forest_label} data found for P{pid}")
                continue
                
            # Store baseline for all variants (1022 or 2043)
            for label in event_labels:
                baseline_stats[pid][label] = bl_df[features].mean()

    # 4. Prepare Dataset for Absolute vs Relative Comparison
    # Filter only for the tasks we want to classify
    all_target_labels = [lbl for sublist in TARGET_CONDITIONS.values() for lbl in sublist]
    task_df = df[df['event_label'].isin(all_target_labels)].copy()
    
    # Map back to workload class
    label_to_class = {lbl: cls for cls, lbls in TARGET_CONDITIONS.items() for lbl in lbls}
    task_df['workload_class'] = task_df['event_label'].map(label_to_class)
    
    # Drop any rows where we couldn't map to a workload class (e.g. Other tasks)
    task_df = task_df.dropna(subset=['workload_class'])
    
    y = (task_df['workload_class'] == 'HighWorkload').astype(int)
    groups = task_df['pid'].values
    
    # 5. Run Evaluations
    abs_acc = run_svm_evaluation(task_df, y, groups, "SVM Baseline: ABSOLUTE FEATURES (Raw Window Power)", features)
    
    # Relative Dataset (X_rel) - Subtract baseline
    rel_rows = []
    # Index task_df to ensure alignment
    task_df_reset = task_df.reset_index(drop=True)
    for idx, row in task_df_reset.iterrows():
        pid = row['pid']
        cond = row['event_label']
        raw_vals = row[features]
        
        if pid in baseline_stats and cond in baseline_stats[pid]:
            bl_vals = baseline_stats[pid][cond]
            rel_vals = raw_vals - bl_vals
            rel_rows.append(rel_vals)
        else:
            rel_rows.append(raw_vals)
    
    rel_df = pd.DataFrame(rel_rows, columns=features)
    rel_df['pid'] = task_df_reset['pid'].values
    y_rel = (task_df_reset['workload_class'] == 'HighWorkload').astype(int)
    groups_rel = task_df_reset['pid'].values
    
    rel_acc = run_svm_evaluation(rel_df, y_rel, groups_rel, "SVM Baseline: RELATIVE FEATURES (Pre-condition Subtraction)", features)
    
    # 6. Save results for decision gate
    results_path = r'C:\vr_tsst_2025\results\workload_baseline_cv.txt'
    with open(results_path, 'w') as f:
        f.write(f"Workload Baseline Evaluation (Subset: {SUBSET_PIDS})\n")
        f.write(f"Targets: {list(TARGET_CONDITIONS.values())}\n")
        f.write(f"Features: {len(features)} bands (Excluded: {EXCLUDE_BANDS}, {EXCLUDE_REGIONS})\n")
        f.write("Evaluation approach: 3-fold GroupKFold (Subject-wise)\n")
        f.write("Normalization: Per-subject Z-scoring applied before CV\n")
        f.write("-" * 50 + "\n")
        f.write(f"Absolute Features Accuracy: {abs_acc:.4f}\n")
        f.write(f"Relative Features Accuracy: {rel_acc:.4f}\n")
        f.write("-" * 50 + "\n")
        
    print(f"\nFinal Results saved to {results_path}")
    if rel_acc >= 0.60:
        print("DECISION GATE: PASSED (>= 0.60 Accuracy)")
    else:
        print(f"DECISION GATE: FAILED ({rel_acc:.4f} < 0.60 Accuracy)")
