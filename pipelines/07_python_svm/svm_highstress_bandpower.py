"""
This script has been renamed to svm_rolling_windows.py.
Please use scripts/modeling/svm_rolling_windows.py for all future work.
"""

import pandas as pd
import numpy as np
import argparse
import yaml
from sklearn.model_selection import GroupKFold
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold


parser = argparse.ArgumentParser(description='SVM classification for EEG rolling window features with flexible label mapping.')
parser.add_argument('--features', type=str, default='C:/vr_tsst_2025/output/aggregated/eeg_features_rolling_windows.csv', help='Path to rolling window features CSV')
parser.add_argument('--label_config', type=str, default='scripts/modeling/model_class_labels.yaml', help='Path to YAML file with label mappings')
parser.add_argument('--task', type=str, required=True, help='Classification task name (must match a key in the YAML config)')
args = parser.parse_args()

# Load data
df = pd.read_csv(args.features)

# Load label mapping
with open(args.label_config, 'r') as f:
    label_map = yaml.safe_load(f)
if args.task not in label_map:
    raise ValueError(f"Task '{args.task}' not found in label config. Available: {list(label_map.keys())}")
task_map = label_map[args.task]

# Map event_label to class label for this task
df = df[df['event_label'].isin(task_map.keys())].copy()
df['class_label'] = df['event_label'].map(task_map)



# Identify columns
group_cols = ['pid', 'event_label', 'window_idx', 'class_label']
feature_cols = [col for col in df.columns if col not in group_cols]

# Remove features containing 'Delta' or 'Occipital' before pruning
feature_cols = [col for col in feature_cols if 'Delta' not in col and 'Occipital' not in col]

# --- Feature pruning ---
# 1. Remove near-zero variance features (threshold very low, e.g. 1e-6)
vt = VarianceThreshold(threshold=1e-6)
X_var = vt.fit_transform(df[feature_cols])
kept_var_idx = vt.get_support(indices=True)
kept_var_cols = [feature_cols[i] for i in kept_var_idx]

 # 2. Remove highly correlated features (correlation > 0.75)
def remove_highly_correlated(X, cols, threshold=0.75):
    X = pd.DataFrame(X, columns=cols)
    corr_matrix = X.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = set()
    while True:
        max_corr = upper.max().max()
        if max_corr < threshold:
            break
        drop_col = upper.stack().idxmax()[1]
        to_drop.add(drop_col)
        upper.loc[:, drop_col] = 0
        upper.loc[drop_col, :] = 0
    keep_cols = [c for c in cols if c not in to_drop]
    return X[keep_cols], keep_cols


X_pruned, pruned_cols = remove_highly_correlated(X_var, kept_var_cols, threshold=0.75)

# Print which columns are being z-scored
print(f"Columns being z-scored: {pruned_cols}")

# Standard z-score function
zscore = lambda series: (series - np.nanmean(series)) / (np.nanstd(series, ddof=0) if np.nanstd(series, ddof=0) > 1e-6 else 1.0)



# Prepare data for SVM
groups = df['pid'].values
labels = df['class_label'].values
results = []
all_true = []
all_pred = []
window_acc = dict()  # window_idx -> accuracy
window_counts = dict()  # window_idx -> count (for averaging if needed)
gkf = GroupKFold(n_splits=min(5, len(np.unique(groups))))

# Print window_idx range before z-scoring
if 'window_idx' in df.columns:
    print(f"window_idx range before z-scoring: min={df['window_idx'].min()}, max={df['window_idx'].max()}")

print(f'Number of features after pruning: {len(pruned_cols)}')
print(f'Features used after pruning: {pruned_cols}')

for train_idx, test_idx in gkf.split(df, labels, groups):
    train = df.iloc[train_idx].copy()
    test = df.iloc[test_idx].copy()
    # Standard z-score within each participant in train, then apply to test
    train_z = train.copy()
    for pid, group in train.groupby('pid'):
        train_z.loc[group.index, pruned_cols] = group[pruned_cols].apply(zscore, axis=0)
    test_z = test.copy()
    for pid, group in test.groupby('pid'):
        if pid in train['pid'].values:
            ref = train[train['pid'] == pid][pruned_cols]
            mu = ref.mean()
            sigma = ref.std(ddof=0).replace(0, 1.0)
        else:
            mu = group[pruned_cols].mean()
            sigma = group[pruned_cols].std(ddof=0).replace(0, 1.0)
        test_z.loc[group.index, pruned_cols] = (group[pruned_cols] - mu) / sigma
    # Print window_idx range after z-scoring features
    if 'window_idx' in train_z.columns:
        print(f"window_idx range after z-scoring: min={train_z['window_idx'].min()}, max={train_z['window_idx'].max()}")
    # Print window_idx range after z-scoring features
    if 'window_idx' in train_z.columns:
        print(f"window_idx range after z-scoring: min={train_z['window_idx'].min()}, max={train_z['window_idx'].max()}")
    X_train = train_z[pruned_cols].values
    y_train = train_z['class_label'].values
    X_test = test_z[pruned_cols].values
    y_test = test_z['class_label'].values
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    # Use default SVC (RBF kernel) without grid search
    clf = SVC(kernel='rbf', random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    results.append(acc)
    all_true.extend(y_test)
    all_pred.extend(y_pred)
    # Track accuracy for each window_idx in the test set
    if 'window_idx' in test_z.columns:
        for idx, win_idx in zip(test_z.index, test_z['window_idx']):
            correct = int(y_pred[list(test_z.index).index(idx)] == y_test[list(test_z.index).index(idx)])
            if win_idx not in window_acc:
                window_acc[win_idx] = 0
                window_counts[win_idx] = 0
            window_acc[win_idx] += correct
            window_counts[win_idx] += 1


print(f'Mean Accuracy: {np.mean(results):.3f}')
print('\nClassification Report:')
print(classification_report(all_true, all_pred, digits=3))

# --- Plot accuracy over window_idx ---
import matplotlib.pyplot as plt
if window_acc:
    window_idxs = sorted(window_acc.keys())
    accs = [window_acc[w] / window_counts[w] for w in window_idxs]
    plt.figure(figsize=(10, 4))
    plt.plot(window_idxs, accs, marker='o', linestyle='-', color='b')
    # Label the point with the highest accuracy
    max_idx = np.argmax(accs)
    max_win = window_idxs[max_idx]
    max_acc = accs[max_idx]
    plt.annotate(f'Max: {max_acc:.2f}', xy=(max_win, max_acc), xytext=(max_win, max_acc+0.05),
                 arrowprops=dict(facecolor='red', shrink=0.05), ha='center', color='red', fontsize=10)
    plt.title('SVM Accuracy Over Condition (Rolling Windows)')
    plt.xlabel('Window Index')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()
else:
    print('window_idx column not found in test set; cannot plot accuracy by window index.')
