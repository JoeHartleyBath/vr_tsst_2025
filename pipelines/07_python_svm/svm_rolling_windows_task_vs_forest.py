import os
import pandas as pd
import numpy as np
import yaml
from sklearn.model_selection import GroupKFold
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
import matplotlib.pyplot as plt

FEATURES_PATH = 'C:/vr_tsst_2025/output/aggregated/eeg_features_rolling_windows.csv'
LABEL_CONFIG_PATH = 'scripts/modeling/model_class_labels.yaml'
PLOT_DIR = 'output/plots'

os.makedirs(PLOT_DIR, exist_ok=True)

print("[INFO] Starting SVM rolling windows classification for task_vs_forest...")

# Load features
try:
    df = pd.read_csv(FEATURES_PATH)
    print(f"[INFO] Loaded features: {df.shape[0]} rows, {df.shape[1]} columns")
except Exception as e:
    print(f"[ERROR] Failed to load features file: {e}")
    raise

# Load label config
def load_label_map(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

label_map = load_label_map(LABEL_CONFIG_PATH)
task = 'task_vs_forest'
if task not in label_map:
    raise ValueError(f"Task '{task}' not found in label config.")
task_map = label_map[task]
task_df = df[df['event_label'].isin(task_map.keys())].copy()
task_df['class_label'] = task_df['event_label'].map(task_map)
print(f"[INFO] Filtered to {task_df.shape[0]} rows for task '{task}'")
if task_df.shape[0] == 0:
    print(f"[WARNING] No data for task {task}, exiting.")
    exit(0)

# Identify columns
group_cols = ['pid', 'event_label', 'window_idx', 'class_label']
feature_cols = [col for col in task_df.columns if col not in group_cols]
feature_cols = [col for col in feature_cols if 'Delta' not in col and 'Occipital' not in col]

# Feature pruning
vt = VarianceThreshold(threshold=1e-6)
X_var = vt.fit_transform(task_df[feature_cols])
kept_var_idx = vt.get_support(indices=True)
kept_var_cols = [feature_cols[i] for i in kept_var_idx]

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
print(f"[INFO] Features after pruning: {len(pruned_cols)}")

zscore = lambda series: (series - np.nanmean(series)) / (np.nanstd(series, ddof=0) if np.nanstd(series, ddof=0) > 1e-6 else 1.0)

groups = task_df['pid'].values
labels = task_df['class_label'].values
results = []
all_true = []
all_pred = []
window_acc = dict()
window_counts = dict()
gkf = GroupKFold(n_splits=min(5, len(np.unique(groups))))

for train_idx, test_idx in gkf.split(task_df, labels, groups):
    train = task_df.iloc[train_idx].copy()
    test = task_df.iloc[test_idx].copy()
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
    X_train = train_z[pruned_cols].values
    y_train = train_z['class_label'].values
    X_test = test_z[pruned_cols].values
    y_test = test_z['class_label'].values
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    clf = SVC(kernel='rbf', random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    results.append(acc)
    all_true.extend(y_test)
    all_pred.extend(y_pred)
    if 'window_idx' in test_z.columns:
        for idx, win_idx in zip(test_z.index, test_z['window_idx']):
            correct = int(y_pred[list(test_z.index).index(idx)] == y_test[list(test_z.index).index(idx)])
            if win_idx not in window_acc:
                window_acc[win_idx] = 0
                window_counts[win_idx] = 0
            window_acc[win_idx] += correct
            window_counts[win_idx] += 1

print(f'[RESULT] {task}: Mean Accuracy: {np.mean(results):.3f}')
print(classification_report(all_true, all_pred, digits=3))

# Plot and save
if window_acc:
    window_idxs = sorted(window_acc.keys())
    accs = [window_acc[w] / window_counts[w] for w in window_idxs]
    plt.figure(figsize=(10, 4))
    plt.plot(window_idxs, accs, marker='o', linestyle='-', color='b')
    max_idx = np.argmax(accs)
    max_win = window_idxs[max_idx]
    max_acc = accs[max_idx]
    plt.annotate(f'Max: {max_acc:.2f}', xy=(max_win, max_acc), xytext=(max_win, max_acc+0.05),
                 arrowprops=dict(facecolor='red', shrink=0.05), ha='center', color='red', fontsize=10)
    plt.title('SVM Accuracy: Task vs Forest Scenes')
    plt.xlabel('Window Index')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plot_path = os.path.join(PLOT_DIR, f'svm_accuracy_{task}.png')
    plt.savefig(plot_path)
    plt.close()
    print(f'[INFO] Saved plot: {plot_path}')
else:
    print(f'[WARNING] window_idx column not found or empty for {task}, plot not saved.')
