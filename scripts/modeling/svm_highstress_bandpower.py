import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold

# Load data
df = pd.read_csv(r'C:/vr_tsst_2025/output/aggregated/eeg_features_highstress.csv')

# Identify columns
group_cols = ['pid', 'label']
feature_cols = [col for col in df.columns if col not in group_cols]


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

# Prepare data for SVM
groups = df['pid'].values
labels = df['label'].values
results = []
all_true = []
all_pred = []
gkf = GroupKFold(n_splits=5)

for train_idx, test_idx in gkf.split(df, labels, groups):
    train = df.iloc[train_idx].copy()
    test = df.iloc[test_idx].copy()
    # Z-score within each participant in train, then apply to test
    train_z = train.copy()
    for pid, group in train.groupby('pid'):
        train_z.loc[group.index, pruned_cols] = (group[pruned_cols] - group[pruned_cols].mean()) / group[pruned_cols].std(ddof=0)
    test_z = test.copy()
    for pid, group in test.groupby('pid'):
        # Use train mean/std for this participant if available, else use test's own
        if pid in train['pid'].values:
            ref = train[train['pid'] == pid][pruned_cols]
            mu = ref.mean()
            sigma = ref.std(ddof=0)
        else:
            mu = group[pruned_cols].mean()
            sigma = group[pruned_cols].std(ddof=0)
        test_z.loc[group.index, pruned_cols] = (group[pruned_cols] - mu) / sigma
    X_train = train_z[pruned_cols].values
    y_train = train_z['label'].values
    X_test = test_z[pruned_cols].values
    y_test = test_z['label'].values
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    clf = SVC(kernel='linear', random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    results.append(acc)
    all_true.extend(y_test)
    all_pred.extend(y_pred)

print(f'Fold Accuracies: {results}')
print(f'Mean Accuracy: {np.mean(results):.3f}')
print('\nClassification Report:')
print(classification_report(all_true, all_pred, digits=3))
