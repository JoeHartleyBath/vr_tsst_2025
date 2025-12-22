import pandas as pd
import json

# Read detailed results
with open('c:/vr_tsst_2025/output/xgboost_results/multimodal_detailed_results.json', 'r') as f:
    detailed = json.load(f)

# Check if tuned_params exist in any result
has_tuned = any('tuned_params' in r and r['tuned_params'] for r in detailed)
has_threshold = any('threshold' in r and r['threshold'] is not None for r in detailed)

print(f'Total strategies: {len(detailed)}')
print(f'Has tuned_params: {has_tuned}')
print(f'Has thresholds: {has_threshold}')

# Count by task
df_all = pd.DataFrame(detailed)
print(f"\nStrategies per task:")
print(df_all.groupby('task').size())

# Show top performers by task
for task in df_all['task'].unique():
    df_task = df_all[df_all['task'] == task].sort_values('accuracy_mean', ascending=False)
    print(f"\n=== TOP 5 for {task} (by accuracy) ===")
    for idx, row in df_task.head(5).iterrows():
        print(f"{row['accuracy_mean']:.4f} ± {row['accuracy_std']:.4f} | AUC {row['auc_mean']:.4f} | {row['model']:25s} | {row['modality']:8s}")
        if row.get('tuned_params'):
            print(f"  Tuned: {row['tuned_params']}")
        if row.get('threshold') is not None:
            print(f"  Threshold: {row['threshold']:.4f}")
    
    print(f"\n=== TOP 5 for {task} (by AUC) ===")
    df_task_auc = df_task.sort_values('auc_mean', ascending=False)
    for idx, row in df_task_auc.head(5).iterrows():
        print(f"AUC {row['auc_mean']:.4f} | Acc {row['accuracy_mean']:.4f} ± {row['accuracy_std']:.4f} | {row['model']:25s} | {row['modality']:8s}")
        if row.get('tuned_params'):
            print(f"  Tuned: {row['tuned_params']}")
        if row.get('threshold') is not None:
            print(f"  Threshold: {row['threshold']:.4f}")
