"""Test baseline adjustment specifically."""
import sys
sys.path.insert(0, 'pipelines/08_xgboost_multimodal')

# Import the main script components
import pandas as pd
from xgboost_multimodal_classification import (
    load_data, identify_feature_columns, apply_baseline_adjustment,
    INPUT_DATA, COUNTERBALANCE, CLASSIFICATION_TASKS
)

print("="*70)
print("BASELINE ADJUSTMENT TEST")
print("="*70)

# Load data
print("\n[1] Loading data...")
df = load_data(INPUT_DATA)
feature_groups = identify_feature_columns(df)

# Load counterbalance
print("\n[2] Loading counterbalance...")
cb_df = pd.read_excel(COUNTERBALANCE)
print(f"Counterbalance: {len(cb_df)} participants")

# Get task data
print("\n[3] Filtering to stress classification tasks...")
task_map = CLASSIFICATION_TASKS['stress_classification']
task_df = df[df['Condition'].isin(task_map.keys())].copy()
task_df['class_label'] = task_df['Condition'].map(task_map)
print(f"Task data: {len(task_df)} windows from {task_df['Participant_ID'].nunique()} participants")

# Test baseline adjustment
print("\n[4] Testing baseline adjustment: subtract, full duration...")
adjusted_df = apply_baseline_adjustment(
    task_df=task_df,
    full_df=df,  # Pass full DF with Forest conditions
    method='subtract',
    duration='full',
    feature_cols=feature_groups['all'],
    counterbalance_data=cb_df
)

if len(adjusted_df) == 0:
    print("\n" + "="*70)
    print("❌ FAILED: Baseline adjustment returned empty DataFrame")
    print("="*70)
    sys.exit(1)
else:
    print(f"\n✓ SUCCESS: Adjusted data has {len(adjusted_df)} windows")
    print(f"✓ Participants: {adjusted_df['Participant_ID'].nunique()}")
    print(f"✓ Conditions: {adjusted_df['Condition'].unique()}")
    print("\n" + "="*70)
    print("✓✓ BASELINE ADJUSTMENT WORKING!")
    print("="*70)
    sys.exit(0)
