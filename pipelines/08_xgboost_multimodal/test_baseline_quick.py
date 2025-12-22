"""Quick test of baseline matching with actual data."""
import pandas as pd

# Load data
print("Loading data...")
df = pd.read_csv('output/aggregated/multimodal_features_rolling_windows.csv', low_memory=False)
cb_df = pd.read_excel('data/experimental_counterbalance.xlsx')

print(f"\nData shape: {df.shape}")
print(f"Conditions: {sorted(df['Condition'].unique())}")
print(f"Participants in data: {df['Participant_ID'].nunique()}")
print(f"Participants in counterbalance: {cb_df['Participant'].nunique()}")

# Test with first participant
pid = 1
print(f"\n\nTesting Participant {pid}:")
print("="*60)

p_data = df[df['Participant_ID'] == pid]
print(f"Participant {pid} conditions: {sorted(p_data['Condition'].unique())}")

# Check counterbalance
p_cb = cb_df[cb_df['Participant'] == pid]
if len(p_cb) > 0:
    print(f"\nCounterbalance rounds:")
    for r in [1, 2, 3, 4]:
        print(f"  Round {r}: {p_cb[f'Round {r}'].values[0]}")
else:
    print("NOT in counterbalance!")

# Test the mapping
condition_map = {
    'Calm Addition': 'LowStress_LowCog_Task',
    'Calm Subtraction': 'LowStress_HighCog_Task',  
    'Stress Addition': 'HighStress_LowCog_Task',
    'Stress Subtraction': 'HighStress_HighCog_Task'
}

print(f"\n\nTesting baseline matching logic:")
print("="*60)

# Test each task condition
task_conditions = [c for c in p_data['Condition'].unique() if 'Task' in c]
print(f"Task conditions for P{pid}: {task_conditions}")

for task_cond in task_conditions:
    print(f"\nTask: {task_cond}")
    
    # Apply the fix
    task_base = task_cond
    task_base = task_base.replace('HighCog1022', 'HighCog')
    task_base = task_base.replace('HighCog2043', 'HighCog')
    print(f"  After stripping: {task_base}")
    
    # Find matching round
    round_num = None
    for r in [1, 2, 3, 4]:
        round_cond = p_cb[f'Round {r}'].values[0]
        mapped_cond = condition_map.get(round_cond, round_cond)
        if mapped_cond == task_base:
            round_num = r
            print(f"  ✓ MATCH! Round {r}: '{round_cond}' → '{mapped_cond}'")
            break
        else:
            print(f"    Round {r}: '{round_cond}' → '{mapped_cond}' (no match)")
    
    if round_num:
        forest_cond = f'Forest{round_num}'
        forest_data = p_data[p_data['Condition'] == forest_cond]
        print(f"  Forest baseline: {forest_cond} ({len(forest_data)} windows)")
        if len(forest_data) > 0:
            print(f"  ✓✓ BASELINE MATCHING WORKS!")
        else:
            print(f"  ✗ No Forest data found")
    else:
        print(f"  ✗ No matching round found")

print("\n" + "="*60)
print("If you see '✓✓ BASELINE MATCHING WORKS!' above, the fix is correct!")
print("="*60)
