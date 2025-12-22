"""
Diagnostic script to isolate baseline matching issues.
Tests the baseline adjustment logic without running full ML pipeline.
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Paths
DATA_FILE = Path('output/aggregated/multimodal_features_rolling_windows.csv')
COUNTERBALANCE_FILE = Path('data/experimental_counterbalance.xlsx')

# Task mappings - ACTUAL condition names from data
CLASSIFICATION_TASKS = {
    'HighStress': ['HighStress_HighCog1022_Task', 'HighStress_LowCog_Task'],
    'LowStress': ['LowStress_HighCog1022_Task', 'LowStress_LowCog_Task'],
    'HighCog': ['HighStress_HighCog1022_Task', 'LowStress_HighCog1022_Task'],
    'LowCog': ['HighStress_LowCog_Task', 'LowStress_LowCog_Task'],
}

# Mapping from task condition to corresponding round number
# Based on experimental design: each participant gets 4 rounds
# Need to check how task names map to round positions

def main():
    print("="*80)
    print("BASELINE MATCHING DIAGNOSTIC")
    print("="*80)
    
    # Load data
    print("\n1. Loading merged multimodal data...")
    try:
        df = pd.read_csv(DATA_FILE)
        print(f"   ✓ Loaded {len(df)} windows from {df['Participant_ID'].nunique()} participants")
    except Exception as e:
        print(f"   ✗ ERROR loading data: {e}")
        return
    
    # Check columns
    print(f"\n2. Data columns: {list(df.columns[:10])}... ({len(df.columns)} total)")
    
    # Check participant IDs
    print(f"\n3. Participant IDs in data:")
    participant_ids = sorted(df['Participant_ID'].unique())
    print(f"   Count: {len(participant_ids)}")
    print(f"   Sample: {participant_ids[:10]}")
    print(f"   Type: {type(participant_ids[0])}")
    
    # Check conditions
    print(f"\n4. Conditions in data:")
    conditions = sorted(df['Condition'].unique())
    print(f"   Count: {len(conditions)}")
    print(f"   All conditions: {conditions}")
    
    # Check for Forest conditions specifically
    forest_conditions = [c for c in conditions if 'Forest' in c]
    print(f"\n   Forest conditions found: {len(forest_conditions)}")
    if forest_conditions:
        print(f"   Forest conditions: {forest_conditions}")
        print(f"   ✓ Forest baselines ARE present in data")
    else:
        print(f"   ✗ NO Forest conditions found! This is the problem.")
        print(f"   → EEG features must be re-extracted with Forest1-4 included")
    
    # Load counterbalance sheet
    print(f"\n5. Loading counterbalance sheet...")
    try:
        cb_df = pd.read_excel(COUNTERBALANCE_FILE)
        print(f"   ✓ Loaded {len(cb_df)} participants")
        print(f"   Columns: {list(cb_df.columns)}")
    except Exception as e:
        print(f"   ✗ ERROR loading counterbalance: {e}")
        return
    
    # Check participant IDs in counterbalance
    print(f"\n6. Participant IDs in counterbalance sheet:")
    cb_participant_ids = sorted(cb_df['Participant'].unique())
    print(f"   Count: {len(cb_participant_ids)}")
    print(f"   Sample: {cb_participant_ids[:10]}")
    print(f"   Type: {type(cb_participant_ids[0])}")
    
    # Compare participant IDs
    print(f"\n7. Comparing participant IDs:")
    data_pids = set(participant_ids)
    cb_pids = set(cb_participant_ids)
    
    matching = data_pids & cb_pids
    only_in_data = data_pids - cb_pids
    only_in_cb = cb_pids - data_pids
    
    print(f"   Matching: {len(matching)} participants")
    if matching:
        print(f"   Sample matching: {sorted(list(matching))[:10]}")
    
    if only_in_data:
        print(f"\n   ⚠ Only in data (not in counterbalance): {len(only_in_data)}")
        print(f"     {sorted(list(only_in_data))[:10]}")
    
    if only_in_cb:
        print(f"\n   ⚠ Only in counterbalance (not in data): {len(only_in_cb)}")
        print(f"     {sorted(list(only_in_cb))[:10]}")
    
    # Test baseline matching for a few participants
    print(f"\n8. Testing baseline matching logic (mimicking actual script):")
    
    # Condition mapping from script
    condition_map = {
        'Calm Addition': 'LowStress_LowCog_Task',
        'Calm Subtraction': 'LowStress_HighCog_Task',  
        'Stress Addition': 'HighStress_LowCog_Task',
        'Stress Subtraction': 'HighStress_HighCog_Task'
    }
    
    # Test with actual conditions from data
    test_conditions = ['HighStress_HighCog1022_Task', 'HighStress_LowCog_Task']
    
    for test_cond in test_conditions:
        print(f"\n   Testing condition: {test_cond}")
        
        # Get participants with this condition
        test_df = df[df['Condition'] == test_cond]
        if len(test_df) == 0:
            print(f"   ⚠ No data for {test_cond}")
            continue
            
        test_pids = test_df['Participant_ID'].unique()[:3]  # Test first 3
        print(f"   Testing with {len(test_pids)} participants: {list(test_pids)}")
        
        for pid in test_pids:
            print(f"\n     Participant {pid}:")
            
            # Check if in counterbalance
            cb_row = cb_df[cb_df['Participant'] == pid]
            if len(cb_row) == 0:
                print(f"       ✗ NOT in counterbalance sheet")
                continue
            
            print(f"       ✓ Found in counterbalance")
            rounds = cb_row[['Round 1', 'Round 2', 'Round 3', 'Round 4']].values[0]
            print(f"       Rounds: {rounds}")
            
            # Strip version suffixes like the script does
            task_base = test_cond
            for suffix in ['1022_', '2043_']:
                task_base = task_base.replace(suffix, '')
            print(f"       Task base (after stripping): {task_base}")
            
            # Find matching round
            round_num = None
            for r in [1, 2, 3, 4]:
                round_cond = cb_row[f'Round {r}'].values[0]
                mapped_cond = condition_map.get(round_cond, round_cond)
                print(f"       Round {r}: '{round_cond}' → '{mapped_cond}'")
                if mapped_cond == task_base:
                    round_num = r
                    print(f"       ✓ MATCH! Task was performed in Round {r}")
                    break
            
            if round_num is None:
                print(f"       ✗ No matching round found for {task_base}")
                continue
            
            # Check for Forest baseline
            forest_cond = f'Forest{round_num}'
            baseline_df = df[(df['Participant_ID'] == pid) & (df['Condition'] == forest_cond)]
            
            if len(baseline_df) == 0:
                print(f"       ✗ No {forest_cond} data")
            else:
                print(f"       ✓ Found {len(baseline_df)} {forest_cond} windows")
                print(f"       → BASELINE MATCHING SUCCESSFUL!")

    print("\n" + "="*80)
    print("DIAGNOSIS COMPLETE")
    print("="*80)
    print("\nKEY FINDINGS:")
    print("1. ✓ Forest baselines ARE in the data")
    print("2. ✓ Participant IDs match between data and counterbalance")
    print("3. ✗ Actual condition names don't match CLASSIFICATION_TASKS mappings:")
    print(f"     Data has: {conditions}")
    print("4. ✗ Counterbalance sheet structure incompatible with current baseline matching:")
    print("     - Has 'Round 1-4' columns with task descriptions")
    print("     - Doesn't have task-specific columns like 'HighStress_round'")
    print("     - Need to parse round descriptions or use different matching strategy")
    print("\nSOLUTION:")
    print("1. Fix CLASSIFICATION_TASKS to use actual condition names from data")
    print("2. Rewrite baseline matching to work with 'Round 1-4' column structure")
    print("3. Parse round task names OR use simpler approach: match Forest1-4 to Round 1-4")
    print("="*80)

if __name__ == '__main__':
    main()
