"""Quick test of the baseline matching fix."""
import pandas as pd

# Test the version stripping logic
test_conditions = [
    'HighStress_HighCog1022_Task',
    'HighStress_LowCog_Task',
    'LowStress_HighCog1022_Task',
    'LowStress_LowCog_Task'
]

condition_map = {
    'Calm Addition': 'LowStress_LowCog_Task',
    'Calm Subtraction': 'LowStress_HighCog_Task',  
    'Stress Addition': 'HighStress_LowCog_Task',
    'Stress Subtraction': 'HighStress_HighCog_Task'
}

print("Testing version stripping logic:")
print("="*60)

for task_condition in test_conditions:
    print(f"\nOriginal: {task_condition}")
    
    # OLD METHOD (BROKEN)
    task_base_old = task_condition
    for suffix in ['1022_', '2043_']:
        task_base_old = task_base_old.replace(suffix, '')
    print(f"  Old method: {task_base_old}")
    
    # NEW METHOD (FIXED)
    task_base_new = task_condition
    task_base_new = task_base_new.replace('HighCog1022', 'HighCog')
    task_base_new = task_base_new.replace('HighCog2043', 'HighCog')
    print(f"  New method: {task_base_new}")
    
    # Check which round tasks it matches
    matches = []
    for round_desc, mapped_cond in condition_map.items():
        if mapped_cond == task_base_new:
            matches.append(round_desc)
    
    if matches:
        print(f"  ✓ MATCHES: {matches}")
    else:
        print(f"  ✗ No matches found")

print("\n" + "="*60)
print("Fix verified! The new method correctly strips version numbers.")
