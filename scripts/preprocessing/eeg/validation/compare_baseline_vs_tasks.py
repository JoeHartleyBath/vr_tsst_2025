"""
Compare EEG features between baseline and task conditions.

This script analyzes whether features show expected changes from
baseline to task conditions, checking both:
- Frontal features (relevant for cognitive load)
- Parietal/Occipital alpha (should decrease during tasks)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
import seaborn as sns

# Configuration
PROJECT_ROOT = Path('c:/vr_tsst_2025')
FEATURES_FILE = PROJECT_ROOT / 'output' / 'aggregated' / 'eeg_features.csv'
OUTPUT_DIR = PROJECT_ROOT / 'output' / 'qc'
OUTPUT_DIR.mkdir(exist_ok=True)

print("=" * 80)
print("BASELINE vs TASK COMPARISON")
print("=" * 80)
print()

# Load data
print(f"Loading features from: {FEATURES_FILE}")
df = pd.read_csv(FEATURES_FILE)
print(f"✓ Loaded {len(df)} rows, {len(df.columns)} columns\n")

# Identify baseline and task conditions
print("Identifying conditions...")
baseline_conditions = [cond for cond in df['Condition'].unique() 
                      if 'Baseline' in cond or 'baseline' in cond]
task_conditions = [cond for cond in df['Condition'].unique() 
                  if 'Task' in cond]

print(f"  Baseline conditions ({len(baseline_conditions)}):")
for cond in baseline_conditions:
    count = len(df[df['Condition'] == cond])
    print(f"    - {cond} (n={count})")

print(f"\n  Task conditions ({len(task_conditions)}):")
for cond in task_conditions:
    count = len(df[df['Condition'] == cond])
    print(f"    - {cond} (n={count})")
print()

if len(baseline_conditions) == 0:
    print("⚠ WARNING: No baseline conditions found!")
    print("Looking for alternative baseline markers...")
    
    # Look for fixation cross or other pre-task markers
    potential_baseline = [cond for cond in df['Condition'].unique() 
                         if any(marker in cond for marker in ['Fixation', 'Pre_Exposure', 'Calibration'])]
    
    if potential_baseline:
        print(f"Found {len(potential_baseline)} potential baseline conditions:")
        for cond in potential_baseline:
            print(f"  - {cond}")
        baseline_conditions = potential_baseline
    else:
        print("❌ No suitable baseline conditions found. Cannot proceed.")
        exit(1)

if len(task_conditions) == 0:
    print("❌ No task conditions found. Cannot proceed.")
    exit(1)

# Filter data
df_baseline = df[df['Condition'].isin(baseline_conditions)]
df_task = df[df['Condition'].isin(task_conditions)]

print(f"Data split:")
print(f"  Baseline: {len(df_baseline)} observations")
print(f"  Task: {len(df_task)} observations")
print()

# ============================================================================
# 1. FRONTAL ALPHA ANALYSIS
# ============================================================================
print("=" * 80)
print("1. FRONTAL ALPHA (Cognitive Load Indicator)")
print("=" * 80)
print("Expected: Alpha decreases during cognitive tasks\n")

frontal_alpha_cols = [col for col in df.columns 
                     if 'Frontal' in col and 'Alpha_Power' in col 
                     and 'Low' not in col and 'High' not in col]

if frontal_alpha_cols:
    for feat in frontal_alpha_cols[:3]:  # Check first 3 frontal alpha features
        print(f"\n{feat}:")
        print(f"{'Participant':<12} {'Baseline':<12} {'Task':<12} {'Change':<12} {'p-value':<12} {'Sig'}")
        print("-" * 80)
        
        decreased_count = 0
        increased_count = 0
        sig_count = 0
        
        for pid in sorted(df['Participant'].unique()):
            baseline_vals = df_baseline[df_baseline['Participant'] == pid][feat].values
            task_vals = df_task[df_task['Participant'] == pid][feat].values
            
            if len(baseline_vals) > 0 and len(task_vals) > 0:
                baseline_mean = baseline_vals.mean()
                task_mean = task_vals.mean()
                change = task_mean - baseline_mean
                change_pct = (change / baseline_mean) * 100 if baseline_mean != 0 else 0
                
                # Paired t-test if we have multiple samples
                if len(baseline_vals) > 1 and len(task_vals) > 1:
                    t_stat, p_val = stats.ttest_ind(baseline_vals, task_vals)
                else:
                    p_val = np.nan
                
                sig_marker = "✓" if p_val < 0.05 else "✗" if not np.isnan(p_val) else "-"
                direction = "↓" if change < 0 else "↑"
                
                print(f"P{pid:02d}          {baseline_mean:>10.4f}  {task_mean:>10.4f}  "
                      f"{change:>+10.4f} {direction}  {p_val:>10.4f}  {sig_marker}")
                
                if change < 0:
                    decreased_count += 1
                else:
                    increased_count += 1
                
                if p_val < 0.05:
                    sig_count += 1
        
        print("\nSummary:")
        total = decreased_count + increased_count
        print(f"  Decreased (expected): {decreased_count}/{total} ({decreased_count/total*100:.0f}%)")
        print(f"  Increased: {increased_count}/{total} ({increased_count/total*100:.0f}%)")
        print(f"  Significant changes: {sig_count}/{total}")
        
        if decreased_count > increased_count:
            print("  ✓ Majority show expected decrease")
        else:
            print("  ⚠ Unexpected: majority show increase or no clear pattern")

# ============================================================================
# 2. PARIETAL/OCCIPITAL ALPHA ANALYSIS
# ============================================================================
print("\n" + "=" * 80)
print("2. PARIETAL/OCCIPITAL ALPHA (Visual Attention Indicator)")
print("=" * 80)
print("Expected: Posterior alpha decreases during visual tasks\n")

posterior_alpha_cols = [col for col in df.columns 
                       if ('Parietal_' in col) 
                       and 'Alpha_Power' in col
                       and 'Low' not in col and 'High' not in col]

if posterior_alpha_cols:
    for feat in posterior_alpha_cols[:2]:  # Check first 2 posterior alpha features
        print(f"\n{feat}:")
        print(f"{'Participant':<12} {'Baseline':<12} {'Task':<12} {'Change':<12} {'p-value':<12} {'Sig'}")
        print("-" * 80)
        
        decreased_count = 0
        increased_count = 0
        sig_count = 0
        
        for pid in sorted(df['Participant'].unique()):
            baseline_vals = df_baseline[df_baseline['Participant'] == pid][feat].values
            task_vals = df_task[df_task['Participant'] == pid][feat].values
            
            if len(baseline_vals) > 0 and len(task_vals) > 0:
                baseline_mean = baseline_vals.mean()
                task_mean = task_vals.mean()
                change = task_mean - baseline_mean
                
                # Statistical test
                if len(baseline_vals) > 1 and len(task_vals) > 1:
                    t_stat, p_val = stats.ttest_ind(baseline_vals, task_vals)
                else:
                    p_val = np.nan
                
                sig_marker = "✓" if p_val < 0.05 else "✗" if not np.isnan(p_val) else "-"
                direction = "↓" if change < 0 else "↑"
                
                print(f"P{pid:02d}          {baseline_mean:>10.4f}  {task_mean:>10.4f}  "
                      f"{change:>+10.4f} {direction}  {p_val:>10.4f}  {sig_marker}")
                
                if change < 0:
                    decreased_count += 1
                else:
                    increased_count += 1
                
                if p_val < 0.05:
                    sig_count += 1
        
        print("\nSummary:")
        total = decreased_count + increased_count
        print(f"  Decreased (expected): {decreased_count}/{total} ({decreased_count/total*100:.0f}%)")
        print(f"  Increased: {increased_count}/{total} ({increased_count/total*100:.0f}%)")
        print(f"  Significant changes: {sig_count}/{total}")
        
        if decreased_count > increased_count:
            print("  ✓ Majority show expected decrease")
        else:
            print("  ⚠ Unexpected: majority show increase or no clear pattern")

# ============================================================================
# 3. FRONTAL THETA (Mental Workload Indicator)
# ============================================================================
print("\n" + "=" * 80)
print("3. FRONTAL THETA (Mental Workload Indicator)")
print("=" * 80)
print("Expected: Theta increases during cognitive tasks\n")

frontal_theta_cols = [col for col in df.columns 
                     if 'FrontalM' in col and 'Theta_Power' in col
                     and 'Low' not in col and 'High' not in col]

if frontal_theta_cols:
    feat = frontal_theta_cols[0]
    print(f"\n{feat}:")
    print(f"{'Participant':<12} {'Baseline':<12} {'Task':<12} {'Change':<12} {'p-value':<12} {'Sig'}")
    print("-" * 80)
    
    increased_count = 0
    decreased_count = 0
    sig_count = 0
    
    for pid in sorted(df['Participant'].unique()):
        baseline_vals = df_baseline[df_baseline['Participant'] == pid][feat].values
        task_vals = df_task[df_task['Participant'] == pid][feat].values
        
        if len(baseline_vals) > 0 and len(task_vals) > 0:
            baseline_mean = baseline_vals.mean()
            task_mean = task_vals.mean()
            change = task_mean - baseline_mean
            
            # Statistical test
            if len(baseline_vals) > 1 and len(task_vals) > 1:
                t_stat, p_val = stats.ttest_ind(baseline_vals, task_vals)
            else:
                p_val = np.nan
            
            sig_marker = "✓" if p_val < 0.05 else "✗" if not np.isnan(p_val) else "-"
            direction = "↑" if change > 0 else "↓"
            
            print(f"P{pid:02d}          {baseline_mean:>10.4f}  {task_mean:>10.4f}  "
                  f"{change:>+10.4f} {direction}  {p_val:>10.4f}  {sig_marker}")
            
            if change > 0:
                increased_count += 1
            else:
                decreased_count += 1
            
            if p_val < 0.05:
                sig_count += 1
    
    print("\nSummary:")
    total = increased_count + decreased_count
    print(f"  Increased (expected): {increased_count}/{total} ({increased_count/total*100:.0f}%)")
    print(f"  Decreased: {decreased_count}/{total} ({decreased_count/total*100:.0f}%)")
    print(f"  Significant changes: {sig_count}/{total}")
    
    if increased_count > decreased_count:
        print("  ✓ Majority show expected increase")
    else:
        print("  ⚠ Unexpected: majority show decrease")

# ============================================================================
# 4. OVERALL ASSESSMENT
# ============================================================================
print("\n" + "=" * 80)
print("4. OVERALL ASSESSMENT")
print("=" * 80)

print("\n📊 Summary of baseline-to-task changes:")
print("  Expected patterns for cognitive/visual tasks:")
print("    • Frontal alpha ↓ (reduced with attention)")
print("    • Posterior alpha ↓ (reduced with visual processing)")
print("    • Frontal theta ↑ (increased with mental workload)")
print()
print("  If features show opposite patterns or no changes:")
print("    - Data may not capture task-related neural activity")
print("    - Baseline and task states may be too similar")
print("    - Feature extraction timing may be incorrect")

print("\n" + "=" * 80)
print("Analysis complete!")
print("=" * 80)
