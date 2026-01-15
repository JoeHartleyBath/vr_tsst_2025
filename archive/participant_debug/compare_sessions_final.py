import pandas as pd
import numpy as np

# Load physio features
physio = pd.read_csv('output/aggregated/physio_features.csv')
p16_data = physio[physio['Participant_ID'] == 16].copy()

# Add row numbers to identify sessions
p16_data = p16_data.reset_index()
p16_data['row_num'] = p16_data.index

# Separate the two sessions based on the pattern we saw
# First 21 rows = Session 1 (rows 299-319)
# Next 21 rows = Session 2 (rows 341-361)
session1 = p16_data.iloc[:21].copy()
session2 = p16_data.iloc[21:].copy()

print("SESSION 1 (First 21 rows - likely P16.csv, recorded first):")
print("="*70)
session1['missing_pct'] = (session1.isnull().sum(axis=1) / (len(session1.columns)-2)) * 100
print(f"Average missing %: {session1['missing_pct'].mean():.2f}%")
print(f"Rows with >20% missing: {(session1['missing_pct'] > 20).sum()}/21")
print(f"Rows with >10% missing: {(session1['missing_pct'] > 10).sum()}/21")
print(f"Task conditions with 0% missing: {(session1[session1['Condition'].str.contains('Task', na=False)]['missing_pct'] == 0).sum()}/4")

print("\nSESSION 2 (Last 21 rows - likely P14.csv, recorded second):")
print("="*70)
session2['missing_pct'] = (session2.isnull().sum(axis=1) / (len(session2.columns)-2)) * 100
print(f"Average missing %: {session2['missing_pct'].mean():.2f}%")
print(f"Rows with >20% missing: {(session2['missing_pct'] > 20).sum()}/21")
print(f"Rows with >10% missing: {(session2['missing_pct'] > 10).sum()}/21")
print(f"Task conditions with 0% missing: {(session2[session2['Condition'].str.contains('Task', na=False)]['missing_pct'] == 0).sum()}/4")

print("\n" + "="*70)
print("COMPARISON SUMMARY:")
print("="*70)

# Calculate completeness for main task conditions only
task_conditions = ['LowStress_HighCog_Task', 'LowStress_LowCog_Task', 
                   'HighStress_HighCog_Task', 'HighStress_LowCog_Task']

print("\nTask condition completeness:")
for cond in task_conditions:
    s1_missing = session1[session1['Condition'] == cond]['missing_pct'].values[0]
    s2_missing = session2[session2['Condition'] == cond]['missing_pct'].values[0]
    print(f"{cond}:")
    print(f"  Session 1: {s1_missing:.1f}% missing")
    print(f"  Session 2: {s2_missing:.1f}% missing")

print("\n" + "="*70)
print("RECOMMENDATION:")
if session1['missing_pct'].mean() < session2['missing_pct'].mean():
    print("✓ SESSION 1 has better data quality (lower missing %)")
    print("  → Keep P16.csv (first session)")
    print("  → Discard P14.csv (second session)")
else:
    print("✓ SESSION 2 has better data quality (lower missing %)")
    print("  → Keep P14.csv (second session)")
    print("  → Discard P16.csv (first session)")
