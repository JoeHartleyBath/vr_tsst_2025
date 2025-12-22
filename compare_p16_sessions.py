import pandas as pd
import numpy as np

# Load physio features
physio = pd.read_csv('output/aggregated/physio_features.csv')

# Filter for Participant_ID = 16 (both sessions)
p16_data = physio[physio['Participant_ID'] == 16].copy()

print(f"Total rows for Participant_ID=16: {len(p16_data)}")
print(f"Unique conditions: {p16_data['Condition'].nunique()}")
print(f"Conditions per participant: {p16_data['Condition'].value_counts()}")

# Check which file each row came from by examining timestamps or other metadata
# Group by some identifier to separate the two sessions
print("\n" + "="*70)
print("Data quality comparison:\n")

# Calculate missing values per row
p16_data['missing_count'] = p16_data.isnull().sum(axis=1)
p16_data['missing_pct'] = (p16_data.isnull().sum(axis=1) / len(p16_data.columns)) * 100

# Get feature columns (exclude metadata)
meta_cols = ['Participant_ID', 'Condition', 'Window', 'missing_count', 'missing_pct']
feature_cols = [col for col in p16_data.columns if col not in meta_cols]

print(f"Total feature columns: {len(feature_cols)}")
print(f"\nRows sorted by condition:")
print(p16_data[['Condition', 'Window', 'missing_count', 'missing_pct']].sort_values('Condition'))

print("\n" + "="*70)
print("Summary statistics:")
print(f"Average missing % per row: {p16_data['missing_pct'].mean():.2f}%")
print(f"Rows with >50% missing: {(p16_data['missing_pct'] > 50).sum()}")
print(f"Rows with >25% missing: {(p16_data['missing_pct'] > 25).sum()}")

# Check if we can identify which rows came from which file
# P14.csv should have ~21 rows (4 conditions * ~5 windows)
# P16.csv should have ~21 rows (4 conditions * ~5 windows)
print("\n" + "="*70)
print("Attempting to identify sessions...")

# If there are exactly 42 rows, they're probably mixed
# We need to look at the raw extraction to see which is which
# For now, let's look at the condition-level aggregates

condition_summary = p16_data.groupby('Condition').agg({
    'missing_pct': 'mean',
    'Window': 'count'
}).round(2)

print("\nPer-condition summary:")
print(condition_summary)

# Save detailed view
output = p16_data[['Condition', 'Window', 'missing_count', 'missing_pct'] + feature_cols[:10]]
print("\n" + "="*70)
print("First 10 feature columns by condition:")
print(output.to_string())
