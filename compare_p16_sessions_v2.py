import pandas as pd
import numpy as np

# Load physio features
physio = pd.read_csv('output/aggregated/physio_features.csv')

# Filter for Participant_ID = 16 (both sessions)
p16_data = physio[physio['Participant_ID'] == 16].copy()

print(f"Total rows for Participant_ID=16: {len(p16_data)}")
print(f"Available columns: {p16_data.columns.tolist()}")

print("\n" + "="*70)
print("Data quality comparison:\n")

# Calculate missing values per row
p16_data['missing_count'] = p16_data.isnull().sum(axis=1)
p16_data['missing_pct'] = (p16_data.isnull().sum(axis=1) / len(p16_data.columns)) * 100

# Get feature columns (exclude metadata)
meta_cols = ['Participant_ID', 'Condition', 'missing_count', 'missing_pct']
feature_cols = [col for col in p16_data.columns if col not in meta_cols]

print(f"Total feature columns: {len(feature_cols)}")

# Group by condition and look at both rows
print("\n" + "="*70)
print("Data completeness by condition (both sessions):\n")

for condition in sorted(p16_data['Condition'].unique()):
    cond_data = p16_data[p16_data['Condition'] == condition]
    print(f"{condition}:")
    print(f"  Rows: {len(cond_data)}")
    print(f"  Missing %: {cond_data['missing_pct'].values}")
    print()

print("="*70)
print("Overall statistics:")
print(f"\nMissing data per row:")
print(p16_data['missing_pct'].describe())

print(f"\nRows with >50% missing: {(p16_data['missing_pct'] > 50).sum()}")
print(f"\nRows with >25% missing: {(p16_data['missing_pct'] > 25).sum()}")

# Try to identify which session is which by looking at row indices
print("\n" + "="*70)
print("Row analysis:")
print(p16_data[['Condition', 'missing_pct']].head(25))
