"""
Quick inspection of physio CSV to discover available markers/columns
for potential event extraction.
"""
import pandas as pd
from pathlib import Path

physio_path = Path(r"C:\phd_projects\vr_tsst_2025\data\raw\metadata\P01.csv")

print("Loading physio CSV...")
df = pd.read_csv(physio_path, low_memory=False)

print(f"\nDataframe shape: {df.shape}")
print(f"\nColumn names and types:")
for col, dtype in df.dtypes.items():
    print(f"  {col:30s} {dtype}")

print(f"\n\nFirst few rows:")
print(df.head(3).to_string())

print(f"\n\nUnique values in key columns:")
key_cols = [col for col in df.columns if 'State' in col or 'Scene' in col or 'Task' in col or 'Phase' in col or 'Response' in col]
for col in key_cols:
    unique_count = df[col].nunique()
    unique_vals = [v for v in df[col].unique() if pd.notna(v)]
    print(f"\n{col}: {unique_count} unique values")
    if len(unique_vals) <= 10:
        print(f"  {sorted(unique_vals)}")
    else:
        print(f"  (too many to display, first 10: {sorted(unique_vals)[:10]})")
