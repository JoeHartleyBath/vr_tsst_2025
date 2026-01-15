#!/usr/bin/env python3
import pandas as pd
from pathlib import Path

path = Path('data/experimental_counterbalance.xlsx')

# Read all sheets
xl_file = pd.ExcelFile(path)
print(f"Sheets: {xl_file.sheet_names}")
print()

for sheet_name in xl_file.sheet_names:
    df = pd.read_excel(path, sheet_name=sheet_name)
    print(f"=== {sheet_name} ===")
    print(f"Shape: {df.shape}")
    print()
    print(df.head(15))
    print()

