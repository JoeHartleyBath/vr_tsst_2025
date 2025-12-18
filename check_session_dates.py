import pandas as pd

# Load both files
p14 = pd.read_csv('data/raw/metadata/P14.csv', low_memory=False)
p16 = pd.read_csv('data/raw/metadata/P16.csv', low_memory=False)

print("P14.csv:")
print(f"  Participant_ID: {p14['Participant_ID'].unique()}")
print(f"  Total rows: {len(p14)}")
if 'LSL_Timestamp' in p14.columns:
    print(f"  First timestamp: {p14['LSL_Timestamp'].min()}")
    print(f"  Last timestamp: {p14['LSL_Timestamp'].max()}")
if 'Date' in p14.columns:
    print(f"  Date column: {p14['Date'].unique()[:3]}")

print("\nP16.csv:")
print(f"  Participant_ID: {p16['Participant_ID'].unique()}")
print(f"  Total rows: {len(p16)}")
if 'LSL_Timestamp' in p16.columns:
    print(f"  First timestamp: {p16['LSL_Timestamp'].min()}")
    print(f"  Last timestamp: {p16['LSL_Timestamp'].max()}")
if 'Date' in p16.columns:
    print(f"  Date column: {p16['Date'].unique()[:3]}")

# Convert timestamps to datetime if possible
print("\n" + "="*70)
print("Session dates:")
if 'LSL_Timestamp' in p14.columns and 'LSL_Timestamp' in p16.columns:
    from datetime import datetime
    
    p14_start = pd.to_datetime(p14['LSL_Timestamp'].min(), unit='s')
    p16_start = pd.to_datetime(p16['LSL_Timestamp'].min(), unit='s')
    
    print(f"P14.csv session started: {p14_start}")
    print(f"P16.csv session started: {p16_start}")
    
    if p14_start < p16_start:
        days_diff = (p16_start - p14_start).days
        print(f"\n✓ P14.csv was recorded FIRST ({days_diff} days before P16.csv)")
    else:
        days_diff = (p14_start - p16_start).days
        print(f"\n✓ P16.csv was recorded FIRST ({days_diff} days before P14.csv)")
