import pandas as pd
import numpy as np

print("\n=== EXTRACTION OUTPUT VALIDATION ===\n")

# Check if output exists
try:
    df = pd.read_csv("output/aggregated/all_data_aggregated.csv")
    
    print(f"✅ Output file loaded successfully")
    print(f"   Rows: {len(df):,}")
    print(f"   Columns: {len(df.columns)}")
    print(f"   Participants: {df['Participant_ID'].nunique()}")
    if 'Condition' in df.columns:
        print(f"   Conditions: {sorted(df['Condition'].unique())}")
    
    # Feature breakdown
    hr_cols = [c for c in df.columns if 'hr' in c.lower() or 'heart' in c.lower()]
    gsr_cols = [c for c in df.columns if 'gsr' in c.lower() or 'eda' in c.lower()]
    pupil_cols = [c for c in df.columns if 'pupil' in c.lower()]
    eeg_cols = [c for c in df.columns if any(b in c.lower() for b in ['alpha','beta','theta','delta','gamma'])]
    
    print(f"\n   Feature counts:")
    print(f"   - HR/HRV: {len(hr_cols)}")
    print(f"   - GSR/EDA: {len(gsr_cols)}")
    print(f"   - Pupil: {len(pupil_cols)}")
    print(f"   - EEG: {len(eeg_cols)}")
    
    # Sanity checks
    if hr_cols:
        hr_sample = df[hr_cols[0]].dropna()
        if len(hr_sample) > 0:
            print(f"\n   HR range: {hr_sample.min():.1f} - {hr_sample.max():.1f} BPM")
            if 40 <= hr_sample.min() and hr_sample.max() <= 220:
                print("   ✅ Within physiological range")
    
    # Missing data
    missing_pct = (df.isnull().sum() / len(df) * 100)
    high_missing = missing_pct[missing_pct > 70]
    if len(high_missing) > 0:
        print(f"\n   ⚠️  {len(high_missing)} features with >70% missing")
    else:
        print(f"\n   ✅ No excessive missing data")
    
    print("\n=== READY FOR STAGE 6: R PREPROCESSING ===")
    
except FileNotFoundError:
    print("❌ Output file not found yet")
except Exception as e:
    print(f"❌ Error: {e}")
