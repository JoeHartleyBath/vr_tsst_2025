import pyreadr
import pandas as pd
import statsmodels.formula.api as smf
import sys

try:
    result = pyreadr.read_r('output/full_data_for_reference.rds')
    df = result[None] # standard pyreadr behavior
    print("RDS Loaded successfully.")
    print("Columns:", df.columns.tolist())
    
    relax_df = df[df['condition_type'] == 'Relaxation'].copy()
    print("Relaxation rows:", len(relax_df))

    # Identify block/order
    # Looking for 'round' or something indicating order
    if 'round' in relax_df.columns:
        print("Found 'round' column.")
        order_col = 'round'
    elif 'condition' in relax_df.columns:
        print("Unique conditions:", relax_df['condition'].unique())
        # Try to parse order or assume numeric
        order_col = None
    else:
        print("No obvious order column.")
        order_col = None

    if order_col:
        features = ["hr_med", "eda_tonic_mean", "eeg_fm_theta_power"]
        for feat in features:
            if feat in relax_df.columns:
                print(f"\nAnalyzing {feat}...")
                # LMM: feature ~ round + (1|participant)
                # using statsmodels mixedlm
                model = smf.mixedlm(f"{feat} ~ {order_col}", relax_df, groups=relax_df["participant_id"])
                res = model.fit()
                print(res.summary())

except Exception as e:
    print("Error:", e)
