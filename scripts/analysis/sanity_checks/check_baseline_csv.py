import pandas as pd
import statsmodels.formula.api as smf
import sys

try:
    df = pd.read_csv('output/final_data.csv')
    print("CSV Loaded successfully.")
    
    # We want to check stability of Pre-Condition Baselines across Rounds (1-4)
    # Features of interest:
    features = [
        "hr_med_precond", 
        "eda_tonic_mean_precond", 
        "eeg_fm_theta_power_precond",
        "eeg_f_beta_power_precond"
    ]
    
    print("\n--- BASELINE STABILITY CHECK (Drift Analysis) ---")
    print("Model: Feature_Baseline ~ Round + (1|Participant)")
    
    for feat in features:
        if feat in df.columns:
            # We use mixedlm (Linear Mixed Model) to account for repeated measures
            model = smf.mixedlm(f"{feat} ~ round", df, groups=df["participant_id"])
            res = model.fit()
            
            print(f"\nFeature: {feat}")
            print(f"Intercept: {res.params['Intercept']:.2f}")
            print(f"Slope (Effect of Round): {res.params['round']:.3f}")
            print(f"P-value (Round): {res.pvalues['round']:.4f}")
            
            if res.pvalues['round'] < 0.05:
                direction = "INCREASING" if res.params['round'] > 0 else "DECREASING"
                print(f"-> SIGNIFICANT DRIFT ({direction})")
            else:
                print("-> STABLE (No significant drift)")

except Exception as e:
    print("Error:", e)
