import pandas as pd
from scipy import stats
import os

def main():
    csv_path = r'output/anova_features_precond.csv'
    
    if not os.path.exists(csv_path):
        # Fallback for when running from dev folder
        csv_path = r'../output/anova_features_precond.csv'
        if not os.path.exists(csv_path):
             # Fallback for absolute path if needed, though relative is preferred
             csv_path = r'c:/vr_tsst_2025/output/anova_features_precond.csv'

    print(f"Reading data from: {csv_path}")
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print("Error: File not found.")
        return

    # Filter QC failed
    # Convert to string and normalize to uppercase to handle boolean or string variations
    df['qc_failed_str'] = df['qc_failed'].astype(str).str.strip().str.upper()
    df_clean = df[df['qc_failed_str'] != 'TRUE'].copy()
    
    print("-" * 40)
    print(f"Total participants/rows: {len(df)}")
    print(f"Valid rows (QC passed):   {len(df_clean)}")
    print("-" * 40)

    target_col = 'response_rate_per_min_precond'

    # --- Workload Analysis ---
    print("\n--- Workload Effect (High vs Low) ---")
    workload_group = df_clean.groupby('workload_level')[target_col].agg(['mean', 'std', 'count'])
    print(workload_group)
    
    high_work = df_clean[df_clean['workload_level'] == 'High'][target_col].dropna()
    low_work = df_clean[df_clean['workload_level'] == 'Low'][target_col].dropna()
    
    t_stat_work, p_val_work = stats.ttest_ind(high_work, low_work, equal_var=False)
    print(f"T-test: t = {t_stat_work:.4f}, p = {p_val_work:.4e}")

    # --- Stress Analysis ---
    print("\n--- Stress Effect (High vs Low) ---")
    stress_group = df_clean.groupby('stress_level')[target_col].agg(['mean', 'std', 'count'])
    print(stress_group)
    
    high_stress = df_clean[df_clean['stress_level'] == 'High'][target_col].dropna()
    low_stress = df_clean[df_clean['stress_level'] == 'Low'][target_col].dropna()
    
    t_stat_stress, p_val_stress = stats.ttest_ind(high_stress, low_stress, equal_var=False)
    print(f"T-test: t = {t_stat_stress:.4f}, p = {p_val_stress:.4f}")

if __name__ == "__main__":
    main()
