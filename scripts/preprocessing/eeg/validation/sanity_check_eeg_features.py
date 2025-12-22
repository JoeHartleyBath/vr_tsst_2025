"""
Sanity check for EEG features to validate data quality for machine learning.

This script examines:
1. Data completeness (missing values, valid ranges)
2. Signal quality indicators (power distributions, SNR estimates)
3. Physiological plausibility (frequency band relationships)
4. Condition differences (variance across experimental conditions)
5. Feature distributions and outliers
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns
from scipy import stats

# Configuration
PROJECT_ROOT = Path('c:/vr_tsst_2025')
FEATURES_FILE = PROJECT_ROOT / 'output' / 'aggregated' / 'eeg_features.csv'
OUTPUT_DIR = PROJECT_ROOT / 'output' / 'qc'
OUTPUT_DIR.mkdir(exist_ok=True)

print("=" * 80)
print("EEG FEATURES SANITY CHECK FOR MACHINE LEARNING")
print("=" * 80)
print()

# Load data
print(f"Loading features from: {FEATURES_FILE}")
df = pd.read_csv(FEATURES_FILE)
print(f"✓ Loaded {len(df)} rows, {len(df.columns)} columns\n")

# ============================================================================
# 1. DATA COMPLETENESS
# ============================================================================
print("=" * 80)
print("1. DATA COMPLETENESS")
print("=" * 80)

# Participants and conditions
n_participants = df['Participant'].nunique()
n_conditions = df['Condition'].nunique()
print(f"Participants: {n_participants}")
print(f"Conditions: {n_conditions}")
print(f"Expected rows: {n_participants * n_conditions}")
print(f"Actual rows: {len(df)}")
print()

# Missing values
missing_summary = df.isnull().sum()
missing_pct = (missing_summary / len(df)) * 100

# Filter out expected missing entropy features
entropy_cols = [col for col in df.columns if 'Entropy' in col]
other_missing = missing_pct[(missing_pct > 0) & (~missing_pct.index.isin(entropy_cols))]

if len(other_missing) > 0:
    print("⚠ WARNING: Missing values detected:")
    for col, pct in other_missing.items():
        print(f"  {col}: {pct:.1f}% missing")
else:
    print("✓ No missing values (entropy features excluded as expected)")
print()

# Infinite values
feature_cols = [col for col in df.columns if col not in ['Participant', 'Condition']]
inf_counts = {}
for col in feature_cols:
    inf_count = np.isinf(df[col]).sum()
    if inf_count > 0:
        inf_counts[col] = inf_count

if inf_counts:
    print("⚠ WARNING: Infinite values detected:")
    for col, count in inf_counts.items():
        print(f"  {col}: {count} infinite values")
else:
    print("✓ No infinite values")
print()

# ============================================================================
# 2. SIGNAL QUALITY INDICATORS
# ============================================================================
print("=" * 80)
print("2. SIGNAL QUALITY - POWER DISTRIBUTIONS")
print("=" * 80)

# Check band power columns
band_power_cols = [col for col in feature_cols if '_Power' in col and 'Ratio' not in col]
print(f"Found {len(band_power_cols)} band power features\n")

# Summary statistics for power features
if len(band_power_cols) > 0:
    power_stats = df[band_power_cols].describe()
    print("Power feature statistics (log10 scale):")
    print(power_stats.loc[['mean', 'std', 'min', 'max']].T.head(10))
    print(f"... ({len(band_power_cols)} total power features)")
    print()
    
    # Note about log scale
    print("NOTE: Power values are in log10 scale (dB).")
    print("Example: -4.0 corresponds to 10^-4 = 0.0001 μV²")
    print()
else:
    print("No power features found.\n")

# ============================================================================
# 3. PHYSIOLOGICAL PLAUSIBILITY
# ============================================================================
print("=" * 80)
print("3. PHYSIOLOGICAL PLAUSIBILITY")
print("=" * 80)

# Check for expected frequency band relationships per participant
# In awake, attentive states: alpha power typically > delta power
# Note: Since values are in log scale, we're comparing log(alpha) vs log(delta)
frontal_alpha_cols = [col for col in df.columns if 'Frontal' in col and 'Alpha_Power' in col and 'Low' not in col and 'High' not in col]
frontal_delta_cols = [col for col in df.columns if 'Frontal' in col and 'Delta_Power' in col]

if frontal_alpha_cols and frontal_delta_cols:
    alpha_col = frontal_alpha_cols[0]
    delta_col = frontal_delta_cols[0]
    
    print(f"Checking {alpha_col} vs {delta_col} per participant:\n")
    
    good_participants = []
    bad_participants = []
    
    for pid in sorted(df['Participant'].unique()):
        participant_data = df[df['Participant'] == pid]
        alpha_gt_delta = (participant_data[alpha_col] > participant_data[delta_col]).sum()
        total_obs = len(participant_data)
        alpha_gt_delta_pct = alpha_gt_delta / total_obs * 100
        
        alpha_mean = participant_data[alpha_col].mean()
        delta_mean = participant_data[delta_col].mean()
        
        status = "✓" if alpha_gt_delta_pct >= 50 else "⚠"
        print(f"  P{pid:02d}: {alpha_gt_delta_pct:5.1f}% alpha>delta  |  α={alpha_mean:.3f}, δ={delta_mean:.3f}  {status}")
        
        if alpha_gt_delta_pct >= 50:
            good_participants.append(pid)
        else:
            bad_participants.append(pid)
    
    print(f"\nSummary:")
    print(f"  Good participants (≥50% alpha>delta): {len(good_participants)} - {good_participants}")
    print(f"  Problematic participants (<50%): {len(bad_participants)} - {bad_participants}")
    
    if len(bad_participants) > len(good_participants):
        print("  ⚠ WARNING: Majority of participants show delta > alpha (unusual)")
    elif len(bad_participants) > 0:
        print(f"  ⚠ {len(bad_participants)} participant(s) may have data quality issues")
    else:
        print("  ✓ All participants show healthy alpha/delta relationships")
        
print()

# Check power ratios for reasonable values
ratio_cols = [col for col in feature_cols if 'Ratio' in col]
if ratio_cols:
    print("Power ratio statistics:")
    for col in ratio_cols:
        mean_val = df[col].mean()
        std_val = df[col].std()
        print(f"  {col}: {mean_val:.3f} ± {std_val:.3f}")

# Check expected power ordering: delta > theta > alpha > beta
print("\nExpected power hierarchy: delta > theta > alpha > beta")
frontal_bands = {}
for band in ['Delta', 'Theta', 'Alpha', 'Beta']:
    cols = [col for col in df.columns if f'FrontalLeft_{band}_Power' in col and 'Low' not in col and 'High' not in col]
    if cols:
        frontal_bands[band] = df[cols[0]].mean()

if len(frontal_bands) == 4:
    print(f"  Mean frontal power (log scale):")
    for band in ['Delta', 'Theta', 'Alpha', 'Beta']:
        print(f"    {band}: {frontal_bands[band]:.3f}")
    
    # Check hierarchy per participant
    correct_hierarchy_count = 0
    for pid in sorted(df['Participant'].unique()):
        participant_data = df[df['Participant'] == pid]
        p_bands = {}
        for band in ['Delta', 'Theta', 'Alpha', 'Beta']:
            cols = [col for col in df.columns if f'FrontalLeft_{band}_Power' in col and 'Low' not in col and 'High' not in col]
            if cols:
                p_bands[band] = participant_data[cols[0]].mean()
        
        if len(p_bands) == 4:
            hierarchy_ok = (p_bands['Delta'] > p_bands['Theta'] > 
                          p_bands['Alpha'] > p_bands['Beta'])
            marker = "✓" if hierarchy_ok else "✗"
            print(f"  P{pid:02d}: δ={p_bands['Delta']:.2f} θ={p_bands['Theta']:.2f} α={p_bands['Alpha']:.2f} β={p_bands['Beta']:.2f} {marker}")
            if hierarchy_ok:
                correct_hierarchy_count += 1
    
    if correct_hierarchy_count == 0:
        print(f"  ⚠ WARNING: No participants show expected power hierarchy")
    elif correct_hierarchy_count < len(df['Participant'].unique()) / 2:
        print(f"  ⚠ Only {correct_hierarchy_count}/{len(df['Participant'].unique())} participants show expected hierarchy")
    else:
        print(f"  ✓ {correct_hierarchy_count}/{len(df['Participant'].unique())} participants show expected hierarchy")

# Check posterior alpha > frontal alpha (spatial distribution)
print("\nExpected spatial distribution: Posterior alpha > Frontal alpha")
occipital_alpha_cols = [col for col in df.columns if 'Occipital_Alpha_Power' in col and 'Low' not in col and 'High' not in col]
frontal_alpha_cols = [col for col in df.columns if 'FrontalLeft_Alpha_Power' in col]

if occipital_alpha_cols and frontal_alpha_cols:
    occipital_col = occipital_alpha_cols[0]
    frontal_col = frontal_alpha_cols[0]
    
    occipital_mean = df[occipital_col].mean()
    frontal_mean = df[frontal_col].mean()
    
    print(f"  Mean alpha power (log scale):")
    print(f"    Occipital: {occipital_mean:.3f}")
    print(f"    Frontal:   {frontal_mean:.3f}")
    
    # Check per participant
    posterior_dominant_count = 0
    for pid in sorted(df['Participant'].unique()):
        participant_data = df[df['Participant'] == pid]
        occ_mean = participant_data[occipital_col].mean()
        fro_mean = participant_data[frontal_col].mean()
        
        posterior_dominant = occ_mean > fro_mean
        marker = "✓" if posterior_dominant else "✗"
        print(f"  P{pid:02d}: Occipital={occ_mean:.3f}, Frontal={fro_mean:.3f} {marker}")
        if posterior_dominant:
            posterior_dominant_count += 1
    
    if posterior_dominant_count == 0:
        print(f"  ⚠ WARNING: No participants show posterior alpha dominance")
        print(f"     This is highly unusual - alpha is typically strongest in occipital regions")
    elif posterior_dominant_count < len(df['Participant'].unique()) / 2:
        print(f"  ⚠ Only {posterior_dominant_count}/{len(df['Participant'].unique())} participants show posterior dominance")
    else:
        print(f"  ✓ {posterior_dominant_count}/{len(df['Participant'].unique())} participants show expected posterior dominance")

print()

# ============================================================================
# 4. CONDITION DIFFERENCES PER PARTICIPANT (Critical for ML)
# ============================================================================
print("=" * 80)
print("4. CONDITION DIFFERENCES PER PARTICIPANT (ML RELEVANCE)")
print("=" * 80)

print("Checking if features vary across conditions WITHIN each participant...\n")

# Select key task conditions to compare
task_conditions = [cond for cond in df['Condition'].unique() if 'Task' in cond]
print(f"Analyzing {len(task_conditions)} task conditions: {task_conditions[:3]}...\n")

if len(task_conditions) >= 2:
    # Select a few key features to test
    test_features_cond = []
    power_features = [col for col in df.columns if '_Power' in col and 'Alpha' in col and 'Low' not in col and 'High' not in col]
    test_features_cond.extend(power_features[:3])
    ratio_features = [col for col in df.columns if 'Ratio' in col]
    test_features_cond.extend(ratio_features[:2])
    
    if test_features_cond:
        print("Per-participant ANOVA for condition differences:")
        print("(p < 0.05 indicates significant within-person condition effects)\n")
        
        # Track results across participants
        participant_results = {feat: {'sig': 0, 'total': 0} for feat in test_features_cond}
        
        for feat in test_features_cond:
            if feat in df.columns:
                print(f"  {feat}:")
                
                for pid in sorted(df['Participant'].unique()):
                    participant_data = df[df['Participant'] == pid]
                    
                    # Get data for each task condition
                    condition_groups = []
                    for cond in task_conditions:
                        cond_data = participant_data[participant_data['Condition'] == cond][feat].dropna()
                        if len(cond_data) > 0:
                            condition_groups.append(cond_data)
                    
                    if len(condition_groups) >= 2:
                        try:
                            f_stat, p_val = stats.f_oneway(*condition_groups)
                            sig_marker = "✓" if p_val < 0.05 else "✗"
                            participant_results[feat]['total'] += 1
                            if p_val < 0.05:
                                participant_results[feat]['sig'] += 1
                            
                            print(f"    P{pid:02d}: F={f_stat:6.3f}, p={p_val:.4f} {sig_marker}")
                        except:
                            print(f"    P{pid:02d}: [insufficient data]")
                
                # Summary for this feature
                sig_count = participant_results[feat]['sig']
                total_count = participant_results[feat]['total']
                if total_count > 0:
                    sig_pct = (sig_count / total_count) * 100
                    print(f"    Summary: {sig_count}/{total_count} participants ({sig_pct:.0f}%) show condition effects\n")
        
        # Overall summary
        print("\n📊 Within-participant condition sensitivity:")
        overall_sig = sum(r['sig'] for r in participant_results.values())
        overall_total = sum(r['total'] for r in participant_results.values())
        
        if overall_sig == 0:
            print("  ⚠ WARNING: No participants show condition effects!")
            print("     Features may not capture task-related changes.")
        elif overall_sig < overall_total * 0.3:
            print(f"  ⚠ Only {overall_sig}/{overall_total} tests show within-person condition effects")
            print("     Features show weak sensitivity to experimental conditions.")
        else:
            print(f"  ✓ {overall_sig}/{overall_total} tests show within-person condition effects")
            print("     Features capture task-related changes in multiple participants.")

print()

# ============================================================================
# 5. PARTICIPANT DIFFERENCES (Between-subjects)
# ============================================================================
print("=" * 80)
print("5. PARTICIPANT-LEVEL ANALYSIS (BETWEEN-SUBJECTS)")
print("=" * 80)

# Group by participant and check variance
print("Checking if features vary across participants...\n")

# Select key features to test
test_features = []
power_features = [col for col in df.columns if '_Power' in col and 'Alpha' in col]
test_features.extend(power_features[:5])
ratio_features = [col for col in df.columns if 'Ratio' in col]
test_features.extend(ratio_features[:2])

if test_features:
    print("ANOVA test for participant differences (p < 0.05 indicates significant difference):")
    significant_features = 0
    
    for feat in test_features:
        if feat in df.columns:
            # Group by participant
            participant_groups = [df[df['Participant'] == p][feat].dropna() 
                                 for p in df['Participant'].unique()]
            if all(len(g) > 0 for g in participant_groups):
                f_stat, p_val = stats.f_oneway(*participant_groups)
                sig_marker = "✓" if p_val < 0.05 else "✗"
                print(f"  {sig_marker} {feat}: F={f_stat:.3f}, p={p_val:.4f}")
                if p_val < 0.05:
                    significant_features += 1
    
    if significant_features == 0:
        print("\n⚠ WARNING: No significant differences between participants!")
        print("  Features show no inter-individual variance.")
    else:
        print(f"\n✓ {significant_features}/{len(test_features)} features show participant differences")
    
    # Check variance within vs between participants
    print("\nVariance analysis (between vs within participants):")
    for feat in test_features[:5]:
        if feat in df.columns:
            # Within-participant variance (average across participants)
            within_var = df.groupby('Participant')[feat].var().mean()
            # Between-participant variance
            between_var = df.groupby('Participant')[feat].mean().var()
            variance_ratio = between_var / within_var if within_var > 0 else 0
            
            print(f"  {feat}:")
            print(f"    Between-participant variance: {between_var:.4f}")
            print(f"    Within-participant variance (avg): {within_var:.4f}")
            print(f"    Ratio: {variance_ratio:.4f} {'✓' if variance_ratio > 0.1 else '⚠'}")

# Per-participant summary statistics
print("\nPer-participant feature means (first 3 features):")
for feat in test_features[:3]:
    if feat in df.columns:
        print(f"\n  {feat}:")
        participant_means = df.groupby('Participant')[feat].mean()
        for pid, mean_val in participant_means.items():
            print(f"    P{pid:02d}: {mean_val:.4f}")

print()

# ============================================================================
# 6. FEATURE DISTRIBUTIONS AND OUTLIERS
# ============================================================================
print("=" * 80)
print("6. FEATURE DISTRIBUTIONS")
print("=" * 80)

# Check skewness and outliers
print("Checking feature distributions for ML suitability...\n")

highly_skewed = []
zero_variance = []
outlier_heavy = []

for col in feature_cols[:20]:  # Check first 20 features
    if col in df.columns:
        data = df[col].dropna()
        
        # Zero variance check
        if data.var() < 1e-10:
            zero_variance.append(col)
            continue
        
        # Skewness check
        skew = stats.skew(data)
        if abs(skew) > 3:
            highly_skewed.append(f"{col} (skew={skew:.2f})")
        
        # Outlier check (using IQR method)
        q1, q3 = data.quantile([0.25, 0.75])
        iqr = q3 - q1
        outliers = ((data < q1 - 3*iqr) | (data > q3 + 3*iqr)).sum()
        outlier_pct = outliers / len(data) * 100
        if outlier_pct > 10:
            outlier_heavy.append(f"{col} ({outlier_pct:.1f}%)")

if zero_variance:
    print(f"⚠ Zero/near-zero variance features ({len(zero_variance)}):")
    for col in zero_variance[:10]:
        print(f"  {col}")
else:
    print("✓ No zero-variance features")

if highly_skewed:
    print(f"\n⚠ Highly skewed features ({len(highly_skewed)}) - consider log transform:")
    for item in highly_skewed[:10]:
        print(f"  {item}")

if outlier_heavy:
    print(f"\n⚠ Features with >10% outliers ({len(outlier_heavy)}):")
    for item in outlier_heavy[:10]:
        print(f"  {item}")

print()

# ============================================================================
# 7. OVERALL ASSESSMENT
# ============================================================================
print("=" * 80)
print("7. OVERALL ASSESSMENT FOR MACHINE LEARNING")
print("=" * 80)

issues = []
warnings = []

# Count non-entropy missing values
non_entropy_missing = missing_summary[~missing_summary.index.isin(entropy_cols)]

# Critical issues
if non_entropy_missing.sum() > len(df) * 0.1:
    issues.append("High percentage of missing values (non-entropy)")
if inf_counts:
    issues.append("Infinite values present")
if len(zero_variance) > len(feature_cols) * 0.1:
    issues.append(f"{len(zero_variance)} features have zero variance")

# Warnings
if len(highly_skewed) > len(feature_cols) * 0.3:
    warnings.append(f"{len(highly_skewed)} features are highly skewed")
if len(outlier_heavy) > len(feature_cols) * 0.2:
    warnings.append(f"{len(outlier_heavy)} features have many outliers")

print("\n📊 Data Quality Summary:")
print(f"  Total features: {len(feature_cols)}")
print(f"  Complete cases: {df.dropna().shape[0]} / {len(df)} ({df.dropna().shape[0]/len(df)*100:.1f}%)")
print(f"  Participants: {n_participants}")
print(f"  Conditions: {n_conditions}")

if issues:
    print("\n❌ CRITICAL ISSUES:")
    for issue in issues:
        print(f"  • {issue}")
    print("\n⚠ Data quality is POOR - address critical issues before ML modeling")
elif warnings:
    print("\n⚠ WARNINGS:")
    for warning in warnings:
        print(f"  • {warning}")
    print("\n✓ Data quality is ACCEPTABLE - consider preprocessing steps")
else:
    print("\n✓ NO MAJOR ISSUES DETECTED")
    print("✓ Data quality is GOOD for ML modeling")

print("\n" + "=" * 80)
print("Sanity check complete!")
print("=" * 80)

# Generate a summary report
report_file = OUTPUT_DIR / 'eeg_features_sanity_check.txt'
with open(report_file, 'w') as f:
    f.write("EEG FEATURES SANITY CHECK REPORT\n")
    f.write("=" * 80 + "\n\n")
    f.write(f"Date: {pd.Timestamp.now()}\n")
    f.write(f"Features file: {FEATURES_FILE}\n")
    f.write(f"Total rows: {len(df)}\n")
    f.write(f"Total features: {len(feature_cols)}\n")
    f.write(f"Participants: {n_participants}\n")
    f.write(f"Conditions: {n_conditions}\n\n")
    
    f.write("ISSUES:\n")
    if issues:
        for issue in issues:
            f.write(f"  • {issue}\n")
    else:
        f.write("  None\n")
    
    f.write("\nWARNINGS:\n")
    if warnings:
        for warning in warnings:
            f.write(f"  • {warning}\n")
    else:
        f.write("  None\n")
    
    f.write(f"\nOVERALL: {'POOR' if issues else 'ACCEPTABLE' if warnings else 'GOOD'}\n")

print(f"\n📝 Report saved to: {report_file}")
