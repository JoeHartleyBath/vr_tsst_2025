#!/usr/bin/env python3
"""
MVP Pipeline: Merge EEG + Subjective + Placeholder Physio

Creates all_data_aggregated.csv with minimal viable structure:
- 127 EEG features (from MATLAB extraction)
- 2 subjective ratings (Stress, Workload)  
- 24 placeholder physio features (5 HR + 13 GSR + 6 Pupil)

Total: ~153 columns needed for R preprocessing script.

This MVP validates the structure works end-to-end before building
full cleaning/extraction modules.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).parent.parent.parent

def load_eeg_features():
    """Load extracted EEG features from MATLAB pipeline"""
    eeg_path = PROJECT_ROOT / "output" / "aggregated" / "eeg_features.csv"
    
    if not eeg_path.exists():
        raise FileNotFoundError(f"EEG features not found: {eeg_path}")
    
    df = pd.read_csv(eeg_path)
    
    # Rename Participant → Participant_ID for consistency
    df = df.rename(columns={'Participant': 'Participant_ID'})
    
    # Prefix all non-metadata columns with Full_ (R pipeline expects this)
    metadata_cols = ['Participant_ID', 'Condition']
    feature_cols = [c for c in df.columns if c not in metadata_cols]
    rename_map = {c: f'Full_{c}' for c in feature_cols}
    df = df.rename(columns=rename_map)
    
    print(f"✓ EEG features: {len(df)} rows, {len(feature_cols)} features")
    print(f"  Participants: {sorted(df['Participant_ID'].unique())}")
    print(f"  Conditions: {df['Condition'].nunique()} unique")
    
    return df

def create_placeholder_subjective(eeg_df):
    """
    Create placeholder subjective ratings
    
    TODO: Replace with real subjective data from:
    - data/raw/metadata/P*.csv (parsed for ratings)
    - OR separate subjective ratings CSV
    """
    df = eeg_df[['Participant_ID', 'Condition']].copy()
    
    # Placeholder NaN values (will be filled with real data)
    df['Stress'] = np.nan
    df['Workload'] = np.nan
    
    print(f"✓ Subjective ratings: {len(df)} rows (placeholders)")
    print(f"  TODO: Extract real ratings from metadata")
    
    return df

def create_placeholder_physio(eeg_df):
    """
    Create placeholder physio features matching R pipeline requirements
    
    Based on required_features_analysis.md:
    - 5 HR features (keep Mean/Median/MIN/MAX + RMSSD)
    - 13 GSR/EDA features (keep EDA-processed versions)  
    - 6 Pupil features (bilateral computed features)
    
    All feature names must start with 'Full_' prefix.
    """
    df = eeg_df[['Participant_ID', 'Condition']].copy()
    
    # === HR Features (5) ===
    hr_features = [
        'Full_Polar_HeartRate_BPM_Mean',
        'Full_Polar_HeartRate_BPM_Median',
        'Full_Polar_HeartRate_BPM_MIN',
        'Full_Polar_HeartRate_BPM_MAX',
        'Full_RMSSD',  # Canonical feature
    ]
    
    # === GSR/EDA Features (13) ===
    gsr_features = [
        # Raw GSR statistics
        'Full_Shimmer_D36A_GSR_Skin_Conductance_uS_CLEANED_ABS_CLEANED_NK_Mean',
        'Full_Shimmer_D36A_GSR_Skin_Conductance_uS_CLEANED_ABS_CLEANED_NK_Median',
        'Full_Shimmer_D36A_GSR_Skin_Conductance_uS_CLEANED_ABS_CLEANED_NK_SD',
        'Full_Shimmer_D36A_GSR_Skin_Conductance_uS_CLEANED_ABS_CLEANED_NK_MIN',
        'Full_Shimmer_D36A_GSR_Skin_Conductance_uS_CLEANED_ABS_CLEANED_NK_MAX',
        # EDA decomposition (tonic/phasic)
        'Full_Shimmer_D36A_GSR_Skin_Conductance_uS_CLEANED_ABS_CLEANED_NK_EDA_Tonic_Mean',  # Canonical
        'Full_Shimmer_D36A_GSR_Skin_Conductance_uS_CLEANED_ABS_CLEANED_NK_EDA_Tonic_SD',
        # SCR features
        'Full_Shimmer_D36A_GSR_Skin_Conductance_uS_CLEANED_ABS_CLEANED_NK_EDA_PeakRate',
        'Full_Shimmer_D36A_GSR_Skin_Conductance_uS_CLEANED_ABS_CLEANED_NK_EDA_PeakHeight_Mean',  # Canonical
        'Full_Shimmer_D36A_GSR_Skin_Conductance_uS_CLEANED_ABS_CLEANED_NK_EDA_PeakHeight_Max',
        'Full_Shimmer_D36A_GSR_Skin_Conductance_uS_CLEANED_ABS_CLEANED_NK_EDA_PeakHeight_Median',
        'Full_Shimmer_D36A_GSR_Skin_Conductance_uS_CLEANED_ABS_CLEANED_NK_EDA_PeakArea',
        'Full_Shimmer_D36A_GSR_Skin_Conductance_uS_CLEANED_ABS_CLEANED_NK_EDA_TotalSCRs',
    ]
    
    # === Pupil Features (6 - bilateral averages) ===
    # Note: R pipeline computes these from left/right, but we can provide pre-computed
    pupil_features = [
        'Full_Pupil_Dilation_Mean',
        'Full_Pupil_Dilation_Min',
        'Full_Pupil_Dilation_Max',
        'Full_Pupil_Dilation_Median',
        'Full_Pupil_Dilation_SD',
        'Full_Pupil_Asymmetry',
    ]
    
    # Also need the individual left/right features for R pipeline to compute bilaterals
    pupil_lr_features = [
        'Full_Foveal_Corrected_Dilation_Left_CLEANED_ABS_Mean',
        'Full_Foveal_Corrected_Dilation_Right_CLEANED_ABS_Mean',
        'Full_Foveal_Corrected_Dilation_Left_CLEANED_ABS_MIN',
        'Full_Foveal_Corrected_Dilation_Right_CLEANED_ABS_MIN',
        'Full_Foveal_Corrected_Dilation_Left_CLEANED_ABS_MAX',
        'Full_Foveal_Corrected_Dilation_Right_CLEANED_ABS_MAX',
        'Full_Foveal_Corrected_Dilation_Left_CLEANED_ABS_Median',
        'Full_Foveal_Corrected_Dilation_Right_CLEANED_ABS_Median',
        'Full_Foveal_Corrected_Dilation_Left_CLEANED_ABS_SD',
        'Full_Foveal_Corrected_Dilation_Right_CLEANED_ABS_SD',
    ]
    
    # Initialize all features with NaN
    all_physio_features = hr_features + gsr_features + pupil_features + pupil_lr_features
    for feat in all_physio_features:
        df[feat] = np.nan
    
    print(f"✓ Physio features: {len(df)} rows, {len(all_physio_features)} features (placeholders)")
    print(f"  - HR: {len(hr_features)} features")
    print(f"  - GSR/EDA: {len(gsr_features)} features")
    print(f"  - Pupil: {len(pupil_features)} bilateral + {len(pupil_lr_features)} L/R")
    print(f"  TODO: Replace with real cleaned/extracted data")
    
    return df

def merge_all(eeg, subjective, physio):
    """Merge EEG, subjective, and physio into single dataframe"""
    # Merge on Participant_ID and Condition
    merged = eeg.merge(subjective, on=['Participant_ID', 'Condition'], how='left')
    merged = merged.merge(physio, on=['Participant_ID', 'Condition'], how='left', suffixes=('', '_dup'))
    
    # Drop any duplicate columns from merge
    dup_cols = [c for c in merged.columns if c.endswith('_dup')]
    if dup_cols:
        merged = merged.drop(columns=dup_cols)
    
    # Add Round column (placeholder - will come from counterbalance mapping)
    merged['Round'] = 0
    
    # Reorder columns: metadata first, then features
    metadata_cols = ['Participant_ID', 'Condition', 'Round', 'Stress', 'Workload']
    feature_cols = [c for c in merged.columns if c not in metadata_cols]
    merged = merged[metadata_cols + sorted(feature_cols)]
    
    print(f"\n✓ Merged dataset: {len(merged)} rows × {len(merged.columns)} columns")
    print(f"  - Metadata: {len(metadata_cols)} columns")
    print(f"  - Features: {len(feature_cols)} columns")
    
    return merged

def save_output(df, filename="all_data_aggregated_mvp.csv"):
    """Save merged data to output directory"""
    output_dir = PROJECT_ROOT / "output" / "aggregated"
    output_dir.mkdir(exist_ok=True, parents=True)
    
    output_path = output_dir / filename
    df.to_csv(output_path, index=False)
    
    print(f"\n✓ Saved to: {output_path}")
    print(f"  Shape: {df.shape}")
    print(f"\nColumn summary:")
    print(f"  Participant_ID: {df['Participant_ID'].nunique()} participants")
    print(f"  Condition: {df['Condition'].nunique()} conditions")
    print(f"  Features: {len([c for c in df.columns if c.startswith('Full_')])} with 'Full_' prefix")
    
    return output_path

def main():
    print("="*70)
    print("MVP PIPELINE: Merge EEG + Subjective + Physio (Placeholders)")
    print("="*70)
    print()
    
    try:
        # Step 1: Load EEG features
        print("[1/4] Loading EEG features...")
        eeg = load_eeg_features()
        print()
        
        # Step 2: Create placeholder subjective ratings
        print("[2/4] Creating placeholder subjective ratings...")
        subjective = create_placeholder_subjective(eeg)
        print()
        
        # Step 3: Create placeholder physio features
        print("[3/4] Creating placeholder physio features...")
        physio = create_placeholder_physio(eeg)
        print()
        
        # Step 4: Merge and save
        print("[4/4] Merging all data...")
        merged = merge_all(eeg, subjective, physio)
        output_path = save_output(merged)
        
        print("\n" + "="*70)
        print("✓ MVP PIPELINE COMPLETE")
        print("="*70)
        print(f"\nNext steps:")
        print(f"  1. Test with R: source('scripts/preproccess_for_xgb.R')")
        print(f"  2. Verify baseline adjustment works")
        print(f"  3. Build real physio cleaning modules")
        print(f"  4. Replace placeholders with real features")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
