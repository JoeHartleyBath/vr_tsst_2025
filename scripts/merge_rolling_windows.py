"""
Merge EEG and Physiological Rolling Window Features

Merges EEG and physio features that were extracted using rolling windows.
Match on: Participant_ID, Condition, Window_Start

Author: VR-TSST Project
Date: December 2025
"""

import pandas as pd
import numpy as np
import logging
import sys
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def merge_rolling_window_features(
    eeg_file: str = 'output/aggregated/eeg_features_rolling_windows.csv',
    physio_file: str = 'output/aggregated/physio_features_rolling_windows.csv',
    output_file: str = 'output/aggregated/multimodal_features_rolling_windows.csv',
    time_tolerance: float = 0.5
) -> pd.DataFrame:
    """
    Merge EEG and physiological rolling window features.
    
    Parameters
    ----------
    eeg_file : str
        Path to EEG rolling window features
    physio_file : str
        Path to physio rolling window features
    output_file : str
        Path to save merged features
    time_tolerance : float
        Maximum time difference (seconds) for matching windows
        
    Returns
    -------
    DataFrame
        Merged multimodal features
    """
    
    logging.info("="*70)
    logging.info("MERGE ROLLING WINDOW FEATURES")
    logging.info("="*70)
    
    # Load data
    logging.info(f"Loading EEG features from: {eeg_file}")
    eeg_df = pd.read_csv(eeg_file)
    logging.info(f"  Loaded: {eeg_df.shape[0]} rows, {eeg_df.shape[1]} columns")
    logging.info(f"  Participants: {eeg_df['Participant_ID'].nunique()}")
    logging.info(f"  Conditions: {eeg_df['Condition'].nunique()}")
    
    logging.info(f"\nLoading Physio features from: {physio_file}")
    physio_df = pd.read_csv(physio_file)
    logging.info(f"  Loaded: {physio_df.shape[0]} rows, {physio_df.shape[1]} columns")
    logging.info(f"  Participants: {physio_df['Participant_ID'].nunique()}")
    logging.info(f"  Conditions: {physio_df['Condition'].nunique()}")
    
    # Ensure consistent types
    eeg_df['Participant_ID'] = eeg_df['Participant_ID'].astype(int)
    physio_df['Participant_ID'] = physio_df['Participant_ID'].astype(int)
    
    # Check for common merge keys
    logging.info("\nChecking merge keys...")
    if 'Window_Start' in eeg_df.columns and 'Window_Start' in physio_df.columns:
        merge_keys = ['Participant_ID', 'Condition', 'Window_Start']
        logging.info(f"  Using exact merge on: {merge_keys}")
        
        # Merge on exact window times
        merged = eeg_df.merge(
            physio_df,
            on=merge_keys,
            how='inner',
            suffixes=('_eeg', '_physio')
        )
        
    elif 'Window_Index' in eeg_df.columns and 'Window_Index' in physio_df.columns:
        merge_keys = ['Participant_ID', 'Condition', 'Window_Index']
        logging.info(f"  Using exact merge on: {merge_keys}")
        
        # Merge on window index
        merged = eeg_df.merge(
            physio_df,
            on=merge_keys,
            how='inner',
            suffixes=('_eeg', '_physio')
        )
    else:
        logging.error("No suitable merge keys found!")
        logging.error("Need either Window_Start or Window_Index in both datasets")
        sys.exit(1)
    
    logging.info(f"\nMerge results:")
    logging.info(f"  Merged rows: {len(merged)}")
    logging.info(f"  Merged columns: {len(merged.columns)}")
    logging.info(f"  EEG rows lost: {len(eeg_df) - len(merged)} ({(len(eeg_df) - len(merged))/len(eeg_df)*100:.1f}%)")
    logging.info(f"  Physio rows lost: {len(physio_df) - len(merged)} ({(len(physio_df) - len(merged))/len(physio_df)*100:.1f}%)")
    
    # Handle duplicate columns
    logging.info("\nHandling duplicate columns...")
    
    # Window timing columns - keep from EEG (more precise)
    time_cols = ['Window_Start', 'Window_End', 'Window_Index']
    for col in time_cols:
        if f'{col}_eeg' in merged.columns:
            merged[col] = merged[f'{col}_eeg']
            merged.drop(columns=[f'{col}_eeg', f'{col}_physio'], inplace=True, errors='ignore')
    
    # For other duplicates, prefer EEG version
    dup_suffixes = [col for col in merged.columns if col.endswith('_eeg') or col.endswith('_physio')]
    if dup_suffixes:
        logging.info(f"  Found {len(dup_suffixes)} duplicate columns")
        
        # Get unique base names
        base_names = set([col.rsplit('_', 1)[0] for col in dup_suffixes])
        
        for base in base_names:
            eeg_col = f'{base}_eeg'
            physio_col = f'{base}_physio'
            
            if eeg_col in merged.columns and physio_col in merged.columns:
                # Keep EEG version if available, otherwise physio
                merged[base] = merged[eeg_col].fillna(merged[physio_col])
                merged.drop(columns=[eeg_col, physio_col], inplace=True)
    
    # Identify feature columns
    meta_cols = ['Participant_ID', 'Condition', 'Window_Start', 'Window_End', 'Window_Index']
    feature_cols = [col for col in merged.columns if col not in meta_cols]
    
    logging.info(f"\nFeature summary:")
    logging.info(f"  Total features: {len(feature_cols)}")
    
    # Count EEG vs physio features
    eeg_features = [col for col in feature_cols if any(x in col for x in 
                    ['Theta', 'Alpha', 'Beta', 'Gamma', 'Delta', 'Frontal', 'Parietal', 'Temporal', 'Occipital'])]
    physio_features = [col for col in feature_cols if any(x in col for x in 
                       ['HR', 'HRV', 'GSR', 'Pupil', 'Blink', 'Shimmer', 'Polar'])]
    
    logging.info(f"  EEG features: {len(eeg_features)}")
    logging.info(f"  Physio features: {len(physio_features)}")
    logging.info(f"  Other features: {len(feature_cols) - len(eeg_features) - len(physio_features)}")
    
    # Check for missing values
    nan_counts = merged[feature_cols].isna().sum()
    features_with_nans = (nan_counts > 0).sum()
    
    if features_with_nans > 0:
        logging.warning(f"  {features_with_nans} features have NaN values")
        top_nan = nan_counts[nan_counts > 0].sort_values(ascending=False).head(5)
        for feat, count in top_nan.items():
            logging.warning(f"    {feat}: {count} ({count/len(merged)*100:.1f}%)")
    
    # Save merged data
    logging.info(f"\nSaving merged features to: {output_file}")
    merged.to_csv(output_file, index=False)
    logging.info(f"  File size: {Path(output_file).stat().st_size / 1024 / 1024:.2f} MB")
    
    # Final summary
    logging.info("\n" + "="*70)
    logging.info("MERGE COMPLETE")
    logging.info("="*70)
    logging.info(f"Output: {output_file}")
    logging.info(f"Rows: {len(merged)}")
    logging.info(f"Columns: {len(merged.columns)}")
    logging.info(f"Participants: {merged['Participant_ID'].nunique()}")
    logging.info(f"Conditions: {merged['Condition'].nunique()}")
    logging.info(f"Total features: {len(feature_cols)}")
    logging.info("="*70)
    
    return merged


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Merge EEG and physio rolling window features")
    parser.add_argument('--eeg', type=str, default='output/aggregated/eeg_features_rolling_windows.csv',
                       help='Path to EEG features')
    parser.add_argument('--physio', type=str, default='output/aggregated/physio_features_rolling_windows.csv',
                       help='Path to physio features')
    parser.add_argument('--output', type=str, default='output/aggregated/multimodal_features_rolling_windows.csv',
                       help='Output path')
    
    args = parser.parse_args()
    
    result = merge_rolling_window_features(
        eeg_file=args.eeg,
        physio_file=args.physio,
        output_file=args.output
    )
