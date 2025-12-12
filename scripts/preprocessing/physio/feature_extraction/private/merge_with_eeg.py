"""
Merge Module - Combine EEG, Physio, and Subjective Data

Merges all features into final aggregated dataset following the legacy notebook structure.
Handles duplicate columns and ensures R preprocessing compatibility.

Author: VR-TSST Project
Date: December 2025
"""

import logging
import pandas as pd
from typing import Optional


def merge_physio_with_eeg(
    physio_features: pd.DataFrame,
    eeg_data: pd.DataFrame,
    subjective_data: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """
    Merge physiological features with EEG and subjective data.
    
    Merge strategy:
    1. Merge EEG + Physio on (Participant_ID, Condition)
    2. Add subjective ratings on (Participant_ID, Condition)
    3. Clean up duplicate columns (keep _x, drop _y)
    4. Remove _x suffix for clarity
    
    Parameters
    ----------
    physio_features : DataFrame
        Physiological features per condition (from extract_all_features)
    eeg_data : DataFrame
        EEG features with Participant_ID, Condition, Sample_Frame
    subjective_data : DataFrame, optional
        Subjective ratings (Stress, Workload, etc.)
        
    Returns
    -------
    DataFrame
        Final merged dataset ready for R preprocessing
    """
    logging.info("Merging EEG + Physio + Subjective data...")
    
    # 1) Ensure consistent types
    eeg_data['Participant_ID'] = eeg_data['Participant_ID'].astype(int)
    physio_features['Participant_ID'] = physio_features['Participant_ID'].astype(int)
    
    # 2) Merge EEG with physio features
    logging.info(f"  Merging EEG ({len(eeg_data)} rows) with physio ({len(physio_features)} rows)...")
    
    merged = eeg_data.merge(
        physio_features,
        on=['Participant_ID', 'Condition'],
        how='left'
    )
    
    logging.info(f"  After EEG+Physio merge: {len(merged)} rows, {len(merged.columns)} columns")
    
    # 3) Merge with subjective data if available
    if subjective_data is not None and not subjective_data.empty:
        logging.info(f"  Merging with subjective data ({len(subjective_data)} rows)...")
        
        # Ensure consistent types
        subjective_data['Participant_ID'] = subjective_data['Participant_ID'].astype(str).str.strip()
        merged['Participant_ID_str'] = merged['Participant_ID'].astype(str).str.strip()
        
        merged = merged.merge(
            subjective_data,
            left_on=['Participant_ID_str', 'Condition'],
            right_on=['Participant_ID', 'Condition'],
            how='left',
            suffixes=('', '_subj')
        )
        
        # Clean up temporary columns
        if 'Participant_ID_str' in merged.columns:
            merged.drop(columns=['Participant_ID_str'], inplace=True)
        if 'Participant_ID_subj' in merged.columns:
            merged.drop(columns=['Participant_ID_subj'], inplace=True)
        
        logging.info(f"  After subjective merge: {len(merged)} rows, {len(merged.columns)} columns")
    else:
        logging.warning("  No subjective data provided - skipping subjective merge")
    
    # 4) Handle duplicate columns (from merge conflicts)
    # Keep _x versions (from EEG), drop _y versions (from physio if conflicting)
    cols_to_drop = [col for col in merged.columns if col.endswith('_y')]
    if cols_to_drop:
        logging.info(f"  Dropping {len(cols_to_drop)} duplicate _y columns")
        merged.drop(columns=cols_to_drop, inplace=True)
    
    # Remove _x suffix for clarity
    merged.columns = [col.replace('_x', '') for col in merged.columns]
    
    # 5) Clean up time columns
    # If Window_Start_Second exists, that's our time reference
    if 'Window_Start_Second' in merged.columns:
        # Rename for clarity if needed
        pass  # Keep as is
    
    logging.info(f"✅ Merge completed: {len(merged)} rows, {len(merged.columns)} columns")
    logging.info(f"  Participants: {merged['Participant_ID'].nunique()}")
    logging.info(f"  Conditions: {merged['Condition'].nunique()}")
    
    return merged


def validate_merged_data(
    merged_data: pd.DataFrame,
    expected_participants: int = 48,
    expected_conditions: int = 12
) -> dict:
    """
    Validate the merged dataset structure and completeness.
    
    Parameters
    ----------
    merged_data : DataFrame
        Final merged dataset
    expected_participants : int
        Expected number of participants
    expected_conditions : int
        Expected number of conditions per participant
        
    Returns
    -------
    dict
        Validation results with flags and messages
    """
    validation = {
        'valid': True,
        'warnings': [],
        'errors': []
    }
    
    # Check participant count
    n_participants = merged_data['Participant_ID'].nunique()
    if n_participants < expected_participants:
        validation['warnings'].append(
            f"Only {n_participants}/{expected_participants} participants present"
        )
    
    # Check for missing conditions per participant
    cond_per_participant = merged_data.groupby('Participant_ID')['Condition'].nunique()
    incomplete = cond_per_participant[cond_per_participant < expected_conditions]
    
    if len(incomplete) > 0:
        validation['warnings'].append(
            f"{len(incomplete)} participants have <{expected_conditions} conditions"
        )
    
    # Check for required columns
    required_cols = ['Participant_ID', 'Condition', 'Sample_Frame']
    missing_cols = [col for col in required_cols if col not in merged_data.columns]
    
    if missing_cols:
        validation['errors'].append(f"Missing required columns: {missing_cols}")
        validation['valid'] = False
    
    # Check for excessive NaNs in key features
    key_features = [
        'Full_RMSSD',
        'Full_Polar_HeartRate_BPM_CLEANED_ABS_Mean',
        'Full_Shimmer_D36A_GSR_Skin_Conductance_uS_CLEANED_ABS_CLEANED_NK_Mean'
    ]
    
    for feat in key_features:
        if feat in merged_data.columns:
            nan_pct = merged_data[feat].isna().mean() * 100
            if nan_pct > 50:
                validation['warnings'].append(
                    f"Feature '{feat}' has {nan_pct:.1f}% NaN values"
                )
    
    # Log results
    if validation['valid']:
        logging.info("✅ Merged data validation passed")
    else:
        logging.error("❌ Merged data validation failed")
    
    for warning in validation['warnings']:
        logging.warning(f"  ⚠️  {warning}")
    
    for error in validation['errors']:
        logging.error(f"  ❌ {error}")
    
    return validation
