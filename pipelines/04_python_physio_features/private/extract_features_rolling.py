"""
Rolling Window Feature Extraction

Implements windowing logic for temporal physiological feature extraction.
Reuses core feature computation functions from extract_features.py.

Window Parameters:
    - Default: 10s windows with 50% overlap (5s stride)
    - Configurable via command-line arguments
    - Outputs timestamp metadata for multimodal alignment

Author: VR-TSST Project
Date: December 2025
"""

import logging
import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from tqdm import tqdm

# Import core feature extraction functions (reuse existing code)
from extract_features import (
    calculate_stats,
    extract_hrv_features,
    extract_gsr_features,
    extract_pupil_features,
    extract_blink_features,
    extract_response_features
)


def create_rolling_windows(
    data: pd.DataFrame,
    time_column: str,
    window_size: float,
    overlap: float
) -> List[Dict]:
    """
    Create rolling windows from time-series data.
    
    Args:
        data: DataFrame with time-series physiological data
        time_column: Name of time column (seconds)
        window_size: Window duration in seconds
        overlap: Overlap fraction (0-1), e.g., 0.5 = 50% overlap
        
    Returns:
        List of dictionaries with window metadata and data slices
    """
    if time_column not in data.columns:
        logging.error(f"Time column '{time_column}' not found in data")
        return []
    
    # Calculate stride (step size between windows)
    stride = window_size * (1 - overlap)
    
    # Get time range
    time_data = data[time_column].values
    min_time = np.min(time_data)
    max_time = np.max(time_data)
    duration = max_time - min_time
    
    # Generate window start times
    window_starts = np.arange(min_time, max_time - window_size + stride, stride)
    
    windows = []
    for idx, window_start in enumerate(window_starts):
        window_end = window_start + window_size
        
        # Extract data within window
        mask = (time_data >= window_start) & (time_data < window_end)
        window_data = data[mask].copy()
        
        # Skip windows with insufficient data
        if len(window_data) < 5:  # Minimum 5 samples per window
            continue
        
        windows.append({
            'window_index': idx,
            'window_start': window_start,
            'window_end': window_end,
            'window_duration': window_end - window_start,
            'n_samples': len(window_data),
            'data': window_data
        })
    
    return windows


def extract_features_from_window(
    window_data: pd.DataFrame,
    gsr_data: pd.DataFrame,
    participant_id: int,
    condition: str,
    window_metadata: Dict
) -> Dict:
    """
    Extract features from a single window using existing feature functions.
    
    Args:
        window_data: Cleaned physio data for this window
        gsr_data: GSR-specific data for this window
        participant_id: Participant ID
        condition: Condition name
        window_metadata: Dict with window_index, window_start, window_end, etc.
        
    Returns:
        Dictionary of features for this window
    """
    features = {
        'Participant_ID': participant_id,
        'Condition': condition,
        'Window_Index': window_metadata['window_index'],
        'Window_Start': window_metadata['window_start'],
        'Window_End': window_metadata['window_end']
    }
    
    try:
        # HR/HRV features (reuse existing functions)
        hr_col = 'Shimmer_D36A_Internal_ADC_13_CLEANED_HR'
        if hr_col in window_data.columns:
            hr_stats = calculate_stats(
                window_data, 
                columns=[hr_col],
                prefix='HR'
            )
            features.update(hr_stats)
            
            # HRV features (if RR intervals available)
            rr_col = 'Shimmer_D36A_Internal_ADC_13_CLEANED_RR'
            if rr_col in window_data.columns:
                rr_intervals = window_data[rr_col].dropna()
                if len(rr_intervals) >= 5:  # Need multiple beats for HRV
                    hrv_features = extract_hrv_features(rr_intervals)
                    features.update(hrv_features.iloc[0].to_dict())
        
        # GSR features (reuse existing function)
        gsr_col = 'Shimmer_D36A_GSR_Skin_Conductance_uS_CLEANED_ABS_CLEANED_NK'
        if gsr_col in window_data.columns:
            gsr_stats = calculate_stats(
                window_data,
                columns=[gsr_col],
                prefix='GSR'
            )
            features.update(gsr_stats)
            
            # Detailed GSR features from neurokit
            gsr_signal = window_data[gsr_col].dropna().values
            if len(gsr_signal) > 10:
                gsr_detailed = extract_gsr_features(
                    signal_data=gsr_signal,
                    sampling_rate=10  # GSR sampled at 10 Hz
                )
                features.update(gsr_detailed)
        
        # Pupil features (reuse existing function)
        pupil_cols = [
            'PupilLabs_Gaze_Pupil_diameter_left',
            'PupilLabs_Gaze_Pupil_diameter_right'
        ]
        
        available_pupil_cols = [col for col in pupil_cols if col in window_data.columns]
        if available_pupil_cols:
            pupil_features = extract_pupil_features(
                window_data[available_pupil_cols + ['PupilLabs_Gaze_Pupil_confidence_left', 
                                                      'PupilLabs_Gaze_Pupil_confidence_right']]
                if all(c in window_data.columns for c in ['PupilLabs_Gaze_Pupil_confidence_left', 
                                                            'PupilLabs_Gaze_Pupil_confidence_right'])
                else window_data[available_pupil_cols]
            )
            features.update(pupil_features)
        
        # Blink features (reuse existing function)
        blink_col = 'Shimmer_D36A_Blinking'
        if blink_col in window_data.columns:
            blink_features = extract_blink_features(
                window_data[blink_col],
                sampling_rate=90  # Eye tracking at 90 Hz
            )
            features.update(blink_features)
        
        # Response features (button presses, accuracy)
        response_cols = ['Unity_VR_Button', 'Unity_VR_Accuracy']
        if all(col in window_data.columns for col in response_cols):
            response_features = extract_response_features(
                window_data[response_cols]
            )
            features.update(response_features)
            
    except Exception as e:
        logging.warning(f"Feature extraction error for P{participant_id} {condition} "
                       f"window {window_metadata['window_index']}: {e}")
    
    return features


def extract_rolling_window_features(
    phys_data: pd.DataFrame,
    gsr_data: pd.DataFrame,
    participants: List[int],
    window_size: float = 10.0,
    overlap: float = 0.5,
    parallel: bool = False
) -> pd.DataFrame:
    """
    Extract features from rolling windows for all participants and conditions.
    
    Args:
        phys_data: Cleaned physiological data
        gsr_data: GSR-specific cleaned data
        participants: List of participant IDs to process
        window_size: Window duration in seconds (default: 10s)
        overlap: Window overlap fraction (default: 0.5 = 50%)
        parallel: Enable parallel processing (not implemented yet)
        
    Returns:
        DataFrame with features for all windows (rows: windows, cols: features)
    """
    all_features = []
    
    logging.info(f"Processing {len(participants)} participants with rolling windows...")
    
    # Process each participant
    for participant_id in tqdm(participants, desc="Participants"):
        
        # Filter data for this participant
        p_data = phys_data[phys_data['Participant_ID'] == participant_id].copy()
        p_gsr = gsr_data[gsr_data['Participant_ID'] == participant_id].copy()
        
        if len(p_data) == 0:
            logging.warning(f"No data for participant {participant_id}")
            continue
        
        # Get unique conditions for this participant
        conditions = p_data['Condition'].dropna().unique()
        
        # Process each condition
        for condition in conditions:
            
            # Filter data for this condition
            cond_data = p_data[p_data['Condition'] == condition].copy()
            cond_gsr = p_gsr[p_gsr['Condition'] == condition].copy()
            
            if len(cond_data) == 0:
                continue
            
            # Create rolling windows
            time_col = 'Adjusted_Time'  # Relative time within condition
            if time_col not in cond_data.columns:
                logging.warning(f"Time column '{time_col}' not found for P{participant_id} {condition}")
                continue
            
            windows = create_rolling_windows(
                cond_data,
                time_column=time_col,
                window_size=window_size,
                overlap=overlap
            )
            
            # Extract features from each window
            for window_meta in windows:
                window_features = extract_features_from_window(
                    window_data=window_meta['data'],
                    gsr_data=cond_gsr,  # Pass full condition GSR data for context
                    participant_id=participant_id,
                    condition=condition,
                    window_metadata=window_meta
                )
                all_features.append(window_features)
    
    # Convert to DataFrame
    features_df = pd.DataFrame(all_features)
    
    logging.info(f"Extracted features from {len(features_df)} total windows")
    logging.info(f"Features per window: {len(features_df.columns) - 5}")  # Exclude metadata
    
    return features_df


def align_with_eeg_windows(
    physio_windows: pd.DataFrame,
    eeg_windows: pd.DataFrame,
    time_tolerance: float = 0.1
) -> pd.DataFrame:
    """
    Align physiological windows with EEG windows based on timestamps.
    
    Args:
        physio_windows: DataFrame with physio features and Window_Start/Window_End
        eeg_windows: DataFrame with EEG features and window timestamps
        time_tolerance: Maximum time difference for alignment (seconds)
        
    Returns:
        Merged DataFrame with aligned multimodal features
    """
    # Merge on Participant_ID, Condition, and approximate window start time
    merged = pd.merge(
        physio_windows,
        eeg_windows,
        on=['Participant_ID', 'Condition'],
        how='inner',
        suffixes=('_physio', '_eeg')
    )
    
    # Filter to windows with aligned timestamps (within tolerance)
    if 'Window_Start_physio' in merged.columns and 'Window_Start_eeg' in merged.columns:
        time_diff = np.abs(merged['Window_Start_physio'] - merged['Window_Start_eeg'])
        aligned = merged[time_diff <= time_tolerance].copy()
        
        logging.info(f"Aligned {len(aligned)} / {len(merged)} windows "
                    f"(tolerance: {time_tolerance}s)")
        
        return aligned
    else:
        logging.warning("Window timestamp columns not found, returning simple merge")
        return merged
