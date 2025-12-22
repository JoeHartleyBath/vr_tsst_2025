"""
Rolling Window Feature Extraction

Implements windowing logic for temporal physiological feature extraction.
Extracts only features suitable for 10s windows.

Window Parameters:
    - Default: 10s windows with 50% overlap (5s stride)
    - Configurable via command-line arguments
    - Outputs timestamp metadata for multimodal alignment

Features Extracted:
    ✅ Heart Rate (mean, median, SD)
    ✅ GSR Tonic (mean, median, SD) - baseline skin conductance
    ✅ Pupil Dilation (mean, median, SD, asymmetry)
    ✅ Blink Metrics (inter-blink interval, duration)

Features EXCLUDED (unsuitable for 10s windows):
    ❌ HRV (RMSSD) - requires ≥2 minutes for reliability
    ❌ GSR Phasic (SCR peaks, counts) - need ≥30s for reliable SCR counts
    ❌ Response Metrics (RT, accuracy) - aggregated version only

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
    calculate_stats
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
        # Define physiological columns (excluding GSR phasic features)
        HR_COLUMNS = [
            'Polar_HeartRate_BPM_CLEANED_ABS',
            'Polar_HeartRate_RR_Interval_CLEANED_ABS'
        ]
        
        GSR_COLUMNS = [
            'Shimmer_D36A_GSR_Skin_Conductance_uS_CLEANED_ABS_CLEANED_NK'
        ]
        
        EYE_COLUMNS = [
            'Foveal_Corrected_Dilation_Left_CLEANED_ABS',
            'Foveal_Corrected_Dilation_Right_CLEANED_ABS',
            'Inter_Blink_Interval_CLEANED_ABS',
            'Current_Blink_Duration_CLEANED_ABS'
        ]
        
        # Only use HR (no HRV), GSR tonic (no phasic), and eye tracking
        VALID_COLUMNS = HR_COLUMNS + GSR_COLUMNS + EYE_COLUMNS
        
        # Extract basic stats for all physio columns (mean, median, SD only)
        # This excludes HRV (RMSSD), GSR phasic (SCR counts/peaks), and response metrics
        phys_stats = calculate_stats(
            window_data,
            columns=VALID_COLUMNS,
            participant_id=participant_id,
            condition=condition
        )
        features.update(phys_stats)
            
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
        if 'Condition' not in p_data.columns:
            logging.error(f"'Condition' column not found in data for P{participant_id}")
            continue
            
        conditions = p_data['Condition'].dropna().unique()
        logging.info(f"P{participant_id}: Found {len(conditions)} conditions")
        
        # Process each condition
        for condition in conditions:
            
            # Filter data for this condition
            cond_data = p_data[p_data['Condition'] == condition].copy()
            cond_gsr = p_gsr[p_gsr['Condition'] == condition].copy()
            
            if len(cond_data) == 0:
                continue
            
            # Use Time_From_Start_Seconds as time column
            time_col = 'Time_From_Start_Seconds'
            if time_col not in cond_data.columns:
                logging.warning(f"Time column '{time_col}' not found for P{participant_id} {condition}")
                continue
            
            # Convert to numeric seconds if datetime
            if pd.api.types.is_datetime64_any_dtype(cond_data[time_col]):
                cond_data[time_col] = (cond_data[time_col] - cond_data[time_col].iloc[0]).dt.total_seconds()
            
            windows = create_rolling_windows(
                cond_data,
                time_column=time_col,
                window_size=window_size,
                overlap=overlap
            )
            
            logging.info(f"P{participant_id} {condition}: Created {len(windows)} windows")
            
            # Extract features from each window
            for window_meta in windows:
                window_features = extract_features_from_window(
                    window_data=window_meta['data'],
                    gsr_data=cond_gsr,
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
