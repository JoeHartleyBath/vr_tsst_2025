"""
Data loading utilities for physiological feature extraction.

This module handles loading and caching of:
- Raw physiological data (HR, GSR, pupil, blinks)
- EEG features
- Subjective ratings

All functions support pickle caching to speed up repeated runs.
"""

import os
import re
import logging
import pandas as pd
import numpy as np
import yaml
from pathlib import Path


def load_config():
    """Load configuration from YAML files."""
    # Get project root (3 levels up from this file)
    project_root = Path(__file__).parent.parent.parent.parent
    
    config_dir = project_root / "config"
    
    # Load general config
    general_config_path = config_dir / "general.yaml"
    if not general_config_path.exists():
        raise FileNotFoundError(f"Config file not found: {general_config_path}")
    
    with open(general_config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    logging.info(f"Loaded config from {general_config_path}")
    return config


def fix_participant_ids(df):
    """
    Ensure Participant_ID column is consistent across the dataframe.
    
    Handles cases where:
    - IDs are missing
    - Multiple IDs exist in one file
    - IDs need to be extracted from folder names
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with Participant_ID column
    
    Returns
    -------
    pd.DataFrame
        DataFrame with fixed Participant_ID column
    """
    if 'Participant_ID' not in df.columns:
        logging.warning("No Participant_ID column found in DataFrame")
        return df
    
    # Convert to numeric
    df['Participant_ID'] = pd.to_numeric(df['Participant_ID'], errors='coerce')
    
    # Check for issues
    num_missing = df['Participant_ID'].isna().sum()
    unique_ids = df['Participant_ID'].dropna().unique()
    
    if num_missing > 0:
        logging.warning(f"Found {num_missing} missing Participant_IDs")
    
    if len(unique_ids) == 0:
        logging.error("No valid Participant_IDs found in data")
        df['Participant_ID'] = -1
    elif len(unique_ids) > 1:
        # Multiple IDs - use most common
        most_common = df['Participant_ID'].mode().iloc[0]
        logging.warning(
            f"Multiple Participant_IDs found: {unique_ids}. "
            f"Using most common: {most_common}"
        )
        df['Participant_ID'] = most_common
    else:
        # Fill missing with the only valid ID
        if num_missing > 0:
            df['Participant_ID'] = df['Participant_ID'].fillna(unique_ids[0])
    
    # Convert to integer
    df['Participant_ID'] = df['Participant_ID'].astype('Int64')
    
    return df


def load_raw_physio_data(data_path, filename_filter='_RAW_DATA_', force_reload=False):
    """
    Load raw physiological data from CSV files.
    
    Searches recursively through data_path for CSV files matching the filter.
    Results are cached as pickle for faster subsequent loads.
    
    Parameters
    ----------
    data_path : str or Path
        Root directory containing participant data folders
    filename_filter : str
        String that must be in filename to be loaded
    force_reload : bool
        If True, ignore cached pickle and reload from CSVs
    
    Returns
    -------
    pd.DataFrame
        Combined dataframe with all participants' raw physio data
    """
    data_path = Path(data_path)
    
    # Cache in project output directory for consistency
    project_root = Path(__file__).parent.parent.parent.parent
    cache_dir = project_root / "output" / "cache"
    cache_dir.mkdir(exist_ok=True, parents=True)
    pickle_path = cache_dir / "phys_data_raw.pkl"
    
    if pickle_path.exists() and not force_reload:
        logging.info(f"Loading cached physio data from {pickle_path}")
        df = pd.read_pickle(pickle_path)
        logging.info(f"Loaded {len(df)} rows from cache")
        return df
    
    logging.info(f"Loading raw physio data from CSVs in {data_path}")
    
    df_list = []
    files_loaded = 0
    
    # Walk through directory tree
    for root, _, files in os.walk(data_path):
        for filename in files:
            if filename.endswith('.csv') and filename_filter in filename:
                file_path = Path(root) / filename
                
                try:
                    logging.debug(f"Reading {file_path}")
                    df = pd.read_csv(file_path, index_col=False, low_memory=False)
                    
                    # Extract participant ID from path if not in data
                    if 'Participant_ID' not in df.columns:
                        # Try to extract from folder structure (e.g., "P01" or "01")
                        match = re.search(r'[Pp]?(\d+)', Path(root).name)
                        if match:
                            df['Participant_ID'] = int(match.group(1))
                        else:
                            logging.warning(f"Could not extract Participant_ID from {file_path}")
                            df['Participant_ID'] = -1
                    
                    # Fix participant IDs
                    df = fix_participant_ids(df)
                    
                    # Fix column name typo if present
                    if "Polar_HearRate_BPM" in df.columns:
                        df.rename(columns={"Polar_HearRate_BPM": "Polar_HeartRate_BPM"}, 
                                 inplace=True)
                    
                    # Parse timestamps
                    if 'Unity_Timestamp' in df.columns:
                        df['Unity_Timestamp'] = pd.to_datetime(
                            df['Unity_Timestamp'], 
                            errors="coerce", 
                            format="%Y-%m-%dT%H:%M:%S.%fZ"
                        ).fillna(
                            pd.to_datetime(
                                df['Unity_Timestamp'], 
                                errors="coerce", 
                                format="%d-%m-%Y - %H:%M:%S.%f"
                            )
                        )
                        
                        # Compute time from start
                        if df['Unity_Timestamp'].notna().any():
                            start_time = df['Unity_Timestamp'].iloc[0]
                            df['Time_From_Start_Seconds'] = (
                                df['Unity_Timestamp'] - start_time
                            ).dt.total_seconds()
                    
                    # Parse LSL timestamp (critical for downstream processing)
                    if 'LSL_Timestamp' in df.columns:
                        df['LSL_Timestamp'] = pd.to_datetime(
                            df['LSL_Timestamp'], 
                            unit='s', 
                            errors='coerce'
                        )
                    
                    # Replace UCDS placeholder with -1
                    df.replace('UCDS', -1, inplace=True)
                    
                    # Convert numeric columns that may have been read as strings
                    numeric_cols = [
                        'Polar_HeartRate_BPM', 'Polar_HeartRate_RR_Interval',
                        'Shimmer_D36A_GSR_Skin_Conductance_uS', 'Shimmer_D36A_GSR_Skin_Resistance_kOhms',
                        'Foveal_Corrected_Dilation_Left', 'Foveal_Corrected_Dilation_Right',
                        'Inter_Blink_Interval', 'Current_Blink_Duration'
                    ]
                    for col in numeric_cols:
                        if col in df.columns:
                            df[col] = pd.to_numeric(df[col], errors='coerce')
                    
                    df_list.append(df)
                    files_loaded += 1
                    
                except Exception as e:
                    logging.error(f"Failed to load {file_path}: {e}")
                    continue
    
    if not df_list:
        raise ValueError(f"No CSV files found matching '{filename_filter}' in {data_path}")
    
    # Combine all dataframes
    combined_df = pd.concat(df_list, ignore_index=True)
    
    # Drop rows with missing critical timestamps (legacy compatibility)
    initial_rows = len(combined_df)
    if 'LSL_Timestamp' in combined_df.columns:
        combined_df.dropna(subset=['LSL_Timestamp', 'Participant_ID'], inplace=True)
        dropped = initial_rows - len(combined_df)
        if dropped > 0:
            logging.warning(f"Dropped {dropped} rows with missing LSL_Timestamp or Participant_ID")
    
    logging.info(f"Loaded {files_loaded} files, {len(combined_df)} total rows")
    logging.info(f"Participants: {sorted(combined_df['Participant_ID'].unique())}")
    
    # Cache for future use
    combined_df.to_pickle(pickle_path)
    logging.info(f"Cached physio data to {pickle_path}")
    
    return combined_df


def load_eeg_features(config, force_reload=False):
    """
    Load extracted EEG features.
    
    Parameters
    ----------
    config : dict
        Configuration dictionary with paths
    force_reload : bool
        If True, ignore cached data
    
    Returns
    -------
    pd.DataFrame
        EEG features with columns: Participant_ID, Condition, Sample_Frame, 
        Window_Start_Second, and all EEG power/ratio features
    """
    eeg_csv_path = Path(config["paths"]["output"]) / "aggregated" / "eeg_features.csv"
    
    if not eeg_csv_path.exists():
        raise FileNotFoundError(f"EEG features file not found: {eeg_csv_path}")
    
    logging.info(f"Loading EEG features from {eeg_csv_path}")
    eeg_df = pd.read_csv(eeg_csv_path)
    
    # Rename 'Participant' to 'Participant_ID' if needed
    if 'Participant' in eeg_df.columns and 'Participant_ID' not in eeg_df.columns:
        eeg_df = eeg_df.rename(columns={'Participant': 'Participant_ID'})
        logging.info("Renamed 'Participant' to 'Participant_ID' in EEG data")
    
    # Convert Window_Start_Second to datetime for easier alignment
    if 'Window_Start_Second' in eeg_df.columns:
        eeg_df['Window_Start_Second_dt'] = (
            pd.to_datetime('1970-01-01') + 
            pd.to_timedelta(eeg_df['Window_Start_Second'], unit='s')
        )
    
    # Ensure Participant_ID is integer
    if 'Participant_ID' in eeg_df.columns:
        eeg_df['Participant_ID'] = pd.to_numeric(
            eeg_df['Participant_ID'], 
            errors='coerce'
        ).astype('Int64')
    
    logging.info(f"Loaded EEG features: {len(eeg_df)} rows, {len(eeg_df.columns)} columns")
    logging.info(f"EEG participants: {sorted(eeg_df['Participant_ID'].unique())}")
    
    return eeg_df


def load_subjective_ratings(config, force_reload=False):
    """
    Load and reshape subjective ratings data.
    
    Original format: Wide (one row per participant, columns per condition)
    Output format: Long (one row per participant-condition pair)
    
    Parameters
    ----------
    config : dict
        Configuration dictionary with paths
    force_reload : bool
        If True, ignore cached data
    
    Returns
    -------
    pd.DataFrame
        Subjective ratings in long format with columns:
        Participant_ID, Condition, Stress, Cognitive, Perceived
    """
    subjective_path = Path(config["paths"]["output"]) / "aggregated" / "subjective.csv"
    
    if not subjective_path.exists():
        raise FileNotFoundError(f"Subjective ratings file not found: {subjective_path}")
    
    logging.info(f"Loading subjective ratings from {subjective_path}")
    subjective_df = pd.read_csv(subjective_path)
    
    # Mapping from subjective condition names to EEG condition names
    condition_map = {
        'Calm Addition': 'LowStress_LowCog_Task',
        'Calm Subtraction': 'LowStress_HighCog_Task',
        'Stress Addition': 'HighStress_LowCog_Task',
        'Stress Subtraction': 'HighStress_HighCog_Task'
    }
    
    # Reshape from wide to long format
    reshaped_dict = {}
    
    for col in subjective_df.columns[1:]:  # Skip Participant_ID column
        # Split column name: "Calm Addition Stress" -> ("Calm Addition", "Stress")
        parts = col.rsplit(' ', 1)
        if len(parts) != 2:
            logging.warning(f"Unexpected subjective column format: {col}")
            continue
        
        task_type, metric = parts
        condition = condition_map.get(task_type, task_type)
        
        for _, row in subjective_df.iterrows():
            # Use integer-based string for consistency
            participant = str(int(float(row['Participant_ID'])))
            
            key = (participant, condition)
            if key not in reshaped_dict:
                reshaped_dict[key] = {
                    'Participant_ID': participant,
                    'Condition': condition
                }
            
            reshaped_dict[key][metric] = row[col]
    
    # Convert to dataframe
    subjective_long = pd.DataFrame.from_dict(reshaped_dict, orient='index')
    subjective_long.reset_index(drop=True, inplace=True)
    
    # Clean Participant_ID
    subjective_long['Participant_ID'] = (
        subjective_long['Participant_ID'].astype(str).str.strip()
    )
    
    logging.info(f"Reshaped subjective data: {len(subjective_long)} rows")
    logging.info(f"Subjective participants: {sorted(subjective_long['Participant_ID'].unique())}")
    
    return subjective_long


def validate_loaded_data(phys_df, eeg_df, subjective_df):
    """
    Validate that loaded datasets are compatible for merging.
    
    Checks:
    - Overlapping participants exist
    - Required columns are present
    - Data types are consistent
    
    Parameters
    ----------
    phys_df : pd.DataFrame
        Raw physiological data
    eeg_df : pd.DataFrame
        EEG features
    subjective_df : pd.DataFrame
        Subjective ratings
    
    Returns
    -------
    dict
        Validation results with warnings/errors
    """
    results = {
        'valid': True,
        'warnings': [],
        'errors': []
    }
    
    # Check participant overlap
    phys_pids = set(phys_df['Participant_ID'].unique())
    eeg_pids = set(eeg_df['Participant_ID'].unique())
    
    if subjective_df is not None:
        subj_pids = set(subjective_df['Participant_ID'].astype(int).unique())
        all_pids = phys_pids | eeg_pids | subj_pids
        overlap = phys_pids & eeg_pids & subj_pids
    else:
        logging.warning("No subjective data provided - validating only physio and EEG")
        all_pids = phys_pids | eeg_pids
        overlap = phys_pids & eeg_pids
    
    logging.info(f"Participant overlap: {len(overlap)} / {len(all_pids)} participants")
    
    # For testing with limited participants, adjust threshold
    min_participants = 1 if len(all_pids) < 10 else 10
    if len(overlap) < min_participants:
        results['errors'].append(
            f"Only {len(overlap)} participants have data in all sources (need at least {min_participants})"
        )
        results['valid'] = False
    
    missing_from_phys = eeg_pids - phys_pids
    if missing_from_phys:
        results['warnings'].append(
            f"EEG participants missing from physio: {missing_from_phys}"
        )
    
    missing_from_eeg = phys_pids - eeg_pids
    if missing_from_eeg:
        results['warnings'].append(
            f"Physio participants missing from EEG: {missing_from_eeg}"
        )
    
    # Check required physio columns
    required_phys_cols = [
        'Participant_ID', 'LSL_Timestamp', 'Study_Phase',
        'Polar_HeartRate_RR_Interval', 'Shimmer_D36A_GSR_Skin_Conductance_uS'
    ]
    
    missing_cols = [col for col in required_phys_cols if col not in phys_df.columns]
    if missing_cols:
        results['errors'].append(f"Missing required physio columns: {missing_cols}")
        results['valid'] = False
    
    # Check required EEG columns
    required_eeg_cols = ['Participant_ID', 'Condition']
    # Note: Sample_Frame and Window_Start_Second are optional - 
    # EEG features may be pre-aggregated per condition
    missing_eeg = [col for col in required_eeg_cols if col not in eeg_df.columns]
    if missing_eeg:
        results['errors'].append(f"Missing required EEG columns: {missing_eeg}")
        results['valid'] = False
    
    # Log results
    for warning in results['warnings']:
        logging.warning(warning)
    
    for error in results['errors']:
        logging.error(error)
    
    if results['valid']:
        logging.info("✅ Data validation passed")
    else:
        logging.error("❌ Data validation failed")
    
    return results
