"""
GSR/EDA Signal Cleaning Pipeline

Implements professional GSR cleaning following best practices:
1. Resampling to consistent 10 Hz (GSR is often sampled irregularly)
2. Absolute threshold filtering (physiologically plausible conductance range)
3. NeuroKit2 eda_clean (Butterworth lowpass filter)
4. Flatline detection and removal (sensor disconnection artifacts)
5. Interpolation for small gaps only

Extracted from legacy notebook: step05_preprocess_physio_and_merge.ipynb
Author: VR-TSST Project
Date: December 2025
"""

import logging
import numpy as np
import pandas as pd
import neurokit2 as nk
from typing import Dict, Optional, List


# Physiologically plausible thresholds
GSR_CONDUCTANCE_RANGE = (0.01, 30)  # µS (microsiemens)
TARGET_FS = 10  # Hz - standard for GSR
TARGET_INTERVAL = pd.Timedelta('100ms')  # 10 Hz = 100ms


def resample_gsr_to_10hz(
    phys_data: pd.DataFrame,
    gsr_cols: List[str],
    timestamp_col: str = "LSL_Timestamp",
    target_fs: int = TARGET_FS
) -> pd.DataFrame:
    """
    Resample GSR data to consistent 10 Hz sampling rate.
    
    GSR is often sampled at irregular intervals. This function:
    1. Creates regular 10 Hz time grid
    2. Uses nearest-neighbor resampling (appropriate for GSR's slow dynamics)
    3. Merges back metadata (Study_Phase, Shown_Scene, etc.)
    
    Parameters
    ----------
    phys_data : DataFrame
        Raw physiological data
    gsr_cols : list of str
        GSR column names to resample
    timestamp_col : str
        Timestamp column name
    target_fs : int
        Target sampling frequency (Hz)
        
    Returns
    -------
    DataFrame
        Resampled GSR data with metadata
    """
    resampled_parts = []
    
    logging.info(f"Resampling GSR to {target_fs} Hz...")
    
    for pid, df_participant in phys_data.groupby("Participant_ID"):
        # 1) Sort & dedupe, set timestamp index
        df = df_participant.sort_values(timestamp_col).drop_duplicates(subset=timestamp_col)
        df = df.set_index(timestamp_col)
        
        # 2) Build regular grid at target_fs
        new_index = pd.date_range(
            start=df.index.min(),
            end=df.index.max(),
            freq=pd.Timedelta(seconds=1/target_fs)
        )
        
        # 3) Nearest-neighbor reindex for GSR columns
        df_nn = df[gsr_cols].reindex(
            new_index,
            method='nearest',
            tolerance=pd.Timedelta('50ms')  # only snap if within 50 ms
        )
        
        # 4) Fill any remaining gaps
        df_nn = df_nn.ffill().bfill()
        
        # 5) Reset index back to timestamp column
        df_gsr_resampled = df_nn.reset_index().rename(columns={'index': timestamp_col})
        
        # 6) Prepare metadata for merge_asof
        metadata_cols = ['Study_Phase', 'Shown_Scene', 'Arithmetic_Task', 'Time_From_Start_Seconds']
        available_metadata = [col for col in metadata_cols if col in df.columns]
        
        if available_metadata:
            df_meta = df[available_metadata].reset_index()
            df_meta[timestamp_col] = df_meta[timestamp_col].dt.round('1ms')
            df_gsr_resampled[timestamp_col] = df_gsr_resampled[timestamp_col].dt.round('1ms')
            
            # 7) Merge metadata back on nearest timestamp
            df_final = pd.merge_asof(
                df_gsr_resampled.sort_values(timestamp_col),
                df_meta.sort_values(timestamp_col),
                on=timestamp_col,
                direction='nearest',
                tolerance=pd.Timedelta("200ms")
            )
        else:
            df_final = df_gsr_resampled
        
        # 8) Add participant + sample rate columns
        df_final['Participant_ID'] = int(pid)
        df_final['Sample_Rate_Hz'] = target_fs
        
        resampled_parts.append(df_final)
    
    resampled_data = pd.concat(resampled_parts, ignore_index=True)
    logging.info(f"✅ GSR resampled to {target_fs} Hz for {len(resampled_parts)} participants")
    
    return resampled_data


def remove_flat_signals(
    df: pd.DataFrame,
    col: str,
    max_flat_duration_sec: float,
    fs: int,
    logger: Optional[logging.Logger] = None
) -> pd.DataFrame:
    """
    Detect and remove flatline segments (sensor disconnection artifacts).
    
    Parameters
    ----------
    df : DataFrame
        DataFrame containing GSR signal
    col : str
        Column name to check for flatlines
    max_flat_duration_sec : float
        Maximum consecutive flat duration before flagging (seconds)
    fs : int
        Sampling frequency (Hz)
    logger : Logger, optional
        QC logger for this participant
        
    Returns
    -------
    DataFrame
        Updated DataFrame with flatlines set to NaN
    """
    max_flat_samples = int(max_flat_duration_sec * fs)
    
    if col not in df.columns:
        logging.warning(f"[GSR] Column '{col}' not found in DataFrame.")
        return df
    
    values = df[col].values
    flat_mask = np.zeros_like(values, dtype=bool)
    total_flagged = 0
    run_counter = 0
    
    start_idx = 0
    while start_idx < len(values):
        current_val = values[start_idx]
        run_len = 1
        
        # Count consecutive identical values (excluding NaNs)
        while (
            start_idx + run_len < len(values)
            and values[start_idx + run_len] == current_val
            and not np.isnan(current_val)
        ):
            run_len += 1
        
        # Flag runs longer than threshold
        if run_len >= max_flat_samples:
            flat_mask[start_idx:start_idx + run_len] = True
            total_flagged += run_len
            run_counter += 1
        
        start_idx += run_len
    
    # Set flagged values to NaN
    df.loc[flat_mask, col] = np.nan
    
    if logger:
        logger.info(
            f"[GSR] {col}: Removed {total_flagged} samples across {run_counter} "
            f"flat segments > {max_flat_duration_sec}s"
        )
    
    return df


def apply_absolute_threshold_cleaning(
    df: pd.DataFrame,
    col: str,
    min_val: float,
    max_val: float,
    logger: Optional[logging.Logger] = None,
    participant_id: Optional[int] = None,
    condition_name: Optional[str] = None
) -> pd.DataFrame:
    """
    Apply absolute threshold cleaning: remove values outside physiological range.
    
    Parameters
    ----------
    df : DataFrame
        DataFrame containing the signal column
    col : str
        Column name to clean
    min_val : float
        Minimum physiologically plausible value
    max_val : float
        Maximum physiologically plausible value
    logger : Logger, optional
        QC logger for this participant
    participant_id : int, optional
        Participant ID for logging
    condition_name : str, optional
        Condition name for logging
        
    Returns
    -------
    DataFrame
        Updated DataFrame with new CLEANED_ABS column
    """
    cleaned_col = f"{col}_CLEANED_ABS"
    
    total = df[col].shape[0]
    n_out_of_bounds = (~df[col].between(min_val, max_val)).sum()
    
    df[cleaned_col] = df[col].where((df[col] >= min_val) & (df[col] <= max_val))
    
    if logger:
        prefix = ""
        if participant_id:
            prefix += f"[{participant_id}] "
        if condition_name:
            prefix += f"[{condition_name}] "
        
        logger.info(
            f"{prefix}{col}: {n_out_of_bounds} / {total} values set to NaN "
            f"(outside [{min_val}, {max_val}])"
        )
    
    return df


def clean_gsr_pipeline(
    gsr_resampled: pd.DataFrame,
    qc_loggers: Dict[int, logging.Logger],
    conductance_col: str = "Shimmer_D36A_GSR_Skin_Conductance_uS",
    target_fs: int = TARGET_FS
) -> pd.DataFrame:
    """
    Complete GSR cleaning pipeline.
    
    Steps:
    1. Apply absolute threshold cleaning (0.01-30 µS)
    2. Apply NeuroKit2's eda_clean (Butterworth lowpass)
    3. Remove flatline segments (>5s of identical values)
    4. Interpolate small gaps only (≤5s)
    5. Log retention statistics per condition
    
    Parameters
    ----------
    gsr_resampled : DataFrame
        GSR data resampled to 10 Hz
    qc_loggers : dict
        Dictionary mapping participant_id -> logging.Logger
    conductance_col : str
        GSR conductance column name
    target_fs : int
        Sampling frequency (Hz)
        
    Returns
    -------
    DataFrame
        Cleaned GSR data with *_CLEANED_ABS and *_CLEANED_NK columns
    """
    all_cleaned_parts = []
    
    logging.info("Starting GSR cleaning pipeline...")
    
    for pid, df_gsr in gsr_resampled.groupby("Participant_ID"):
        participant_id = int(pid)
        logger = qc_loggers.get(participant_id)
        
        if logger is None:
            logging.warning(f"[GSR] Logger missing for P{participant_id}, skipping.")
            continue
        
        logging.info(f"[GSR] Cleaning GSR for P{participant_id}")
        
        # 1) Absolute threshold cleaning
        if conductance_col in df_gsr.columns:
            df_gsr = apply_absolute_threshold_cleaning(
                df_gsr, conductance_col, *GSR_CONDUCTANCE_RANGE,
                logger=logger,
                participant_id=participant_id,
                condition_name="GSR_Resampled"
            )
        else:
            logger.warning(f"[GSR] Column '{conductance_col}' not found for P{participant_id}")
            continue
        
        # 2) Apply NeuroKit2's eda_clean (Butterworth lowpass filter)
        cleaned_abs_col = f"{conductance_col}_CLEANED_ABS"
        cleaned_nk_col = f"{cleaned_abs_col}_CLEANED_NK"
        
        try:
            df_gsr[cleaned_nk_col] = nk.eda_clean(
                df_gsr[cleaned_abs_col], 
                sampling_rate=target_fs
            )
        except Exception as e:
            logger.warning(f"[GSR] nk.eda_clean failed for P{participant_id}: {e}")
            df_gsr[cleaned_nk_col] = df_gsr[cleaned_abs_col]
        
        # Log retention BEFORE flatline removal and interpolation
        if "Study_Phase" in df_gsr.columns:
            for condition_name, df_cond in df_gsr.groupby("Study_Phase"):
                for col in [cleaned_abs_col, cleaned_nk_col]:
                    if col in df_cond.columns:
                        non_nan = df_cond[col].notna().sum()
                        total = len(df_cond[col])
                        logger.info(
                            f"[{participant_id}] [{condition_name}] {col}: "
                            f"{non_nan} / {total} retained BEFORE flatline removal"
                        )
        
        # 3) Remove flatline segments (sensor disconnection)
        df_gsr = remove_flat_signals(
            df_gsr, cleaned_abs_col,
            max_flat_duration_sec=5,
            fs=target_fs,
            logger=logger
        )
        
        # 4) Interpolate only small gaps (≤5 seconds)
        interpolation_limit = int(5 * target_fs)  # 50 samples at 10 Hz
        
        for col in [cleaned_abs_col, cleaned_nk_col]:
            if col in df_gsr.columns:
                before = df_gsr[col].isna().sum()
                df_gsr[col] = df_gsr[col].interpolate(
                    method='linear',
                    limit=interpolation_limit,
                    limit_direction='both'
                )
                after = df_gsr[col].isna().sum()
                logger.info(
                    f"[GSR] P{participant_id} interpolated {before - after} values in '{col}' "
                    f"(remaining NaNs: {after})"
                )
        
        all_cleaned_parts.append(df_gsr)
    
    # Reassemble and log final retention per condition
    cleaned_data = pd.concat(all_cleaned_parts, ignore_index=True)
    
    # Final retention logging per condition
    for df_pid in all_cleaned_parts:
        participant_id = int(df_pid['Participant_ID'].iloc[0])
        logger = qc_loggers.get(participant_id)
        
        if logger is None or "Study_Phase" not in df_pid.columns:
            continue
        
        for condition_name, df_cond in df_pid.groupby("Study_Phase"):
            for col in [f"{conductance_col}_CLEANED_ABS", f"{conductance_col}_CLEANED_ABS_CLEANED_NK"]:
                if col in df_cond.columns:
                    total = len(df_cond[col])
                    non_nan = df_cond[col].notna().sum()
                    num_nan = total - non_nan
                    percent_retained = 100 * non_nan / total if total > 0 else 0
                    logger.info(
                        f"[{participant_id}] [{condition_name}] {col}: "
                        f"{non_nan} / {total} retained "
                        f"({percent_retained:.1f}%), {num_nan} missing"
                    )
    
    logging.info("✅ GSR cleaning pipeline completed")
    
    return cleaned_data
