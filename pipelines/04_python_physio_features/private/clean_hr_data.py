"""
Heart Rate Signal Cleaning Pipeline

Implements professional heart rate and HRV data cleaning following best practices:
1. Absolute threshold filtering (physiologically plausible ranges)
2. Ectopic beat correction (using NeuroKit2's Kubios method)
3. MAD-based outlier rejection (robust to artifacts)
4. Interpolation for small gaps

Extracted from legacy notebook: step05_preprocess_physio_and_merge.ipynb
Author: VR-TSST Project
Date: December 2025
"""

import logging
import numpy as np
import pandas as pd
import neurokit2 as nk
from typing import Dict, Optional


# Physiologically plausible thresholds
BPM_THRESHOLDS = (40, 220)  # BPM range for adults
RR_THRESHOLDS = (100, 2000)  # RR interval range in milliseconds


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


def correct_ectopic_beats(
    df: pd.DataFrame,
    rr_col: str = "Polar_HeartRate_RR_Interval_CLEANED_ABS",
    sampling_rate: int = 1000
) -> pd.DataFrame:
    """
    Correct ectopic beats using NeuroKit2's signal_fixpeaks (Kubios method).
    
    This method:
    1. Converts RR intervals to peak indices
    2. Applies Kubios ectopic beat correction
    3. Converts back to RR intervals
    
    Parameters
    ----------
    df : DataFrame
        DataFrame containing RR interval column
    rr_col : str
        Column name of RR intervals (ms)
    sampling_rate : int
        Sampling rate for peak reconstruction (Hz)
        
    Returns
    -------
    DataFrame
        Updated DataFrame with corrected RR intervals
    """
    rr_ms = df[rr_col].dropna().values
    
    if len(rr_ms) < 10:
        return df  # Not enough data for correction
    
    try:
        # Convert RR intervals to peak indices
        peaks = nk.intervals_to_peaks(rr_ms, sampling_rate=sampling_rate)
        
        # Apply ectopic beat correction
        _, peaks_clean = nk.signal_fixpeaks(
            peaks, 
            sampling_rate=sampling_rate, 
            method="Kubios"
        )
        
        # Convert back to RR intervals
        diffs = np.diff(peaks_clean)
        rr_clean_ms = diffs * (1000.0 / sampling_rate)
        
        # Assign back to original indices (skip first value as diff reduces length by 1)
        orig_idx = df.index[df[rr_col].notna()].to_numpy()
        n_assign = min(len(orig_idx) - 1, len(rr_clean_ms))
        assign_idx = orig_idx[1 : 1 + n_assign]
        df.loc[assign_idx, rr_col] = rr_clean_ms[:n_assign]
        
    except Exception as e:
        logging.warning(f"[HR] Ectopic beat correction failed: {e}")
    
    return df


def apply_mad_outlier_rejection(
    df: pd.DataFrame,
    col: str,
    window_size: int = 11,
    mad_threshold: float = 4.0
) -> pd.DataFrame:
    """
    Apply MAD (Median Absolute Deviation) based outlier rejection.
    
    This is more robust than z-score for physiological signals with artifacts.
    
    Parameters
    ----------
    df : DataFrame
        DataFrame containing the signal column
    col : str
        Column name to clean
    window_size : int
        Rolling window size for MAD calculation (must be odd)
    mad_threshold : float
        Number of MADs beyond which values are considered outliers
        
    Returns
    -------
    DataFrame
        Updated DataFrame with outliers set to NaN
    """
    series = df[col]
    
    # Calculate rolling MAD
    mad = series.rolling(window_size, center=True).apply(
        lambda x: np.median(np.abs(x - np.median(x))), raw=True
    )
    
    # Avoid division by zero: set floor to smallest non-zero MAD or 10
    floor = mad[mad > 0].min() if mad[mad > 0].any() else 10
    mad = mad.fillna(floor).clip(lower=floor)
    
    # Calculate rolling median
    med = series.rolling(window_size, center=True).median()
    
    # Flag outliers
    outliers = np.abs(series - med) > (mad_threshold * mad)
    
    # Set outliers to NaN
    df.loc[outliers, col] = np.nan
    
    return df


def clean_hr_pipeline(
    phys_data: pd.DataFrame,
    qc_loggers: Dict[int, logging.Logger],
    bpm_col: str = "Polar_HeartRate_BPM",
    rr_col: str = "Polar_HeartRate_RR_Interval",
    group_cols: list = None
) -> pd.DataFrame:
    """
    Complete heart rate cleaning pipeline.
    
    Steps:
    1. Bootstrap cleaned RR column if missing
    2. Clamp extreme RR values outside physiological range
    3. Apply absolute threshold cleaning to BPM
    4. Interpolate small gaps in BPM
    5. Correct ectopic beats in RR intervals
    6. Apply MAD-based outlier rejection to RR
    7. Interpolate small gaps in RR
    
    Parameters
    ----------
    phys_data : DataFrame
        Raw physiological data
    qc_loggers : dict
        Dictionary mapping participant_id -> logging.Logger
    bpm_col : str
        BPM column name
    rr_col : str
        RR interval column name
    group_cols : list, optional
        Columns to group by (default: ["Participant_ID", "Study_Phase"])
        
    Returns
    -------
    DataFrame
        Cleaned physiological data with *_CLEANED_ABS columns
    """
    if group_cols is None:
        group_cols = ["Participant_ID", "Study_Phase"]
    
    all_cleaned_parts = []
    
    logging.info("Starting HR cleaning pipeline...")
    
    for group_keys, df_cond in phys_data.groupby(group_cols):
        if len(group_cols) == 2:
            participant_id, condition_name = group_keys
        else:
            participant_id = group_keys
            condition_name = "All"
        
        participant_id = int(participant_id)
        logger = qc_loggers.get(participant_id)
        
        if logger is None:
            logging.warning(f"⚠️ No QC logger for P{participant_id}, skipping.")
            continue
        
        logger.info(f"[{participant_id}] [{condition_name}] Starting HR cleaning pipeline")
        
        # Make a copy for safety
        df = df_cond.copy()
        
        # 0) Bootstrap cleaned RR column if missing
        rr_clean_col = f"{rr_col}_CLEANED_ABS"
        if rr_clean_col not in df and rr_col in df:
            df[rr_clean_col] = df[rr_col].copy()
        
        # 1) Clamp extremes on RR
        if rr_clean_col in df:
            df.loc[~df[rr_clean_col].between(*RR_THRESHOLDS), rr_clean_col] = np.nan
        
        # 2) BPM threshold cleaning
        if bpm_col in df:
            df = apply_absolute_threshold_cleaning(
                df, bpm_col, *BPM_THRESHOLDS,
                logger=logger,
                participant_id=participant_id,
                condition_name=condition_name
            )
        
        # 3) Interpolate BPM
        bpm_clean_col = f"{bpm_col}_CLEANED_ABS"
        if bpm_clean_col in df:
            df[bpm_clean_col] = df[bpm_clean_col].interpolate(
                method="linear", limit=10, limit_direction="both"
            )
        
        # 4) Ectopic beat correction
        if rr_clean_col in df:
            df = correct_ectopic_beats(df, rr_clean_col)
        
        # 5) MAD-based outlier rejection on RR
        if rr_clean_col in df:
            df = apply_mad_outlier_rejection(df, rr_clean_col)
        
        # 6) Interpolate small gaps in RR
        if rr_clean_col in df:
            df[rr_clean_col] = df[rr_clean_col].interpolate(
                method="cubic", limit=2, limit_direction="both"
            )
        
        # Final logging
        cleaned_cols = [c for c in df.columns if c.endswith("_CLEANED_ABS")]
        total = len(df)
        for col in cleaned_cols:
            retained = df[col].notna().sum()
            logger.info(
                f"[{participant_id}] [{condition_name}] {col}: {retained} / {total} retained"
            )
        
        all_cleaned_parts.append(df)
    
    # Reassemble
    cleaned_data = pd.concat(all_cleaned_parts, ignore_index=True)
    logging.info("✅ HR cleaning pipeline completed")
    
    return cleaned_data
