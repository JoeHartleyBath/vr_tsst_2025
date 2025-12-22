"""
Eye Tracking Signal Cleaning Pipeline

Implements professional pupil dilation and blink cleaning following best practices:
1. Absolute threshold filtering (physiologically plausible ranges)
2. Blink/closure detection (based on dilation thresholds and drop rates)
3. Closure classification (short/medium/prolonged for different handling)
4. Interpolation for short closures only (≤0.3s)

Extracted from legacy notebook: step05_preprocess_physio_and_merge.ipynb
Author: VR-TSST Project
Date: December 2025
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Optional


# Physiologically plausible thresholds
EYE_THRESHOLDS = {
    "Inter_Blink_Interval": (0.1, 20.0),  # seconds
    "Current_Blink_Duration": (0.05, 1.0),  # seconds
    "Foveal_Corrected_Dilation_Left": (-4.0, 4.0),  # mm
    "Foveal_Corrected_Dilation_Right": (-4.0, 4.0),  # mm
}

# Blink detection parameters
BLINK_PARAMS = {
    "low_margin": 0.5,  # mm above absolute threshold for "low dilation"
    "max_interp_duration": 0.3,  # seconds - max duration to interpolate
    "drop_thresh": 1.0,  # mm/s - rate of change threshold for closure detection
    "min_remove_duration": 2.0,  # seconds - prolonged closures to leave as NaN
}

TIME_COL = "LSL_Timestamp"


def apply_absolute_threshold_cleaning(
    df: pd.DataFrame,
    feat: str,
    min_val: float,
    max_val: float,
    logger: Optional[logging.Logger] = None,
    participant_id: Optional[int] = None,
    condition_name: Optional[str] = None
) -> pd.DataFrame:
    """
    Apply absolute threshold cleaning to eye tracking features.
    
    Parameters
    ----------
    df : DataFrame
        DataFrame containing eye tracking data (must have TIME_COL as index)
    feat : str
        Feature name to clean
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
    clean_col = f"{feat}_CLEANED_ABS"
    
    before = df[feat].notna().sum()
    df[feat] = pd.to_numeric(df[feat], errors="coerce")
    df[clean_col] = df[feat].where(df[feat].between(min_val, max_val), other=np.nan)
    after = df[clean_col].notna().sum()
    total = len(df)
    
    if logger:
        prefix = ""
        if participant_id:
            prefix += f"[{participant_id}] "
        if condition_name:
            prefix += f"[{condition_name}] "
        logger.info(
            f"{prefix}{clean_col}: retained {after}/{total} (dropped {before-after})"
        )
    
    return df


def detect_and_classify_closures(
    df: pd.DataFrame,
    left_col: str,
    right_col: str,
    thresholds: dict,
    params: dict,
    logger: Optional[logging.Logger] = None,
    participant_id: Optional[int] = None,
    condition_name: Optional[str] = None
) -> pd.DataFrame:
    """
    Detect eye closures (blinks) and classify by duration.
    
    Closure detection logic:
    - Both eyes have low dilation (< threshold + margin)
    - AND at least one eye shows rapid drop (rate of change threshold)
    
    Classification:
    - Short (≤0.3s): Interpolate (typical blinks)
    - Medium (0.3s - 2.0s): Set to NaN (longer blinks, squints)
    - Prolonged (≥2.0s): Set to NaN (eyes closed, look-away)
    
    Parameters
    ----------
    df : DataFrame
        DataFrame with cleaned pupil columns (indexed by TIME_COL)
    left_col : str
        Left pupil cleaned column name
    right_col : str
        Right pupil cleaned column name
    thresholds : dict
        Eye tracking thresholds
    params : dict
        Blink detection parameters
    logger : Logger, optional
        QC logger
    participant_id : int, optional
        Participant ID for logging
    condition_name : str, optional
        Condition name for logging
        
    Returns
    -------
    DataFrame
        Updated DataFrame with closures masked
    """
    # Extract thresholds
    mnL, _ = thresholds["Foveal_Corrected_Dilation_Left"]
    mnR, _ = thresholds["Foveal_Corrected_Dilation_Right"]
    lowL = mnL + params["low_margin"]
    lowR = mnR + params["low_margin"]
    drop_thr = params.get("drop_thresh", 1.0)
    
    # Calculate time deltas and rates of change
    dt = df.index.to_series().diff().dt.total_seconds()
    rate_L = df[left_col].diff() / dt
    rate_R = df[right_col].diff() / dt
    
    # Detect closures: both eyes low AND at least one dropping rapidly
    is_low_L = df[left_col] <= lowL
    is_low_R = df[right_col] <= lowR
    is_drop_L = rate_L <= -drop_thr
    is_drop_R = rate_R <= -drop_thr
    is_closure = (is_low_L & is_low_R) & (is_drop_L | is_drop_R)
    
    # Group consecutive closures
    grp = (is_closure != is_closure.shift(fill_value=False)).cumsum()
    durations = (
        is_closure
        .groupby(grp)
        .apply(lambda seg: (seg.index[-1] - seg.index[0]).total_seconds() if seg.any() else 0)
    )
    
    # Classify by duration
    short_ids = durations[durations <= params["max_interp_duration"]].index
    prolonged_ids = durations[durations >= params.get("min_remove_duration", np.inf)].index
    medium_ids = durations.index.difference(short_ids.union(prolonged_ids))
    
    if logger:
        prefix = ""
        if participant_id:
            prefix += f"[{participant_id}] "
        if condition_name:
            prefix += f"[{condition_name}] "
        logger.info(
            f"{prefix}Closures detected → "
            f"short: {len(short_ids)}, medium: {len(medium_ids)}, prolonged: {len(prolonged_ids)}"
        )
    
    # Mask all closures (will interpolate short ones later)
    df.loc[grp.isin(short_ids) & is_closure, [left_col, right_col]] = np.nan
    df.loc[grp.isin(medium_ids) & is_closure, [left_col, right_col]] = np.nan
    df.loc[grp.isin(prolonged_ids) & is_closure, [left_col, right_col]] = np.nan
    
    return df


def clean_eye_pipeline(
    phys_data: pd.DataFrame,
    qc_loggers: Dict[int, logging.Logger]
) -> pd.DataFrame:
    """
    Complete eye tracking cleaning pipeline.
    
    Steps:
    1. Ensure TIME_COL is datetime
    2. For each participant:
       a. Apply absolute threshold cleaning to all eye features
       b. Detect and classify closures (blinks)
       c. Interpolate short closures only
    3. Log retention statistics per condition
    
    Parameters
    ----------
    phys_data : DataFrame
        Raw physiological data with eye tracking columns
    qc_loggers : dict
        Dictionary mapping participant_id -> logging.Logger
        
    Returns
    -------
    DataFrame
        Cleaned eye tracking data with *_CLEANED_ABS columns
    """
    logging.info("Starting eye tracking cleaning pipeline...")
    
    # Ensure datetime format
    if not pd.api.types.is_datetime64_any_dtype(phys_data[TIME_COL]):
        phys_data[TIME_COL] = pd.to_datetime(phys_data[TIME_COL])
    
    all_cleaned_parts = []
    global_acc = {}
    
    for pid, df_pid in phys_data.groupby("Participant_ID", sort=False):
        participant_id = int(pid)
        condition_name = df_pid['Study_Phase'].iloc[0] if 'Study_Phase' in df_pid else 'All'
        logger = qc_loggers.get(participant_id)
        
        if logger is None:
            logging.warning(f"[⚠️] No QC logger for P{participant_id}, skipping eye cleaning.")
            continue
        
        logger.info(f"[{participant_id}] [{condition_name}] Starting eye feature cleaning")
        
        # Set time index
        df = df_pid.sort_values(TIME_COL).set_index(TIME_COL).copy()
        
        # 1) Absolute threshold cleaning for all eye features
        for feat, (mn, mx) in EYE_THRESHOLDS.items():
            if feat not in df.columns:
                continue
            
            clean_col = f"{feat}_CLEANED_ABS"
            df = apply_absolute_threshold_cleaning(
                df, feat, mn, mx,
                logger=logger,
                participant_id=participant_id,
                condition_name=condition_name
            )
            
            # Track global statistics
            global_acc.setdefault(clean_col, {"dropped": 0, "retained": 0, "total": 0})
            before = df[feat].notna().sum()
            after = df[clean_col].notna().sum()
            total = len(df)
            global_acc[clean_col]["dropped"] += before - after
            global_acc[clean_col]["retained"] += after
            global_acc[clean_col]["total"] += total
        
        # 2) Detect and classify closures
        Lcol = "Foveal_Corrected_Dilation_Left_CLEANED_ABS"
        Rcol = "Foveal_Corrected_Dilation_Right_CLEANED_ABS"
        
        if Lcol in df.columns and Rcol in df.columns:
            df = detect_and_classify_closures(
                df, Lcol, Rcol,
                EYE_THRESHOLDS, BLINK_PARAMS,
                logger=logger,
                participant_id=participant_id,
                condition_name=condition_name
            )
        
        # 3) Interpolation on all *_CLEANED_ABS columns
        dt = df.index.to_series().diff().dt.total_seconds()
        dt_med = dt.median()
        max_n = int(BLINK_PARAMS["max_interp_duration"] // dt_med) if dt_med > 0 else None
        clean_cols = [f"{feat}_CLEANED_ABS" for feat in EYE_THRESHOLDS if f"{feat}_CLEANED_ABS" in df.columns]
        
        before_interp = df[clean_cols].isna().sum().sum()
        df[clean_cols] = df[clean_cols].interpolate(
            method='time', limit=max_n, limit_direction='both'
        )
        after_interp = df[clean_cols].isna().sum().sum()
        interpolated = before_interp - after_interp
        
        logger.info(
            f"[{participant_id}] [{condition_name}] "
            f"Interpolated {interpolated} NaNs across {len(clean_cols)} eye columns"
        )
        
        all_cleaned_parts.append(df.reset_index())
    
    # Reassemble
    eye_cleaned = pd.concat(all_cleaned_parts, ignore_index=True)
    
    # Per-condition summary logging
    for pid, df_pid in eye_cleaned.groupby("Participant_ID", sort=False):
        participant_id = int(pid)
        logger = qc_loggers.get(participant_id)
        
        if logger is None:
            continue
        
        for cond, df_cond in df_pid.groupby("Study_Phase", sort=False):
            for col in [f"{feat}_CLEANED_ABS" for feat in EYE_THRESHOLDS]:
                if col in df_cond:
                    retained = df_cond[col].notna().sum()
                    total = len(df_cond)
                    logger.info(
                        f"[{participant_id}] [{cond}] {col}: "
                        f"{retained} / {total} values retained after all cleaning"
                    )
                else:
                    logger.warning(
                        f"[{participant_id}] [{cond}] Column '{col}' not found in eye-cleaned data"
                    )
    
    # Overall dropped/retained summary
    logging.info("=== Overall Dropped/Retained Summary ===")
    for clean_col, stats in global_acc.items():
        total = stats["total"]
        drp = stats["dropped"]
        keep = stats["retained"]
        
        if total == 0:
            continue
        
        logging.info(
            f"{clean_col}: dropped {drp}/{total} ({drp/total:.1%}), "
            f"kept {keep}/{total} ({keep/total:.1%})"
        )
    
    logging.info("✅ Eye tracking cleaning pipeline completed")
    
    return eye_cleaned
