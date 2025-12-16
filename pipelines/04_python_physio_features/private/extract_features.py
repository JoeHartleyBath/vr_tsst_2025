"""
Feature Extraction Module

Extracts physiological features from cleaned signals and aligns them with EEG time windows.
Implements both rolling (30s EEG-aligned) and full condition statistics.

Features computed (following feature_computation_analysis.md):
- HR: Mean/Median/SD (exclude MIN/MAX as per validation)
- HRV: RMSSD, SDNN, pNN50
- GSR: Tonic Mean/SD, Peak Rate/Height/Area, SCR counts
- Pupil: Bilateral Mean/Median/SD/Asymmetry (exclude MIN/MAX/UnrestPower)
- Response: Count, Rate, Latency, Accuracy

Author: VR-TSST Project
Date: December 2025
"""

import logging
import numpy as np
import pandas as pd
import neurokit2 as nk
from typing import Dict, List, Optional
from tqdm import tqdm


def extract_gsr_features(signal_data: np.ndarray, sampling_rate: int = 10,
                        min_scr_amp: float = 0.03, max_scr_amp: float = 3.0,
                        flat_std_thresh: float = 0.05) -> Dict:
    """
    Extract tonic and phasic EDA/GSR features with built-in QC.
    
    Parameters
    ----------
    signal_data : array
        Cleaned skin-conductance signal (µS)
    sampling_rate : int
        Sampling frequency (Hz), default 10
    min_scr_amp, max_scr_amp : float
        Amplitude thresholds (µS) for meaningful SCRs
    flat_std_thresh : float
        SD threshold below which phasic features are unreliable
        
    Returns
    -------
    dict
        Dictionary of EDA features with LowVar_Flag indicator
    """
    base_keys = [
        "EDA_Tonic_Mean", "EDA_Tonic_SD", "EDA_PeakRate",
        "EDA_PeakHeight_Mean", "EDA_PeakHeight_Max",
        "EDA_PeakHeight_Median", "EDA_PeakArea",
        "EDA_TotalSCRs", "EDA_MeaningfulSCRs",
        "EDA_ProportionMeaningful", "LowVar_Flag"
    ]
    
    if len(signal_data) < 10:
        return {k: np.nan for k in base_keys}
    
    try:
        eda_signals, info = nk.eda_process(
            signal_data,
            sampling_rate=sampling_rate,
            method="neurokit"
        )
        
        # Tonic features
        tonic = eda_signals["EDA_Tonic"].dropna()
        tonic_mean = float(np.nanmean(tonic))
        tonic_sd = float(np.nanstd(tonic))
        
        # Check for flat segment
        lowvar_flag = bool(tonic_sd < flat_std_thresh)
        
        # Phasic features (mask if flat)
        if lowvar_flag:
            phasic_feats = {k: np.nan for k in [
                "EDA_PeakRate", "EDA_PeakHeight_Mean", "EDA_PeakHeight_Max",
                "EDA_PeakHeight_Median", "EDA_PeakArea",
                "EDA_TotalSCRs", "EDA_MeaningfulSCRs", "EDA_ProportionMeaningful"
            ]}
        else:
            scr_amplitudes = eda_signals["SCR_Amplitude"].dropna()
            scr_peaks = eda_signals["SCR_Peaks"].sum()
            duration_sec = len(signal_data) / sampling_rate
            peak_rate = scr_peaks / duration_sec if duration_sec > 0 else np.nan
            
            phasic_feats = {
                "EDA_PeakRate": peak_rate,
                "EDA_PeakHeight_Mean": float(np.nanmean(scr_amplitudes)),
                "EDA_PeakHeight_Max": float(np.nanmax(scr_amplitudes)),
                "EDA_PeakHeight_Median": float(np.nanmedian(scr_amplitudes)),
                "EDA_PeakArea": float(np.nansum(scr_amplitudes)),
                "EDA_TotalSCRs": int(scr_peaks),
                "EDA_MeaningfulSCRs": int(
                    ((scr_amplitudes >= min_scr_amp) &
                     (scr_amplitudes <= max_scr_amp)).sum()
                ),
                "EDA_ProportionMeaningful": (
                    ((scr_amplitudes >= min_scr_amp) &
                     (scr_amplitudes <= max_scr_amp)).mean()
                    if len(scr_amplitudes) > 0 else np.nan
                )
            }
        
        return {
            "EDA_Tonic_Mean": tonic_mean,
            "EDA_Tonic_SD": tonic_sd,
            **phasic_feats,
            "LowVar_Flag": lowvar_flag
        }
    
    except Exception as e:
        logging.warning(f"[GSR] Feature extraction error: {e}")
        return {k: np.nan for k in base_keys}


def extract_hrv_features(rr_intervals_ms: np.ndarray) -> pd.DataFrame:
    """
    Extract HRV time-domain features using NeuroKit2.
    
    Parameters
    ----------
    rr_intervals_ms : array
        RR intervals in milliseconds
        
    Returns
    -------
    DataFrame
        Single-row DataFrame with HRV_RMSSD, HRV_SDNN, HRV_pNN50
    """
    # Clean up
    rr = np.asarray(rr_intervals_ms, dtype=float)
    rr = rr[~np.isnan(rr)]
    
    if len(rr) < 2:
        return pd.DataFrame([{
            "HRV_RMSSD": np.nan,
            "HRV_SDNN": np.nan,
            "HRV_pNN50": np.nan
        }])
    
    # Convert to seconds
    rr_s = rr / 1000.0
    rri_time = np.cumsum(rr_s)
    rri_dict = {"RRI": rr_s, "RRI_Time": rri_time}
    
    # Compute time-domain HRV
    try:
        hrv_td = nk.hrv_time(rri_dict, sampling_rate=None, show=False)
        metrics = hrv_td[["HRV_RMSSD", "HRV_SDNN", "HRV_pNN50"]].reset_index(drop=True)
        
        # Convert RMSSD from seconds to milliseconds
        metrics["HRV_RMSSD"] = metrics["HRV_RMSSD"] * 1000
        
        return metrics
    except Exception as e:
        logging.warning(f"[HRV] Extraction error: {e}")
        return pd.DataFrame([{
            "HRV_RMSSD": np.nan,
            "HRV_SDNN": np.nan,
            "HRV_pNN50": np.nan
        }])


def calculate_stats(data: pd.DataFrame, columns: List[str],
                   participant_id: int, condition: str,
                   filter_value: Optional[float] = None,
                   context: str = "Full") -> Dict:
    """
    Calculate summary statistics for physiological signals.
    
    NOTE: MIN/MAX are computed for completeness but will be dropped by R scripts
    as validated in feature_computation_analysis.md.
    
    Parameters
    ----------
    data : DataFrame
        Window of physiological data
    columns : list of str
        Column names to compute stats for
    participant_id : int
        Participant ID for logging
    condition : str
        Condition name for logging
    filter_value : float, optional
        Value to filter out before computing stats
    context : str
        "Rolling" or "Full" for logging context
        
    Returns
    -------
    dict
        Dictionary of computed statistics
    """
    all_stats = {}
    
    # Ensure only string columns
    columns = [col for col in columns if isinstance(col, str)]
    
    for column in columns:
        if column not in data.columns:
            continue
        
        col_data = data[column].dropna()
        
        if filter_value is not None:
            col_data = col_data[col_data != filter_value]
        
        # Compute basic stats
        if len(col_data) < 1:
            stats = {
                f"{column}_Median": np.nan,
                f"{column}_Mean": np.nan,
                f"{column}_SD": np.nan,
                # MIN/MAX will be dropped by R, but compute for reference
                f"{column}_MIN": np.nan,
                f"{column}_MAX": np.nan
            }
        else:
            stats = {
                f"{column}_Median": col_data.median(),
                f"{column}_Mean": col_data.mean(),
                f"{column}_SD": col_data.std(),
                f"{column}_MIN": col_data.min(),
                f"{column}_MAX": col_data.max()
            }
        
        # Detailed GSR features
        if column == "Shimmer_D36A_GSR_Skin_Conductance_uS_CLEANED_ABS_CLEANED_NK":
            try:
                gsr_features = extract_gsr_features(
                    signal_data=np.asarray(col_data, dtype=float),
                    sampling_rate=10
                )
                
                # Check if segment is too flat
                too_flat = (stats[f"{column}_SD"] is not np.nan and
                           stats[f"{column}_SD"] < 0.05)
                
                for feat_name, val in gsr_features.items():
                    # Mask phasic features if flat
                    if too_flat and feat_name.startswith("EDA_Peak"):
                        stats[f"{column}_{feat_name}"] = np.nan
                    else:
                        stats[f"{column}_{feat_name}"] = val
            
            except Exception as e:
                logging.warning(
                    f"[GSR] Feature extraction failed "
                    f"(P{participant_id}, {condition}): {e}"
                )
                for feat_name in [
                    "EDA_Tonic_Mean", "EDA_Tonic_SD", "EDA_PeakRate",
                    "EDA_PeakHeight_Mean", "EDA_PeakHeight_Max",
                    "EDA_PeakHeight_Median", "EDA_PeakArea",
                    "EDA_TotalSCRs", "EDA_MeaningfulSCRs",
                    "EDA_ProportionMeaningful"
                ]:
                    stats[f"{column}_{feat_name}"] = np.nan
        
        all_stats.update(stats)
    
    # Pupil bilateral aggregation (if both eyes present)
    if {"Foveal_Corrected_Dilation_Left_CLEANED_ABS",
        "Foveal_Corrected_Dilation_Right_CLEANED_ABS"} <= set(data.columns):
        
        left = data["Foveal_Corrected_Dilation_Left_CLEANED_ABS"].dropna()
        right = data["Foveal_Corrected_Dilation_Right_CLEANED_ABS"].dropna()
        
        if len(left) > 0 and len(right) > 0:
            # Bilateral mean
            all_stats["Full_Pupil_Dilation_Mean"] = (left.mean() + right.mean()) / 2
            all_stats["Full_Pupil_Dilation_Median"] = (left.median() + right.median()) / 2
            all_stats["Full_Pupil_Dilation_SD"] = np.mean([left.std(), right.std()])
            
            # Asymmetry (left - right)
            # Align by index for proper subtraction
            common_idx = left.index.intersection(right.index)
            if len(common_idx) > 0:
                asymmetry = left.loc[common_idx] - right.loc[common_idx]
                all_stats["Full_Pupil_Dilation_Asymmetry"] = asymmetry.mean()
            else:
                all_stats["Full_Pupil_Dilation_Asymmetry"] = np.nan
    
    return all_stats


def extract_response_metrics(window_df: pd.DataFrame) -> Dict:
    """
    Extract response metrics from behavioral data.
    
    Parameters
    ----------
    window_df : DataFrame
        Window of data with Response and Response_Time columns
        
    Returns
    -------
    dict
        Dictionary of response metrics
    """
    response_rows = window_df[window_df["Response"].notna()]
    response_count = len(response_rows)
    
    # Duration in seconds
    duration_sec = (
        window_df["Time_From_Start_Seconds"].max()
        - window_df["Time_From_Start_Seconds"].min()
    )
    
    if pd.api.types.is_timedelta64_dtype(duration_sec):
        duration_sec = duration_sec.total_seconds()
    elif hasattr(duration_sec, "total_seconds"):
        duration_sec = duration_sec.total_seconds()
    
    duration_sec = duration_sec if duration_sec > 0 else np.nan
    
    response_rate_per_min = (
        (response_count / duration_sec) * 60 if pd.notna(duration_sec) else np.nan
    )
    mean_response_latency = (
        response_rows["Response_Time"].mean() if response_count > 0 else np.nan
    )
    
    correct_count = (response_rows["Response"] == "Correct").sum()
    incorrect_count = (response_rows["Response"] == "Incorrect").sum()
    accuracy = correct_count / response_count if response_count > 0 else np.nan
    
    usable = response_count >= 3
    
    return {
        "Response_Count": response_count,
        "Response_Rate_per_min": response_rate_per_min,
        "Mean_Response_Latency_sec": mean_response_latency,
        "Response_Accuracy": accuracy,
        "Response_Correct": correct_count,
        "Response_Incorrect": incorrect_count,
        "Response_Usable": usable,
    }


def extract_all_features(
    phys_cleaned: pd.DataFrame,
    gsr_cleaned: pd.DataFrame,
    eeg_data: pd.DataFrame,
    participants: List[int],
    parallel: bool = False
) -> pd.DataFrame:
    """
    Extract all physiological features aligned with EEG windows.
    
    This function computes FULL condition statistics only (not rolling windows).
    The full window approach provides more stable estimates for ML models.
    
    Parameters
    ----------
    phys_cleaned : DataFrame
        Cleaned physiological data (HR, pupil, blinks)
    gsr_cleaned : DataFrame
        Cleaned and resampled GSR data (10 Hz)
    eeg_data : DataFrame
        EEG features with Participant_ID, Condition, Window_Start_Second
    participants : list of int
        Participant IDs to process
    parallel : bool
        Enable parallel processing (not yet implemented)
        
    Returns
    -------
    DataFrame
        EEG data merged with physiological features
    """
    logging.info("Extracting physiological features...")
    
    # Define columns to extract features from
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
    
    ALL_PHYS_COLUMNS = HR_COLUMNS + GSR_COLUMNS + EYE_COLUMNS
    
    # Check if EEG has temporal information or is pre-aggregated
    has_temporal_eeg = 'Window_Start_Second' in eeg_data.columns
    
    # Ensure datetime formats if temporal EEG
    if has_temporal_eeg:
        eeg_data['Window_Start_Second_dt'] = pd.to_datetime('1970-01-01') + \
            pd.to_timedelta(eeg_data['Window_Start_Second'], unit='s')
    
    if not pd.api.types.is_datetime64_any_dtype(phys_cleaned['Time_From_Start_Seconds']):
        phys_cleaned['Time_From_Start_Seconds'] = pd.to_datetime('1970-01-01') + \
            pd.to_timedelta(phys_cleaned['Time_From_Start_Seconds'], unit='s')
    
    if not pd.api.types.is_datetime64_any_dtype(gsr_cleaned['Time_From_Start_Seconds']):
        gsr_cleaned['Time_From_Start_Seconds'] = pd.to_datetime('1970-01-01') + \
            pd.to_timedelta(gsr_cleaned['Time_From_Start_Seconds'], unit='s')
    
    # Determine condition time ranges from EEG data
    if has_temporal_eeg:
        cond_time = eeg_data.groupby(['Participant_ID', 'Condition']).agg(
            condition_start=('Window_Start_Second_dt', 'min'),
            condition_end=('Window_Start_Second_dt', 'max')
        ).reset_index()
        
        # Add 30s to capture full last window
        cond_time['condition_end'] = cond_time['condition_end'] + pd.Timedelta(seconds=30)
        
        # Adjusted windows (skip first/last 30s for stable estimates)
        cond_time['adjusted_condition_start'] = cond_time['condition_start']
        cond_time['adjusted_condition_end'] = cond_time['condition_end']
    else:
        # EEG is pre-aggregated - extract physio per full Study_Phase
        logging.info("EEG data is pre-aggregated. Extracting physio features per Study_Phase.")
        # Map conditions from physio data
        cond_time = phys_cleaned.groupby(['Participant_ID', 'Study_Phase']).agg(
            condition_start=('Time_From_Start_Seconds', 'min'),
            condition_end=('Time_From_Start_Seconds', 'max')
        ).reset_index()
        cond_time = cond_time.rename(columns={'Study_Phase': 'Condition'})
        # No temporal adjustment for pre-aggregated data - use full windows
        cond_time['adjusted_condition_start'] = cond_time['condition_start']
        cond_time['adjusted_condition_end'] = cond_time['condition_end']
    
    condition_stats_results = []
    
    for idx, row in tqdm(cond_time.iterrows(), total=len(cond_time), desc="Extracting Features"):
        participant = row['Participant_ID']
        condition = row['Condition']
        start_time = row['adjusted_condition_start']
        end_time = row['adjusted_condition_end']
        
        # Extract window data
        hr_data = phys_cleaned[
            (phys_cleaned['Participant_ID'] == participant) &
            (phys_cleaned['Time_From_Start_Seconds'] >= start_time) &
            (phys_cleaned['Time_From_Start_Seconds'] <= end_time)
        ]
        
        gsr_data = gsr_cleaned[
            (gsr_cleaned['Participant_ID'] == participant) &
            (gsr_cleaned['Time_From_Start_Seconds'] >= start_time) &
            (gsr_cleaned['Time_From_Start_Seconds'] <= end_time)
        ]
        
        group_data = pd.concat([hr_data, gsr_data], axis=0).sort_values('Time_From_Start_Seconds')
        
        stats_dict = {
            'Participant_ID': participant,
            'Condition': condition
        }
        
        # Response metrics
        resp_metrics = extract_response_metrics(hr_data)
        stats_dict.update({f"Full_{k}": v for k, v in resp_metrics.items()})
        
        # HRV features
        rr_intervals = group_data['Polar_HeartRate_RR_Interval_CLEANED_ABS'].dropna()
        if len(rr_intervals) >= 2:
            hrv_metrics = extract_hrv_features(rr_intervals)
            rmssd_ms = hrv_metrics["HRV_RMSSD"].iloc[0]
            sdnn = hrv_metrics["HRV_SDNN"].iloc[0]
            pnn50 = hrv_metrics["HRV_pNN50"].iloc[0]
        else:
            rmssd_ms = sdnn = pnn50 = np.nan
        
        stats_dict.update({
            'Full_RMSSD': rmssd_ms,
            'Full_SDNN': sdnn,
            'Full_pNN50': pnn50,
        })
        
        # Other physiological stats
        if not group_data.empty:
            computed_stats = calculate_stats(
                group_data, ALL_PHYS_COLUMNS,
                participant, condition,
                filter_value=-1, context="Full"
            )
            computed_stats = {f"Full_{key}": value for key, value in computed_stats.items()}
            stats_dict.update(computed_stats)
        else:
            for col in ALL_PHYS_COLUMNS:
                for stat in ['Median', 'Mean', 'SD', 'MIN', 'MAX']:
                    stats_dict[f"Full_{col}_{stat}"] = np.nan
        
        condition_stats_results.append(stats_dict)
    
    condition_stats_df = pd.DataFrame(condition_stats_results)
    
    logging.info(f"✅ Extracted features for {len(condition_stats_df)} condition windows")
    
    return condition_stats_df
