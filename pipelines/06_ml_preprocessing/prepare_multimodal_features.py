"""
Multimodal Feature Preprocessing for Condition Classification

Implements optimal preprocessing strategy for deep learning:
1. Forest baseline adjustment (task-specific, last 90s)
2. Within-subject z-score normalization
3. Per-modality scaling (optional)

Strategy:
- Uses pre-condition forest baselines (Relaxation1-4) 
- Z-score relative to forest baseline: (task - forest_mean) / forest_sd
- Then within-subject normalization to remove trait differences
- Outputs ML-ready features for condition classification

Usage:
    python prepare_multimodal_features.py
    python prepare_multimodal_features.py --baseline-duration last_90s
    python prepare_multimodal_features.py --method zscore

Requirements:
    - Multimodal rolling window features
    - Counterbalance sheet (for forest-task mapping)

Outputs:
    - output/ml_ready/multimodal_condition_classification.csv
    - output/qc/preprocessing_report.txt

Author: VR-TSST Project  
Date: December 2025
"""

import os
import sys
import argparse
import logging
import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Dict, Optional


def setup_logging():
    """Configure logging."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"ml_preprocessing_{timestamp}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return log_file


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Preprocess multimodal features for condition classification"
    )
    parser.add_argument(
        '--input',
        type=str,
        default='output/aggregated/multimodal_features_rolling_windows.csv',
        help='Path to multimodal rolling window features'
    )
    parser.add_argument(
        '--counterbalance',
        type=str,
        default='data/processed/VR-TSST Counterbalance sheet.xlsx',
        help='Path to counterbalance sheet'
    )
    parser.add_argument(
        '--baseline-duration',
        type=str,
        default='last_90s',
        choices=['full', 'last_120s', 'last_90s', 'last_60s'],
        help='Which portion of forest baseline to use'
    )
    parser.add_argument(
        '--method',
        type=str,
        default='zscore',
        choices=['subtract', 'percent', 'zscore'],
        help='Baseline adjustment method'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='output/ml_ready/multimodal_condition_classification.csv',
        help='Output path for preprocessed features'
    )
    
    return parser.parse_args()


def identify_feature_columns(df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Identify EEG and physiological feature columns.
    
    Returns:
        Dictionary with 'eeg' and 'physio' feature lists
    """
    eeg_features = [col for col in df.columns if any(
        x in col for x in ['Theta', 'Alpha', 'Beta' 
                           'Frontal', 'Parietal', 'Temporal', 'Central']
    )]
    
    physio_features = [col for col in df.columns if any(
        x in col for x in ['HR', 'HRV', 'RMSSD', 'GSR', 'EDA', 'Pupil', 'Shimmer']
    )]
    
    # Remove any overlap
    physio_features = [f for f in physio_features if f not in eeg_features]
    
    return {
        'eeg': eeg_features,
        'physio': physio_features,
        'all': eeg_features + physio_features
    }


def get_forest_baseline_for_task(
    df: pd.DataFrame,
    participant_id: int,
    task_condition: str,
    counterbalance_data: pd.DataFrame,
    baseline_duration: str = 'last_90s'
) -> Optional[pd.DataFrame]:
    """
    Get the appropriate forest baseline for a given task condition.
    
    Args:
        df: Full dataset with all conditions
        participant_id: Participant ID
        task_condition: Task condition name
        counterbalance_data: DataFrame with participant condition orders
        baseline_duration: Which portion of forest to use
        
    Returns:
        DataFrame with baseline data or None if not found
    """
    
    # Get counterbalance info for this participant
    p_counterbalance = counterbalance_data[
        counterbalance_data['Participant'] == participant_id
    ]
    
    if len(p_counterbalance) == 0:
        logging.warning(f"No counterbalance data for P{participant_id}")
        return None
    
    # Find which round this task appears in
    round_num = None
    for r in [1, 2, 3, 4]:
        round_condition = p_counterbalance[f'Round {r}'].values[0]
        
        # Map from counterbalance names to our task names
        condition_map = {
            'Calm Addition': 'LowStress_LowCog_Task',
            'Calm Subtraction': 'LowStress_HighCog_Task',
            'Stress Addition': 'HighStress_LowCog_Task',
            'Stress Subtraction': 'HighStress_HighCog_Task'
        }
        
        mapped_condition = condition_map.get(round_condition, round_condition)
        
        if mapped_condition == task_condition:
            round_num = r
            break
    
    if round_num is None:
        logging.warning(f"Task {task_condition} not found in counterbalance for P{participant_id}")
        return None
    
    # Get corresponding forest baseline
    forest_condition = f'Forest{round_num}'
    
    # Also try Relaxation naming
    if forest_condition not in df['Condition'].values:
        forest_condition = f'Relaxation{round_num}'
    
    # Extract forest data
    forest_data = df[
        (df['Participant_ID'] == participant_id) &
        (df['Condition'] == forest_condition)
    ].copy()
    
    if len(forest_data) == 0:
        logging.warning(f"No {forest_condition} data for P{participant_id}")
        return None
    
    # Apply windowing based on baseline_duration
    if baseline_duration == 'full':
        return forest_data
    
    # For windowed approaches, use Adjusted_Time or Window_Start
    time_col = 'Adjusted_Time' if 'Adjusted_Time' in forest_data.columns else 'Window_Start'
    
    if time_col not in forest_data.columns:
        logging.warning(f"No time column found in forest data for P{participant_id}")
        return forest_data  # Return full data as fallback
    
    max_time = forest_data[time_col].max()
    
    if baseline_duration == 'last_120s':
        cutoff = max_time - 120
    elif baseline_duration == 'last_90s':
        cutoff = max_time - 90
    elif baseline_duration == 'last_60s':
        cutoff = max_time - 60
    else:
        cutoff = 0
    
    return forest_data[forest_data[time_col] >= cutoff]


def baseline_adjust_with_forests(
    df: pd.DataFrame,
    counterbalance_data: pd.DataFrame,
    feature_cols: List[str],
    baseline_duration: str = 'last_90s',
    method: str = 'zscore'
) -> pd.DataFrame:
    """
    Baseline-adjust task windows using pre-condition forest baselines.
    
    Each task gets its OWN forest baseline from the same round.
    
    Args:
        df: Rolling windows dataset with all conditions
        counterbalance_data: Counterbalance sheet
        feature_cols: List of feature columns to adjust
        baseline_duration: Which portion of forest to use
        method: Adjustment method ('subtract', 'percent', 'zscore')
        
    Returns:
        DataFrame with baseline-adjusted task windows only
    """
    
    logging.info(f"Forest baseline adjustment: {baseline_duration}, method: {method}")
    
    adjusted_windows = []
    n_adjusted = 0
    n_failed = 0
    
    # Get unique participants
    participants = df['Participant_ID'].unique()
    
    for pid in participants:
        
        p_data = df[df['Participant_ID'] == pid].copy()
        
        # Get task conditions for this participant
        task_conditions = [c for c in p_data['Condition'].unique() if 'Task' in c]
        
        for task_cond in task_conditions:
            
            # Get task windows
            task_windows = p_data[p_data['Condition'] == task_cond].copy()
            
            if len(task_windows) == 0:
                continue
            
            # Get corresponding forest baseline
            baseline_data = get_forest_baseline_for_task(
                df=p_data,
                participant_id=pid,
                task_condition=task_cond,
                counterbalance_data=counterbalance_data,
                baseline_duration=baseline_duration
            )
            
            if baseline_data is None or len(baseline_data) == 0:
                logging.warning(f"  No baseline for P{pid:02d} {task_cond}")
                n_failed += 1
                continue
            
            # Compute baseline statistics
            baseline_mean = baseline_data[feature_cols].mean()
            baseline_sd = baseline_data[feature_cols].std()
            
            # Apply adjustment method
            if method == 'subtract':
                # Simple subtraction: task - baseline_mean
                task_windows[feature_cols] = (
                    task_windows[feature_cols].values - baseline_mean.values
                )
            
            elif method == 'percent':
                # Percentage change: (task - baseline) / |baseline| * 100
                for col in feature_cols:
                    if baseline_mean[col] != 0:
                        task_windows[col] = (
                            (task_windows[col] - baseline_mean[col]) / 
                            abs(baseline_mean[col]) * 100
                        )
                    else:
                        task_windows[col] = 0
            
            elif method == 'zscore':
                # Z-score: (task - baseline_mean) / baseline_sd
                for col in feature_cols:
                    if baseline_sd[col] > 0:
                        task_windows[col] = (
                            (task_windows[col] - baseline_mean[col]) / 
                            baseline_sd[col]
                        )
                    else:
                        task_windows[col] = 0
            
            adjusted_windows.append(task_windows)
            n_adjusted += len(task_windows)
    
    logging.info(f"  Adjusted {n_adjusted} windows")
    logging.info(f"  Failed: {n_failed} participant-task pairs")
    
    if len(adjusted_windows) == 0:
        raise ValueError("No windows were successfully adjusted!")
    
    return pd.concat(adjusted_windows, ignore_index=True)


def normalize_within_subject(
    df: pd.DataFrame,
    feature_cols: List[str]
) -> pd.DataFrame:
    """
    Z-score normalize features within each participant.
    
    Removes between-subject trait differences, preserves within-subject patterns.
    
    Args:
        df: DataFrame with features
        feature_cols: List of feature columns to normalize
        
    Returns:
        DataFrame with within-subject normalized features
    """
    
    logging.info("Within-subject z-score normalization")
    
    df_normalized = df.copy()
    
    for pid in df['Participant_ID'].unique():
        mask = df['Participant_ID'] == pid
        
        # Z-score using this participant's mean and SD across ALL conditions
        for col in feature_cols:
            participant_data = df.loc[mask, col]
            mean = participant_data.mean()
            std = participant_data.std()
            
            if std > 0:
                df_normalized.loc[mask, col] = (participant_data - mean) / std
            else:
                df_normalized.loc[mask, col] = 0
    
    return df_normalized


def generate_preprocessing_report(
    df_original: pd.DataFrame,
    df_final: pd.DataFrame,
    feature_groups: Dict,
    output_path: str
):
    """Generate QC report on preprocessing."""
    
    report = []
    report.append("="*70)
    report.append("MULTIMODAL PREPROCESSING REPORT")
    report.append("="*70)
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")
    
    # Input/output statistics
    report.append("DATA TRANSFORMATION:")
    report.append(f"  Input windows: {len(df_original)}")
    report.append(f"  Output windows: {len(df_final)}")
    report.append(f"  Reduction: {(1 - len(df_final)/len(df_original))*100:.1f}%")
    report.append("")
    
    # Participants and conditions
    report.append("PARTICIPANTS & CONDITIONS:")
    report.append(f"  Participants: {df_final['Participant_ID'].nunique()}")
    report.append(f"  Task conditions: {df_final['Condition'].nunique()}")
    
    for cond in sorted(df_final['Condition'].unique()):
        n_windows = len(df_final[df_final['Condition'] == cond])
        report.append(f"    {cond}: {n_windows} windows")
    report.append("")
    
    # Feature statistics
    report.append("FEATURE STATISTICS:")
    report.append(f"  EEG features: {len(feature_groups['eeg'])}")
    report.append(f"  Physio features: {len(feature_groups['physio'])}")
    report.append(f"  Total features: {len(feature_groups['all'])}")
    report.append("")
    
    # Check for NaN/Inf
    all_features = feature_groups['all']
    nan_counts = df_final[all_features].isna().sum()
    inf_counts = np.isinf(df_final[all_features]).sum()
    
    report.append("DATA QUALITY:")
    report.append(f"  Features with NaN: {(nan_counts > 0).sum()}")
    report.append(f"  Features with Inf: {(inf_counts > 0).sum()}")
    
    if (nan_counts > 0).sum() > 0:
        report.append("  Top features with NaN:")
        top_nan = nan_counts[nan_counts > 0].sort_values(ascending=False).head(5)
        for feat, count in top_nan.items():
            report.append(f"    {feat}: {count} ({count/len(df_final)*100:.1f}%)")
    report.append("")
    
    # Normalization check (values should be ~N(0,1) within each participant)
    report.append("NORMALIZATION CHECK (within-subject):")
    for pid in df_final['Participant_ID'].unique()[:3]:  # Check first 3
        p_data = df_final[df_final['Participant_ID'] == pid]
        p_mean = p_data[all_features].mean().mean()
        p_std = p_data[all_features].std().mean()
        report.append(f"  P{pid:02d}: mean={p_mean:.3f}, std={p_std:.3f}")
    report.append("")
    
    report.append("="*70)
    
    # Write report
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        f.write('\n'.join(report))
    
    logging.info(f"Preprocessing report saved to: {output_path}")


def main():
    """Main preprocessing pipeline."""
    
    args = parse_arguments()
    log_file = setup_logging()
    
    logging.info("="*70)
    logging.info("MULTIMODAL FEATURE PREPROCESSING")
    logging.info("="*70)
    logging.info(f"Baseline duration: {args.baseline_duration}")
    logging.info(f"Adjustment method: {args.method}")
    logging.info(f"Log file: {log_file}")
    logging.info("")
    
    try:
        # Load data
        logging.info("Loading data...")
        df = pd.read_csv(args.input)
        logging.info(f"  Input data: {df.shape}")
        
        counterbalance = pd.read_excel(args.counterbalance)
        logging.info(f"  Counterbalance: {counterbalance.shape}")
        logging.info("")
        
        # Identify feature columns
        logging.info("Identifying feature columns...")
        feature_groups = identify_feature_columns(df)
        logging.info(f"  EEG features: {len(feature_groups['eeg'])}")
        logging.info(f"  Physio features: {len(feature_groups['physio'])}")
        logging.info(f"  Total features: {len(feature_groups['all'])}")
        logging.info("")
        
        # STEP 1: Forest baseline adjustment
        logging.info("STEP 1: Forest baseline adjustment...")
        df_baseline_adj = baseline_adjust_with_forests(
            df=df,
            counterbalance_data=counterbalance,
            feature_cols=feature_groups['all'],
            baseline_duration=args.baseline_duration,
            method=args.method
        )
        logging.info(f"  Output shape: {df_baseline_adj.shape}")
        logging.info("")
        
        # STEP 2: Within-subject normalization
        logging.info("STEP 2: Within-subject normalization...")
        df_normalized = normalize_within_subject(
            df=df_baseline_adj,
            feature_cols=feature_groups['all']
        )
        logging.info(f"  Output shape: {df_normalized.shape}")
        logging.info("")
        
        # STEP 3: Handle missing values
        logging.info("STEP 3: Handling missing values...")
        n_nan_before = df_normalized[feature_groups['all']].isna().sum().sum()
        df_normalized[feature_groups['all']] = df_normalized[feature_groups['all']].fillna(0)
        n_nan_after = df_normalized[feature_groups['all']].isna().sum().sum()
        logging.info(f"  NaN values filled: {n_nan_before} → {n_nan_after}")
        logging.info("")
        
        # STEP 4: Save output
        logging.info("STEP 4: Saving output...")
        output_path = args.output
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df_normalized.to_csv(output_path, index=False)
        logging.info(f"  Saved to: {output_path}")
        logging.info(f"  File size: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")
        logging.info("")
        
        # STEP 5: Generate QC report
        logging.info("STEP 5: Generating QC report...")
        report_path = "output/qc/preprocessing_report.txt"
        generate_preprocessing_report(
            df_original=df,
            df_final=df_normalized,
            feature_groups=feature_groups,
            output_path=report_path
        )
        logging.info("")
        
        # Summary
        logging.info("="*70)
        logging.info("PREPROCESSING COMPLETE")
        logging.info("="*70)
        logging.info(f"Input: {len(df)} windows")
        logging.info(f"Output: {len(df_normalized)} task windows")
        logging.info(f"Participants: {df_normalized['Participant_ID'].nunique()}")
        logging.info(f"Conditions: {df_normalized['Condition'].nunique()}")
        logging.info(f"Features: {len(feature_groups['all'])}")
        logging.info(f"Output: {output_path}")
        logging.info(f"Report: {report_path}")
        logging.info("="*70)
        
        return df_normalized
        
    except Exception as e:
        logging.error(f"FATAL ERROR: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    result = main()
