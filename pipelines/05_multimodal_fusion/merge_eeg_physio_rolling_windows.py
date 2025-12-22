"""
Multimodal Feature Fusion - Rolling Windows

Merges EEG and physiological features from rolling windows for multimodal ML.
Aligns timestamps across modalities and prepares data for LSTM/temporal models.

Usage:
    python merge_eeg_physio_rolling_windows.py
    python merge_eeg_physio_rolling_windows.py --time-tolerance 0.5
    python merge_eeg_physio_rolling_windows.py --participants 1 2 3

Outputs:
    - output/aggregated/multimodal_features_rolling_windows.csv
    - output/qc/multimodal_alignment_report.txt

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


def setup_logging():
    """Configure logging."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"multimodal_fusion_{timestamp}.log")
    
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
        description="Merge EEG and physio rolling window features"
    )
    parser.add_argument(
        '--eeg-features',
        type=str,
        default='output/aggregated/eeg_features_rolling_windows.csv',
        help='Path to EEG rolling window features'
    )
    parser.add_argument(
        '--physio-features',
        type=str,
        default='output/aggregated/physio_features_rolling_windows.csv',
        help='Path to physio rolling window features'
    )
    parser.add_argument(
        '--time-tolerance',
        type=float,
        default=0.1,
        help='Max time difference for window alignment (seconds)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='output/aggregated/multimodal_features_rolling_windows.csv',
        help='Output path for merged features'
    )
    parser.add_argument(
        '--participants',
        type=int,
        nargs='+',
        help='Filter specific participants'
    )
    
    return parser.parse_args()


def load_features(file_path: str, modality: str) -> pd.DataFrame:
    """Load feature file with validation."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{modality} features not found: {file_path}")
    
    df = pd.read_csv(file_path)
    
    # Standardize column names based on modality
    if modality.upper() == "EEG":
        # EEG files have: pid, event_label, window_idx, window_start, window_end
        if 'pid' in df.columns:
            df = df.rename(columns={'pid': 'Participant_ID'})
        if 'event_label' in df.columns:
            df = df.rename(columns={'event_label': 'Condition'})
        if 'window_idx' in df.columns:
            df = df.rename(columns={'window_idx': 'Window_Index'})
            # CRITICAL FIX: Convert EEG from 1-indexed to 0-indexed to match physio
            df['Window_Index'] = df['Window_Index'] - 1
            logging.info(f"  Applied index offset: EEG Window_Index converted from 1-indexed to 0-indexed")
        if 'window_start' in df.columns:
            df = df.rename(columns={'window_start': 'Window_Start'})
        if 'window_end' in df.columns:
            df = df.rename(columns={'window_end': 'Window_End'})
    elif modality.upper() == "PHYSIO":
        # Physio files have: participant_id, condition
        if 'participant_id' in df.columns:
            df = df.rename(columns={'participant_id': 'Participant_ID'})
        if 'condition' in df.columns:
            df = df.rename(columns={'condition': 'Condition'})
    
    # Verify required columns exist
    if 'Participant_ID' not in df.columns:
        raise ValueError(f"{modality} file missing participant ID column. Found columns: {df.columns.tolist()}")
    if 'Condition' not in df.columns:
        raise ValueError(f"{modality} file missing condition column. Found columns: {df.columns.tolist()}")
    
    logging.info(f"Loaded {modality} features: {df.shape}")
    logging.info(f"  Participants: {df['Participant_ID'].nunique()}")
    logging.info(f"  Conditions: {df['Condition'].nunique()}")
    logging.info(f"  Windows: {len(df)}")
    
    return df


def align_windows(
    eeg_features: pd.DataFrame,
    physio_features: pd.DataFrame,
    time_tolerance: float
) -> pd.DataFrame:
    """
    Align EEG and physio windows based on window index and participant/condition.
    
    Args:
        eeg_features: EEG rolling window features
        physio_features: Physio rolling window features
        time_tolerance: Maximum time difference for alignment (seconds) - unused if no timestamps
        
    Returns:
        Merged DataFrame with aligned features
    """
    logging.info("Aligning windows based on Window_Index, Participant_ID, and Condition")
    
    # Check if timestamp columns exist
    has_eeg_timestamps = 'Window_Start' in eeg_features.columns
    has_physio_timestamps = 'Window_Start' in physio_features.columns
    
    # PRIMARY: Use index-based alignment (1-to-1 window matching)
    logging.info("Using index-based alignment on Window_Index")
    
    # Rename columns to track source modality
    eeg_features = eeg_features.rename(columns={
        'Window_Start': 'Window_Start_EEG',
        'Window_End': 'Window_End_EEG'
    })
    
    physio_features = physio_features.rename(columns={
        'Window_Start': 'Window_Start_Physio',
        'Window_End': 'Window_End_Physio'
    })
    
    # Merge on participant, condition, and window index (1-to-1 matching)
    aligned = pd.merge(
        eeg_features,
        physio_features,
        on=['Participant_ID', 'Condition', 'Window_Index'],
        how='inner',
        suffixes=('_EEG', '_Physio')
    )
    
    logging.info(f"Aligned windows: {len(aligned)}")
    
    # SECONDARY: Validate alignment with timestamp comparison (if available)
    if has_eeg_timestamps and has_physio_timestamps:
        logging.info(f"Validating alignment with timestamp tolerance: {time_tolerance}s")
        
        # Calculate time difference between matched windows
        aligned['Time_Diff'] = np.abs(
            aligned['Window_Start_EEG'] - aligned['Window_Start_Physio']
        )
        
        # Report alignment quality
        n_misaligned = (aligned['Time_Diff'] > time_tolerance).sum()
        if n_misaligned > 0:
            logging.warning(f"  {n_misaligned}/{len(aligned)} window pairs exceed time tolerance of {time_tolerance}s")
            max_diff = aligned['Time_Diff'].max()
            logging.warning(f"  Maximum time difference: {max_diff:.3f}s")
        else:
            logging.info(f"  All {len(aligned)} window pairs within {time_tolerance}s tolerance")
            max_diff = aligned['Time_Diff'].max()
            logging.info(f"  Maximum time difference: {max_diff:.3f}s")
        
        # Use average window start/end for final timestamps
        aligned['Window_Start'] = (aligned['Window_Start_EEG'] + 
                                   aligned['Window_Start_Physio']) / 2
        aligned['Window_End'] = (aligned['Window_End_EEG'] + 
                                aligned['Window_End_Physio']) / 2
        
        # Drop redundant columns
        aligned = aligned.drop(columns=[
            'Window_Start_EEG', 'Window_Start_Physio',
            'Window_End_EEG', 'Window_End_Physio',
            'Time_Diff'
        ])
    elif has_physio_timestamps:
        # Use physio timestamps if EEG doesn't have them
        logging.info("  Using physio timestamps for Window_Start/Window_End")
        aligned['Window_Start'] = aligned['Window_Start_Physio']
        aligned['Window_End'] = aligned['Window_End_Physio']
        aligned = aligned.drop(columns=['Window_Start_Physio', 'Window_End_Physio'])
    
    return aligned


def generate_alignment_report(
    eeg_features: pd.DataFrame,
    physio_features: pd.DataFrame,
    merged_features: pd.DataFrame,
    output_path: str
):
    """Generate QC report on alignment quality."""
    
    report = []
    report.append("="*70)
    report.append("MULTIMODAL WINDOW ALIGNMENT REPORT")
    report.append("="*70)
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")
    
    # Input statistics
    report.append("INPUT FEATURES:")
    report.append(f"  EEG windows: {len(eeg_features)}")
    report.append(f"  Physio windows: {len(physio_features)}")
    report.append("")
    
    # Alignment statistics
    report.append("ALIGNMENT RESULTS:")
    report.append(f"  Aligned windows: {len(merged_features)}")
    report.append(f"  EEG coverage: {len(merged_features)/len(eeg_features)*100:.1f}%")
    report.append(f"  Physio coverage: {len(merged_features)/len(physio_features)*100:.1f}%")
    report.append("")
    
    # Per-participant statistics
    report.append("PER-PARTICIPANT ALIGNMENT:")
    for pid in sorted(merged_features['Participant_ID'].unique()):
        n_windows = len(merged_features[merged_features['Participant_ID'] == pid])
        n_conditions = merged_features[merged_features['Participant_ID'] == pid]['Condition'].nunique()
        report.append(f"  P{pid:02d}: {n_windows} windows across {n_conditions} conditions")
    report.append("")
    
    # Window count discrepancies per participant-condition
    report.append("WINDOW COUNT ANALYSIS (EEG vs Physio per condition):")
    mismatches = []
    for pid in sorted(merged_features['Participant_ID'].unique()):
        for cond in sorted(merged_features[merged_features['Participant_ID'] == pid]['Condition'].unique()):
            eeg_count = len(eeg_features[(eeg_features['Participant_ID'] == pid) & (eeg_features['Condition'] == cond)])
            physio_count = len(physio_features[(physio_features['Participant_ID'] == pid) & (physio_features['Condition'] == cond)])
            merged_count = len(merged_features[(merged_features['Participant_ID'] == pid) & (merged_features['Condition'] == cond)])
            
            if abs(eeg_count - physio_count) > 1:  # Flag discrepancies > 1 window
                diff = eeg_count - physio_count
                mismatches.append(f"  P{pid:02d} {cond}: EEG={eeg_count}, Physio={physio_count}, Diff={diff:+d}, Merged={merged_count}")
            
            # Calculate window loss percentage
            if eeg_count > 0:
                loss_pct = (1 - merged_count / eeg_count) * 100
                if loss_pct > 10:  # Flag >10% window loss
                    mismatches.append(f"  P{pid:02d} {cond}: {loss_pct:.1f}% window loss (EEG={eeg_count}, Merged={merged_count})")
    
    if mismatches:
        report.append("  DISCREPANCIES DETECTED:")
        report.extend(mismatches)
    else:
        report.append("  No significant discrepancies detected (all within ±1 window)")
    report.append("")
    
    # Feature counts
    eeg_cols = [col for col in merged_features.columns if 'EEG' in col or any(
        band in col for band in ['Theta', 'Alpha', 'Beta']
    )]
    physio_cols = [col for col in merged_features.columns if any(
        x in col for x in ['HR','GSR', 'EDA', 'Pupil']
    )]
    
    report.append("FEATURE COUNTS:")
    report.append(f"  EEG features: {len(eeg_cols)}")
    report.append(f"  Physio features: {len(physio_cols)}")
    report.append(f"  Total features: {len(merged_features.columns) - 5}")  # Exclude metadata
    report.append("")
    
    report.append("="*70)
    
    # Write report
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        f.write('\n'.join(report))
    
    logging.info(f"Alignment report saved to: {output_path}")


def main():
    """Main fusion pipeline."""
    
    args = parse_arguments()
    log_file = setup_logging()
    
    logging.info("="*70)
    logging.info("MULTIMODAL FEATURE FUSION - ROLLING WINDOWS")
    logging.info("="*70)
    logging.info(f"Time tolerance: {args.time_tolerance}s")
    logging.info(f"Log file: {log_file}")
    logging.info("")
    
    try:
        # Load features
        logging.info("Loading features...")
        eeg_features = load_features(args.eeg_features, "EEG")
        physio_features = load_features(args.physio_features, "Physio")
        logging.info("")
        
        # Filter participants if specified
        if args.participants:
            logging.info(f"Filtering to participants: {args.participants}")
            eeg_features = eeg_features[eeg_features['Participant_ID'].isin(args.participants)]
            physio_features = physio_features[physio_features['Participant_ID'].isin(args.participants)]
            logging.info("")
        
        # Align windows
        logging.info("Aligning windows...")
        merged_features = align_windows(
            eeg_features,
            physio_features,
            time_tolerance=args.time_tolerance
        )
        logging.info("")
        
        # Save merged features
        logging.info("Saving merged features...")
        output_path = args.output
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        merged_features.to_csv(output_path, index=False)
        logging.info(f"  Saved to: {output_path}")
        logging.info(f"  Shape: {merged_features.shape}")
        logging.info(f"  File size: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")
        logging.info("")
        
        # Generate QC report
        logging.info("Generating alignment report...")
        report_path = "output/qc/multimodal_alignment_report.txt"
        generate_alignment_report(
            eeg_features,
            physio_features,
            merged_features,
            report_path
        )
        logging.info("")
        
        # Summary
        logging.info("="*70)
        logging.info("FUSION COMPLETE")
        logging.info("="*70)
        logging.info(f"Participants: {merged_features['Participant_ID'].nunique()}")
        logging.info(f"Conditions: {merged_features['Condition'].nunique()}")
        logging.info(f"Aligned windows: {len(merged_features)}")
        logging.info(f"Output: {output_path}")
        logging.info(f"Report: {report_path}")
        logging.info("="*70)
        
        return merged_features
        
    except Exception as e:
        logging.error(f"FATAL ERROR: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    result = main()
