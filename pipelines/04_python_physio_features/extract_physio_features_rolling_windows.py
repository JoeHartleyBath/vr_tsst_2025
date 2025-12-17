"""
Physiological Feature Extraction Pipeline - Rolling Windows Version

Extracts heart rate, GSR, pupil, and blink features from VR-TSST physiological data
using rolling windows for temporal analysis and multimodal fusion with EEG.

Usage:
    python extract_physio_features_rolling_windows.py                          # All participants
    python extract_physio_features_rolling_windows.py --participants 1 2 3     # Specific participants
    python extract_physio_features_rolling_windows.py --parallel               # Enable parallel processing
    python extract_physio_features_rolling_windows.py --window-size 10 --overlap 0.5

Requirements:
    - Must be run from project root: c:/vr_tsst_2025/
    - Config file: config/general.yaml
    - Input data: data/raw/metadata/P01.csv through P48.csv
    - Python packages: pandas, numpy, yaml, neurokit2, tqdm

Outputs:
    - output/aggregated/physio_features_rolling_windows.csv  # Windows with timestamps
    - logs/physio_extraction_rolling_<timestamp>.log         # Processing log
    - output/qc/physio_rolling/P##_qc.txt                    # Per-participant QC logs

Key Differences from Aggregated Version:
    - Output: Thousands of rows (multiple windows per condition)
    - Includes: Window_Start, Window_End, Window_Index columns
    - Window size: 10s default (configurable)
    - Overlap: 50% default (5s stride)
    - Enables: LSTM, temporal modeling, multimodal synchronization

Author: VR-TSST Project
Date: December 2025
"""

import sys
import os
import argparse
import logging
from datetime import datetime
import pandas as pd
import numpy as np
from tqdm import tqdm

# Add private modules to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'private'))

from load_data import load_raw_physio_data, load_eeg_features
from clean_hr_data import clean_hr_pipeline
from clean_gsr_data import clean_gsr_pipeline
from clean_eye_data import clean_eye_data
from extract_features_rolling import extract_rolling_window_features


def setup_logging():
    """Configure logging to file and console."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"physio_extraction_rolling_{timestamp}.log")
    
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
        description="Extract physiological features with rolling windows"
    )
    parser.add_argument(
        '--participants', 
        type=int, 
        nargs='+',
        help='List of participant IDs to process (default: all)'
    )
    parser.add_argument(
        '--parallel',
        action='store_true',
        help='Enable parallel processing'
    )
    parser.add_argument(
        '--window-size',
        type=float,
        default=10.0,
        help='Window size in seconds (default: 10.0)'
    )
    parser.add_argument(
        '--overlap',
        type=float,
        default=0.5,
        help='Window overlap fraction 0-1 (default: 0.5 = 50%%)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='output/aggregated/physio_features_rolling_windows.csv',
        help='Output file path'
    )
    
    return parser.parse_args()


def main():
    """Main processing pipeline."""
    
    # Parse arguments and setup logging
    args = parse_arguments()
    log_file = setup_logging()
    
    logging.info("="*70)
    logging.info("Physiological Feature Extraction - ROLLING WINDOWS")
    logging.info("="*70)
    logging.info(f"Window size: {args.window_size}s")
    logging.info(f"Overlap: {args.overlap*100:.0f}% (stride: {args.window_size*(1-args.overlap):.1f}s)")
    logging.info(f"Parallel processing: {args.parallel}")
    logging.info(f"Log file: {log_file}")
    logging.info("")
    
    # Determine participants to process
    if args.participants:
        participants = args.participants
        logging.info(f"Processing {len(participants)} specified participants: {participants}")
    else:
        participants = list(range(1, 49))
        logging.info(f"Processing all {len(participants)} participants")
    
    try:
        # STEP 1: Load raw physiological data
        logging.info("STEP 1: Loading raw physiological data...")
        phys_data = load_raw_physio_data(
            participants=participants,
            use_cache=True,
            cache_path='output/cache/phys_data_raw.pkl'
        )
        logging.info(f"  Loaded data shape: {phys_data.shape}")
        logging.info(f"  Participants found: {phys_data['Participant_ID'].nunique()}")
        logging.info("")
        
        # STEP 2: Clean data (HR, GSR, Eye tracking)
        logging.info("STEP 2: Cleaning physiological signals...")
        
        # HR cleaning
        logging.info("  Cleaning heart rate data...")
        phys_data_cleaned = clean_hr_pipeline(
            phys_data.copy(),
            output_dir='output/qc/physio_rolling'
        )
        
        # GSR cleaning
        logging.info("  Cleaning GSR data...")
        gsr_cleaned = clean_gsr_pipeline(
            phys_data.copy(),
            output_dir='output/qc/physio_rolling'
        )
        
        # Eye tracking cleaning
        logging.info("  Cleaning eye tracking data...")
        eye_cleaned = clean_eye_data(phys_data.copy())
        
        # Merge cleaned signals
        phys_data_cleaned = phys_data_cleaned.copy()
        gsr_col = 'Shimmer_D36A_GSR_Skin_Conductance_uS_CLEANED_ABS_CLEANED_NK'
        if gsr_col in gsr_cleaned.columns:
            phys_data_cleaned[gsr_col] = gsr_cleaned[gsr_col]
        
        # Merge eye data
        eye_cols = [col for col in eye_cleaned.columns if 'Pupil' in col or 'Valid' in col]
        for col in eye_cols:
            if col in eye_cleaned.columns:
                phys_data_cleaned[col] = eye_cleaned[col]
        
        logging.info(f"  Cleaned data shape: {phys_data_cleaned.shape}")
        logging.info("")
        
        # STEP 3: Extract rolling window features
        logging.info("STEP 3: Extracting rolling window features...")
        logging.info(f"  Window parameters: {args.window_size}s window, {args.overlap*100:.0f}% overlap")
        
        all_features = extract_rolling_window_features(
            phys_data_cleaned,
            gsr_cleaned,
            participants=participants,
            window_size=args.window_size,
            overlap=args.overlap,
            parallel=args.parallel
        )
        
        logging.info(f"  Extracted features shape: {all_features.shape}")
        logging.info(f"  Total windows: {len(all_features)}")
        logging.info(f"  Windows per participant (avg): {len(all_features) / len(participants):.1f}")
        logging.info("")
        
        # STEP 4: Save output
        logging.info("STEP 4: Saving output...")
        output_path = args.output
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        all_features.to_csv(output_path, index=False)
        logging.info(f"  Saved to: {output_path}")
        logging.info(f"  File size: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")
        
        # Summary statistics
        logging.info("")
        logging.info("="*70)
        logging.info("PROCESSING COMPLETE")
        logging.info("="*70)
        logging.info(f"Total participants: {all_features['Participant_ID'].nunique()}")
        logging.info(f"Total conditions: {all_features['Condition'].nunique()}")
        logging.info(f"Total windows: {len(all_features)}")
        logging.info(f"Features per window: {len(all_features.columns) - 5}")  # Exclude metadata cols
        logging.info(f"Output: {output_path}")
        logging.info("="*70)
        
        return all_features
        
    except Exception as e:
        logging.error(f"FATAL ERROR: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    result = main()
