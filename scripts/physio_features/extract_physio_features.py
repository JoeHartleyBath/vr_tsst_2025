"""
Physiological Feature Extraction Pipeline (Refactored)

Main orchestration script for extracting heart rate, GSR, pupil, and blink features
from VR-TSST experiment physiological data and merging with EEG features.

Usage:
    python extract_physio_features.py                              # All participants
    python extract_physio_features.py --participants 1 2 3         # Specific participants
    python extract_physio_features.py --parallel                    # Enable parallel processing
    python extract_physio_features.py --output custom_output.csv   # Custom output path

Author: VR-TSST Project
Date: December 2025
"""

import os
import sys
import argparse
import logging
from datetime import datetime
import pandas as pd
import numpy as np

# Import helper modules
from private.load_data import (
    load_config,
    load_raw_physio_data,
    load_eeg_features,
    load_subjective_ratings,
    validate_loaded_data
)

# Import cleaning modules
from private.clean_hr_data import clean_hr_pipeline
from private.clean_gsr_data import clean_gsr_pipeline, resample_gsr_to_10hz
from private.clean_eye_data import clean_eye_pipeline

# Import feature extraction and merge modules
from private.extract_features import extract_all_features
from private.merge_with_eeg import merge_physio_with_eeg, validate_merged_data


def setup_logging():
    """Configure logging with timestamps."""
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_filename = os.path.join(log_dir, f"physio_extraction_{timestamp}.log")
    
    logging.basicConfig(
        filename=log_filename,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        force=True
    )
    
    # Also log to console
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(levelname)s - %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)
    
    logging.info(f"Logging to: {log_filename}")
    return log_filename


def setup_qc_loggers(config, participants):
    """Create QC loggers for each participant."""
    qc_loggers = {}
    
    physio_qc_path = config['paths'].get('physio_qc', 'output/qc/physio')
    os.makedirs(physio_qc_path, exist_ok=True)
    
    for participant_id in participants:
        qc_log_path = os.path.join(physio_qc_path, f'P{participant_id:02d}_qc.txt')
        
        logger = logging.getLogger(f'qc_logger_{participant_id}')
        logger.setLevel(logging.INFO)
        
        # Avoid duplicate handlers
        if not logger.handlers:
            handler = logging.FileHandler(qc_log_path, mode='a')
            formatter = logging.Formatter('%(asctime)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.propagate = False
        
        qc_loggers[participant_id] = logger
    
    logging.info(f"✅ Created QC loggers for {len(participants)} participants")
    return qc_loggers


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Extract physiological features from VR-TSST experiment data"
    )
    
    parser.add_argument(
        '--participants',
        type=int,
        nargs='+',
        default=None,
        help='Participant IDs to process (default: all participants 1-48)'
    )
    
    parser.add_argument(
        '--parallel',
        action='store_true',
        help='Enable parallel processing across participants'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='output/aggregated/all_data_aggregated.csv',
        help='Output CSV file path (default: output/aggregated/all_data_aggregated.csv)'
    )
    
    parser.add_argument(
        '--force-reprocess',
        action='store_true',
        help='Force reprocessing even if cached data exists'
    )
    
    parser.add_argument(
        '--skip-cleaning',
        action='store_true',
        help='Skip signal cleaning steps (use for testing only)'
    )
    
    return parser.parse_args()


def main():
    """Main execution pipeline."""
    print("=" * 80)
    print("PHYSIOLOGICAL FEATURE EXTRACTION PIPELINE")
    print("=" * 80)
    
    # Setup
    args = parse_arguments()
    log_file = setup_logging()
    
    logging.info("Starting physiological feature extraction pipeline")
    logging.info(f"Arguments: {args}")
    
    # Determine participants to process
    if args.participants:
        participants = args.participants
        logging.info(f"Processing specific participants: {participants}")
    else:
        participants = list(range(1, 49))
        logging.info("Processing all 48 participants")
    
    try:
        # STEP 0: Load configuration
        logging.info("STEP 0: Loading configuration...")
        config = load_config()
        data_path = config["paths"]["raw_data"]
        
        # STEP 1: Load data
        logging.info("STEP 1: Loading raw physiological data...")
        phys_data_raw = load_raw_physio_data(
            data_path,
            filename_filter='P',  # Match P01.csv, P02.csv, etc.
            force_reload=args.force_reprocess
        )
        logging.info(f"  Loaded {len(phys_data_raw)} physio rows")
        
        logging.info("Loading EEG features...")
        eeg_data = load_eeg_features(config, force_reload=args.force_reprocess)
        logging.info(f"  Loaded {len(eeg_data)} EEG rows")
        
        logging.info("Loading subjective ratings...")
        try:
            subjective_data = load_subjective_ratings(config, force_reload=args.force_reprocess)
            logging.info(f"  Loaded {len(subjective_data)} subjective rows")
        except FileNotFoundError as e:
            logging.warning(f"Subjective ratings file not found: {e}")
            logging.warning("Continuing without subjective ratings - they may already be in EEG data")
            subjective_data = None
        
        # Validate loaded data
        logging.info("Validating loaded datasets...")
        validation = validate_loaded_data(phys_data_raw, eeg_data, subjective_data)
        if not validation['valid']:
            raise ValueError("Data validation failed. Check log for details.")
        
        # Setup QC loggers for signal cleaning
        logging.info("Setting up QC loggers...")
        qc_loggers = setup_qc_loggers(config, participants)
        
        # STEP 2: Signal cleaning
        if not args.skip_cleaning:
            logging.info("STEP 2: Cleaning physiological signals...")
            
            logging.info("  - Cleaning heart rate data...")
            phys_data_cleaned = clean_hr_pipeline(phys_data_raw, qc_loggers)
            
            logging.info("  - Resampling and cleaning GSR data...")
            gsr_resampled = resample_gsr_to_10hz(
                phys_data_raw,
                gsr_cols=['Shimmer_D36A_GSR_Skin_Conductance_uS',
                         'Shimmer_D36A_GSR_Skin_Resistance_kOhms']
            )
            gsr_cleaned = clean_gsr_pipeline(gsr_resampled, qc_loggers)
            
            logging.info("  - Cleaning eye tracking data...")
            phys_data_cleaned = clean_eye_pipeline(phys_data_cleaned, qc_loggers)
        else:
            logging.warning("Skipping signal cleaning (--skip-cleaning flag set)")
            phys_data_cleaned = phys_data_raw.copy()
            gsr_cleaned = phys_data_raw.copy()
            
            # Create cleaned column aliases for feature extraction
            phys_data_cleaned['Polar_HeartRate_BPM_CLEANED_ABS'] = phys_data_cleaned['Polar_HeartRate_BPM']
            phys_data_cleaned['Polar_HeartRate_RR_Interval_CLEANED_ABS'] = phys_data_cleaned['Polar_HeartRate_RR_Interval']
            gsr_cleaned['Shimmer_D36A_GSR_Skin_Conductance_uS_CLEANED_ABS_CLEANED_NK'] = gsr_cleaned['Shimmer_D36A_GSR_Skin_Conductance_uS']
            phys_data_cleaned['Foveal_Corrected_Dilation_Left_CLEANED_ABS'] = phys_data_cleaned['Foveal_Corrected_Dilation_Left']
            phys_data_cleaned['Foveal_Corrected_Dilation_Right_CLEANED_ABS'] = phys_data_cleaned['Foveal_Corrected_Dilation_Right']
            phys_data_cleaned['Inter_Blink_Interval_CLEANED_ABS'] = phys_data_cleaned['Inter_Blink_Interval']
            phys_data_cleaned['Current_Blink_Duration_CLEANED_ABS'] = phys_data_cleaned['Current_Blink_Duration']
        
        # STEP 3: Feature extraction
        logging.info("STEP 3: Extracting physiological features...")
        physio_features = extract_all_features(
            phys_data_cleaned, 
            gsr_cleaned,
            eeg_data,
            participants,
            parallel=args.parallel
        )
        
        # STEP 4: Merge with EEG and subjective data
        logging.info("STEP 4: Merging physio features with EEG and subjective data...")
        final_data = merge_physio_with_eeg(
            physio_features,
            eeg_data,
            subjective_data
        )
        
        # Validate final merged data
        logging.info("Validating final merged dataset...")
        merge_validation = validate_merged_data(final_data)
        if not merge_validation['valid']:
            logging.warning("Merged data validation found issues - check warnings above")
        
        # STEP 5: Export
        logging.info(f"STEP 5: Exporting final dataset to {args.output}...")
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        final_data.to_csv(args.output, index=False)
        
        logging.info("=" * 80)
        logging.info("✅ Physiological feature extraction completed successfully!")
        logging.info(f"Output saved to: {args.output}")
        logging.info(f"Log file: {log_file}")
        logging.info("=" * 80)
        
        return 0
    
    except Exception as e:
        logging.error(f"Pipeline failed with error: {e}", exc_info=True)
        print(f"\n[ERROR] {e}")
        print(f"Check log file for details: {log_file}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
