"""
Physiological Feature Extraction Pipeline (Refactored)

Main orchestration script for extracting heart rate, GSR, pupil, and blink features
from VR-TSST experiment physiological data and merging with EEG features.

Usage:
    python extract_physio_features.py                              # All participants
    python extract_physio_features.py --participants 1 2 3         # Specific participants
    python extract_physio_features.py --parallel                    # Enable parallel processing
    python extract_physio_features.py --output custom_output.csv   # Custom output path

Requirements:
    - Must be run from project root: c:/vr_tsst_2025/
    - Config file: config/general.yaml
    - Input data: data/raw/metadata/P01.csv through P48.csv
    - EEG features: output/aggregated/eeg_features.csv
    - Python packages: pandas, numpy, yaml, neurokit2, tqdm

Outputs:
    - output/aggregated/physio_features.csv         # Standalone physio features
    - output/aggregated/all_data_aggregated.csv     # Merged with EEG & subjective
    - output/cache/phys_data_raw.pkl                # Cached raw data
    - logs/physio_extraction_<timestamp>.log        # Processing log
    - output/qc/physio/P##_qc.txt                   # Per-participant QC logs

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
    validate_loaded_data,
    load_cached_cleaned_data,
    save_cleaned_data_cache
)

# Import cleaning modules
from private.clean_hr_data import clean_hr_pipeline
from private.clean_gsr_data import clean_gsr_pipeline, resample_gsr_to_10hz
from private.clean_eye_data import clean_eye_pipeline

# Import feature extraction module
from private.extract_features import extract_all_features
from private.assign_conditions import load_conditions_config, assign_conditions_to_dataframe


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
        default='output/aggregated/physio_features.csv',
        help='Output CSV file path (default: output/aggregated/physio_features.csv)'
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
    
    # Check for existing output and determine participants to process
    physio_output_path = 'output/aggregated/physio_features.csv'
    
    if os.path.exists(physio_output_path) and not args.force_reprocess:
        logging.info(f"Found existing output: {physio_output_path}")
        existing_df = pd.read_csv(physio_output_path)
        processed_participants = set(existing_df['Participant_ID'].unique())
        logging.info(f"  Already processed: {len(processed_participants)} participants")
    else:
        processed_participants = set()
        if args.force_reprocess:
            logging.info("Force reprocess flag set - will reprocess all participants")
    
    # Determine participants to process
    if args.participants:
        requested_participants = args.participants
        participants = [p for p in requested_participants if p not in processed_participants]
        if len(participants) < len(requested_participants):
            skipped = set(requested_participants) - set(participants)
            logging.info(f"Skipping already processed participants: {sorted(skipped)}")
        logging.info(f"Processing specific participants: {participants}")
    else:
        participants = [p for p in range(1, 49) if p not in processed_participants]
        logging.info(f"Processing {len(participants)} participants (skipping {len(processed_participants)} already completed)")
    
    # Exit if nothing to process
    if not participants:
        logging.info("=" * 80)
        logging.info("✅ All participants already processed!")
        logging.info(f"Output file: {physio_output_path}")
        logging.info("Use --force-reprocess to reprocess all participants")
        logging.info("=" * 80)
        return 0
    
    try:
        # STEP 0: Load configuration
        logging.info("STEP 0: Loading configuration...")
        config = load_config()
        data_path = config["paths"]["raw_data"]
        
        # STEP 1: Load EEG and subjective data once (shared across participants)
        logging.info("STEP 1: Loading EEG and subjective data...")
        
        logging.info("Loading EEG features...")
        eeg_data = load_eeg_features(config, force_reload=args.force_reprocess)
        logging.info(f"  Loaded {len(eeg_data)} EEG rows")
        
        # Load condition configuration for physio data assignment
        logging.info("Loading condition configuration...")
        conditions_config = load_conditions_config()
        logging.info(f"  Loaded {len(conditions_config)} condition definitions")
        
        logging.info("Loading subjective ratings...")
        try:
            subjective_data = load_subjective_ratings(config, force_reload=args.force_reprocess)
            logging.info(f"  Loaded {len(subjective_data)} subjective rows")
        except FileNotFoundError as e:
            logging.warning(f"Subjective ratings file not found: {e}")
            logging.warning("Continuing without subjective ratings")
            subjective_data = None
        
        # Setup QC loggers
        logging.info("Setting up QC loggers...")
        qc_loggers = setup_qc_loggers(config, participants)
        
        # STEP 2 & 3: Process each participant individually
        logging.info(f"STEP 2-3: Processing {len(participants)} participants individually...")
        all_features = []
        
        for i, pid in enumerate(participants, 1):
            logging.info(f"\n[{i}/{len(participants)}] Processing P{pid:02d}...")
            
            try:
                # Check for cached cleaned data
                cached_cleaned = load_cached_cleaned_data(pid) if not args.force_reprocess else None
                
                if cached_cleaned is not None:
                    logging.info(f"  Using cached cleaned data for P{pid:02d}")
                    phys_data_cleaned = cached_cleaned
                    gsr_cleaned = cached_cleaned
                else:
                    # Load raw data for this participant only
                    logging.info(f"  Loading raw data for P{pid:02d}...")
                    phys_data_raw = load_raw_physio_data(
                        data_path,
                        filename_filter='P',
                        participants=[pid],
                        force_reload=True
                    )
                    
                    # Assign conditions from config
                    phys_data_raw = assign_conditions_to_dataframe(phys_data_raw, conditions_config)
                    
                    if len(phys_data_raw) == 0:
                        logging.warning(f"  No data found for P{pid:02d}, skipping")
                        continue
                    
                    # Signal cleaning
                    if not args.skip_cleaning:
                        logging.info(f"  Cleaning signals for P{pid:02d}...")
                        phys_data_cleaned = clean_hr_pipeline(phys_data_raw, qc_loggers)
                        
                        gsr_resampled = resample_gsr_to_10hz(
                            phys_data_raw,
                            gsr_cols=['Shimmer_D36A_GSR_Skin_Conductance_uS',
                                     'Shimmer_D36A_GSR_Skin_Resistance_kOhms']
                        )
                        gsr_cleaned = clean_gsr_pipeline(gsr_resampled, qc_loggers)
                        phys_data_cleaned = clean_eye_pipeline(phys_data_cleaned, qc_loggers)
                    else:
                        phys_data_cleaned = phys_data_raw.copy()
                        gsr_cleaned = phys_data_raw.copy()
                        # Create cleaned column aliases
                        phys_data_cleaned['Polar_HeartRate_BPM_CLEANED_ABS'] = phys_data_cleaned['Polar_HeartRate_BPM']
                        phys_data_cleaned['Polar_HeartRate_RR_Interval_CLEANED_ABS'] = phys_data_cleaned['Polar_HeartRate_RR_Interval']
                        gsr_cleaned['Shimmer_D36A_GSR_Skin_Conductance_uS_CLEANED_ABS_CLEANED_NK'] = gsr_cleaned['Shimmer_D36A_GSR_Skin_Conductance_uS']
                        phys_data_cleaned['Foveal_Corrected_Dilation_Left_CLEANED_ABS'] = phys_data_cleaned['Foveal_Corrected_Dilation_Left']
                        phys_data_cleaned['Foveal_Corrected_Dilation_Right_CLEANED_ABS'] = phys_data_cleaned['Foveal_Corrected_Dilation_Right']
                        phys_data_cleaned['Inter_Blink_Interval_CLEANED_ABS'] = phys_data_cleaned['Inter_Blink_Interval']
                        phys_data_cleaned['Current_Blink_Duration_CLEANED_ABS'] = phys_data_cleaned['Current_Blink_Duration']
                    
                    # Cache cleaned data
                    save_cleaned_data_cache(pid, phys_data_cleaned)
                
                # Extract features for this participant
                logging.info(f"  Extracting features for P{pid:02d}...")
                participant_features = extract_all_features(
                    phys_data_cleaned,
                    gsr_cleaned,
                    eeg_data,
                    [pid],  # Process just this participant
                    parallel=False
                )
                
                all_features.append(participant_features)
                logging.info(f"  ✓ P{pid:02d} complete ({len(participant_features)} rows)")
                
            except Exception as e:
                logging.error(f"  ✗ Failed to process P{pid:02d}: {e}")
                logging.error(f"  {e}", exc_info=True)
                continue
        
        # Combine all participant features
        if not all_features:
            logging.error("No participants were successfully processed")
            return 1
        
        physio_features = pd.concat(all_features, ignore_index=True)
        logging.info(f"\nExtracted features for {len(participants)} participants ({len(physio_features)} total rows)")
        
        # Save standalone physio features (append mode for incremental processing)
        physio_output_path = 'output/aggregated/physio_features.csv'
        os.makedirs(os.path.dirname(physio_output_path), exist_ok=True)
        
        if os.path.exists(physio_output_path) and not args.force_reprocess:
            # Append to existing file
            existing_df = pd.read_csv(physio_output_path)
            combined_df = pd.concat([existing_df, physio_features], ignore_index=True)
            # Remove duplicates (in case of overlap)
            combined_df = combined_df.drop_duplicates(subset=['Participant_ID', 'Condition'], keep='last')
            combined_df.to_csv(physio_output_path, index=False)
            logging.info(f"  Appended {len(physio_features)} rows to existing physio features")
            logging.info(f"  Total rows now: {len(combined_df)}")
        else:
            # Create new file
            physio_features.to_csv(physio_output_path, index=False)
            logging.info(f"  Created new physio features file: {physio_output_path}")
        
        logging.info(f"  Physio features saved to: {physio_output_path}")
        
        logging.info("=" * 80)
        logging.info("✅ Physiological feature extraction completed successfully!")
        logging.info(f"Physio features: {physio_output_path}")
        logging.info(f"Processed participants: {sorted(participants)}")
        logging.info("=" * 80)
        logging.info("Next step: Run merge script to combine with EEG and subjective data")
        logging.info("  python pipelines/05_multimodal_fusion/merge_all_features.py")
        logging.info("=" * 80)
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
