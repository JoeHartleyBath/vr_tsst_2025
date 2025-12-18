"""
Multimodal Feature Fusion Script

Merges EEG, physiological, and subjective features into a single unified dataset.

Usage:
    python pipelines/05_multimodal_fusion/merge_all_features.py
    python pipelines/05_multimodal_fusion/merge_all_features.py --output custom_output.csv
    python pipelines/05_multimodal_fusion/merge_all_features.py --force

Requirements:
    - Must be run from project root: c:/vr_tsst_2025/
    - Input files:
        - output/aggregated/eeg_features.csv
        - output/aggregated/physio_features.csv
        - output/aggregated/subjective.csv

Outputs:
    - output/aggregated/all_data_aggregated.csv    # Full merged dataset

Author: VR-TSST Project
Date: December 2025
"""

import os
import sys
import argparse
import logging
import re
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / '04_python_physio_features'))
from private.merge_with_eeg import merge_physio_with_eeg, validate_merged_data


def setup_logging():
    """Configure logging with timestamps."""
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_filename = os.path.join(log_dir, f"multimodal_merge_{timestamp}.log")
    
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


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Merge EEG, physio, and subjective features into unified dataset"
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='output/aggregated/all_data_aggregated.csv',
        help='Output CSV file path (default: output/aggregated/all_data_aggregated.csv)'
    )
    
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force merge even if output file already exists'
    )
    
    parser.add_argument(
        '--eeg',
        type=str,
        default='output/aggregated/eeg_features.csv',
        help='EEG features input path'
    )
    
    parser.add_argument(
        '--physio',
        type=str,
        default='output/aggregated/physio_features.csv',
        help='Physio features input path'
    )
    
    parser.add_argument(
        '--subjective',
        type=str,
        default='output/aggregated/subjective.csv',
        help='Subjective ratings input path'
    )
    
    return parser.parse_args()


def load_input_data(args):
    """Load EEG, physio, and subjective data."""
    
    # Check if all input files exist
    missing_files = []
    for name, path in [('EEG', args.eeg), ('Physio', args.physio), ('Subjective', args.subjective)]:
        if not os.path.exists(path):
            missing_files.append(f"{name}: {path}")
    
    if missing_files:
        raise FileNotFoundError(
            f"Missing required input files:\n" + "\n".join(missing_files)
        )
    
    # Load EEG features
    logging.info(f"Loading EEG features from {args.eeg}")
    eeg_df = pd.read_csv(args.eeg)
    
    # Standardize column name
    if 'Participant' in eeg_df.columns:
        eeg_df.rename(columns={'Participant': 'Participant_ID'}, inplace=True)
    
    # Clean condition names: remove numeric suffixes like 1022, 2043
    # This ensures compatibility between EEG and physio condition names
    if 'Condition' in eeg_df.columns:
        eeg_df['Condition'] = eeg_df['Condition'].apply(
            lambda x: re.sub(r'(\d{4})', '', x) if pd.notna(x) else x
        )
        logging.info("  Cleaned numeric suffixes from EEG condition names")
    
    logging.info(f"  Loaded {len(eeg_df)} rows, {len(eeg_df.columns)} columns")
    logging.info(f"  Participants: {len(eeg_df['Participant_ID'].unique())}")
    
    # Load physio features
    logging.info(f"Loading physio features from {args.physio}")
    physio_df = pd.read_csv(args.physio)
    logging.info(f"  Loaded {len(physio_df)} rows, {len(physio_df.columns)} columns")
    logging.info(f"  Participants: {len(physio_df['Participant_ID'].unique())}")
    
    # Load subjective ratings
    logging.info(f"Loading subjective ratings from {args.subjective}")
    subjective_df = pd.read_csv(args.subjective)
    logging.info(f"  Loaded {len(subjective_df)} rows, {len(subjective_df.columns)} columns")
    logging.info(f"  Participants: {len(subjective_df['Participant_ID'].unique())}")
    
    return eeg_df, physio_df, subjective_df


def main():
    """Main execution pipeline."""
    print("=" * 80)
    print("MULTIMODAL FEATURE FUSION")
    print("=" * 80)
    
    # Setup
    args = parse_arguments()
    log_file = setup_logging()
    
    logging.info("Starting multimodal feature fusion")
    logging.info(f"Arguments: {args}")
    
    # Check if output already exists
    if os.path.exists(args.output) and not args.force:
        logging.info(f"Output file already exists: {args.output}")
        logging.info("Use --force to overwrite")
        print("=" * 80)
        print(f"✅ Output already exists: {args.output}")
        print("Use --force to regenerate")
        print("=" * 80)
        return 0
    
    try:
        # Load input data
        logging.info("Loading input datasets...")
        eeg_df, physio_df, subjective_df = load_input_data(args)
        
        # Merge datasets
        logging.info("Merging datasets...")
        merged_df = merge_physio_with_eeg(
            physio_df,
            eeg_df,
            subjective_df
        )
        
        logging.info(f"Merged dataset shape: {merged_df.shape}")
        logging.info(f"  Rows: {len(merged_df)}")
        logging.info(f"  Columns: {len(merged_df.columns)}")
        logging.info(f"  Participants: {len(merged_df['Participant_ID'].unique())}")
        
        # Validate merged data
        logging.info("Validating merged dataset...")
        validation_result = validate_merged_data(merged_df)
        
        if validation_result['valid']:
            logging.info("✅ Data validation passed")
        else:
            logging.warning("⚠️ Data validation found issues - check warnings above")
        
        # Save output
        logging.info(f"Saving merged dataset to {args.output}...")
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        merged_df.to_csv(args.output, index=False)
        
        # Summary
        logging.info("=" * 80)
        logging.info("✅ Multimodal feature fusion completed successfully!")
        logging.info(f"Output: {args.output}")
        logging.info(f"Shape: {merged_df.shape} (rows × columns)")
        logging.info(f"Participants: {len(merged_df['Participant_ID'].unique())}")
        logging.info("=" * 80)
        
        print("=" * 80)
        print("✅ Merge completed successfully!")
        print(f"Output: {args.output}")
        print(f"Shape: {merged_df.shape}")
        print("=" * 80)
        
        return 0
        
    except Exception as e:
        logging.error(f"Pipeline failed with error: {e}", exc_info=True)
        print(f"\n[ERROR] {e}")
        print(f"Check log file for details: {log_file}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
