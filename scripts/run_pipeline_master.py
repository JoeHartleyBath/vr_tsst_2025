#!/usr/bin/env python
"""
Pipeline orchestrator for VR-TSST data processing.
Runs stages 1–6 (raw data → ML-ready features) with logging and error handling.
Usage: python scripts/run_pipeline_master.py [--participants 1 2 3] [--stages 1 2 3 4 5 6]
"""
import sys
import subprocess
import logging
from pathlib import Path
from datetime import datetime
import argparse
import json

# Setup logging
log_dir = Path("output/logs")
log_dir.mkdir(parents=True, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = log_dir / f"pipeline_{timestamp}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Stage definitions
STAGES = {
    1: {
        "name": "XDF → SET",
        "script": "scripts/preprocessing/raw_conversion/run/run_xdf_to_set_end2end.py",
        "type": "python",
        "duration_est": "5-10 min (P01-P03)"
    },
    2: {
        "name": "EEG Cleaning (AMICA)",
        "script": "scripts/preprocessing/eeg/cleaning/run_clean_eeg_pipeline.m",
        "type": "matlab",
        "duration_est": "2-3 hours (P01-P03, AMICA heavy)"
    },
    3: {
        "name": "EEG Feature Extraction",
        "script": "scripts/preprocessing/eeg/feature_extraction/extract_eeg_features.m",
        "type": "matlab",
        "duration_est": "15-30 min"
    },
    4: {
        "name": "Physio Feature Extraction",
        "script": "scripts/preprocessing/physio/feature_extraction/extract_physio_features.py",
        "type": "python",
        "duration_est": "10-20 min"
    },
    5: {
        "name": "Merge Features",
        "script": "scripts/preprocessing/physio/feature_extraction/mvp_merge_pipeline.py",
        "type": "python",
        "duration_est": "<1 min"
    },
    6: {
        "name": "R Preprocessing (Final data prep)",
        "script": "scripts/preproccess_for_xgb.R",
        "type": "r",
        "duration_est": "5-10 min"
    }
}

def run_stage(stage_num, stage_info):
    """Run a single pipeline stage."""
    logger.info(f"\n{'='*70}")
    logger.info(f"STAGE {stage_num}: {stage_info['name']}")
    logger.info(f"Est. duration: {stage_info['duration_est']}")
    logger.info(f"{'='*70}\n")
    
    script = stage_info['script']
    stage_type = stage_info['type']
    
    if not Path(script).exists():
        logger.error(f"Script not found: {script}")
        return False
    
    try:
        if stage_type == "python":
            cmd = [sys.executable, script]
            logger.info(f"Running: {' '.join(cmd)}")
            result = subprocess.run(cmd, check=True, capture_output=False)
        
        elif stage_type == "matlab":
            # Run MATLAB batch
            matlab_script = f"cd('{Path.cwd()}'); addpath(genpath('scripts')); run('{script}'); exit;"
            cmd = ["matlab", "-batch", matlab_script]
            logger.info(f"Running MATLAB: {script}")
            result = subprocess.run(cmd, check=True, capture_output=False)
        
        elif stage_type == "r":
            cmd = ["Rscript", script]
            logger.info(f"Running: {' '.join(cmd)}")
            result = subprocess.run(cmd, check=True, capture_output=False)
        
        logger.info(f"✓ Stage {stage_num} completed successfully")
        return True
    
    except subprocess.CalledProcessError as e:
        logger.error(f"✗ Stage {stage_num} failed with exit code {e.returncode}")
        logger.error(f"Error: {e}")
        return False
    except Exception as e:
        logger.error(f"✗ Stage {stage_num} failed with exception: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(
        description="Run VR-TSST pipeline stages 1-6 (raw → ML-ready)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/run_pipeline_master.py                    # Run all stages
  python scripts/run_pipeline_master.py --stages 1 4 5     # Run specific stages
  python scripts/run_pipeline_master.py --stages 1-3       # Run stages 1, 2, 3
        """
    )
    parser.add_argument(
        "--stages",
        nargs="+",
        type=str,
        default=["1", "2", "3", "4", "5", "6"],
        help="Space-separated list of stages to run (e.g., '1 4 5' or '1-3')"
    )
    
    args = parser.parse_args()
    
    # Parse stage input (handle ranges and lists)
    stage_list = []
    for item in args.stages:
        if isinstance(item, str) and "-" in item:
            start, end = map(int, item.split("-"))
            stage_list.extend(range(start, end + 1))
        elif isinstance(item, str) and item.strip():
            stage_list.append(int(item.strip()))
        elif isinstance(item, int):
            stage_list.append(item)
    
    stage_list = sorted(set(stage_list))  # Remove duplicates, sort
    
    # Validate stages
    invalid = [s for s in stage_list if s not in STAGES]
    if invalid:
        logger.error(f"Invalid stages: {invalid}. Valid stages: {list(STAGES.keys())}")
        sys.exit(1)
    
    logger.info(f"\n{'='*70}")
    logger.info(f"VR-TSST PIPELINE ORCHESTRATOR")
    logger.info(f"{'='*70}")
    logger.info(f"Stages to run: {stage_list}")
    logger.info(f"Log file: {log_file}")
    logger.info(f"{'='*70}\n")
    
    # Run stages
    results = {}
    for stage_num in stage_list:
        success = run_stage(stage_num, STAGES[stage_num])
        results[stage_num] = "✓ OK" if success else "✗ FAILED"
        
        if not success:
            print(f"\nStage {stage_num} failed. Continue? (y/n): ", end="", flush=True)
            if input().lower() != "y":
                logger.info("Pipeline stopped by user.")
                break
    
    # Summary
    logger.info(f"\n{'='*70}")
    logger.info("PIPELINE SUMMARY")
    logger.info(f"{'='*70}")
    for stage_num in sorted(results.keys()):
        logger.info(f"Stage {stage_num}: {results[stage_num]}")
    logger.info(f"{'='*70}\n")
    
    # Save results to JSON
    results_file = log_dir / f"pipeline_results_{timestamp}.json"
    with open(results_file, "w") as f:
        json.dump({
            "timestamp": timestamp,
            "stages": stage_list,
            "results": results,
            "log_file": str(log_file)
        }, f, indent=2)
    logger.info(f"Results saved to: {results_file}")

if __name__ == "__main__":
    main()
