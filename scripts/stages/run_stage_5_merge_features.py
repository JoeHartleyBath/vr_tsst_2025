#!/usr/bin/env python
"""Stage 5: Merge EEG and physio features into final dataset"""
import subprocess
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

def main():
    """Run feature merge pipeline."""
    script = Path(__file__).parent.parent / "preprocessing" / "physio" / "feature_extraction" / "mvp_merge_pipeline.py"
    
    if not script.exists():
        logger.error(f"Script not found: {script}")
        logger.info("Checking what exists in physio/feature_extraction/...")
        physio_dir = Path(__file__).parent.parent / "preprocessing" / "physio" / "feature_extraction"
        if physio_dir.exists():
            logger.info(f"Contents: {list(physio_dir.glob('*.py'))}")
        return 1
    
    logger.info(f"Running Stage 5: Feature merge")
    logger.info(f"Script: {script}")
    
    result = subprocess.run(
        ["python", str(script)],
        cwd=Path(__file__).parent.parent.parent,
        capture_output=False
    )
    
    return result.returncode

if __name__ == "__main__":
    exit(main())
