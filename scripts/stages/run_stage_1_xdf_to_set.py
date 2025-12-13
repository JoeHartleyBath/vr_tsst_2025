#!/usr/bin/env python
"""Stage 1: Convert XDF to SET format (EEGLAB)"""
import subprocess
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

def main():
    """Run XDF to SET conversion for all participants."""
    script = Path(__file__).parent.parent / "preprocessing" / "raw_conversion" / "run" / "run_xdf_to_set_end2end.py"
    
    if not script.exists():
        logger.error(f"Script not found: {script}")
        return 1
    
    logger.info(f"Running Stage 1: XDF → SET conversion")
    logger.info(f"Script: {script}")
    
    result = subprocess.run(
        ["python", str(script)],
        cwd=Path(__file__).parent.parent.parent,
        capture_output=False
    )
    
    return result.returncode

if __name__ == "__main__":
    exit(main())
