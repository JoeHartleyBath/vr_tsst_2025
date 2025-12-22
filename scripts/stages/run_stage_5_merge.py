#!/usr/bin/env python
"""Stage 5: Merge EEG, physio, and subjective features"""
import subprocess
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

def main():
    """Run multimodal feature fusion."""
    script = Path(__file__).parent.parent.parent / "pipelines" / "05_multimodal_fusion" / "merge_all_features.py"
    
    if not script.exists():
        logger.error(f"Script not found: {script}")
        return 1
    
    logger.info(f"Running Stage 5: Multimodal feature fusion")
    logger.info(f"Script: {script}")
    
    result = subprocess.run(
        ["python", str(script)],
        cwd=Path(__file__).parent.parent.parent,
        capture_output=False
    )
    
    return result.returncode

if __name__ == "__main__":
    exit(main())
