#!/usr/bin/env python
"""
Pipeline health check: Validates all dependencies and data before running.
"""
import sys
import subprocess
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

def check_python_packages():
    """Verify Python packages are installed."""
    logger.info("\n[1/5] Checking Python packages...")
    required = ["pyxdf", "mne", "numpy", "pandas", "yaml", "scipy", "neurokit2"]
    missing = []
    
    for pkg in required:
        try:
            __import__(pkg)
            logger.info(f"  ✓ {pkg}")
        except ImportError:
            logger.warning(f"  ✗ {pkg} (missing)")
            missing.append(pkg)
    
    if missing:
        logger.error(f"Install with: pip install {' '.join(missing)}")
        return False
    return True

def check_r_installation():
    """Verify R is installed."""
    logger.info("\n[2/5] Checking R installation...")
    try:
        result = subprocess.run(
            ["Rscript", "--version"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            version = result.stderr.split('\n')[0]
            logger.info(f"  ✓ {version}")
            return True
    except FileNotFoundError:
        logger.warning("  ✗ Rscript not found in PATH")
    except Exception as e:
        logger.warning(f"  ✗ Error: {e}")
    
    logger.error("  Install R or add C:/Program Files/R/R-X.X.X/bin to PATH")
    return False

def check_matlab_installation():
    """Verify MATLAB is installed."""
    logger.info("\n[3/5] Checking MATLAB installation...")
    try:
        result = subprocess.run(
            ["matlab", "-v"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            logger.info(f"  ✓ MATLAB found")
            return True
    except FileNotFoundError:
        logger.warning("  ✗ MATLAB not found in PATH")
    except Exception as e:
        logger.warning(f"  ✗ Error: {e}")
    
    logger.error("  Install MATLAB or add to PATH (e.g., C:/Program Files/MATLAB/R2023a/bin)")
    return False

def check_eeglab_installation():
    """Verify EEGLAB toolbox exists."""
    logger.info("\n[4/5] Checking EEGLAB installation...")
    eeglab_paths = [
        Path("c:/MATLAB/toolboxes/eeglab"),
        Path("C:/MATLAB/toolboxes/eeglab"),
        Path("C:/Program Files/MATLAB/R2023a/toolbox/eeglab")
    ]
    
    for path in eeglab_paths:
        if (path / "eeglab.m").exists():
            logger.info(f"  ✓ Found at {path}")
            
            # Check AMICA
            amica_path = Path("c:/MATLAB/toolboxes/amica")
            if (amica_path / "amicarunner.m").exists():
                logger.info(f"  ✓ AMICA found at {amica_path}")
            else:
                logger.warning(f"  ⚠ AMICA not found (stage 2 will fail)")
                logger.warning(f"    Expected: {amica_path}/amicarunner.m")
            
            return True
    
    logger.warning(f"  ✗ EEGLAB not found")
    logger.error(f"  Expected one of: {eeglab_paths}")
    return False

def check_raw_data():
    """Verify raw data is staged."""
    logger.info("\n[5/5] Checking raw data staging...")
    
    eeg_dir = Path("data/raw/eeg")
    meta_dir = Path("data/raw/metadata")
    subj_dir = Path("data/raw/subjective")
    
    checks = {
        "EEG files": (eeg_dir, "*.xdf"),
        "Metadata files": (meta_dir, "*.csv"),
        "Subjective files": (subj_dir, "*.csv")
    }
    
    all_ok = True
    for name, (dir_path, pattern) in checks.items():
        files = list(dir_path.glob(pattern)) if dir_path.exists() else []
        if files:
            logger.info(f"  ✓ {name}: {len(files)} files")
        else:
            logger.warning(f"  ✗ {name}: not found or empty ({dir_path})")
            all_ok = False
    
    return all_ok

def main():
    """Run all checks and report status."""
    logger.info("=" * 70)
    logger.info("PIPELINE HEALTH CHECK")
    logger.info("=" * 70)
    
    checks = [
        ("Python packages", check_python_packages),
        ("R installation", check_r_installation),
        ("MATLAB installation", check_matlab_installation),
        ("EEGLAB installation", check_eeglab_installation),
        ("Raw data staging", check_raw_data),
    ]
    
    results = []
    for name, check_fn in checks:
        try:
            results.append((name, check_fn()))
        except Exception as e:
            logger.error(f"Error checking {name}: {e}")
            results.append((name, False))
    
    logger.info("\n" + "=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)
    
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        logger.info(f"  {status}: {name}")
    
    passed_count = sum(1 for _, p in results if p)
    total_count = len(results)
    
    logger.info("=" * 70)
    
    if passed_count == total_count:
        logger.info(f"✓ All checks passed! Ready to run pipeline.")
        logger.info(f"\nNext: python scripts/run_pipeline_master.py")
        return 0
    else:
        logger.error(f"✗ {total_count - passed_count} check(s) failed. Fix issues above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
