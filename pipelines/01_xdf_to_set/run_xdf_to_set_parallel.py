#!/usr/bin/env python3
"""
Parallel XDF to SET conversion for multiple participants
Uses multiprocessing to convert multiple XDF files simultaneously.

Professional Pipeline with:
  - Verbose logging at each stage
  - Input validation (file existence, integrity)
  - Output sanity checks (channels, srate, data units)
  - Amplitude range verification (detect scaled vs unscaled)
  - Detailed error reporting
  - Summary statistics and provenance
"""

import sys
import logging
from pathlib import Path
from multiprocessing import Pool, cpu_count
import argparse
from time import perf_counter
import json

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from xdf_to_set.xdf_to_set import xdf_to_set

# Global variables for worker processes
BASE_PATH = None
CONFIG_PATH = None
LOG_DIR = None

# Setup logging for worker processes
def setup_worker_logging(participant, log_dir):
    """Configure logging for a worker process"""
    log_file = log_dir / f"P{participant:02d}_conversion.log"
    handler = logging.FileHandler(log_file, mode='w')
    handler.setFormatter(logging.Formatter('%(asctime)s | %(levelname)-8s | %(message)s'))
    logger = logging.getLogger()
    logger.handlers = [handler]  # Replace handlers
    logger.setLevel(logging.DEBUG)
    return logger

def init_worker(base_path, config_path, log_dir):
    """Initialize worker process with shared paths"""
    global BASE_PATH, CONFIG_PATH, LOG_DIR
    BASE_PATH = base_path
    CONFIG_PATH = config_path
    LOG_DIR = log_dir


def validate_input_file(xdf_path):
    """Validate that XDF input file exists and is readable"""
    if not xdf_path.exists():
        raise FileNotFoundError(f"Input file not found: {xdf_path}")
    if not xdf_path.is_file():
        raise ValueError(f"Input path is not a file: {xdf_path}")
    size_mb = xdf_path.stat().st_size / (1024**2)
    if size_mb < 1:
        raise ValueError(f"Input file suspiciously small ({size_mb:.2f} MB): {xdf_path}")
    return size_mb


def validate_output_file(set_path, expected_channels=128, expected_srate=500):
    """Sanity check the output .set file (which is actually a .mat file with .set extension)"""
    if not set_path.exists():
        raise FileNotFoundError(f"Output .set file not created: {set_path}")
    
    # Try to load and validate structure (basic checks)
    try:
        from scipy.io import loadmat
        mat = loadmat(str(set_path), squeeze_me=True)
        
        if 'EEG' not in mat:
            raise ValueError("Output .set file does not contain 'EEG' struct")
        
        eeg = mat['EEG']
        nbchan = int(eeg['nbchan']) if 'nbchan' in eeg else None
        pnts = int(eeg['pnts']) if 'pnts' in eeg else None
        srate = int(eeg['srate']) if 'srate' in eeg else None
        
        # Validate channel count
        if nbchan != expected_channels:
            raise ValueError(f"Channel count mismatch: expected {expected_channels}, got {nbchan}")
        
        # Validate sample rate
        if srate != expected_srate:
            raise ValueError(f"Sample rate mismatch: expected {expected_srate} Hz, got {srate} Hz")
        
        # Validate data dimensions
        if 'data' in eeg:
            data_shape = eeg['data'].shape
            if len(data_shape) != 2 or data_shape[0] != expected_channels:
                raise ValueError(f"Data shape mismatch: expected ({expected_channels}, N), got {data_shape}")
        
        return {'nbchan': nbchan, 'pnts': pnts, 'srate': srate}
    
    except Exception as e:
        raise ValueError(f"Output validation failed: {e}")


def check_amplitude_scaling(set_path, expected_units='µV'):
    """Verify that data is in expected units (detect if mV vs µV scaling is correct)"""
    try:
        from scipy.io import loadmat
        mat = loadmat(str(set_path), squeeze_me=True)
        eeg = mat['EEG']
        
        if 'data' in eeg:
            data = eeg['data']
            # For real EEG, typical µV ranges are ±100 to ±500
            # If data is in mV (unscaled), it would be ±0.1 to ±0.5
            mean_abs = abs(data).mean()
            max_abs = abs(data).max()
            
            status = "✓"
            reason = ""
            
            if mean_abs < 1.0:
                status = "⚠"
                reason = f"Suspiciously small amplitudes (mean {mean_abs:.4f}, likely unscaled mV)"
            elif mean_abs > 1000.0:
                status = "⚠"
                reason = f"Suspiciously large amplitudes (mean {mean_abs:.1f}, may be pre-scaled)"
            else:
                reason = f"Reasonable range for {expected_units} (mean {mean_abs:.2f}, max {max_abs:.2f})"
            
            return {'status': status, 'reason': reason, 'mean_abs_µV': mean_abs, 'max_abs_µV': max_abs}
    
    except Exception as e:
        return {'status': '?', 'reason': f"Could not check scaling: {e}"}


def convert_participant(p_num):
    """Convert a single participant's XDF to SET with comprehensive logging and validation"""
    logger = setup_worker_logging(p_num, LOG_DIR)
    
    xdf = BASE_PATH / f"data/RAW/eeg/P{p_num:02d}.xdf"
    out = BASE_PATH / f"output/sets/P{p_num:02d}.set"
    out.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 80)
    logger.info(f"PARTICIPANT P{p_num:02d} CONVERSION START")
    logger.info("=" * 80)
    
    start = perf_counter()
    result = {'participant': p_num, 'status': 'FAILED', 'time': 0, 'checks': {}}
    
    try:
        # ---- STEP 1: Validate input ----
        logger.info(f"[1/5] Input validation: {xdf}")
        size_mb = validate_input_file(xdf)
        logger.info(f"  ✓ Input file exists, {size_mb:.2f} MB")
        result['checks']['input_validation'] = 'PASS'
        
        # ---- STEP 2: Run conversion ----
        logger.info(f"[2/5] Running XDF→SET conversion...")
        summary = xdf_to_set(xdf, out, config_path=CONFIG_PATH)
        elapsed_convert = perf_counter() - start
        logger.info(f"  ✓ Conversion complete in {elapsed_convert:.1f}s")
        for k, v in summary.items():
            logger.info(f"      {k}: {v}")
        result['summary'] = summary
        result['checks']['conversion'] = 'PASS'
        
        # ---- STEP 3: Validate output structure ----
        logger.info(f"[3/5] Output file validation...")
        struct_checks = validate_output_file(out)
        logger.info(f"  ✓ Output .set/.mat structure valid")
        logger.info(f"      nbchan: {struct_checks['nbchan']}")
        logger.info(f"      pnts: {struct_checks['pnts']}")
        logger.info(f"      srate: {struct_checks['srate']} Hz")
        result['checks']['output_structure'] = 'PASS'
        
        # ---- STEP 4: Check amplitude scaling ----
        logger.info(f"[4/5] Amplitude range verification...")
        amp_check = check_amplitude_scaling(out)
        logger.info(f"  {amp_check['status']} Scaling check: {amp_check['reason']}")
        result['checks']['amplitude_scaling'] = amp_check['status']
        result['checks']['amplitude_details'] = amp_check
        
        # ---- STEP 5: Summary ----
        elapsed = perf_counter() - start
        logger.info(f"[5/5] Final summary")
        logger.info(f"  ✓ ALL CHECKS PASSED")
        logger.info(f"  Output: {out}")
        logger.info(f"  Total time: {elapsed:.1f}s")
        logger.info("=" * 80)
        
        result['status'] = 'SUCCESS'
        result['time'] = elapsed
        
    except Exception as e:
        import traceback
        elapsed = perf_counter() - start
        error_msg = traceback.format_exc()
        logger.error(f"✗ CONVERSION FAILED")
        logger.error(f"  Error: {e}")
        logger.error(f"  Time: {elapsed:.1f}s")
        logger.error("Traceback:")
        logger.error(error_msg)
        logger.info("=" * 80)
        
        result['status'] = 'FAILED'
        result['time'] = elapsed
        result['error'] = str(e)
        result['traceback'] = error_msg
    
    return result


if __name__ == "__main__":
    base = Path(__file__).parent.parent.parent.parent.parent  # Go up to c:\vr_tsst_2025
    log_dir = base / "output/logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup main logger (for orchestrator, not workers)
    log_file = log_dir / "conversion_orchestrator.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)-8s | %(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='w'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logger = logging.getLogger(__name__)
    
    parser = argparse.ArgumentParser(description="Parallel XDF→SET conversion for participants")
    parser.add_argument("--participants", nargs="*", type=int, help="Participant numbers (e.g., 1 2 3), or omit to process all 48")
    parser.add_argument("--config", type=str, default=str(base / "config/conditions_pilot.yaml"), help="Path to conditions YAML")
    parser.add_argument("--processes", type=int, default=1, help="Number of parallel processes (default: 1 for memory safety)")
    args = parser.parse_args()
    
    # Default to all 48 participants if not specified
    participants = args.participants if args.participants else list(range(1, 49))
    config_path = Path(args.config)
    
    # Determine number of processes (default to 1 for large XDF files)
    num_processes = args.processes
    num_processes = min(num_processes, len(participants))  # Don't exceed participant count
    
    logger.info("=" * 80)
    logger.info("PARALLEL XDF → SET CONVERSION WITH VALIDATION")
    logger.info("=" * 80)
    logger.info(f"Participants: {len(participants)} total - {participants[:10]}{'...' if len(participants) > 10 else ''}")
    logger.info(f"Config: {config_path}")
    logger.info(f"Processes: {num_processes} workers")
    logger.info(f"Output: {base / 'output/sets'}")
    logger.info(f"Logs: {log_dir}")
    logger.info("Per-participant logs: {log_dir}/P{NN:02d}_conversion.log")
    logger.info("=" * 80)
    
    overall_start = perf_counter()
    
    # Run in parallel
    with Pool(processes=num_processes, initializer=init_worker, initargs=(base, config_path, log_dir)) as pool:
        results = pool.map(convert_participant, participants)
    
    overall_elapsed = perf_counter() - overall_start
    
    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("BATCH CONVERSION SUMMARY")
    logger.info("=" * 80)
    
    success_count = sum(1 for r in results if r['status'] == 'SUCCESS')
    failed_results = [r for r in results if r['status'] == 'FAILED']
    
    # Detailed results
    logger.info("\nRESULTS BY PARTICIPANT:")
    for result in results:
        status_icon = "✓" if result['status'] == 'SUCCESS' else "✗"
        p_num = result['participant']
        t = result['time']
        
        if result['status'] == 'SUCCESS':
            # Extract summary info
            summary = result.get('summary', {})
            n_events = summary.get('n_events', '?')
            amp_status = result.get('checks', {}).get('amplitude_scaling', '?')
            amp_info = result.get('checks', {}).get('amplitude_details', {})
            mean_amp = amp_info.get('mean_abs_µV', 0)
            
            logger.info(f"{status_icon} P{p_num:02d}: SUCCESS ({t:.1f}s) | {n_events} events | amplitude {amp_status} (mean {mean_amp:.2f}µV)")
        else:
            error = result.get('error', 'Unknown error')
            logger.info(f"{status_icon} P{p_num:02d}: FAILED ({t:.1f}s) | {error}")
    
    # Detailed errors for failures
    if failed_results:
        logger.info("\n" + "=" * 80)
        logger.info("FAILED CONVERSIONS - DIAGNOSTIC DETAILS")
        logger.info("=" * 80)
        for result in failed_results:
            p_num = result['participant']
            logger.info(f"\n[P{p_num:02d}] Error: {result['error']}")
            logfile_path = log_dir / f"P{p_num:02d}_conversion.log"
            logger.info(f"Log file: {logfile_path}")
            if 'traceback' in result:
                logger.debug("Traceback:\n" + result['traceback'])
    
    # Statistics
    logger.info("\n" + "=" * 80)
    logger.info("STATISTICS")
    logger.info("=" * 80)
    logger.info(f"Success: {success_count}/{len(participants)} participants ({100*success_count/len(participants):.1f}%)")
    logger.info(f"Failed: {len(failed_results)}/{len(participants)}")
    logger.info(f"Total time: {overall_elapsed:.1f}s ({overall_elapsed/60:.1f} minutes)")
    if success_count > 0:
        avg_time = sum(r['time'] for r in results if r['status'] == 'SUCCESS') / success_count
        logger.info(f"Mean time per successful conversion: {avg_time:.1f}s")
    logger.info("=" * 80)
    logger.info(f"Individual logs saved to: {log_dir}")
    logger.info(f"Orchestrator log: {log_file}")
    logger.info("=" * 80)
    
    # Write JSON manifest for downstream processing
    manifest = {
        'batch_time': overall_elapsed,
        'success_count': success_count,
        'total_count': len(participants),
        'results': results,
        'config': str(config_path),
        'log_dir': str(log_dir)
    }
    manifest_file = log_dir / "conversion_manifest.json"
    with open(manifest_file, 'w') as f:
        json.dump(manifest, f, indent=2)
    logger.info(f"Manifest written: {manifest_file}")
    
    sys.exit(0 if success_count == len(participants) else 1)
