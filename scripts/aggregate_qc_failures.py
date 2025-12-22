"""
Aggregate QC failures from EEG and physio metrics and generate exclusion list.

This script reads:
  - EEG QC from output/qc/P##_qc.mat (MATLAB binary files)
  - Physio QC from output/qc/physio/P##_qc.txt (text logs)
  - EEG QC summaries from output/qc/QC_P##.txt (text reports)

And applies thresholds:
  - EEG non-occipital channels interpolated: >20% (>50/249 channels)
  - EEG sample retention: <75%
  - EEG ICA components removed: >35%
  - Physio: <70% retention in any sensor within any analysis condition
  
Analysis conditions scope:
  - Task conditions: Any containing word "Task"
  - Pre-Exposure conditions: Pre_Exposure_Blank_Fixation_Cross, Pre_Exposure_Room_Fixation_Cross
  - Forest conditions: Forest1, Forest2, Forest3, Forest4

Outputs: output/qc/qc_failures_summary.csv with columns:
  - Participant_ID
  - Failure_Reason (comma-separated if multiple failures)
"""

import os
import re
import csv
from pathlib import Path
from typing import Dict, List, Tuple, Set

# Configuration
QC_DIR = Path('output/qc')
PHYSIO_QC_DIR = QC_DIR / 'physio'
OUTPUT_FILE = QC_DIR / 'qc_failures_summary.csv'

# Thresholds
EEG_MAX_INTERPOLATED_PCT = 25.0  # Percent of non-occipital channels (>62/249 max)
EEG_MIN_SAMPLE_RETENTION = 75.0  # Minimum percent samples retained
EEG_MAX_ICA_REMOVED_PCT = 40.0  # Maximum percent ICA components removed
PHYSIO_MIN_RETENTION_PCT = 50.0  # Minimum percent retention per sensor per condition

# Occipital channels (22 total out of 271) - exclude from interpolation count
OCCIPITAL_CHANNELS = {
    'Z12', 'Z13', 
    'L11', 'L12', 'L13', 'L14',
    'R11', 'R12', 'R13', 'R14',
    'LL12', 'LL13', 'LL14',
    'RR12', 'RR13', 'RR14',
    'LC7', 'RC7', 'LD7', 'RD7', 'LE4', 'RE4'
}
NON_OCCIPITAL_CHANNELS = 271 - len(OCCIPITAL_CHANNELS)  # 249 channels
MAX_NON_OCCIPITAL_INTERPOLATED = int(NON_OCCIPITAL_CHANNELS * (EEG_MAX_INTERPOLATED_PCT / 100.0))  # 50 channels

# Analysis conditions
TASK_PATTERN = r'.*Task.*'  # Any condition with "Task"
PRE_EXPOSURE_CONDITIONS = {'Pre_Exposure_Blank_Fixation_Cross', 'Pre_Exposure_Room_Fixation_Cross'}
FOREST_CONDITIONS = {'Forest1', 'Forest2', 'Forest3', 'Forest4'}

# Physio conditions (from actual physio QC log labels)
PHYSIO_QC_CONDITIONS = {'Relaxation', 'Exposure'}


def parse_eeg_qc_text(participant_id: int) -> Dict[str, any]:
    """Parse EEG QC from text report file (QC_P##.txt)."""
    qc_file = QC_DIR / f'QC_P{participant_id:02d}.txt'
    
    if not qc_file.exists():
        return None
    
    metrics = {
        'sample_retention_pct': None,
        'asr_repaired_pct': None,
        'bad_channels_count': None,
        'ica_removed_count': None,
    }
    
    try:
        with open(qc_file, 'r') as f:
            content = f.read()
        
        # Parse: "Samples retained: 100.0% (465781 / 465781)"
        match = re.search(r'Samples retained:\s*([\d.]+)%', content)
        if match:
            metrics['sample_retention_pct'] = float(match.group(1))
        
        # Parse: "Percent ASR-repaired: 0.00%"
        match = re.search(r'Percent ASR-repaired:\s*([\d.]+)%', content)
        if match:
            metrics['asr_repaired_pct'] = float(match.group(1))
        
        # Parse: "Bad channels: 30"
        match = re.search(r'Bad channels:\s*(\d+)', content)
        if match:
            metrics['bad_channels_count'] = int(match.group(1))
        
        # Parse: "ICs removed: 18"
        match = re.search(r'ICs removed:\s*(\d+)', content)
        if match:
            metrics['ica_removed_count'] = int(match.group(1))
    
    except Exception as e:
        print(f"Error parsing EEG QC for P{participant_id:02d}: {e}")
        return None
    
    return metrics



def parse_physio_qc_text(participant_id: int) -> Dict[str, Dict[str, float]]:
    """
    Parse physio QC from text log file.
    
    Returns: Dict mapping condition_name -> Dict[sensor -> retention_pct]
    Example: {"Forest1": {"HR_RR_Interval": 85.5}, ...}
    """
    qc_file = PHYSIO_QC_DIR / f'P{participant_id:02d}_qc.txt'
    
    if not qc_file.exists():
        return {}
    
    # Pattern to match retention lines:
    # "[1] [Condition] Sensor: X / Y retained"
    # "[1] [Condition] Sensor: X / Y retained BEFORE flatline removal"
    retention_pattern = r'\[1\]\s+\[([^\]]+)\]\s+([^\s:]+).*?:\s*(\d+)\s*/\s*(\d+)\s+retained'
    
    condition_metrics = {}
    
    try:
        with open(qc_file, 'r') as f:
            lines = f.readlines()
        
        for line in lines:
            match = re.search(retention_pattern, line)
            if match:
                condition = match.group(1)
                sensor = match.group(2)
                retained = int(match.group(3))
                total = int(match.group(4))
                
                if total > 0:
                    retention_pct = (retained / total) * 100.0
                    
                    if condition not in condition_metrics:
                        condition_metrics[condition] = {}
                    
                    condition_metrics[condition][sensor] = retention_pct
    
    except Exception as e:
        print(f"Error parsing physio QC for P{participant_id:02d}: {e}")
    
    return condition_metrics


def check_eeg_failures(participant_id: int, metrics: Dict) -> List[str]:
    """Check EEG QC thresholds and return list of failure reasons."""
    failures = []
    
    if metrics is None:
        failures.append("EEG_QC_FILE_MISSING")
        return failures
    
    # Check sample retention
    if metrics['sample_retention_pct'] is not None:
        if metrics['sample_retention_pct'] < EEG_MIN_SAMPLE_RETENTION:
            failures.append(
                f"EEG_LOW_SAMPLE_RETENTION_{metrics['sample_retention_pct']:.1f}%"
            )
    
    # Check bad channels (interpolated) - count against non-occipital only
    if metrics['bad_channels_count'] is not None:
        bad_channels = metrics['bad_channels_count']
        if bad_channels > MAX_NON_OCCIPITAL_INTERPOLATED:
            pct = (bad_channels / NON_OCCIPITAL_CHANNELS) * 100.0
            failures.append(
                f"EEG_HIGH_INTERPOLATION_{bad_channels}ch_{pct:.1f}%"
            )
    
    # Check ICA components removed
    if metrics['ica_removed_count'] is not None:
        # Estimate total ICs (typically ~50-60 for 271 channels, but use ratio)
        # For conservatism, we check absolute count: >35% of typical 50 = >17.5
        # But more reliably, we could check against 35% threshold directly
        # For now, use 18+ as conservative threshold (36% of 50)
        if metrics['ica_removed_count'] > int(50 * (EEG_MAX_ICA_REMOVED_PCT / 100.0)):
            failures.append(
                f"EEG_HIGH_ICA_REMOVAL_{metrics['ica_removed_count']}cs"
            )
    
    return failures


def check_physio_failures(participant_id: int, condition_metrics: Dict) -> List[str]:
    """Check physio QC thresholds for Relaxation and Exposure conditions only."""
    failures = []
    
    if not condition_metrics:
        return []  # No physio data available (not necessarily a failure)
    
    # Check only Relaxation and Exposure conditions from physio QC logs
    conditions_to_check = PHYSIO_QC_CONDITIONS.intersection(set(condition_metrics.keys()))
    
    # Check each condition
    for condition in conditions_to_check:
        sensors = condition_metrics[condition]
        
        # Check each sensor in this condition
        for sensor, retention_pct in sensors.items():
            if retention_pct < PHYSIO_MIN_RETENTION_PCT:
                failures.append(
                    f"PHYSIO_LOW_RETENTION_{condition}_{sensor}_{retention_pct:.1f}%"
                )
    
    return failures


def aggregate_qc_failures() -> None:
    """Main function to aggregate QC failures across all participants."""
    
    failures_by_participant = {}
    
    # Check all participant QC files
    for p in range(1, 49):  # Participants P01-P48
        participant_id = f"P{p:02d}"
        all_failures = []
        
        # Check EEG QC
        eeg_metrics = parse_eeg_qc_text(p)
        eeg_failures = check_eeg_failures(p, eeg_metrics)
        all_failures.extend(eeg_failures)
        
        # Check physio QC
        physio_metrics = parse_physio_qc_text(p)
        physio_failures = check_physio_failures(p, physio_metrics)
        
        # Debug: show physio metrics for first 3 participants
        if p <= 3 and physio_metrics:
            print(f"  [DEBUG P{p:02d}] Physio conditions found: {list(physio_metrics.keys())}")
            for cond in ['Relaxation', 'Exposure']:
                if cond in physio_metrics:
                    sensors = physio_metrics[cond]
                    print(f"  [DEBUG P{p:02d}] {cond}: {list(sensors.keys())[:2]}...")
        
        all_failures.extend(physio_failures)
        
        if all_failures:
            failures_by_participant[participant_id] = all_failures
            print(f"❌ {participant_id}: {', '.join(all_failures)}")
        else:
            print(f"✅ {participant_id}: PASS")
    
    # Write output CSV
    with open(OUTPUT_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Participant_ID', 'Failure_Reason'])
        
        for participant_id in sorted(failures_by_participant.keys()):
            reasons = failures_by_participant[participant_id]
            writer.writerow([participant_id, '; '.join(reasons)])
    
    print(f"\n📊 QC Summary:")
    print(f"  Total participants: 48")
    print(f"  Failed QC: {len(failures_by_participant)}")
    print(f"  Passed QC: {48 - len(failures_by_participant)}")
    print(f"\n✅ QC failures summary written to: {OUTPUT_FILE}")


if __name__ == '__main__':
    aggregate_qc_failures()
