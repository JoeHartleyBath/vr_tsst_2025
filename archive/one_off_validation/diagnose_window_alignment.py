"""
Window Alignment Diagnostic Tool

Validates temporal alignment of EEG and physio rolling windows after merge.
Tests a subset of participants (P01, P10, P20) across all conditions.

Checks:
1. Window_Index sequence is contiguous (0, 1, 2, ... no gaps)
2. Window duration is ~10s ±0.1s
3. Window overlap is 50% (5s stride between starts)
4. No temporal gaps or overlaps between windows
5. EEG and physio timestamps align correctly

Usage:
    python diagnose_window_alignment.py
    python diagnose_window_alignment.py --output-dir output/qc/window_alignment

Author: VR-TSST Project
Date: December 2025
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from typing import List, Dict, Tuple


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Diagnose window alignment issues in multimodal features"
    )
    parser.add_argument(
        '--merged-data',
        type=str,
        default='output/aggregated/multimodal_features_rolling_windows.csv',
        help='Path to merged multimodal features'
    )
    parser.add_argument(
        '--eeg-data',
        type=str,
        default='output/aggregated/eeg_features_rolling_windows.csv',
        help='Path to EEG rolling window features'
    )
    parser.add_argument(
        '--physio-data',
        type=str,
        default='output/aggregated/physio_features_rolling_windows.csv',
        help='Path to physio rolling window features'
    )
    parser.add_argument(
        '--test-participants',
        type=int,
        nargs='+',
        default=[1, 10, 20],
        help='Participant IDs to test (default: 1 10 20)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='output/qc/window_alignment',
        help='Output directory for diagnostic plots and reports'
    )
    
    return parser.parse_args()


def load_data(file_path: str) -> pd.DataFrame:
    """Load feature data with validation."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    df = pd.read_csv(file_path)
    print(f"Loaded {file_path}:")
    print(f"  Shape: {df.shape}")
    print(f"  Columns: {df.columns.tolist()[:10]}...")  # Show first 10 columns
    
    return df


def check_window_index_sequence(
    df: pd.DataFrame,
    participant_id: int,
    condition: str
) -> Dict[str, any]:
    """Check if Window_Index is contiguous (no gaps)."""
    
    mask = (df['Participant_ID'] == participant_id) & (df['Condition'] == condition)
    data = df[mask].sort_values('Window_Index')
    
    if len(data) == 0:
        return {
            'status': 'MISSING',
            'message': f'No data for P{participant_id:02d} {condition}'
        }
    
    indices = data['Window_Index'].values
    expected = np.arange(indices.min(), indices.max() + 1)
    
    is_contiguous = np.array_equal(indices, expected)
    
    if is_contiguous:
        return {
            'status': 'PASS',
            'message': f'Window_Index contiguous: {len(indices)} windows (0-{indices.max()})',
            'n_windows': len(indices),
            'min_index': int(indices.min()),
            'max_index': int(indices.max())
        }
    else:
        gaps = set(expected) - set(indices)
        return {
            'status': 'FAIL',
            'message': f'Window_Index has gaps: missing indices {sorted(gaps)}',
            'n_windows': len(indices),
            'gaps': sorted(gaps)
        }


def check_window_durations(
    df: pd.DataFrame,
    participant_id: int,
    condition: str,
    expected_duration: float = 10.0,
    tolerance: float = 0.1
) -> Dict[str, any]:
    """Check if window durations are correct (~10s)."""
    
    mask = (df['Participant_ID'] == participant_id) & (df['Condition'] == condition)
    data = df[mask]
    
    if len(data) == 0 or 'Window_Start' not in data.columns or 'Window_End' not in data.columns:
        return {
            'status': 'SKIP',
            'message': 'No timestamp data available'
        }
    
    durations = data['Window_End'] - data['Window_Start']
    mean_duration = durations.mean()
    std_duration = durations.std()
    min_duration = durations.min()
    max_duration = durations.max()
    
    is_valid = np.all(np.abs(durations - expected_duration) <= tolerance)
    
    if is_valid:
        return {
            'status': 'PASS',
            'message': f'All window durations within {tolerance}s of {expected_duration}s',
            'mean': mean_duration,
            'std': std_duration,
            'min': min_duration,
            'max': max_duration
        }
    else:
        n_invalid = np.sum(np.abs(durations - expected_duration) > tolerance)
        return {
            'status': 'FAIL',
            'message': f'{n_invalid} windows outside tolerance (mean={mean_duration:.3f}s, std={std_duration:.3f}s)',
            'mean': mean_duration,
            'std': std_duration,
            'min': min_duration,
            'max': max_duration,
            'n_invalid': n_invalid
        }


def check_window_overlap(
    df: pd.DataFrame,
    participant_id: int,
    condition: str,
    expected_stride: float = 5.0,
    tolerance: float = 0.1
) -> Dict[str, any]:
    """Check if windows have correct 50% overlap (5s stride)."""
    
    mask = (df['Participant_ID'] == participant_id) & (df['Condition'] == condition)
    data = df[mask].sort_values('Window_Index')
    
    if len(data) < 2 or 'Window_Start' not in data.columns:
        return {
            'status': 'SKIP',
            'message': 'Insufficient data or no timestamps'
        }
    
    strides = np.diff(data['Window_Start'].values)
    mean_stride = strides.mean()
    std_stride = strides.std()
    
    is_valid = np.all(np.abs(strides - expected_stride) <= tolerance)
    
    if is_valid:
        return {
            'status': 'PASS',
            'message': f'All strides within {tolerance}s of {expected_stride}s',
            'mean': mean_stride,
            'std': std_stride
        }
    else:
        n_invalid = np.sum(np.abs(strides - expected_stride) > tolerance)
        return {
            'status': 'FAIL',
            'message': f'{n_invalid} strides outside tolerance (mean={mean_stride:.3f}s, std={std_stride:.3f}s)',
            'mean': mean_stride,
            'std': std_stride,
            'n_invalid': n_invalid
        }


def check_temporal_gaps(
    df: pd.DataFrame,
    participant_id: int,
    condition: str
) -> Dict[str, any]:
    """Check for gaps or overlaps between consecutive windows."""
    
    mask = (df['Participant_ID'] == participant_id) & (df['Condition'] == condition)
    data = df[mask].sort_values('Window_Index')
    
    if len(data) < 2 or 'Window_Start' not in data.columns or 'Window_End' not in data.columns:
        return {
            'status': 'SKIP',
            'message': 'Insufficient data or no timestamps'
        }
    
    # Check if end of window i overlaps with start of window i+1 (50% overlap expected)
    gaps = []
    overlaps = []
    
    for i in range(len(data) - 1):
        end_i = data.iloc[i]['Window_End']
        start_next = data.iloc[i+1]['Window_Start']
        
        # For 50% overlap, start_next should be 5s after start_i (halfway through window_i)
        # So we expect: end_i - start_next = 5s (window_i extends 5s beyond start of window_i+1)
        expected_overlap = 5.0
        actual_overlap = end_i - start_next
        
        if actual_overlap < (expected_overlap - 0.1):  # Gap
            gaps.append({
                'window_i': int(data.iloc[i]['Window_Index']),
                'gap': expected_overlap - actual_overlap
            })
        elif actual_overlap > (expected_overlap + 0.1):  # Excessive overlap
            overlaps.append({
                'window_i': int(data.iloc[i]['Window_Index']),
                'overlap': actual_overlap - expected_overlap
            })
    
    if len(gaps) == 0 and len(overlaps) == 0:
        return {
            'status': 'PASS',
            'message': 'No temporal gaps or excessive overlaps detected'
        }
    else:
        messages = []
        if gaps:
            messages.append(f'{len(gaps)} gaps detected')
        if overlaps:
            messages.append(f'{len(overlaps)} excessive overlaps detected')
        
        return {
            'status': 'FAIL',
            'message': ', '.join(messages),
            'gaps': gaps,
            'overlaps': overlaps
        }


def check_eeg_physio_alignment(
    eeg_df: pd.DataFrame,
    physio_df: pd.DataFrame,
    merged_df: pd.DataFrame,
    participant_id: int,
    condition: str
) -> Dict[str, any]:
    """Check if EEG and physio windows align correctly after merge."""
    
    eeg_mask = (eeg_df['Participant_ID'] == participant_id) & (eeg_df['Condition'] == condition)
    physio_mask = (physio_df['Participant_ID'] == participant_id) & (physio_df['Condition'] == condition)
    merged_mask = (merged_df['Participant_ID'] == participant_id) & (merged_df['Condition'] == condition)
    
    eeg_data = eeg_df[eeg_mask]
    physio_data = physio_df[physio_mask]
    merged_data = merged_df[merged_mask]
    
    n_eeg = len(eeg_data)
    n_physio = len(physio_data)
    n_merged = len(merged_data)
    
    if n_eeg == 0 or n_physio == 0:
        return {
            'status': 'MISSING',
            'message': f'Missing data (EEG={n_eeg}, Physio={n_physio})'
        }
    
    # Check window counts
    count_match = (n_eeg == n_physio == n_merged)
    loss_pct = (1 - n_merged / max(n_eeg, n_physio)) * 100 if max(n_eeg, n_physio) > 0 else 0
    
    # Check timestamp alignment if available
    timestamp_check = None
    if 'Window_Start' in eeg_data.columns and 'Window_Start' in physio_data.columns:
        # After merge, check a few sample windows
        sample_indices = [0, len(merged_data)//2, len(merged_data)-1] if len(merged_data) > 0 else []
        max_time_diff = 0
        
        for idx in sample_indices:
            if idx < len(merged_data):
                win_idx = merged_data.iloc[idx]['Window_Index']
                
                eeg_row = eeg_data[eeg_data['Window_Index'] == win_idx]
                physio_row = physio_data[physio_data['Window_Index'] == win_idx]
                
                if len(eeg_row) > 0 and len(physio_row) > 0:
                    eeg_start = eeg_row.iloc[0]['Window_Start']
                    physio_start = physio_row.iloc[0]['Window_Start']
                    time_diff = abs(eeg_start - physio_start)
                    max_time_diff = max(max_time_diff, time_diff)
        
        timestamp_check = f'Max timestamp difference: {max_time_diff:.3f}s'
    
    if count_match and loss_pct < 5:
        return {
            'status': 'PASS',
            'message': f'Window counts match (EEG={n_eeg}, Physio={n_physio}, Merged={n_merged})',
            'n_eeg': n_eeg,
            'n_physio': n_physio,
            'n_merged': n_merged,
            'timestamp_check': timestamp_check
        }
    else:
        return {
            'status': 'WARN' if loss_pct < 10 else 'FAIL',
            'message': f'Window count mismatch: EEG={n_eeg}, Physio={n_physio}, Merged={n_merged} ({loss_pct:.1f}% loss)',
            'n_eeg': n_eeg,
            'n_physio': n_physio,
            'n_merged': n_merged,
            'loss_pct': loss_pct,
            'timestamp_check': timestamp_check
        }


def plot_window_timeline(
    df: pd.DataFrame,
    participant_id: int,
    condition: str,
    output_path: str
):
    """Plot window timeline to visualize alignment."""
    
    mask = (df['Participant_ID'] == participant_id) & (df['Condition'] == condition)
    data = df[mask].sort_values('Window_Index')
    
    if len(data) == 0 or 'Window_Start' not in data.columns or 'Window_End' not in data.columns:
        print(f"  Skipping plot for P{participant_id:02d} {condition} (no timestamp data)")
        return
    
    fig, ax = plt.subplots(figsize=(14, 4))
    
    for i, row in data.iterrows():
        win_idx = row['Window_Index']
        start = row['Window_Start']
        end = row['Window_End']
        
        # Plot window as horizontal bar
        ax.barh(0, end - start, left=start, height=0.5, alpha=0.6, edgecolor='black')
        
        # Label every 5th window to avoid clutter
        if win_idx % 5 == 0:
            ax.text(start + (end - start) / 2, 0, str(int(win_idx)), 
                   ha='center', va='center', fontsize=8)
    
    ax.set_xlabel('Time (seconds)', fontsize=12)
    ax.set_ylabel('')
    ax.set_yticks([])
    ax.set_title(f'P{participant_id:02d} {condition} - Window Timeline', fontsize=14)
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved timeline plot: {output_path}")


def generate_diagnostic_report(
    results: Dict[str, Dict],
    output_path: str
):
    """Generate comprehensive diagnostic report."""
    
    report = []
    report.append("="*70)
    report.append("WINDOW ALIGNMENT DIAGNOSTIC REPORT")
    report.append("="*70)
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")
    
    # Summary statistics
    total_tests = len(results)
    passed = sum(1 for r in results.values() if all(
        check.get('status') == 'PASS' for check in r.values() if isinstance(check, dict)
    ))
    
    report.append("SUMMARY:")
    report.append(f"  Total participant-condition pairs tested: {total_tests}")
    report.append(f"  Fully passed: {passed}")
    report.append(f"  With issues: {total_tests - passed}")
    report.append("")
    
    # Detailed results
    report.append("DETAILED RESULTS:")
    report.append("")
    
    for key, checks in results.items():
        participant_id, condition = key
        report.append(f"P{participant_id:02d} {condition}:")
        
        for check_name, result in checks.items():
            if isinstance(result, dict):
                status = result.get('status', 'UNKNOWN')
                message = result.get('message', '')
                
                status_symbol = {
                    'PASS': '✓',
                    'FAIL': '✗',
                    'WARN': '⚠',
                    'SKIP': '○',
                    'MISSING': '○'
                }.get(status, '?')
                
                report.append(f"  {status_symbol} {check_name}: {message}")
                
                # Add additional details for failures
                if status == 'FAIL':
                    for k, v in result.items():
                        if k not in ['status', 'message'] and not k.startswith('n_'):
                            report.append(f"      {k}: {v}")
        
        report.append("")
    
    report.append("="*70)
    
    # Write report
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))
    
    print(f"\nDiagnostic report saved: {output_path}")
    
    # Print summary to console
    print("\n" + "="*70)
    print("DIAGNOSTIC SUMMARY")
    print("="*70)
    print(f"Total tests: {total_tests}")
    print(f"Fully passed: {passed} ({passed/total_tests*100:.1f}%)")
    print(f"With issues: {total_tests - passed} ({(total_tests-passed)/total_tests*100:.1f}%)")
    print("="*70)


def main():
    """Main diagnostic routine."""
    
    args = parse_arguments()
    
    print("="*70)
    print("WINDOW ALIGNMENT DIAGNOSTICS")
    print("="*70)
    print(f"Testing participants: {args.test_participants}")
    print()
    
    # Load data
    print("Loading data...")
    merged_df = load_data(args.merged_data)
    eeg_df = load_data(args.eeg_data)
    physio_df = load_data(args.physio_data)
    print()
    
    # Standardize column names for EEG (apply same transformations as merge script)
    if 'pid' in eeg_df.columns:
        eeg_df = eeg_df.rename(columns={'pid': 'Participant_ID'})
    if 'event_label' in eeg_df.columns:
        eeg_df = eeg_df.rename(columns={'event_label': 'Condition'})
    if 'window_idx' in eeg_df.columns:
        eeg_df = eeg_df.rename(columns={'window_idx': 'Window_Index'})
        # Apply same offset correction as merge script
        eeg_df['Window_Index'] = eeg_df['Window_Index'] - 1
    if 'window_start' in eeg_df.columns:
        eeg_df = eeg_df.rename(columns={'window_start': 'Window_Start'})
    if 'window_end' in eeg_df.columns:
        eeg_df = eeg_df.rename(columns={'window_end': 'Window_End'})
    
    # Standardize physio column names
    if 'participant_id' in physio_df.columns:
        physio_df = physio_df.rename(columns={'participant_id': 'Participant_ID'})
    if 'condition' in physio_df.columns:
        physio_df = physio_df.rename(columns={'condition': 'Condition'})
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run diagnostics
    results = {}
    
    for participant_id in args.test_participants:
        # Get conditions for this participant
        conditions = merged_df[merged_df['Participant_ID'] == participant_id]['Condition'].unique()
        
        if len(conditions) == 0:
            print(f"P{participant_id:02d}: No data found")
            continue
        
        print(f"\nP{participant_id:02d}: Testing {len(conditions)} conditions...")
        
        for condition in conditions:
            print(f"  {condition}...")
            
            checks = {}
            
            # 1. Check Window_Index sequence
            checks['Index Sequence'] = check_window_index_sequence(
                merged_df, participant_id, condition
            )
            
            # 2. Check window durations
            checks['Window Durations'] = check_window_durations(
                merged_df, participant_id, condition
            )
            
            # 3. Check window overlap
            checks['Window Overlap'] = check_window_overlap(
                merged_df, participant_id, condition
            )
            
            # 4. Check temporal gaps
            checks['Temporal Gaps'] = check_temporal_gaps(
                merged_df, participant_id, condition
            )
            
            # 5. Check EEG-physio alignment
            checks['EEG-Physio Alignment'] = check_eeg_physio_alignment(
                eeg_df, physio_df, merged_df, participant_id, condition
            )
            
            results[(participant_id, condition)] = checks
            
            # Generate timeline plot
            plot_path = os.path.join(
                args.output_dir,
                f'P{participant_id:02d}_{condition}_timeline.png'
            )
            plot_window_timeline(merged_df, participant_id, condition, plot_path)
    
    # Generate report
    report_path = os.path.join(args.output_dir, 'window_alignment_diagnostic_report.txt')
    generate_diagnostic_report(results, report_path)
    
    print("\n✓ Diagnostics complete!")
    print(f"Results saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
