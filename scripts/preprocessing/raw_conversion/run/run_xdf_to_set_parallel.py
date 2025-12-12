#!/usr/bin/env python3
"""
Parallel XDF to SET conversion for multiple participants
Uses multiprocessing to convert multiple XDF files simultaneously
"""

import sys
from pathlib import Path
from multiprocessing import Pool, cpu_count

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from xdf_to_set.xdf_to_set import xdf_to_set


def convert_participant(p_num):
    """Convert a single participant's XDF to SET"""
    xdf = Path(rf"C:\phd_projects\vr_tsst_2025\data\raw\eeg\P{p_num:02d}.xdf")
    out = Path(rf"C:\phd_projects\vr_tsst_2025\output\processed\P{p_num:02d}.set")
    
    print(f"\n[P{p_num:02d}] Starting conversion...")
    
    try:
        summary = xdf_to_set(xdf, out)
        print(f"[P{p_num:02d}] ✓ Conversion complete")
        print(f"[P{p_num:02d}] Summary:")
        for k, v in summary.items():
            print(f"  {k}: {v}")
        return {'participant': p_num, 'status': 'SUCCESS', 'summary': summary}
        
    except Exception as e:
        print(f"[P{p_num:02d}] ✗ Conversion failed: {e}")
        return {'participant': p_num, 'status': 'FAILED', 'error': str(e)}


if __name__ == "__main__":
    participants = [1, 2, 3]  # Update with your participant numbers
    
    # Use 3 processes (one per participant)
    num_processes = min(len(participants), cpu_count() - 1)
    
    print("=" * 60)
    print("PARALLEL XDF → SET CONVERSION")
    print("=" * 60)
    print(f"Participants: {participants}")
    print(f"Processes: {num_processes}")
    print("=" * 60)
    
    # Run in parallel
    with Pool(processes=num_processes) as pool:
        results = pool.map(convert_participant, participants)
    
    # Summary
    print("\n" + "=" * 60)
    print("CONVERSION COMPLETE")
    print("=" * 60)
    
    success_count = sum(1 for r in results if r['status'] == 'SUCCESS')
    
    for result in results:
        status_icon = "✓" if result['status'] == 'SUCCESS' else "✗"
        print(f"{status_icon} P{result['participant']:02d}: {result['status']}")
        if result['status'] == 'FAILED':
            print(f"   Error: {result['error']}")
    
    print(f"\nSuccess rate: {success_count}/{len(participants)} participants")
    print("=" * 60)
