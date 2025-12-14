#!/usr/bin/env python3
"""
Sequential XDF to SET conversion (memory-safe for large files)
Processes one participant at a time to avoid memory spikes.
"""

import sys
from pathlib import Path
from time import perf_counter
import argparse

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from xdf_to_set.xdf_to_set import xdf_to_set

if __name__ == "__main__":
    base = Path(__file__).parent.parent.parent.parent.parent
    
    parser = argparse.ArgumentParser(description="Sequential XDF→SET conversion (memory-safe)")
    parser.add_argument("--participants", nargs="*", type=int, help="Participant numbers (e.g., 18 19 20), or omit to process all 48")
    parser.add_argument("--config", type=str, default=str(base / "config/conditions_pilot.yaml"), help="Path to conditions YAML")
    parser.add_argument("--skip-existing", action="store_true", default=True, help="Skip already-converted files (default: True)")
    args = parser.parse_args()
    
    # Default to all 48 participants if not specified
    participants = args.participants if args.participants else list(range(1, 49))
    config_path = Path(args.config)
    
    print("=" * 80)
    print("SEQUENTIAL XDF → SET CONVERSION (Memory-Safe)")
    print("=" * 80)
    print(f"Participants: {len(participants)} total - {participants[:10]}{'...' if len(participants) > 10 else ''}")
    print(f"Config: {config_path}")
    print(f"Skip existing: {args.skip_existing}")
    print(f"Output: {base / 'output/sets'}")
    print("=" * 80)
    print()
    
    overall_start = perf_counter()
    success_count = 0
    skipped_count = 0
    failed_count = 0
    
    for i, p in enumerate(participants, 1):
        p_str = f"P{p:02d}"
        xdf = base / f"data/RAW/eeg/{p_str}.xdf"
        out = base / f"output/sets/{p_str}.set"
        
        # Skip if already exists
        if args.skip_existing and out.exists():
            print(f"[{i}/{len(participants)}] {p_str}: SKIP (already exists)")
            skipped_count += 1
            continue
        
        print(f"[{i}/{len(participants)}] {p_str}: Starting conversion...")
        start = perf_counter()
        
        try:
            summary = xdf_to_set(xdf, out, config_path=config_path)
            elapsed = perf_counter() - start
            print(f"  ✓ Success in {elapsed:.1f}s - {summary['n_events']} events, {summary['pnts']} samples")
            success_count += 1
            
        except Exception as e:
            elapsed = perf_counter() - start
            print(f"  ✗ FAILED after {elapsed:.1f}s: {e}")
            failed_count += 1
            continue
    
    overall_elapsed = perf_counter() - overall_start
    
    print()
    print("=" * 80)
    print("CONVERSION COMPLETE")
    print("=" * 80)
    print(f"Success: {success_count}/{len(participants)}")
    print(f"Skipped: {skipped_count}/{len(participants)}")
    print(f"Failed: {failed_count}/{len(participants)}")
    print(f"Total time: {overall_elapsed:.1f}s ({overall_elapsed/60:.1f} minutes)")
    print("=" * 80)
    
    sys.exit(0 if failed_count == 0 else 1)
