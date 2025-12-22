import sys
from pathlib import Path
import argparse
from time import perf_counter
from tqdm import tqdm

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent))

from xdf_to_set import xdf_to_set

def main():
    base = Path(__file__).parent.parent.parent  # Go up to c:\vr_tsst_2025

    parser = argparse.ArgumentParser(description="XDF→SET conversion for participants")
    parser.add_argument(
    "--participants",
    nargs="+",
    type=int,
    required=True,
    help="Participant numbers, e.g. --participants 10 17 23"
)
    parser.add_argument("--config", type=str, default=str(base / "config/conditions.yaml"), help="Path to conditions YAML")
    args = parser.parse_args()

    participants = args.participants
    pilot_config = Path(args.config)

    print(f"Starting XDF→SET conversion for: {participants}")
    print(f"Config: {pilot_config}")

    success_count = 0
    failure_count = 0

    for p in tqdm(participants, desc="Stage 1: XDF→SET", unit="participant"):
        xdf = base / f"data/RAW/eeg/P{p:02d}.xdf"
        out = base / f"output/sets/P{p:02d}.set"
        out.parent.mkdir(parents=True, exist_ok=True)
        start = perf_counter()
        try:
            print(f"\n[P{p:02d}] Load: {xdf}")
            summary = xdf_to_set(xdf, out, config_path=pilot_config)
            elapsed = perf_counter() - start
            print(f"[P{p:02d}] Save: {out}")
            print(f"✓ [P{p:02d}] Completed in {elapsed:.1f}s")
            for k, v in summary.items():
                print(f"  {k}: {v}")
            success_count += 1
        except Exception as e:
            import traceback
            elapsed = perf_counter() - start
            print(f"✗ [P{p:02d}] Failed after {elapsed:.1f}s: {e}")
            print(traceback.format_exc())
            failure_count += 1

    print(f"\nStage 1 Complete: {success_count} succeeded, {failure_count} failed")
    if failure_count > 0:
        sys.exit(1)  # Exit with error if any conversion failed

if __name__ == "__main__":
    main()
