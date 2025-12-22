#!/usr/bin/env python
"""Quick test of the complete xdf_to_set pipeline with response markers."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "scripts"))

from xdf_to_set.xdf_to_set import xdf_to_set

print("Running xdf_to_set pipeline on P01...")
result = xdf_to_set(
    Path("data/raw/eeg/P01.xdf"),
    Path("output/processed/P01.set"),
    Path("data/raw/metadata/P01.csv"),
    Path("config/conditions.yaml")
)

print("\n✓ Pipeline completed successfully!")
print("\nResults:")
for k, v in result.items():
    print(f"  {k}: {v}")

print(f"\n✓ File saved to: {result['path']}")
print(f"✓ Total events: {result['n_events']} (including condition onsets + response transitions)")
