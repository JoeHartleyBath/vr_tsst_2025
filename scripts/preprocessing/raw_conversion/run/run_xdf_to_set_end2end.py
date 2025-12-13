import sys
from pathlib import Path

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from xdf_to_set.xdf_to_set import xdf_to_set

if __name__ == "__main__":
    base = Path(__file__).parent.parent.parent.parent.parent  # Go up to c:\vr_tsst_2025
    participants = [1, 2, 3]

    for p in participants:
        xdf = base / f"data/RAW/eeg/P{p:02d}.xdf"
        out = base / f"output/sets/P{p:02d}.set"
        out.parent.mkdir(parents=True, exist_ok=True)
        try:
            summary = xdf_to_set(xdf, out)
            print(f"Conversion summary for P{p:02d}:")
            for k, v in summary.items():
                print(f"{k}: {v}")
        except Exception as e:
            print(f"Conversion failed for P{p:02d}:", e)
