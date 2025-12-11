from pathlib import Path
from scripts.xdf_to_set.xdf_to_set import xdf_to_set

if __name__ == "__main__":
    xdf = Path(r"C:\phd_projects\vr_tsst_2025\data\raw\eeg\P01.xdf")
    out = Path(r"C:\phd_projects\vr_tsst_2025\output\P01.set")
    try:
        summary = xdf_to_set(xdf, out)
        print("Conversion summary:")
        for k, v in summary.items():
            print(f"{k}: {v}")
    except Exception as e:
        print("Conversion failed:", e)
