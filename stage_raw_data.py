#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Stage raw data from data/raw/<PN>/ into pipeline structure.
Copies XDF, CSV metadata, and subjective PQs into separate folders.
"""
import shutil
from pathlib import Path

base = Path(r"c:\vr_tsst_2025\data\raw")
eeg_dir = base / "eeg"
metadata_dir = base / "metadata"
subjective_dir = base / "subjective"

# Create dirs
for d in [eeg_dir, metadata_dir, subjective_dir]:
    d.mkdir(parents=True, exist_ok=True)

success = 0
errors = 0

# Process participants 1-48
for pnum in range(1, 49):
    pdir = base / str(pnum)
    if not pdir.exists():
        print("Skip P{:02d}: folder not found".format(pnum))
        continue
    
    padded = "{:02d}".format(pnum)
    files = list(pdir.glob("*"))
    
    # XDF (PD_<PN>_..._N.xdf)
    xdfs = [f for f in files if f.suffix.lower() == ".xdf" and "_N.xdf" in f.name]
    if xdfs:
        src = xdfs[0]
        dst = eeg_dir / "P{}.xdf".format(padded)
        shutil.copy2(src, dst)
        print("[OK] XDF P{}".format(padded))
        success += 1
    
    # Metadata CSV (PD_<PN>_RAW_DATA_..._C.csv)
    csvs = [f for f in files if f.suffix.lower() == ".csv" and "_C.csv" in f.name and "RAW_DATA" in f.name]
    if csvs:
        src = csvs[0]
        dst = metadata_dir / "P{}.csv".format(padded)
        shutil.copy2(src, dst)
        print("[OK] Metadata P{}".format(padded))
        success += 1
    else:
        print("[ERROR] No metadata CSV for P{}".format(padded))
        errors += 1
    
    # Subjective PQs (PQs_<PN>_compiled.csv)
    pqs = [f for f in files if f.name.startswith("PQs_") and "compiled" in f.name and f.suffix.lower() == ".csv"]
    if pqs:
        src = pqs[0]
        dst = subjective_dir / "PQs_{}_compiled.csv".format(padded)
        shutil.copy2(src, dst)
        print("[OK] PQ P{}".format(padded))
        success += 1

print("\n=== Summary ===")
print("Files copied: {}".format(success))
print("Errors: {}".format(errors))

