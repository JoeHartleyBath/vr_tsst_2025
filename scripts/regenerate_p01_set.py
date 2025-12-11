"""
Regenerate P01.set with complete EEGLAB structure.
"""

import sys
from pathlib import Path

# Add the scripts directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from xdf_to_set.xdf_to_set import xdf_to_set

def main():
    print("=" * 60)
    print("REGENERATING P01.set WITH COMPLETE EEGLAB STRUCTURE")
    print("=" * 60)
    print()
    
    # Define paths
    xdf_path = Path("data/raw/eeg/P01.xdf")
    output_path = Path("output/processed/P01.set")
    physio_path = Path("data/raw/metadata/P01.csv")
    config_path = Path("config/conditions.yaml")
    
    # Check if input files exist
    if not xdf_path.exists():
        print(f"ERROR: XDF file not found: {xdf_path}")
        return 1
    
    if not physio_path.exists():
        print(f"ERROR: Physio CSV not found: {physio_path}")
        return 1
    
    if not config_path.exists():
        print(f"ERROR: Config file not found: {config_path}")
        return 1
    
    print(f"Input XDF: {xdf_path}")
    print(f"Input CSV: {physio_path}")
    print(f"Config: {config_path}")
    print(f"Output: {output_path}")
    print()
    
    # Run conversion
    print("Starting conversion (this will take several minutes)...")
    print()
    
    try:
        summary = xdf_to_set(
            xdf_path=xdf_path,
            output_path=output_path,
            physio_path=physio_path,
            config_path=config_path
        )
        
        print()
        print("=" * 60)
        print("SUCCESS!")
        print("=" * 60)
        print(f"Output file: {summary['path']}")
        print(f"Channels: {summary['nbchan']}")
        print(f"Samples: {summary['pnts']}")
        print(f"Sampling rate: {summary['srate']} Hz")
        print(f"Events: {summary['n_events']}")
        print()
        print("The .set file now includes all required EEGLAB fields.")
        print("You can now test the cleaning pipeline on this file.")
        
        return 0
        
    except Exception as e:
        print()
        print("=" * 60)
        print("ERROR DURING CONVERSION")
        print("=" * 60)
        print(f"{type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
