"""
Create a minimal test .set file to validate event structure format.

This creates a tiny .set file (1 second, 3 events) to quickly test
whether the event structure is compatible with MATLAB/EEGLAB.
"""

from pathlib import Path
import numpy as np
from scipy.io import savemat
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from xdf_to_set.xdf_to_set import build_eeglab_struct, add_events_to_eeg_struct


def create_minimal_test_set():
    """Create a minimal 1-second test .set file with 3 events."""
    
    print("Creating minimal test .set file...")
    print("=" * 60)
    
    # Minimal EEG data: 3 channels, 1 second at 256 Hz
    srate = 256.0
    duration_sec = 1.0
    n_samples = int(srate * duration_sec)
    n_channels = 3
    
    # Random data
    data = np.random.randn(n_channels, n_samples) * 10  # Channels × Samples
    
    # Build basic EEGLAB structure manually (simpler than using build_eeglab_struct)
    import yaml
    template_path = Path("config/eeglab_template.yaml")
    with open(template_path, "r") as f:
        template = yaml.safe_load(f)
    
    times = np.arange(n_samples) / srate * 1000
    
    # Create chanlocs as structured array for MATLAB compatibility
    chanlocs_dtype = np.dtype([('labels', 'O')])
    chanlocs = np.array(
        [("Ch1",), ("Ch2",), ("Ch3",)],
        dtype=chanlocs_dtype
    )
    
    EEG = {
        **template,
        "nbchan": n_channels,
        "trials": 1,
        "pnts": n_samples,
        "srate": srate,
        "xmin": 0,
        "xmax": (n_samples - 1) / srate,
        "times": times,
        "data": data,
        "chanlocs": chanlocs,
        "setname": "test_minimal",
        "filename": "test_minimal.set",
        "filepath": str(Path("output/processed").absolute()),
    }
    
    # Create 3 test events
    events = [
        {"latency": 0, "type": "Start"},
        {"latency": 128, "type": "Middle"},
        {"latency": 255, "type": "End"}
    ]
    
    # Add events using our function
    EEG = add_events_to_eeg_struct(EEG, events)
    
    # Check event structure before saving
    print("\nEvent structure before saving:")
    print(f"  Type: {type(EEG['event'])}")
    print(f"  Dtype: {EEG['event'].dtype}")
    print(f"  Shape: {EEG['event'].shape}")
    print(f"  Fields: {EEG['event'].dtype.names}")
    
    if len(EEG['event']) > 0:
        print(f"\n  First event:")
        for field in EEG['event'].dtype.names:
            print(f"    {field}: {EEG['event'][field][0]} (type: {type(EEG['event'][field][0])})")
    
    # Save
    output_path = Path("output/processed/test_minimal.set")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    mat_dict = {"EEG": EEG}
    savemat(str(output_path), mat_dict, do_compression=True, oned_as='column')
    
    print(f"\n✓ Saved to: {output_path}")
    print(f"  File size: {output_path.stat().st_size:,} bytes")
    
    return output_path


def validate_saved_file(set_path: Path):
    """Load the saved file and validate structure."""
    from scipy.io import loadmat
    
    print(f"\n{'=' * 60}")
    print("Validating saved file...")
    print("=" * 60)
    
    # Load with struct_as_record=True to preserve structures
    mat = loadmat(str(set_path), struct_as_record=True, squeeze_me=False)
    EEG = mat["EEG"]
    
    print(f"\nLoaded EEG structure:")
    print(f"  Type: {type(EEG)}")
    print(f"  Dtype: {EEG.dtype if hasattr(EEG, 'dtype') else 'N/A'}")
    
    if isinstance(EEG, np.ndarray) and EEG.dtype.names:
        print(f"  Fields: {list(EEG.dtype.names)}")
        
        # Extract event field
        event = EEG['event'][0, 0]
        print(f"\nEvent structure:")
        print(f"  Type: {type(event)}")
        
        if isinstance(event, np.ndarray):
            print(f"  Shape: {event.shape}")
            print(f"  Dtype: {event.dtype}")
            
            if event.dtype.names:
                print(f"  Field names: {event.dtype.names}")
                
                if event.size > 0:
                    print(f"\n  First event after loading:")
                    for field in event.dtype.names:
                        val = event[field][0, 0] if event.shape else event[field]
                        print(f"    {field}: {val}")
                
                print("\n✓ Event structure is properly structured")
                return True
            else:
                print("\n✗ Event is ndarray but not structured!")
                return False
        else:
            print(f"\n✗ Event is {type(event)}, not ndarray!")
            return False
    else:
        print("\n✗ EEG is not a structured array!")
        return False


if __name__ == "__main__":
    try:
        # Create test file
        test_file = create_minimal_test_set()
        
        # Validate it
        is_valid = validate_saved_file(test_file)
        
        if is_valid:
            print(f"\n{'=' * 60}")
            print("SUCCESS! Event structure is compatible.")
            print("You can now test in MATLAB with:")
            print(f"  EEG = pop_loadset('{test_file.name}', '{test_file.parent}');")
            print("=" * 60)
            sys.exit(0)
        else:
            print(f"\n{'=' * 60}")
            print("FAILED! Event structure is incompatible.")
            print("=" * 60)
            sys.exit(1)
            
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
