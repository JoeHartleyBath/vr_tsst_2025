"""
Validate EEGLAB .set file structure without loading in MATLAB.

This script loads a .set file with scipy and checks that:
1. Event structure is a numpy structured array (not cell array)
2. All required EEGLAB fields are present
3. Data types are compatible with MATLAB/EEGLAB
"""

from pathlib import Path
import scipy.io as sio
import numpy as np


def validate_set_structure(set_path: Path) -> dict:
    """
    Load and validate EEGLAB .set structure.
    
    Returns dict with validation results.
    """
    
    results = {
        "valid": True,
        "errors": [],
        "warnings": [],
        "info": {}
    }
    
    print(f"Loading: {set_path}")
    
    try:
        mat = sio.loadmat(str(set_path), struct_as_record=False, squeeze_me=True)
    except Exception as e:
        results["valid"] = False
        results["errors"].append(f"Failed to load .set file: {e}")
        return results
    
    if "EEG" not in mat:
        results["valid"] = False
        results["errors"].append("No 'EEG' structure found in .set file")
        return results
    
    EEG = mat["EEG"]
    
    # Check if EEG is a MATLAB struct (scipy loads as object with attributes)
    if not hasattr(EEG, '_fieldnames'):
        results["valid"] = False
        results["errors"].append("EEG is not a MATLAB structure")
        return results
    
    print(f"\nEEG structure fields: {EEG._fieldnames}")
    
    # Required fields
    required = ['nbchan', 'trials', 'pnts', 'srate', 'xmin', 'xmax', 'data']
    for field in required:
        if not hasattr(EEG, field):
            results["errors"].append(f"Missing required field: {field}")
            results["valid"] = False
        else:
            val = getattr(EEG, field)
            if np.isscalar(val):
                results["info"][field] = f"{type(val).__name__}: {val}"
            else:
                shape_info = getattr(val, 'shape', 'N/A')
                results["info"][field] = f"{type(val).__name__}: shape {shape_info}"
    
    # Check event structure
    if hasattr(EEG, 'event'):
        event = EEG.event
        event_type = type(event)
        
        print(f"\nEvent structure:")
        print(f"  Type: {event_type}")
        print(f"  Python type: {type(event)}")
        
        # Check if it's a structured array
        if isinstance(event, np.ndarray):
            print(f"  Shape: {event.shape}")
            print(f"  Dtype: {event.dtype}")
            
            if event.dtype.names is not None:
                print(f"  Field names: {event.dtype.names}")
                results["info"]["event_fields"] = list(event.dtype.names)
                
                # Check required event fields
                required_event_fields = ['latency', 'type']
                for field in required_event_fields:
                    if field not in event.dtype.names:
                        results["errors"].append(f"Event missing required field: {field}")
                        results["valid"] = False
                
                # Show first event as example
                if len(event) > 0:
                    print(f"\n  First event:")
                    for field in event.dtype.names:
                        print(f"    {field}: {event[field][0]}")
            else:
                results["errors"].append("Event is ndarray but not structured (no field names)")
                results["valid"] = False
        else:
            # Check if it's a cell array or other incompatible type
            results["errors"].append(f"Event is {event_type}, not numpy structured array")
            results["valid"] = False
            
            # If it's a cell array (loaded as ndarray of objects), show details
            if isinstance(event, np.ndarray) and event.dtype == object:
                print(f"  WARNING: Event is object array (likely MATLAB cell array)")
                if event.size > 0:
                    print(f"  First element type: {type(event.flat[0])}")
                    print(f"  First element: {event.flat[0]}")
    else:
        results["warnings"].append("No 'event' field found (this may be OK for empty datasets)")
    
    # Check urevent
    if hasattr(EEG, 'urevent'):
        urevent = EEG.urevent
        if isinstance(urevent, np.ndarray) and urevent.dtype.names is not None:
            results["info"]["urevent_fields"] = list(urevent.dtype.names)
        else:
            results["warnings"].append(f"urevent is {type(urevent)}, not structured array")
    
    # Summary
    print(f"\n{'='*60}")
    if results["valid"]:
        print("✓ VALIDATION PASSED")
    else:
        print("✗ VALIDATION FAILED")
        print("\nErrors:")
        for err in results["errors"]:
            print(f"  - {err}")
    
    if results["warnings"]:
        print("\nWarnings:")
        for warn in results["warnings"]:
            print(f"  - {warn}")
    
    return results


if __name__ == "__main__":
    set_file = Path("output/processed/P01.set")
    
    if not set_file.exists():
        print(f"ERROR: {set_file} does not exist")
    else:
        results = validate_set_structure(set_file)
        
        # Exit with error code if validation failed
        import sys
        sys.exit(0 if results["valid"] else 1)
