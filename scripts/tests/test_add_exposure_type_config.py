"""Test add_exposure_type_from_config function."""
import pandas as pd
import numpy as np
from pathlib import Path
from scripts.xdf_to_set.xdf_to_set import add_exposure_type_from_config

def test_add_exposure_type_from_config():
    """Test that add_exposure_type_from_config reads config and assigns labels correctly."""
    
    # Create a small synthetic physio dataframe
    df = pd.DataFrame({
        'Unity Scene': ['PrimaryCalibration', 'Fixation', 'Fixation', 'BlinkCalibration'],
        'Study_Phase': ['', 'Start Blank Fixation', 'End Room Fixation', ''],
        'Shown_Scene': ['', '', '', ''],
        'Arithmetic_Task': ['', '', '', ''],
        'Participant_State': ['', '', '', ''],
    })
    
    # Call the config-based function
    result = add_exposure_type_from_config(df, Path("config/conditions.yaml"))
    
    # Check that exposure_type column was added
    assert 'exposure_type' in result.columns, "exposure_type column not added"
    
    # Check that rows matched expected labels
    assert result.iloc[0]['exposure_type'] == 'Primary_Calibrations', \
        f"Row 0 expected Primary_Calibrations, got {result.iloc[0]['exposure_type']}"
    
    assert result.iloc[1]['exposure_type'] == 'Pre_Exposure_Blank_Fixation_Cross', \
        f"Row 1 expected Pre_Exposure_Blank_Fixation_Cross, got {result.iloc[1]['exposure_type']}"
    
    assert result.iloc[2]['exposure_type'] == 'Post_Exposure_Room_Fixation_Cross', \
        f"Row 2 expected Post_Exposure_Room_Fixation_Cross, got {result.iloc[2]['exposure_type']}"
    
    assert result.iloc[3]['exposure_type'] == 'Blink_Calibration', \
        f"Row 3 expected Blink_Calibration, got {result.iloc[3]['exposure_type']}"
    
    print("✓ All tests passed!")
    print("\nExposure type assignments:")
    print(result[['Unity Scene', 'Study_Phase', 'exposure_type']])

if __name__ == "__main__":
    test_add_exposure_type_from_config()
