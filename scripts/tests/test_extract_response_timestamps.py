"""Test extract_response_timestamps function."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Add parent dir to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from xdf_to_set.xdf_to_set import extract_response_timestamps


def test_extract_response_timestamps_basic():
    """Test that response transitions (state changes) are detected correctly."""
    # Create a simple physio df with response changes
    # Represents: Correct → Correct → Correct → Incorrect → Incorrect → Incorrect → Correct → Correct
    timestamps = pd.date_range("2024-01-01 12:00:00", periods=8, freq="100ms")
    responses = ["Correct", "Correct", "Correct", "Incorrect", "Incorrect", "Incorrect", "Correct", "Correct"]
    
    df_physio = pd.DataFrame({
        "Response": responses,
    }, index=timestamps)
    
    result = extract_response_timestamps(df_physio)
    
    # Should detect 3 state transitions:
    # - Correct (rows 0-2): one marker at first transition (index 0)
    # - Incorrect (rows 3-5): one marker at transition (index 3)
    # - Correct (rows 6-7): one marker at transition (index 6)
    # Total: 2 Correct events + 1 Incorrect event (one per state change)
    
    assert "Response_Correct" in result
    assert "Response_Incorrect" in result
    assert len(result["Response_Correct"]) == 2, f"Expected 2 Correct transitions, got {len(result['Response_Correct'])}"
    assert len(result["Response_Incorrect"]) == 1, f"Expected 1 Incorrect transition, got {len(result['Response_Incorrect'])}"
    
    # Check timestamps are at transition points
    assert result["Response_Correct"][0] == np.datetime64(timestamps[0]), "First Correct at start"
    assert result["Response_Incorrect"][0] == np.datetime64(timestamps[3]), "Incorrect at transition point"
    assert result["Response_Correct"][1] == np.datetime64(timestamps[6]), "Second Correct at transition point"


def test_extract_response_timestamps_no_responses():
    """Test when there are no valid responses."""
    timestamps = pd.date_range("2024-01-01 12:00:00", periods=5, freq="100ms")
    responses = [None, None, None, None, None]
    
    df_physio = pd.DataFrame({
        "Response": responses,
    }, index=timestamps)
    
    result = extract_response_timestamps(df_physio)
    
    assert result == {}


def test_extract_response_timestamps_single_response_type():
    """Test when only one type of response exists."""
    timestamps = pd.date_range("2024-01-01 12:00:00", periods=5, freq="100ms")
    responses = ["Correct", "Correct", "Correct", "Correct", "Correct"]
    
    df_physio = pd.DataFrame({
        "Response": responses,
    }, index=timestamps)
    
    result = extract_response_timestamps(df_physio)
    
    assert "Response_Correct" in result
    assert "Response_Incorrect" not in result
    assert len(result["Response_Correct"]) == 1
    assert result["Response_Correct"][0] == np.datetime64(timestamps[0])


def test_extract_response_timestamps_alternating():
    """Test alternating response types (each is a separate event)."""
    timestamps = pd.date_range("2024-01-01 12:00:00", periods=6, freq="100ms")
    responses = ["Correct", "Incorrect", "Correct", "Incorrect", "Correct", "Incorrect"]
    
    df_physio = pd.DataFrame({
        "Response": responses,
    }, index=timestamps)
    
    result = extract_response_timestamps(df_physio)
    
    # Each row is a new transition
    assert len(result["Response_Correct"]) == 3
    assert len(result["Response_Incorrect"]) == 3
    
    # Verify order matches index
    assert result["Response_Correct"][0] == np.datetime64(timestamps[0])
    assert result["Response_Incorrect"][0] == np.datetime64(timestamps[1])
    assert result["Response_Correct"][1] == np.datetime64(timestamps[2])
    assert result["Response_Incorrect"][1] == np.datetime64(timestamps[3])
    assert result["Response_Correct"][2] == np.datetime64(timestamps[4])
    assert result["Response_Incorrect"][2] == np.datetime64(timestamps[5])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
