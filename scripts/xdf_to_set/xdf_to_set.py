"""
xdf_to_set.py

Refactored XDF → SET conversion module.
Defines the public interface for importing raw .xdf files,
embedding canonical condition events, and saving EEGLAB .set files.

This file contains function signatures only. Implementation comes later.
"""

from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np

def load_xdf(xdf_path: Path) -> Dict:
    """Load a raw .xdf file and return parsed LSL streams.

    Args:
        xdf_path (Path): Full path to the .xdf file.

    Returns:
        Dict: Parsed XDF streams as a list of dicts, plus metadata.
            Example:
                {
                    "streams": [...],
                    "info": {...},
                }

    Raises:
        FileNotFoundError: If the xdf file does not exist.
        ValueError: If the file cannot be parsed or contains no streams.
    """

    if not xdf_path.exists():
        raise FileNotFoundError(f"XDF file not found: {xdf_path}")

    try:
        import pyxdf
    except ImportError as e:
        raise ImportError("pyxdf is required for loading XDF files. Install via pip.") from e

    streams, info = pyxdf.load_xdf(str(xdf_path))

    if not streams:
        raise ValueError(f"No streams found in XDF file: {xdf_path}")

    return {
        "streams": streams,
        "info": info
    }

def find_eeg_streams(xdf_streams: list) -> list:
    """Return the list of EEG streams found in the XDF file."""
        
    # 1) Filter for streams where info.type == "EEG"
    eeg_streams = [
        s for s in xdf_streams
        if s["info"]["type"][0] == "EEG"
    ]

    # 2) Validate count
    if len(eeg_streams) == 0:
        raise ValueError("No EEG streams found in the XDF file.")

    if len(eeg_streams) > 2:
        raise ValueError(
            f"Expected at most 2 EEG streams, found {len(eeg_streams)}."
        )

    return eeg_streams

import numpy as np

def extract_single_stream(stream: dict) -> dict:
    """
    Extract raw EEG data, timestamps, sampling rate, and stream name
    from a single pyxdf EEG stream.

    Returns a dict with:
        - data: np.ndarray, shape (samples, channels)
        - timestamps: np.ndarray, shape (samples,)
        - srate: float
        - name: str
    """

    # Raw EEG samples: shape (samples, channels)
    data = np.asarray(stream["time_series"], dtype=float)

    # Timestamps: shape (samples,)
    timestamps = np.asarray(stream["time_stamps"], dtype=float)

    # Sampling rate: comes as a string, so convert to float
    srate = float(stream["info"]["nominal_srate"][0])

    # Name of the stream (e.g., "EE225-...-eego_laptop")
    name = stream["info"]["name"][0]

    return {
        "data": data,
        "timestamps": timestamps,
        "srate": srate,
        "name": name,
    }

def align_streams(streamA: Dict, streamB: Dict) -> Tuple[Dict, Dict]:
    """
    Align two EEG streams by timestamps.
    Returns trimmed copies of streamA and streamB where:
    - timestamps are the same length
    - data arrays have matching sample counts
    - both streams cover the same overlapping window
    """

    tsA = streamA["timestamps"]
    tsB = streamB["timestamps"]

    # --- 1) Determine the overlap window ---
    start = max(tsA[0], tsB[0])
    end   = min(tsA[-1], tsB[-1])

    if start >= end:
        raise ValueError("Streams do not overlap in time.")

    # --- 2) Get index ranges for the overlap ---
    idxA_start = np.searchsorted(tsA, start)
    idxA_end   = np.searchsorted(tsA, end)

    idxB_start = np.searchsorted(tsB, start)
    idxB_end   = np.searchsorted(tsB, end)

    # --- 3) Slice each stream to its overlapping segment ---
    newA = {
        "name": streamA["name"],
        "srate": streamA["srate"],
        "timestamps": tsA[idxA_start:idxA_end],
        "data": streamA["data"][idxA_start:idxA_end, :],
    }

    newB = {
        "name": streamB["name"],
        "srate": streamB["srate"],
        "timestamps": tsB[idxB_start:idxB_end],
        "data": streamB["data"][idxB_start:idxB_end, :],
    }

    # --- 4) Enforce that the lengths match exactly ---
    nA = newA["data"].shape[0]
    nB = newB["data"].shape[0]

    n = min(nA, nB)

    newA["timestamps"] = newA["timestamps"][:n]
    newA["data"]       = newA["data"][:n, :]

    newB["timestamps"] = newB["timestamps"][:n]
    newB["data"]       = newB["data"][:n, :]

    return newA, newB

def extract_eeg_stream(xdf_streams: Dict) -> Tuple:
    """Extract EEG samples, timestamps, sampling rate, and channel names."""
    pass


def extract_event_timestamps(xdf_streams: Dict) -> Dict[str, List[float]]:
    """Extract condition onset timestamps from metadata/marker streams.
    
    Returns:
        A dict mapping event_type → list of timestamps.
        Example:
            {"HS_LL": [12.4], "HS_HL": [215.2], ...}
    """
    pass


def convert_events_to_latencies(eeg_timestamps, event_ts_dict) -> Dict[str, List[int]]:
    """Convert LSL timestamps to EEG sample latencies."""
    pass


def build_eeglab_struct(eeg_data, srate, ch_names, event_latencies) -> object:
    """Build an EEGLAB-like struct (via MNE or MATLAB via MATLAB Engine)."""
    pass


def save_set(eeg_struct, output_path: Path):
    """Save the .set file to disk."""
    pass


def xdf_to_set(xdf_path: Path, output_path: Path) -> Dict:
    """High-level pipeline:
    
    1. Load XDF
    2. Extract EEG
    3. Extract event timestamps
    4. Convert timestamps → latencies
    5. Construct EEGLAB struct
    6. Save .set
    
    Returns:
        Summary dict with timings, event counts, etc.
    """
    pass
