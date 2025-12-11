"""
xdf_to_set.py

Refactored XDF → SET conversion module.
Defines the public interface for importing raw .xdf files,
embedding canonical condition events, and saving EEGLAB .set files.

This file contains function signatures only. Implementation comes later.
"""

# -------------------------------------------------------------------------
# Channel Handling (Design Note)
# -------------------------------------------------------------------------
# The raw eego streams contain 66 channels:
# - 64 EEG channels
# - 1 trigger channel
# - 1 counter channel
#
# For now, all channels are preserved in `extract_single_stream()` and
# during stream alignment. This ensures:
# - alignment uses the true raw structure
# - future studies can leverage trigger channels if needed
# - no assumptions are baked into the early pipeline stages
#
# The auxiliary channels (trigger + counter) will be removed in a separate
# `strip_aux_channels()` step *after* alignment and *before* merging A/B
# streams. Channel naming logic will also be added at that stage.
#
# This is intentional. Do not remove or rename channels earlier in the
# pipeline unless the design changes.
# -------------------------------------------------------------------------


from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import yaml
import pandas as pd

def load_condition_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config



def load_and_merge(xdf_path):
    """
    Convenience function that runs:
    1. load_xdf
    2. find_eeg_streams
    3. extract_single_stream
    4. align_streams
    5. strip_aux_channels
    6. merge_streams

    Returns:
        merged_stream (dict): {
            "data": (samples, 128),
            "timestamps": np.ndarray,
            "srate": 500,
            "name": "A__B",
        }
    """
    # 1. Load XDF
    xdf = load_xdf(xdf_path)

    # 2. Find streams
    eegA, eegB = find_eeg_streams(xdf["streams"])

    # 3. Extract
    A = extract_single_stream(eegA)
    B = extract_single_stream(eegB)

    # 4. Align
    A_aligned, B_aligned = align_streams(A, B)

    # 5. Strip aux channels
    A_clean = strip_aux_channels(A_aligned)
    B_clean = strip_aux_channels(B_aligned)

    # 6. Merge
    merged = merge_streams(A_clean, B_clean)

    return merged


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

def strip_aux_channels(stream: Dict) -> Dict:
    """
    Remove auxiliary channels (trigger + counter) from a 66-channel eego stream.
    Returns a new stream dict with shape (samples, 64).
    """
    data = stream["data"]

    if data.shape[1] != 66:
        raise ValueError(f"Expected 66 channels before stripping, got {data.shape[1]}")

    # EEG channels are always the first 64
    eeg_only = data[:, :64]

    return {
        "name": stream["name"],
        "srate": stream["srate"],
        "timestamps": stream["timestamps"],
        "data": eeg_only,
    }

def merge_streams(streamA: Dict, streamB: Dict) -> Dict:
    """
    Merge two aligned, aux-stripped EEG streams (each 64 channels)
    into a single 128-channel EEG stream.
    Assumes both streams have identical timestamps and sample counts.
    """

    # --- Sanity checks ---
    dataA = streamA["data"]
    dataB = streamB["data"]

    if dataA.shape[0] != dataB.shape[0]:
        raise ValueError(f"Stream lengths differ: {dataA.shape[0]} vs {dataB.shape[0]}")

    if dataA.shape[1] != 64 or dataB.shape[1] != 64:
        raise ValueError("Both streams must have 64 channels before merging.")

    # --- Merge (concat channels) ---
    merged_data = np.concatenate([dataA, dataB], axis=1)   # shape: (N, 128)

    # --- Build output ---
    merged_stream = {
        "srate": streamA["srate"],                  # same for both streams
        "timestamps": streamA["timestamps"],        # identical after alignment
        "data": merged_data,                        # (N, 128)
        "name": f"{streamA['name']}__{streamB['name']}",
    }

    return merged_stream

def build_eeglab_struct(merged_stream: dict) -> dict:
    """
    Minimal EEGLAB-style container.
    Adds:
      - data (channels x samples)
      - sampling rate
      - nbchan
      - pnts
      - trials=1
      - times (ms)
      - chanlocs with real channel names
    """
    data = merged_stream["data"]          # (samples, 128)
    srate = merged_stream["srate"]

    # EEGLAB format: (channels, samples)
    data_ch_by_time = data.T

    nbchan = data_ch_by_time.shape[0]
    pnts   = data_ch_by_time.shape[1]

    # time axis in milliseconds
    times = np.arange(pnts) / srate * 1000
    
    # --- Load channel names from config ---
    meta_path = Path("config/eeg_metadata.yaml")
    with open(meta_path, "r") as f:
        eeg_meta = yaml.safe_load(f)

    channel_names = eeg_meta["channel_names"]

    # Safety check (optional but good practice)
    if len(channel_names) != nbchan:
        raise ValueError(f"Expected {nbchan} channel names, found {len(channel_names)}")

    # --- Chanlocs ---
    chanlocs = [{"labels": name} for name in channel_names]


    EEG = {
        "data": data_ch_by_time,
        "srate": srate,
        "nbchan": nbchan,
        "pnts": pnts,
        "trials": 1,          # continuous data
        "times": times,       # (pnts,) in ms
        "chanlocs": chanlocs,
    }

    return EEG


def add_exposure_type(df_physio: pd.DataFrame, exposure_labels: list[str]) -> pd.DataFrame:
    """
    Assigns experimental condition labels to each physio-row based on predefined boolean
    expressions. The function reproduces the study’s legacy condition-mapping logic while
    sourcing label names from an external configuration file.

    Parameters
    ----------
    df_physio : pandas.DataFrame
        Input dataframe containing study-phase annotations and sensor timestamps.
    exposure_labels : list of str
        Ordered list of condition labels corresponding to the boolean expressions.

    Returns
    -------
    pandas.DataFrame
        The input dataframe with an additional column `exposure_type` representing the
        assigned condition label for each row.
    """


    # Define conditions and corresponding labels
    conditions = [
        #Calibrations
        (df_physio['Unity Scene'] == 'PrimaryCalibration'),
        (df_physio['Unity Scene'] == 'BlinkCalibration'),
        (df_physio['Unity Scene'] == 'BaseLine'),
        #Fixation cross scenes
        ((df_physio['Unity Scene'] == 'Fixation') &
         (df_physio['Study_Phase'] == 'Start Blank Fixation')),
         ((df_physio['Unity Scene'] == 'Fixation') &
         (df_physio['Study_Phase'] == 'Start Room Fixation')),
         ((df_physio['Unity Scene'] == 'Fixation') &
         (df_physio['Study_Phase'] == 'End Room Fixation')),
         ((df_physio['Unity Scene'] == 'Fixation') &
         (df_physio['Study_Phase'] == 'End Blank Fixation')),
        #Preambles
    #Stress 1022 preamble
    (
        (df_physio['Shown_Scene'] == 'StressRoom') & 
        (df_physio['Arithmetic_Task'] == 'Subtraction1022') & 
        (df_physio['Participant_State'].isin(['Instructions', 'Begin', 'TaskAction']))
    ),
    #Stress 2043 preamble
    (
        (df_physio['Shown_Scene'] == 'StressRoom') & 
        (df_physio['Arithmetic_Task'] == 'Subtraction2043') & 
        (df_physio['Participant_State'].isin(['Instructions', 'Begin', 'TaskAction']))
    ),
    # Preamble for Stress Addition
    (
        (df_physio['Shown_Scene'] == 'StressRoom') & 
        (df_physio['Arithmetic_Task'] == 'Addition') & 
        (df_physio['Participant_State'].isin(['Instructions', 'Begin', 'TaskAction']))
    ),
    #Calm 1022 preamble
    (
        (df_physio['Shown_Scene'] == 'CalmRoom') & 
        (df_physio['Arithmetic_Task'] == 'Subtraction1022') & 
        (df_physio['Participant_State'].isin(['Instructions', 'Begin', 'TaskAction']))
    ),
    #Calm 2043 preamble
    (
        (df_physio['Shown_Scene'] == 'CalmRoom') & 
        (df_physio['Arithmetic_Task'] == 'Subtraction2043') & 
        (df_physio['Participant_State'].isin(['Instructions', 'Begin', 'TaskAction']))
    ),
    # Preamble for Calm Addition
    (
        (df_physio['Shown_Scene'] == 'CalmRoom') & 
        (df_physio['Arithmetic_Task'] == 'Addition') & 
        (df_physio['Participant_State'].isin(['Instructions', 'Begin', 'TaskAction']))
    ),
    # TaskTime for high cognitive effort
    (
        (df_physio['Shown_Scene'] == 'StressRoom') & 
        (df_physio['Arithmetic_Task'] == 'Subtraction1022') & 
        (df_physio['Participant_State'] == 'TaskTime')
    ),
    (
        (df_physio['Shown_Scene'] == 'StressRoom') & 
        (df_physio['Arithmetic_Task'] == 'Subtraction2043') & 
        (df_physio['Participant_State'] == 'TaskTime')
    ),
    # TaskTime for low cognitive effort
    (
        (df_physio['Shown_Scene'] == 'StressRoom') & 
        (df_physio['Arithmetic_Task'] == 'Addition') & 
        (df_physio['Participant_State'] == 'TaskTime')
    ),
    # Same for CalmRoom
    (
        (df_physio['Shown_Scene'] == 'CalmRoom') & 
        (df_physio['Arithmetic_Task'] == 'Subtraction1022') & 
        (df_physio['Participant_State'] == 'TaskTime')
    ),
    (
        (df_physio['Shown_Scene'] == 'CalmRoom') & 
        (df_physio['Arithmetic_Task'] == 'Subtraction2043') & 
        (df_physio['Participant_State'] == 'TaskTime')
    ),
    (
        (df_physio['Shown_Scene'] == 'CalmRoom') & 
        (df_physio['Arithmetic_Task'] == 'Addition') & 
        (df_physio['Participant_State'] == 'TaskTime')
    ),
# Thanks Action for high cognitive effort
    (
        (df_physio['Shown_Scene'] == 'StressRoom') & 
        (df_physio['Arithmetic_Task'] == 'Subtraction1022') & 
        (df_physio['Participant_State'] == 'ThanksAction')
    ),
    (
        (df_physio['Shown_Scene'] == 'StressRoom') & 
        (df_physio['Arithmetic_Task'] == 'Subtraction2043') & 
        (df_physio['Participant_State'] == 'ThanksAction')
    ),
    # Thanks Action for low cognitive effort
    (
        (df_physio['Shown_Scene'] == 'StressRoom') & 
        (df_physio['Arithmetic_Task'] == 'Addition') & 
        (df_physio['Participant_State'] == 'ThanksAction')
    ),
    # Same for CalmRoom
    (
        (df_physio['Shown_Scene'] == 'CalmRoom') & 
        (df_physio['Arithmetic_Task'] == 'Subtraction1022') & 
        (df_physio['Participant_State'] == 'ThanksAction')
    ),
    (
        (df_physio['Shown_Scene'] == 'CalmRoom') & 
        (df_physio['Arithmetic_Task'] == 'Subtraction2043') & 
        (df_physio['Participant_State'] == 'ThanksAction')
    ),
    (
        (df_physio['Shown_Scene'] == 'CalmRoom') & 
        (df_physio['Arithmetic_Task'] == 'Addition') & 
        (df_physio['Participant_State'] == 'ThanksAction')
    ),

        (df_physio['Shown_Scene'] == 'Forest1'),
        (df_physio['Shown_Scene'] == 'Forest2'),
        (df_physio['Shown_Scene'] == 'Forest3'),
        (df_physio['Shown_Scene'] == 'Forest4'),
    ]

    #2.) load labels from config

    exposure_labels = exposure_labels

    # Safety check — same number?
    if len(conditions) != len(exposure_labels):
        raise ValueError(
            f"Number of conditions ({len(conditions)}) does not match "
            f"number of labels ({len(exposure_labels)})."
        )
    
    #3.) assign exposure labels as exposure type
    df_physio["exposure_type"] = np.select(
        conditions,
        exposure_labels,
        default="no exposure"
    )

    return df_physio

def align_timestamps(df_physio: pd.DataFrame,
                     eeg_data: np.ndarray,
                     eeg_ts: np.ndarray,
                     srate: float) -> np.ndarray:
    """
    Reproduce the legacy timestamp alignment logic exactly.

    Parameters
    ----------
    df_physio : pd.DataFrame
        Contains 'LSL_Timestamp' column in LSL seconds.
    eeg_data : np.ndarray
        EEG data array (samples x channels). Only the sample count is used.
    eeg_ts : np.ndarray
        Raw EEG timestamps from XDF (only the first element is used).
    srate : float
        EEG sampling rate.

    Returns
    -------
    aligned_eeg_dt : np.ndarray of datetime64[ns]
        EEG timestamps aligned to physio's datetime reference frame.
    """

    # --- 1) Convert physio timestamps to datetime and set index ---
    df_physio["LSL_Timestamp"] = pd.to_datetime(df_physio["LSL_Timestamp"], unit="s", origin="unix")
    df_physio = df_physio.set_index("LSL_Timestamp")
    physio_first = df_physio.index[0]

    # --- 2) Convert EEG first timestamp to datetime ---
    eeg_first_dt = pd.to_datetime(eeg_ts[0], unit="s", origin="unix")

    # --- 3) Construct a proper datetime index for EEG using sample count ---
    n_samples = eeg_data.shape[0]
    eeg_time_deltas = pd.to_timedelta(np.arange(n_samples) / srate, unit="s")
    eeg_dt = eeg_first_dt + eeg_time_deltas

    # --- 4) Compute offset between physio start and EEG start ---
    gap = eeg_first_dt - physio_first

    # --- 5) Shift EEG timestamps into physio's reference frame ---
    aligned_eeg_dt = eeg_dt - gap

    # --- 6) Return clean datetime64 array ---
    return aligned_eeg_dt.to_numpy(dtype='datetime64[ns]')



def extract_event_timestamps(df_physio: pd.DataFrame) -> Dict[str, np.datetime64]:
    """
    Extract the FIRST onset timestamp for each exposure_type.

    Parameters
    ----------
    df_physio : pd.DataFrame
        Must contain:
            - 'exposure_type' column with condition labels
            - datetime64 index (LSL_Timestamp)

    Returns
    -------
    Dict[str, np.datetime64]
        Mapping of exposure label → first timestamp where it appears.
    """

    # Safety check: ensure the index is datetime
    if not np.issubdtype(df_physio.index.dtype, np.datetime64):
        raise ValueError("df_physio index must be datetime64[ns].")

    # Remove rows with missing exposure labels
    valid = df_physio[df_physio["exposure_type"] != "no exposure"]

    # Group by exposure_type and take the FIRST timestamp for each
    first_ts = valid.groupby("exposure_type").apply(lambda x: x.index[0])

    # Convert to Python dict of numpy.datetime64
    event_ts_dict = {label: np.datetime64(ts) for label, ts in first_ts.items()}

    return event_ts_dict
  

def extract_event_timestamps(df_phsyio: pd.DataFrame) -> Dict[str, List[float]]:
    """Extract condition onset timestamps from df_phsyio.
    
    Returns:
        A dict mapping event_type → list of timestamps.
        Example:
            {"HS_LL": [12.4], "HS_HL": [215.2], ...}
    """




    pass


def convert_events_to_latencies(eeg_timestamps, event_ts_dict) -> Dict[str, List[int]]:
    """Convert LSL timestamps to EEG sample latencies."""
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
