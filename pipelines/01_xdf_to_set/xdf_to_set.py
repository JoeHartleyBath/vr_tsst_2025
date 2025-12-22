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

    # Scale data: ANT eego streams appear to be in millivolts; EEGLAB expects microvolts
    data = data * 1000.0

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
    data_ch_by_time = data.T.astype(np.float32)

    nbchan = data_ch_by_time.shape[0]
    pnts   = data_ch_by_time.shape[1]

    # time axis in milliseconds (float32 to avoid bloating MAT files)
    times = (np.arange(pnts, dtype=np.float32) / np.float32(srate)) * np.float32(1000)* np.float32(1000)

    
    # --- Load channel names from config ---
    # Determine workspace root
    current = Path.cwd()
    workspace_root = current
    while workspace_root != workspace_root.parent:
        if (workspace_root / "config").exists():
            break
        workspace_root = workspace_root.parent
    
    meta_path = workspace_root / "config/eeg_metadata.yaml"
    with open(meta_path, "r") as f:
        eeg_meta = yaml.safe_load(f)

    channel_names = eeg_meta["channel_names"]

    # Safety check (optional but good practice)
    if len(channel_names) != nbchan:
        raise ValueError(f"Expected {nbchan} channel names, found {len(channel_names)}")

    # --- Chanlocs aworkspace_root / "config/eeglab_template.yaml"tibility ---
    chanlocs_dtype = np.dtype([('labels', 'O')])
    chanlocs = np.array(
        [(name,) for name in channel_names],
        dtype=chanlocs_dtype
    )

    # --- Load EEGLAB template from config ---
    template_path = workspace_root / "config/eeglab_template.yaml"
    with open(template_path, "r") as f:
        template = yaml.safe_load(f)

    # --- Build complete EEGLAB structure ---
    EEG = {
        **template,  # Start with template defaults
        # Override with actual data (ensure all numeric types are float for MATLAB compatibility)
        "nbchan": int(nbchan),
        "trials": int(1),
        "pnts": int(pnts),
        "srate": float(srate),
        "xmin": float(0),
        "xmax": float((pnts - 1) / srate),
        "times": times,
        "data": data_ch_by_time,
        "chanlocs": chanlocs,
    }

    return EEG


def add_events_to_eeg_struct(EEG: dict, events: list) -> dict:
    """
    Attach EEGLAB-style `event` and `urevent` structures to the EEG dict.

    Parameters
    ----------
    EEG : dict
        EEG structure produced by `build_eeglab_struct`.
    events : list of dict
        Each event should be {'latency': int, 'type': <code>} where
        `latency` is a zero-based sample index (as produced by
        `build_eeg_event_list`).

    Returns
    -------
    EEG : dict
        EEG with added `event` and `urevent` fields compatible with
        MATLAB/EEGLAB conventions (latency is 1-based integer).
    """

    # Defensive copy (don't mutate caller's dict unexpectedly)
    EEG = dict(EEG)

    ev_list = []

    for i, ev in enumerate(events, start=1):
        # Convert latency to 1-based sample index for EEGLAB
        latency_1b = int(ev["latency"]) + 1

        # EEGLAB stores type as a string or numeric; use numeric where possible
        ev_type = ev.get("type")

        event_struct = {
            "latency": latency_1b,
            "type": ev_type,
            "duration": 0,
        }

        ev_list.append(event_struct)

    # Convert to numpy structured array for MATLAB compatibility
    if ev_list:
        event_dtype = np.dtype([('latency', 'f8'), ('type', 'O'), ('duration', 'f8')])
        event_array = np.array(
            [(e['latency'], e['type'], e['duration']) for e in ev_list],
            dtype=event_dtype
        )
        EEG["event"] = event_array
    else:
        EEG["event"] = np.array([], dtype=np.dtype([('latency', 'f8'), ('type', 'O'), ('duration', 'f8')]))

    return EEG


def save_set(eeg_struct, output_path: Path):
    """Save the .set file to disk."""
    # Ensure parent exists
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        from scipy.io import savemat
    except Exception as e:
        raise ImportError(
            "Saving EEGLAB .set/.mat requires scipy. Install with `pip install scipy`."
        ) from e

    # MATLABs/EEGLAB expects a struct named `EEG` in the .mat file.
    # `savemat` will convert nested Python dicts/lists into MATLAB structs/arrays
    # where possible. We keep the structure simple and numeric where feasible.
    mat_dict = {"EEG": eeg_struct}

    # Write to .set (MATFILE). Many EEGLAB users use .set files backed by a
    # MATLAB .mat; here we write a .set file that is actually a .mat container.
    # Use the provided extension if present; otherwise default to .set
    if output_path.suffix == "":
        output_path = output_path.with_suffix(".mat")

    savemat(str(output_path), mat_dict, do_compression=True)

    return output_path
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


def add_exposure_type_from_config(
    df_physio: pd.DataFrame,
    config_path: Path = Path("config/conditions.yaml")
) -> pd.DataFrame:
    """
    Assigns experimental condition labels to each physio-row based on
    filter definitions loaded from a YAML config file.

    Parameters
    ----------
    df_physio : pandas.DataFrame
        Input dataframe containing study-phase annotations.
    config_path : Path, optional
        Path to conditions.yaml containing the `conditions` section.

    Returns
    -------
    pandas.DataFrame
        The input dataframe with an additional column `exposure_type`
        representing the assigned condition label for each row.
    """

    # Normalize key columns to improve matching (trim and collapse whitespace)
    cols_to_norm = [
        "Unity Scene",
        "Study_Phase",
        "Shown_Scene",
        "Arithmetic_Task",
        "Participant_State",
    ]
    for col in cols_to_norm:
        if col in df_physio.columns:
            df_physio[col] = (
                df_physio[col].astype(str)
                .str.strip()
                .str.replace(r"\s+", " ", regex=True)
            )

    # Load conditions from config
    config_path = Path(config_path)
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    conditions_dict = config.get("conditions", {})

    if not conditions_dict:
        raise ValueError(f"No 'conditions' section found in {config_path}")

    # Build boolean masks from filter definitions
    condition_masks = []
    label_order = []
    match_counts = {}

    for label, filters in conditions_dict.items():
        # Start with all True
        mask = pd.Series([True] * len(df_physio), index=df_physio.index)

        # AND together all filters for this condition
        for col, value in filters.items():
            # Skip filters for columns not in the dataframe (e.g., 'duration' used only for feature extraction)
            if col not in df_physio.columns:
                continue

            if isinstance(value, list):
                # Multiple allowed values: check membership
                mask &= df_physio[col].isin(value)
            else:
                # Single value: check equality
                mask &= (df_physio[col] == value)

        match_count = int(mask.sum())
        match_counts[label] = match_count
        condition_masks.append(mask)
        label_order.append(label)

    # Store match counts in dataframe as metadata for caller to log
    df_physio.attrs['label_match_counts'] = match_counts

    # Assign exposure_type using np.select
    df_physio["exposure_type"] = np.select(
        condition_masks,
        label_order,
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

    if not np.issubdtype(df_physio.index.dtype, np.datetime64):
        raise ValueError("df_physio index must be datetime64[ns].")

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
    # Simple loop approach to avoid pandas version compatibility issues
    event_ts_dict = {}
    for label in valid["exposure_type"].unique():
        first_ts = valid[valid["exposure_type"] == label].index[0]
        event_ts_dict[label] = np.datetime64(first_ts)

    return event_ts_dict


def extract_response_timestamps(df_physio: pd.DataFrame) -> Dict[str, List[np.datetime64]]:
    """
    Identify true response events from the physio metadata.

    Rules (based on metadata behavior):
    - The Response column latches the last value across many rows.
    - The Response_Time column is a per-trial timer that resets on each response.
    - Emit an event when Response_Time decreases (reset), and also emit the
      first non-null Response within each task block so single-response tasks
      are not missed.

    Assumptions:
    - df_physio index is datetime64[ns]
    - 'Response' contains exactly: 'Correct', 'Incorrect', or NaN
    - 'Response_Time' exists and resets to a low value (often 0) on each response
    """

    required_cols = {"Response", "Response_Time"}
    if not required_cols.issubset(df_physio.columns):
        return {}

    # Ensure we have datetime index for downstream conversions
    if not np.issubdtype(df_physio.index.dtype, np.datetime64):
        raise ValueError("df_physio index must be datetime64[ns].")

    resp = df_physio["Response"]
    resp_time = pd.to_numeric(df_physio["Response_Time"], errors="coerce")
    resp_notna = resp.notna()

    # Detect timer resets: Response_Time drops relative to the previous sample.
    reset_mask = (
        resp_time.notna()
        & resp_time.shift().notna()
        & (resp_time < resp_time.shift())
        & resp_notna
    )

    candidate_idx = pd.Index(df_physio.index[reset_mask])

    # Also capture the first non-null Response within each task block so we
    # still emit an event when only a single response occurs in that block.
    task_key = None
    if "Arithmetic_Task" in df_physio.columns:
        task_key = df_physio["Arithmetic_Task"].ffill()
    elif "Task_State" in df_physio.columns:
        task_key = df_physio["Task_State"].ffill()

    if task_key is not None:
        task_df = (
            df_physio.assign(_task=task_key)
            .loc[resp_notna & task_key.notna(), ["Response", "_task"]]
        )
        if not task_df.empty:
            first_per_task = task_df.groupby("_task", sort=False).head(1)
            candidate_idx = candidate_idx.union(first_per_task.index)
    else:
        # Fallback: at least include the first observed non-null response overall
        if resp_notna.any():
            candidate_idx = candidate_idx.union(resp.index[resp_notna].take([0]))

    if candidate_idx.empty:
        return {}

    # Build response event dict with de-duplicated, time-sorted indices
    response_ts_dict: Dict[str, List[np.datetime64]] = {}
    for ts in candidate_idx.sort_values():
        val = resp.loc[ts]
        if pd.isna(val):
            continue
        label = f"Response_{val}"
        response_ts_dict.setdefault(label, []).append(np.datetime64(ts))

    return response_ts_dict


  
def build_eeg_event_list(
        eeg_dt_aligned: np.ndarray,
        event_ts_dict: dict,
        srate: float,
        export_event_labels: dict,
        ignore_missing: bool = True,
    ) -> list:
    """
    Convert per-condition onset timestamps into EEGLAB-style event markers.

    Parameters
    ----------
    eeg_dt_aligned : np.ndarray of datetime64[ns]
        Aligned EEG timestamps (one per sample).
    event_ts_dict : dict
        Mapping {exposure_type: timestamp} from extract_event_timestamps().
    srate : float
        EEG sampling rate.
    export_event_labels : dict
        Maps exposure_type → short event code to use in EEG.
    ignore_missing : bool
        If True, skip events whose timestamps fall outside EEG recording.

    Returns
    -------
    List[dict]
        Each element is {"latency": int, "type": str}
    """

    events = []
    eeg_start = eeg_dt_aligned[0]
    eeg_end   = eeg_dt_aligned[-1]

    for exposure, onset_ts_or_list in event_ts_dict.items():

        # Skip if label not selected for export
        # In pilot config, export_event_labels is a list, not a mapping.
        # Use the exposure label itself as the event type when included.
        if isinstance(export_event_labels, dict):
            if exposure not in export_event_labels:
                continue
            event_type = export_event_labels[exposure]
        else:
            if exposure not in export_event_labels:
                continue
            event_type = exposure

        # Handle both single timestamp and list of timestamps
        timestamps = onset_ts_or_list if isinstance(onset_ts_or_list, list) else [onset_ts_or_list]

        for onset_ts in timestamps:
            # Check EEG spans
            if onset_ts < eeg_start or onset_ts > eeg_end:
                if ignore_missing:
                    continue
                else:
                    raise ValueError(f"Timestamp {onset_ts} for {exposure} is outside EEG span.")

            # Convert timestamp → sample index
            delta_sec = (onset_ts - eeg_start) / np.timedelta64(1, "s")
            sample_idx = int(round(delta_sec * srate))

            events.append({
                "latency": sample_idx,
                "type": event_type
            })

    # Sort by sample index
    events.sort(key=lambda e: e["latency"])

    return events


def save_set(eeg_struct, output_path: Path):
    """Save the .set file to disk."""
    # Ensure parent exists
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        from scipy.io import savemat
    except Exception as e:
        raise ImportError(
            "Saving EEGLAB .set/.mat requires scipy. Install with `pip install scipy`."
        ) from e

    # MATLABs/EEGLAB expects a struct named `EEG` in the .mat file.
    # `savemat` will convert nested Python dicts/lists into MATLAB structs/arrays
    # where possible. We keep the structure simple and numeric where feasible.
    mat_dict = {"EEG": eeg_struct}

    # Write to .set (MATFILE). Many EEGLAB users use .set files backed by a
    # MATLAB .mat; here we write a .set file that is actually a .mat container.
    # Use the provided extension if present; otherwise default to .set
    if output_path.suffix == "":
        output_path = output_path.with_suffix(".set")

    # Write with row-oriented vectors so event arrays are 1xN
    savemat(
        str(output_path),
        mat_dict,
        do_compression=True,
        oned_as='row'
    )

    return output_path


def add_events_to_eeg_struct(EEG: dict, events: list) -> dict:
    """
    Attach EEGLAB-style `event` and `urevent` structures to the EEG dict.

    Parameters
    ----------
    EEG : dict
        EEG structure produced by `build_eeglab_struct`.
    events : list of dict
        Each event should be {'latency': int, 'type': <code>} where
        `latency` is a zero-based sample index (as produced by
        `build_eeg_event_list`).

    Returns
    -------
    EEG : dict
        EEG with added `event` and `urevent` fields compatible with
        MATLAB/EEGLAB conventions (latency is 1-based integer).
    """

    # Defensive copy (don't mutate caller's dict unexpectedly)
    EEG = dict(EEG)

    ev_list = []
    urevent_list = []

    for i, ev in enumerate(events, start=1):
        # Convert latency to 1-based sample index for EEGLAB
        latency_1b = int(ev["latency"]) + 1

        # EEGLAB stores type as a string or numeric; use numeric where possible
        ev_type = ev.get("type")

        event_struct = {
            "latency": latency_1b,
            "type": ev_type,
            "duration": 0,
        }

        ev_list.append(event_struct)

    # Convert to numpy structured array for MATLAB compatibility
    if ev_list:
        event_dtype = np.dtype([('latency', 'f8'), ('type', 'O'), ('duration', 'f8')])
        event_array = np.array(
            [(e['latency'], e['type'], e['duration']) for e in ev_list],
            dtype=event_dtype
        )
        EEG["event"] = event_array
    else:
        EEG["event"] = np.array([], dtype=np.dtype([('latency', 'f8'), ('type', 'O'), ('duration', 'f8')]))

    return EEG


def xdf_to_set(
    xdf_path: Path,
    output_path: Path,
    physio_path: Path | None = None,
    config_path: Path = Path("config/conditions.yaml"),
    log_label_matches: bool = True,
) -> Dict:
    """High-level pipeline that converts an XDF to an EEGLAB .set (MAT) file.

    Steps:
      - load and merge EEG streams
      - load physio metadata and assign exposure labels
      - align timestamps and extract first-onset timestamps
      - convert timestamps into event latencies
      - build EEGLAB struct and attach events
      - save .set/.mat to disk

    Returns a summary dict with keys: `path`, `n_events`, `nbchan`, `pnts`, `srate`, `label_matches`, `event_types`.
    """

    xdf_path = Path(xdf_path)
    output_path = Path(output_path)

    # Determine workspace root (look for config/ directory as marker)
    current = Path.cwd()
    workspace_root = current
    while workspace_root != workspace_root.parent:
        if (workspace_root / "config").exists():
            break
        workspace_root = workspace_root.parent
    
    # 1) Load/merge EEG streams
    merged = load_and_merge(xdf_path)

    eeg_ts = merged["timestamps"]
    eeg_data = merged["data"]
    srate = merged["srate"]

    # 2) Resolve physio CSV path if not provided (handle RAW/raw case-insensitive)
    if physio_path is None:
        candidates = [
            workspace_root / "data/RAW/metadata" / f"{xdf_path.stem}.csv",
            workspace_root / "data/raw/metadata" / f"{xdf_path.stem}.csv",
        ]
        physio_found = None
        for physio_candidate in candidates:
            if physio_candidate.exists():
                physio_found = physio_candidate
                break
        if physio_found is None:
            raise FileNotFoundError(
                "Physio metadata not provided and could not find any of: "
                + ", ".join(str(c) for c in candidates)
            )
        physio_path = physio_found

    # 3) Load physio and condition config
    df_physio = pd.read_csv(physio_path, low_memory=False)
    cfg = load_condition_config(Path(config_path))

    exposure_labels = cfg.get("exposure_labels")
    export_labels = cfg.get("export_event_labels")

    # 4) Assign exposure types and prepare datetime index
    df_physio = add_exposure_type_from_config(df_physio, Path(config_path))
    df_physio["LSL_Timestamp"] = pd.to_datetime(df_physio["LSL_Timestamp"], unit="s", origin="unix")
    df_physio = df_physio.set_index("LSL_Timestamp")

    # 5) Align timestamps (produce datetime64 array aligned to physio)
    aligned = align_timestamps(df_physio, eeg_data, eeg_ts, srate)

    # 6) Extract per-condition first timestamps and response state transitions
    event_ts = extract_event_timestamps(df_physio)
    response_ts = extract_response_timestamps(df_physio)

    # Merge exposure and response events
    merged_events = {**event_ts, **response_ts}

    # 7) Convert to EEG event markers (latencies)
    event_list = build_eeg_event_list(
        eeg_dt_aligned=aligned,
        event_ts_dict=merged_events,
        srate=srate,
        export_event_labels=export_labels,
    )

    # 8) Build EEGLAB struct and attach events
    EEG = build_eeglab_struct(merged)
    EEG = add_events_to_eeg_struct(EEG, event_list)

    # 9) Set metadata before saving
    EEG["setname"] = output_path.stem
    EEG["filename"] = output_path.name
    EEG["filepath"] = str(output_path.parent)

    # 10) Save to disk
    saved = save_set(EEG, output_path)
# Collect event type counts
    event_types = {}
    for ev in event_list:
        ev_type = ev.get("type")
        event_types[ev_type] = event_types.get(ev_type, 0) + 1

    # Build summary with label match counts
    summary = {
        "path": str(saved),
        "n_events": len(event_list),
        "nbchan": EEG.get("nbchan"),
        "pnts": EEG.get("pnts"),
        "srate": EEG.get("srate"),
        "label_matches": df_physio.attrs.get('label_match_counts', {}),
        "event_types": event_types,
    }

    # Optional: log label matches if requested
    if log_label_matches and 'label_match_counts' in df_physio.attrs:
        match_counts = df_physio.attrs['label_match_counts']
        print(f"  Label matches (top 10):")
        sorted_matches = sorted(match_counts.items(), key=lambda x: x[1], reverse=True)
        for label, count in sorted_matches[:10]:
            print(f"    {label}: {count}")

    return summary
