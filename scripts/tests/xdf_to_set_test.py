import sys
from pathlib import Path
import pandas as pd

# Add parent dir to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import ONLY the functions you want to test today
from xdf_to_set.xdf_to_set import (
    load_and_merge,
    align_timestamps,
    extract_event_timestamps,
    extract_response_timestamps,
    add_exposure_type_from_config,
    load_condition_config,
    build_eeg_event_list,
)

# ---- File paths ----
xdf_path = Path(r"C:\phd_projects\vr_tsst_2025\data\raw\eeg\P01.xdf")
physio_path = Path(r"C:\phd_projects\vr_tsst_2025\data\raw\metadata\P01.csv")
config_path = Path(r"C:\phd_projects\vr_tsst_2025\config\conditions.yaml")


if __name__ == "__main__":

    # Load physio + EEG
    df_physio = pd.read_csv(physio_path, low_memory=False)
    merged = load_and_merge(xdf_path)

    eeg_ts = merged["timestamps"]
    eeg_data = merged["data"]
    srate = merged["srate"]

    # Timestamp alignment (always needed for downstream tests)
    aligned = align_timestamps(df_physio, eeg_data, eeg_ts, srate)

    # ----------------------------------------------------------------------
    # PLACE YOUR TEST HERE — only modify THIS BLOCK each time
    # ----------------------------------------------------------------------

    cfg = load_condition_config(config_path)
    labels = cfg["exposure_labels"]
    export_labels = cfg["export_event_labels"]

    # --- Add exposure_type labels ---
    df_physio = add_exposure_type_from_config(df_physio, config_path)

    # --- Ensure timestamp index is correct ---
    df_physio["LSL_Timestamp"] = pd.to_datetime(df_physio["LSL_Timestamp"], unit="s", origin="unix")
    df_physio = df_physio.set_index("LSL_Timestamp")

    # --- 1) Test extraction of first timestamps ---
    event_ts = extract_event_timestamps(df_physio)

    print("\n=== Extracted exposure onset timestamps ===")
    for k, v in event_ts.items():
        print(f"{k:35s} {pd.to_datetime(v)}")

    # --- 1b) Test extraction of response state transitions ---
    response_ts = extract_response_timestamps(df_physio)

    print("\n=== Extracted response state transition timestamps ===")
    for k, v_list in response_ts.items():
        print(f"{k:35s} {len(v_list)} transitions")
        for i, ts in enumerate(v_list[:3]):  # show first 3
            print(f"  [{i}] {pd.to_datetime(ts)}")
        if len(v_list) > 3:
            print(f"  ... ({len(v_list) - 3} more)")

    # --- 2) Test conversion to EEG event markers ---
    # Merge both exposure and response events
    merged_events = {**event_ts, **response_ts}

    event_list = build_eeg_event_list(
        eeg_dt_aligned=aligned,
        event_ts_dict=merged_events,
        srate=srate,
        export_event_labels=export_labels
    )

    print("\n=== EEG event list (latency + type) ===")
    for ev in event_list[:10]:   # print first 10 only
        print(ev)

    print(f"\nTotal events created: {len(event_list)}")



    # ----------------------------------------------------------------------
    # Done
    # ----------------------------------------------------------------------
