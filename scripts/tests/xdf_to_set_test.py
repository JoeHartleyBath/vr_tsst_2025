import pandas as pd
from pathlib import Path

# Import ONLY the functions you want to test today
from scripts.xdf_to_set.xdf_to_set import (
    load_and_merge,
    align_timestamps,
    extract_event_timestamps,
    add_exposure_type,
    load_condition_config,
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

    df_physio = add_exposure_type(df_physio, labels)

    # --- Ensure LSL_Timestamp exists and is datetime ---
    df_physio["LSL_Timestamp"] = pd.to_datetime(df_physio["LSL_Timestamp"], unit="s", origin="unix")
    df_physio = df_physio.set_index("LSL_Timestamp")

    events = extract_event_timestamps(df_physio)

    print("Event keys:", list(events.keys())[:10])
    for k, v in events.items():
        print(f"{k}: {pd.to_datetime(v)}")


    # ----------------------------------------------------------------------
    # Done
    # ----------------------------------------------------------------------
