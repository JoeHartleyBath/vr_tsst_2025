import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def load_event_durations(config_path: Path) -> dict:
    """Load event code to (name, duration) mapping from conditions.yaml"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Build reverse mapping: event_code -> (condition_name, duration)
    code_to_condition = {}
    
    export_events = config.get('export_event_labels', {})
    conditions = config.get('conditions', {})
    
    for condition_name, event_code in export_events.items():
        if condition_name in conditions:
            duration = conditions[condition_name].get('duration', 0)
            code_to_condition[str(event_code)] = (condition_name, duration)
    
    return code_to_condition


def export_full_eeg(*, set_path: Path, out_csv: Path, code_to_condition: dict) -> None:
    import mne

    raw = mne.io.read_raw_eeglab(str(set_path), preload=True, verbose="ERROR")

    sfreq = float(raw.info["sfreq"])
    n_channels = raw.info["nchan"]
    n_samples = raw.n_times
    
    print(f"Loaded {set_path.name}: nchan={n_channels}, sfreq={sfreq:.3f} Hz, n_samples={n_samples}")

    # Get all data
    data = raw.get_data()  # shape (n_channels, n_samples)
    time_s = np.arange(n_samples, dtype=np.float64) / sfreq

    # Create DataFrame with time as first column
    df = pd.DataFrame(data.T, columns=list(raw.ch_names))
    df.insert(0, "time_s", time_s)
    
    # Add event column using code_to_condition mapping
    if raw.annotations is not None and len(raw.annotations) > 0:
        print(f"Processing {len(raw.annotations)} event annotations...")
        event_labels = [""] * n_samples
        
        for onset, _, description in zip(
            raw.annotations.onset, 
            raw.annotations.duration, 
            raw.annotations.description
        ):
            event_code = str(description).strip()
            
            # Look up condition name and duration from config
            if event_code in code_to_condition:
                condition_name, duration_s = code_to_condition[event_code]
                
                start_idx = int(onset * sfreq)
                end_idx = int((onset + duration_s) * sfreq)
                
                # Clamp to valid range
                start_idx = max(0, start_idx)
                end_idx = min(n_samples, end_idx)
                
                for i in range(start_idx, end_idx):
                    event_labels[i] = condition_name
        
        df.insert(1, "event", event_labels)
        
        # Count unique events
        unique_events = set([e for e in event_labels if e])
        print(f"Found {len(unique_events)} unique events: {', '.join(sorted(unique_events))}")
    else:
        print("No annotations found in dataset")

    # Export to CSV
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    duration_s = n_samples / sfreq
    print(f"Wrote {out_csv.name}: duration={duration_s:.2f}s, samples={n_samples}, channels={n_channels}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export complete EEG data from EEGLAB .set files to CSV")
    parser.add_argument("--cleaned-set", type=str, required=True, help="Path to cleaned .set")
    parser.add_argument("--raw-set", type=str, required=True, help="Path to raw-with-chanlocs .set")
    parser.add_argument("--out-dir", type=str, required=True, help="Output directory")
    parser.add_argument("--config", type=str, default="config/conditions.yaml", help="Path to conditions.yaml")

    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Load event code to condition mapping
    config_path = Path(args.config)
    print(f"Loading event mappings from {config_path}...")
    code_to_condition = load_event_durations(config_path)
    print(f"Loaded {len(code_to_condition)} event code mappings")

    print("\nExporting cleaned EEG data...")
    export_full_eeg(
        set_path=Path(args.cleaned_set),
        out_csv=out_dir / "P26_cleaned_full.csv",
        code_to_condition=code_to_condition,
    )

    print("\nExporting raw EEG data...")
    try:
        export_full_eeg(
            set_path=Path(args.raw_set),
            out_csv=out_dir / "P26_raw_full.csv",
            code_to_condition=code_to_condition,
        )
    except Exception as e:
        print(f"Warning: Could not export raw data: {e}")
        print("Continuing with cleaned data only...")


if __name__ == "__main__":
    main()
