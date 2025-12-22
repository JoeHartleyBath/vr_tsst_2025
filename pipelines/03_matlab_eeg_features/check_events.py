"""
Check what events are present in participants with missing conditions.
This helps diagnose why some conditions weren't extracted.
"""

import mne
import yaml
from pathlib import Path

# Participants with missing conditions (from completeness check)
participants_with_missing = [5, 8, 9, 13, 19, 22, 24, 26, 29, 31, 33, 34, 35, 37, 38, 45, 48]

# Load config
config_path = Path('config/conditions.yaml')
with open(config_path) as f:
    config_cond = yaml.safe_load(f)

expected_conditions = list(config_cond['conditions'].keys())
print(f"Expected conditions ({len(expected_conditions)}):")
for cond in expected_conditions:
    print(f"  - {cond}")
print()

# Check each participant
cleaned_eeg_dir = Path('output/cleaned_eeg')

for p in participants_with_missing:
    print(f"=== P{p:02d} ===")
    
    set_file = cleaned_eeg_dir / f"P{p:02d}_cleaned.set"
    if not set_file.exists():
        print(f"  ✗ .set file not found: {set_file}")
        print()
        continue
    
    try:
        # Load with MNE
        raw = mne.io.read_raw_eeglab(set_file, preload=False, verbose=False)
        events, event_dict = mne.events_from_annotations(raw, verbose=False)
        
        print(f"  Found {len(events)} events")
        print(f"  Unique event types ({len(event_dict)}):")
        
        # Count events per type
        from collections import Counter
        event_ids = events[:, 2]
        event_counts = Counter(event_ids)
        
        # Map back to labels
        id_to_label = {v: k for k, v in event_dict.items()}
        
        for event_id, count in sorted(event_counts.items()):
            label = id_to_label.get(event_id, f"ID_{event_id}")
            print(f"    - '{label}' (n={count})")
        
    except Exception as e:
        print(f"  ✗ Error loading: {e}")
    
    print()

print("Done.")
