import sys
sys.path.insert(0, 'tools')
from export_p26_csv_segments import load_event_durations
from pathlib import Path

config_path = Path("config/conditions.yaml")
code_to_condition = load_event_durations(config_path)

print(f"Loaded {len(code_to_condition)} event mappings:\n")
for code, (name, duration) in sorted(code_to_condition.items()):
    print(f"  Code {code:>3}: {name:50} ({duration}s)")
