import mne
from pathlib import Path

# Load the cleaned set file
set_path = Path("output/cleaned_eeg/P26_cleaned.set")
raw = mne.io.read_raw_eeglab(str(set_path), preload=False, verbose="ERROR")

print(f"Annotations: {len(raw.annotations)}")
print(f"\nFirst 10 annotations:")

for i, (onset, duration, description) in enumerate(zip(
    raw.annotations.onset[:10], 
    raw.annotations.duration[:10], 
    raw.annotations.description[:10]
)):
    print(f"  {i}: onset={onset:.2f}s, duration={duration:.2f}s, desc='{description}'")

# Check if all durations are 0
durations = raw.annotations.duration
print(f"\nDuration stats:")
print(f"  Min: {durations.min()}")
print(f"  Max: {durations.max()}")
print(f"  Mean: {durations.mean()}")
print(f"  All zero: {(durations == 0).all()}")

# Check descriptions
descriptions = raw.annotations.description
unique_desc = set(descriptions)
print(f"\nUnique descriptions ({len(unique_desc)}):")
for desc in sorted(unique_desc):
    count = sum(1 for d in descriptions if d == desc)
    print(f"  '{desc}': {count} times")
