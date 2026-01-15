import mne
import sys

fif_path = r'C:\vr_tsst_2025\output\cleaned_eeg\P01_cleaned.set'
try:
    raw = mne.io.read_raw_eeglab(fif_path, preload=False)
    print("Annotations found:", raw.annotations)
    # Print unique annotation descriptions
    print("Unique events:", set(raw.annotations.description))
except Exception as e:
    print(f"Error: {e}")
