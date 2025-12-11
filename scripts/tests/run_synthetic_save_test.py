import numpy as np
from pathlib import Path
from scripts.xdf_to_set.xdf_to_set import build_eeglab_struct, add_events_to_eeg_struct, save_set

if __name__ == '__main__':
    # create synthetic 128-channel, 1000-sample data
    samples = 1000
    channels = 128
    data = np.random.randn(samples, channels)
    merged = {"data": data, "srate": 500.0, "timestamps": np.arange(samples) / 500.0, "name": "A__B"}

    EEG = build_eeglab_struct(merged)

    # Add a couple of synthetic events
    events = [{"latency": 10, "type": 1}, {"latency": 200, "type": 2}]
    EEG = add_events_to_eeg_struct(EEG, events)

    out = Path(r"C:\phd_projects\vr_tsst_2025\output\synthetic_test.set")
    saved = save_set(EEG, out)
    print('Saved to', saved)

    # Verify mat file
    try:
        from scipy.io import loadmat
        m = loadmat(str(saved))
        print('MAT keys:', list(m.keys()))
        if 'EEG' in m:
            print('EEG present in .mat — keys under EEG struct may be numeric/obj depending on scipy version')
    except Exception as e:
        print('Could not load saved mat:', e)
