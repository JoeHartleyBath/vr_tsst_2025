import mne
import numpy as np

# Load cleaned data
raw = mne.io.read_raw_eeglab('output/cleaned_eeg/P01_cleaned.set', preload=True, verbose=False)

print('='*70)
print(' DATA STRUCTURE AND UNITS AUDIT')
print('='*70)

print('\n=== DATA STRUCTURE ===')
print(f'Type: {type(raw)}')
print(f'Data type: Continuous Raw EEG')
print(f'N samples: {raw.n_times}')
print(f'N channels: {len(raw.ch_names)}')
print(f'Sampling rate: {raw.info["sfreq"]} Hz')
print(f'Duration: {raw.times[-1]:.1f} seconds')

print('\n=== ANNOTATIONS/EVENTS ===')
events, event_id = mne.events_from_annotations(raw)
print(f'Event types: {list(event_id.keys())}')
print(f'Total events: {len(events)}')

print('\n=== AMPLITUDE STATISTICS (MNE native units = VOLTS) ===')
# Sample 10 channels x 10 seconds
data_sample = raw.get_data()[:10, :int(10*raw.info["sfreq"])]
print(f'Sample shape (10 chans x 10s): {data_sample.shape}')
print(f'Mean: {data_sample.mean():.6e} V')
print(f'Std: {data_sample.std():.6e} V')
print(f'Min: {data_sample.min():.6e} V')
print(f'Max: {data_sample.max():.6e} V')
print(f'Peak-to-peak: {(data_sample.max() - data_sample.min()):.6e} V')

print('\n=== CONVERTED TO MICROVOLTS (standard EEG scale) ===')
data_uv = data_sample * 1e6
print(f'Mean: {data_uv.mean():.2f} µV')
print(f'Std: {data_uv.std():.2f} µV')
print(f'Min: {data_uv.min():.2f} µV')
print(f'Max: {data_uv.max():.2f} µV')
print(f'Peak-to-peak: {(data_uv.max() - data_uv.min()):.2f} µV')

print('\n=== PER-CHANNEL STATISTICS (first 10 channels, in µV) ===')
for i, ch_name in enumerate(raw.ch_names[:10]):
    ch_data = raw.get_data(picks=[i])[0] * 1e6
    print(f'{ch_name}: std={ch_data.std():.2f} µV, p2p={ch_data.max()-ch_data.min():.2f} µV')

print('\n' + '='*70)
print(' SUMMARY')
print('='*70)
print(f'MNE reads data in: VOLTS')
print(f'Typical EEG std: ~7-8 µV (matches your data!)')
print(f'To get µV from MNE: multiply by 1e6')
print('='*70)
