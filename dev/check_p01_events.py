from scipy.io import loadmat
import numpy as np

# Load P01
m = loadmat('output/sets/P01.set')
EEG = m['EEG'][0,0]

print(f"Channels: {EEG['nbchan'][0,0]}")
print(f"Samples: {EEG['pnts'][0,0]}")
print(f"Sample rate: {EEG['srate'][0,0]} Hz")
print()

# Get events
events = EEG['event'][0]
print(f"Total events: {len(events)}")
print()

print("All events:")
for i, ev in enumerate(events, 1):
    latency = int(ev['latency'][0,0])
    ev_type = ev['type'][0,0]
    duration = ev['duration'][0,0]
    time_sec = (latency - 1) / EEG['srate'][0,0]  # Convert to 0-based, then seconds
    print(f"{i:2d}. Latency: {latency:7d} ({time_sec:8.2f}s)  Type: {ev_type}  Duration: {duration}")
