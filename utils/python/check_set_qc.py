"""
Quick QC check for a converted .set file.
Reports: channels, samples, srate, events, amplitude ranges.
"""
from pathlib import Path
from scipy.io import loadmat
import numpy as np
import sys

def qc_set_file(set_path: Path):
    m = loadmat(str(set_path))
    EEG = m['EEG'][0, 0]
    
    # Basic metadata
    nbchan = int(EEG['nbchan'][0, 0])
    pnts = int(EEG['pnts'][0, 0])
    srate = float(EEG['srate'][0, 0])
    duration_s = pnts / srate
    
    # Data amplitude check (first 10k samples to avoid memory)
    data = EEG['data']  # (channels, samples)
    sample_chunk = min(10000, data.shape[1])
    chunk = data[:, :sample_chunk]
    
    amp_min = float(np.min(chunk))
    amp_max = float(np.max(chunk))
    amp_median = float(np.median(chunk))
    amp_std = float(np.std(chunk))
    
    # Events
    events = EEG['event'][0]
    n_events = len(events)
    
    event_types = {}
    if n_events > 0:
        for ev in events:
            ev_type = int(ev['type'][0, 0])
            event_types[ev_type] = event_types.get(ev_type, 0) + 1
        
        first_lat = int(events[0]['latency'][0, 0])
        last_lat = int(events[-1]['latency'][0, 0])
        first_time = (first_lat - 1) / srate
        last_time = (last_lat - 1) / srate
    else:
        first_time = last_time = 0
    
    # Report
    print(f"File: {set_path.name}")
    print(f"  Channels: {nbchan}")
    print(f"  Samples: {pnts:,} ({duration_s:.1f}s @ {srate} Hz)")
    print(f"  Amplitude (µV, first {sample_chunk} samples):")
    print(f"    Min: {amp_min:.2f}, Max: {amp_max:.2f}, Median: {amp_median:.2f}, Std: {amp_std:.2f}")
    print(f"  Events: {n_events}")
    if event_types:
        print(f"    Types: {dict(sorted(event_types.items()))}")
        print(f"    Span: {first_time:.1f}s → {last_time:.1f}s")
    print()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python check_set_qc.py <file.set> [<file2.set> ...]")
        sys.exit(1)
    
    for path_str in sys.argv[1:]:
        path = Path(path_str)
        if not path.exists():
            print(f"ERROR: {path} not found")
            continue
        qc_set_file(path)
