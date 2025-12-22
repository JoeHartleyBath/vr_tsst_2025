import argparse
from pathlib import Path
import h5py


def summarize_mat_v73(path: Path):
    with h5py.File(path, 'r') as f:
        # Common EEGLAB v7.3 structure: /EEG with fields
        summary = {}
        if 'EEG' in f:
            eeg = f['EEG']
            # Attempt to read sizes; EEGLAB stores/refs fields via object references
            # Try typical fields: data, nbchan, pnts, srate, chanlocs, event
            def read_scalar(group, name):
                if name in group:
                    ds = group[name]
                    # scalar may be stored as array; try to read first element
                    try:
                        val = ds[()]  # numpy
                        if hasattr(val, 'shape') and val.shape == ():
                            return float(val)
                        # if array, take first
                        return float(val.flat[0])
                    except Exception:
                        return None
                return None

            summary['nbchan'] = read_scalar(eeg, 'nbchan')
            summary['pnts'] = read_scalar(eeg, 'pnts')
            summary['srate'] = read_scalar(eeg, 'srate')

            # data may be huge; just shape
            if 'data' in eeg:
                d = eeg['data']
                summary['data_shape'] = tuple(d.shape)
            else:
                summary['data_shape'] = None

            # events: count if present
            if 'event' in eeg:
                event = eeg['event']
                # EEGLAB stores event fields separately; count length via one field like 'type'
                ev_count = None
                if 'type' in event:
                    ev_count = event['type'].shape[0]
                elif 'latency' in event:
                    ev_count = event['latency'].shape[0]
                summary['events'] = ev_count
            else:
                summary['events'] = None
        else:
            summary['error'] = 'EEG group not found in MAT file.'
        return summary


def main():
    parser = argparse.ArgumentParser(description='Summarize EEGLAB v7.3 MAT file using h5py')
    parser.add_argument('mat_path', type=Path, help='Path to cleaned .mat')
    args = parser.parse_args()

    summary = summarize_mat_v73(args.mat_path)
    print(summary)


if __name__ == '__main__':
    main()
