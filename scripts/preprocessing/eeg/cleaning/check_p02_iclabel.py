import os
import sys
import pprint

try:
    import scipy.io
    import numpy as np
except Exception as e:
    print('ERROR: missing python packages:', e)
    sys.exit(2)

fn = os.path.join('output','cleaned_eeg','P02_cleaned.mat')
if not os.path.exists(fn):
    print('MISSING:', fn)
    sys.exit(1)

try:
    mat = scipy.io.loadmat(fn, struct_as_record=False, squeeze_me=True)
    # If loadmat succeeds, just show top-level keys
    keys = [k for k in mat.keys() if not k.startswith('__')]
    pprint.pprint({'mat_keys_sample': keys[:20]})
    # best-effort: check for common fields in top-level EEG struct
    EEG = mat.get('EEG', None)
    if EEG is None:
        # try to find a likely EEG struct
        for k,v in mat.items():
            if k.startswith('__'):
                continue
            EEG = v
            break

    def get_field(obj, name):
        try:
            return getattr(obj, name)
        except Exception:
            pass
        try:
            return obj[name]
        except Exception:
            pass
        try:
            if hasattr(obj, 'dtype') and obj.dtype.names:
                if name in obj.dtype.names:
                    return obj[name]
        except Exception:
            pass
        return None

    icaw = get_field(EEG, 'icaweights') if EEG is not None else None
    icas = get_field(EEG, 'icasphere') if EEG is not None else None
    iclabel = None
    etc = get_field(EEG, 'etc') if EEG is not None else None
    if etc is not None:
        icc = get_field(etc, 'ic_classification')
        if icc is not None:
            iclabel = get_field(icc, 'ICLabel') or get_field(icc, 'classifications')
        else:
            iclabel = get_field(etc, 'iclabel') or get_field(etc, 'classifications')

    if iclabel is None and EEG is not None:
        iclabel = get_field(EEG, 'ic_classification') or get_field(EEG, 'classifications')

    out = {}
    out['icaweights_present'] = icaw is not None
    out['icasphere_present'] = icas is not None
    out['iclabel_present'] = iclabel is not None
    pprint.pprint(out)
    print('DONE')
except NotImplementedError:
    # MATLAB v7.3 HDF5 file: use h5py
    try:
        import h5py
    except Exception as e:
        print('HDF5 read required but h5py not installed:', e)
        sys.exit(3)

    def walk(h, prefix=''):
        hits = []
        for k in h.keys():
            path = prefix + '/' + k if prefix else k
            item = h[k]
            # dataset
            if isinstance(item, h5py.Dataset):
                hits.append(('D', path, item.shape))
            else:
                hits.append(('G', path))
                hits.extend(walk(item, path))
        return hits

    with h5py.File(fn, 'r') as f:
        all_entries = walk(f)
        # search for indicative names
        found = {'icaweights': [], 'icasphere': [], 'iclabel': []}
        for e in all_entries:
            if 'icaweight' in e[1].lower():
                found['icaweights'].append(e)
            if 'icasphere' in e[1].lower():
                found['icasphere'].append(e)
            if 'iclabel' in e[1].lower() or 'classificat' in e[1].lower():
                found['iclabel'].append(e)

        pprint.pprint({'hdf5_hits_sample': all_entries[:30]})
        pprint.pprint(found)
        print('DONE')

# Also inspect the .set file if present
set_fn = os.path.join('output','cleaned_eeg','P02_cleaned.set')
if os.path.exists(set_fn):
    print('\nInspecting .set file:', set_fn)
    # try HDF5 open
    try:
        with h5py.File(set_fn, 'r') as sf:
            entries = walk(sf)
            pprint.pprint({'set_hdf5_sample': entries[:30]})
    except Exception as e:
        # not HDF5 — show header bytes
        try:
            with open(set_fn, 'rb') as fh:
                head = fh.read(512)
                print('SET header (first 512 bytes):')
                print(repr(head[:200]))
        except Exception as e2:
            print('Failed to read .set file:', e, e2)
    # try to load with scipy.loadmat (MAT v5)
    try:
        s_mat = scipy.io.loadmat(set_fn, struct_as_record=False, squeeze_me=True)
        keys = [k for k in s_mat.keys() if not k.startswith('__')]
        print('set: top-level keys:', keys)
        # Directly check for common variables at top-level
        for name in ['icaweights','icasphere','icawinv','icaact','ICLabel','classifications','ic_classification','etc']:
            if name in s_mat:
                try:
                    val = s_mat[name]
                    print(f"FOUND {name}: type={type(val)}, shape={(getattr(val,'shape',None))}")
                except Exception as e:
                    print(f"FOUND {name} but failed to introspect: {e}")
            else:
                print(f"{name} not in .set top-level keys")
        # inspect etc struct if present
        etc = s_mat.get('etc', None)
        if etc is not None:
            fnames = getattr(etc, '_fieldnames', None) or getattr(etc, 'dtype', None)
            try:
                if hasattr(etc, '_fieldnames'):
                    print('etc fields:', etc._fieldnames)
                    names = etc._fieldnames
                elif hasattr(etc, 'dtype') and etc.dtype.names:
                    print('etc dtype names:', etc.dtype.names)
                    names = etc.dtype.names
                else:
                    names = []
                for n in names:
                    try:
                        v = getattr(etc, n)
                        print('etc.%s: type=%s shape=%s' % (n, type(v), getattr(v,'shape',None)))
                    except Exception as e:
                        print('etc.%s: failed to read (%s)' % (n, e))
            except Exception as e:
                print('Failed to introspect etc:', e)
            # inspect ic_classification if present
            icc = getattr(etc, 'ic_classification', None)
            if icc is not None:
                try:
                    fname = getattr(icc, '_fieldnames', None) or getattr(icc, 'dtype', None)
                    if hasattr(icc, '_fieldnames'):
                        print('ic_classification fields:', icc._fieldnames)
                        ic_names = icc._fieldnames
                    elif hasattr(icc, 'dtype') and icc.dtype.names:
                        print('ic_classification dtype names:', icc.dtype.names)
                        ic_names = icc.dtype.names
                    else:
                        ic_names = []
                    for n in ic_names:
                        try:
                            v = getattr(icc, n)
                            print('ic_classification.%s: type=%s shape=%s' % (n, type(v), getattr(v,'shape',None)))
                        except Exception as e:
                            print('ic_classification.%s: failed to read (%s)' % (n, e))
                except Exception as e:
                    print('Failed to introspect ic_classification:', e)
                # inspect ICLabel struct inside ic_classification
                try:
                    icl = getattr(icc, 'ICLabel', None)
                    if icl is not None:
                        print('ICLabel struct fields:', getattr(icl, '_fieldnames', None))
                        # try to get classifications matrix
                        cls = getattr(icl, 'classifications', None)
                        if cls is not None:
                            try:
                                arr = np.asarray(cls)
                                print('ICLabel.classifications shape:', arr.shape)
                                print('first 5 rows:', arr[:5,:].tolist())
                                # assume column 2 corresponds to Eye (0-based)
                                eye_col = 2
                                if arr.shape[1] > eye_col:
                                    eye_probs = arr[:, eye_col]
                                    max_eye = float(np.max(eye_probs))
                                    high = np.where(eye_probs >= 0.9)[0].tolist()
                                    print('max_eye_prob:', max_eye, 'num_>=0.9:', len(high), 'indices_0based:', high)
                                else:
                                    print('ICLabel matrix has fewer columns than expected')
                            except Exception as e:
                                print('failed to read ICLabel.classifications:', e)
                        else:
                            print('ICLabel.classifications not found as field')
                except Exception as e:
                    print('Failed to inspect ICLabel substruct:', e)
    except Exception as e:
        print('scipy.loadmat on .set failed:', e)
