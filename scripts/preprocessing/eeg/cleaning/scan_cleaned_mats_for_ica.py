import os
import sys
import glob

try:
    import scipy.io
    import h5py
    import numpy as np
except Exception as e:
    print('Missing packages:', e)
    sys.exit(2)

basedir = os.path.join('output','cleaned_eeg')
files = sorted(glob.glob(os.path.join(basedir,'*_cleaned.mat')))
if not files:
    print('No cleaned .mat files found in', basedir)
    sys.exit(0)

results = []

def walk_h5(fn):
    hits = []
    with h5py.File(fn,'r') as f:
        def walk(group, path=''):
            for k in group.keys():
                item = group[k]
                p = path + '/' + k if path else k
                if isinstance(item, h5py.Dataset):
                    hits.append((p, 'D', item.shape))
                else:
                    hits.append((p, 'G'))
                    walk(item, p)
        walk(f)
    return hits

for fn in files:
    name = os.path.basename(fn)
    info = {'file': name, 'icaweights': False, 'icasphere': False, 'iclabel': False, 'EEG_struct': False}
    try:
        mat = scipy.io.loadmat(fn, struct_as_record=False, squeeze_me=True)
        keys = [k for k in mat.keys() if not k.startswith('__')]
        if 'EEG' in mat:
            info['EEG_struct'] = True
            EEG = mat['EEG']
            # try fields
            try:
                if hasattr(EEG, 'icaweights') and np.asarray(getattr(EEG,'icaweights')).size>0:
                    info['icaweights'] = True
            except Exception:
                pass
            try:
                if hasattr(EEG, 'icasphere') and np.asarray(getattr(EEG,'icasphere')).size>0:
                    info['icasphere'] = True
            except Exception:
                pass
            try:
                etc = getattr(EEG,'etc',None)
                if etc is not None:
                    icc = getattr(etc,'ic_classification', None)
                    if icc is not None:
                        icl = getattr(icc,'ICLabel', None)
                        if icl is not None:
                            cls = getattr(icl,'classifications', None)
                            if cls is not None:
                                info['iclabel'] = True
            except Exception:
                pass
        else:
            # inspect top-level keys for indicative names
            for k in keys:
                lk = k.lower()
                if 'icaweight' in lk:
                    info['icaweights'] = True
                if 'icasphere' in lk:
                    info['icasphere'] = True
                if 'iclabel' in lk or 'classificat' in lk:
                    info['iclabel'] = True
    except NotImplementedError:
        # HDF5 v7.3
        hits = walk_h5(fn)
        lk = [p.lower() for p,_,*rest in hits]
        for p in lk:
            if 'icaweight' in p:
                info['icaweights'] = True
            if 'icasphere' in p:
                info['icasphere'] = True
            if 'iclabel' in p or 'classificat' in p:
                info['iclabel'] = True
            if p.strip().startswith('eeg'):
                info['EEG_struct'] = True
    except Exception as e:
        info['error'] = str(e)
    results.append(info)

# print concise summary
for r in results:
    flags = []
    for k in ('icaweights','icasphere','iclabel','EEG_struct'):
        if r.get(k):
            flags.append(k)
    print(r['file'],'->', ','.join(flags) if flags else 'NONE', r.get('error',''))

# overall counts
from collections import Counter
cnt = Counter()
for r in results:
    for k in ('icaweights','icasphere','iclabel','EEG_struct'):
        if r.get(k):
            cnt[k]+=1
print('\nTotals:', dict(cnt))
