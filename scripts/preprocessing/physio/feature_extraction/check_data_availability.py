import pandas as pd
import os

print('='*60)
print('DATA AVAILABILITY CHECK')
print('='*60)

# Check files
print('\n1. RAW DATA:')
p01_path = 'data/raw/metadata/P01.csv'
print(f'   P01 metadata exists: {os.path.exists(p01_path)}')
if os.path.exists(p01_path):
    size_mb = os.path.getsize(p01_path) / 1e6
    print(f'   Size: {size_mb:.1f} MB')
    
    # Load sample
    df = pd.read_csv(p01_path, nrows=10)
    print(f'   Columns: {len(df.columns)}')
    print(f'   First 15 columns: {list(df.columns[:15])}')
    
    # Check for key columns
    key_cols = ['LSL_Timestamp', 'Participant_ID', 'Polar_HeartRate_BPM', 
                'Shimmer_D36A_GSR_Skin_Conductance_uS',
                'Pupil_Dilation_Left', 'Pupil_Dilation_Right']
    found = [c for c in key_cols if c in df.columns]
    missing = [c for c in key_cols if c not in df.columns]
    print(f'\n   Key columns found: {len(found)}/{len(key_cols)}')
    if missing:
        print(f'   Missing: {missing}')

print('\n2. EEG FEATURES:')
eeg_path = 'output/aggregated/eeg_features.csv'
print(f'   EEG features exists: {os.path.exists(eeg_path)}')
if os.path.exists(eeg_path):
    eeg = pd.read_csv(eeg_path)
    print(f'   Rows: {len(eeg)}')
    print(f'   Columns: {len(eeg.columns)}')
    print(f'   Participants: {sorted(eeg["Participant"].unique())}')
    print(f'   Conditions: {sorted(eeg["Condition"].unique())}')

print('\n3. SUBJECTIVE RATINGS:')
subj_locations = [
    'output/aggregated/subjective.csv',
    'data/subjective/ratings.csv',
    'output/subjective_ratings.csv'
]
found_subj = False
for path in subj_locations:
    if os.path.exists(path):
        print(f'   Found: {path}')
        found_subj = True
        break
if not found_subj:
    print('   No subjective ratings file found')
    print('   Checked:', subj_locations)

print('\n4. LEGACY AGGREGATED DATA:')
legacy_path = 'output/aggregated/all_data_aggregated.csv'
print(f'   Legacy file exists: {os.path.exists(legacy_path)}')
if os.path.exists(legacy_path):
    # Just check shape, don't load full file
    with open(legacy_path) as f:
        header = f.readline()
        ncols = len(header.split(','))
    print(f'   Columns: {ncols}')

print('\n' + '='*60)
print('READY TO PROCEED?')
print('='*60)

ready = True
if not os.path.exists(p01_path):
    print('❌ Missing: P01 raw data')
    ready = False
if not os.path.exists(eeg_path):
    print('❌ Missing: EEG features')
    ready = False
if not found_subj:
    print('⚠️  Missing: Subjective ratings (will need to extract from metadata)')
    
if ready:
    print('✅ All required data available')
    print('\nNext steps:')
    print('1. Implement HR cleaning & feature extraction')
    print('2. Implement GSR cleaning & feature extraction')
    print('3. Implement pupil cleaning & feature extraction')
    print('4. Extract subjective ratings from metadata')
    print('5. Merge all features')
    print('6. Test with R preprocessing script')
