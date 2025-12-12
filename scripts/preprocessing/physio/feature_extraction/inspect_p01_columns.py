import pandas as pd

# Load a sample of P01 data
df = pd.read_csv('data/raw/metadata/P01.csv', nrows=1000)

print(f'Total columns: {len(df.columns)}\n')

# Find physio columns
hr_cols = [c for c in df.columns if 'heart' in c.lower() or 'polar' in c.lower() or 'hr' in c.lower() or 'bpm' in c.lower()]
gsr_cols = [c for c in df.columns if 'gsr' in c.lower() or 'skin' in c.lower() or 'conductance' in c.lower()]
pupil_cols = [c for c in df.columns if 'pupil' in c.lower() or 'dilation' in c.lower()]
subj_cols = [c for c in df.columns if 'stress' in c.lower() or 'workload' in c.lower() or 'rating' in c.lower()]

print('HR/POLAR columns:')
for c in hr_cols[:10]:
    print(f'  {c}')
print(f'  ... ({len(hr_cols)} total)')

print('\nGSR columns:')
for c in gsr_cols[:10]:
    print(f'  {c}')
print(f'  ... ({len(gsr_cols)} total)')

print('\nPupil columns:')
for c in pupil_cols[:10]:
    print(f'  {c}')
print(f'  ... ({len(pupil_cols)} total)')

print('\nSubjective rating columns:')
for c in subj_cols[:10]:
    print(f'  {c}')
print(f'  ... ({len(subj_cols)} total)')

print('\nCondition-related columns:')
cond_cols = ['Unity Scene', 'Study_Phase', 'Shown_Scene', 'Arithmetic_Task', 'Participant_State']
for c in cond_cols:
    if c in df.columns:
        print(f'  {c}: {df[c].nunique()} unique values')
        print(f'    Sample: {df[c].dropna().unique()[:5]}')
