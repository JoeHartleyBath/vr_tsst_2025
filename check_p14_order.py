import pandas as pd

# Load P14.csv and get task condition order
df = pd.read_csv('data/raw/metadata/P14.csv', low_memory=False)
task_rows = df[df['Unity Scene'].str.contains('Task', na=False)]
first_appearance = task_rows.groupby('Unity Scene').first().reset_index()[['Unity Scene', 'LSL_Timestamp']].sort_values('LSL_Timestamp')

print('Task conditions in P14.csv (chronological):')
for scene in first_appearance['Unity Scene'].tolist():
    print('  -', scene)

print('\n' + '='*60)
print('Expected orders from counterbalance:')
print('\nP14 counterbalance:')
print('  - LowStress_HighCog_Task (Calm Subtraction)')
print('  - HighStress_LowCog_Task (Stress Addition)')
print('  - LowStress_LowCog_Task (Calm Addition)')
print('  - HighStress_HighCog_Task (Stress Subtraction)')

print('\nP16 counterbalance:')
print('  - LowStress_HighCog_Task (Calm Subtraction)')
print('  - HighStress_HighCog_Task (Stress Subtraction)')
print('  - LowStress_LowCog_Task (Calm Addition)')
print('  - HighStress_LowCog_Task (Stress Addition)')
