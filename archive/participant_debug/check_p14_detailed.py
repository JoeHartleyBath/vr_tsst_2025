import pandas as pd

# Load P14.csv
df = pd.read_csv('data/raw/metadata/P14.csv', low_memory=False)

print('Column names:')
print(df.columns.tolist()[:15])
print()

print('First 10 unique Unity Scene values:')
unique_scenes = df['Unity Scene'].unique()
for i, scene in enumerate(unique_scenes[:10]):
    print(f'  {i+1}. {scene}')
print()

# Check for Task-containing scenes
task_scenes = df[df['Unity Scene'].str.contains('Task', case=False, na=False)]['Unity Scene'].unique()
print(f'Task-containing scenes (found {len(task_scenes)}):')
for scene in task_scenes:
    print(f'  - {scene}')
