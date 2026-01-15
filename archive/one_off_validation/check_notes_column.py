import pandas as pd

df = pd.read_excel('data/experimental_counterbalance.xlsx')

print('Columns:', df.columns.tolist())
print('\nTotal participants:', len(df))

if 'Notes' in df.columns:
    notes_present = df[df['Notes'].notna()]
    print(f'\nRows with Notes: {len(notes_present)}')
    if len(notes_present) > 0:
        print('\nParticipants with Notes:')
        for idx, row in notes_present.iterrows():
            print(f"P{row['Participant']:02d}: {row['Notes']}")
else:
    print('\nNo Notes column found')
