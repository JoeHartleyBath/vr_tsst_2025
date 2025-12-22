import pandas as pd

# Read the counterbalance file
df = pd.read_excel('data/experimental_counterbalance.xlsx')

# Filter for P14 and P16
p14_p16 = df[df['Participant'].isin([14, 16])]

# Display all columns
print("P14 and P16 entries in counterbalance.xlsx:")
print("=" * 100)
for idx, row in p14_p16.iterrows():
    print(f"\nParticipant: {row['Participant']}")
    print(f"Round 1: {row['Round 1']}")
    print(f"Round 2: {row['Round 2']}")
    print(f"Round 3: {row['Round 3']}")
    print(f"Round 4: {row['Round 4']}")
    if 'Notes' in row and pd.notna(row['Notes']):
        print(f"Notes: {row['Notes']}")
    print("-" * 100)
