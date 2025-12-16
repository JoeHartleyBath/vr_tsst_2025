
import os
import glob
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# List of occipital channels from config/eeg_feature_extraction.yaml
OCCIPITAL_CHANNELS = [
    'Z12', 'Z13', 'L11', 'L12', 'L13', 'L14', 'LL12', 'LL13', 'LC7', 'LD7', 'LE4',
    'R11', 'R12', 'R13', 'R14', 'RR12', 'RR13', 'RC7', 'RD7', 'RE4', 'LL14', 'RR14'
]

QC_DIR = 'output/cleaned_eeg/qc/'
SUMMARY_CSV = 'output/cleaned_eeg/qc_summary.csv'
PLOTS_DIR = 'output/cleaned_eeg/qc_plots/'
TOTAL_CHANNELS = 271  # Update if different
TOTAL_ICS = 64        # Update if different

os.makedirs(PLOTS_DIR, exist_ok=True)

qc_files = sorted(glob.glob(os.path.join(QC_DIR, 'P*_qc.json')))
rows = []

for qc_file in qc_files:
    with open(qc_file, 'r') as f:
        qc = json.load(f)
    pid = os.path.basename(qc_file).split('_')[0]
    # Skip files with missing/null values for required metrics
    if (
        qc.get('bad_channels_count') is None or
        qc.get('ics_removed') is None or
        qc.get('asr_repaired_percent') is None or
        qc.get('bad_channels') is None
    ):
        print(f"Skipping {qc_file} due to missing QC metrics.")
        continue
    # Exclude occipital channels from bad_channels_count
    bad_channels = [ch for ch in qc['bad_channels'] if ch not in OCCIPITAL_CHANNELS]
    percent_channels_interp = len(bad_channels) / TOTAL_CHANNELS * 100
    percent_ics_removed = qc['ics_removed'] / TOTAL_ICS * 100
    percent_data_retained = 100 - qc['asr_repaired_percent']
    excluded = (
        percent_data_retained < 75 or
        percent_channels_interp > 20 or
        percent_ics_removed > 35
    )
    rows.append({
        'participant': pid,
        'percent_data_retained': percent_data_retained,
        'percent_channels_interpolated': percent_channels_interp,
        'percent_ica_components_removed': percent_ics_removed,
        'excluded_and_verified': excluded
    })

df = pd.DataFrame(rows)
df.to_csv(SUMMARY_CSV, index=False)

# Plotting
metrics = [
    ('percent_data_retained', 'Percent Data Retained'),
    ('percent_channels_interpolated', 'Percent Channels Interpolated'),
    ('percent_ica_components_removed', 'Percent ICA Components Removed')
]

for col, label in metrics:
    plt.figure(figsize=(8, 5))
    sns.histplot(df[col], bins=15, kde=True, color='skyblue', edgecolor='black')
    plt.title(f'{label} Distribution')
    plt.xlabel(label)
    plt.ylabel('Count')
    # Highlight excluded participants
    for x in df[df['excluded_and_verified']][col]:
        plt.axvline(x, color='red', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f'{col}_hist.png'))
    plt.close()

    plt.figure(figsize=(8, 5))
    sns.boxplot(x=df[col], color='lightgreen')
    plt.title(f'{label} Boxplot')
    plt.xlabel(label)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f'{col}_box.png'))
    plt.close()

print(f'Summary table saved to {SUMMARY_CSV}')
print(f'Plots saved to {PLOTS_DIR}')
