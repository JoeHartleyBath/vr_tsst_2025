import pandas as pd

summary_path = 'output/cleaned_eeg/qc_summary.csv'
report_path = 'output/cleaned_eeg/qc_exclusion_reasons.csv'

# QC thresholds
THRESHOLDS = {
    'percent_data_retained': 75,
    'percent_channels_interpolated': 20,
    'percent_ica_components_removed': 35
}

# Load summary
qc = pd.read_csv(summary_path)

reasons = []
for _, row in qc.iterrows():
    if not row['excluded_and_verified']:
        continue
    reason = []
    if row['percent_data_retained'] < THRESHOLDS['percent_data_retained']:
        reason.append('Low data retained')
    if row['percent_channels_interpolated'] > THRESHOLDS['percent_channels_interpolated']:
        reason.append('High channels interpolated')
    if row['percent_ica_components_removed'] > THRESHOLDS['percent_ica_components_removed']:
        reason.append('High ICA components removed')
    reasons.append({
        'participant': row['participant'],
        'reason': '; '.join(reason)
    })

pd.DataFrame(reasons).to_csv(report_path, index=False)
print(f'Exclusion reasons saved to {report_path}')
