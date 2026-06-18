import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read the CSV file
df = pd.read_csv(r'c:\vr_tsst_2025\output\aggregated\eeg_features_rolling_windows.csv')

# Filter for participant 26
p26_data = df[df['pid'] == 26].copy()

print(f"Found {len(p26_data)} rows for participant 26")
print(f"Events: {p26_data['event_label'].unique()}")

# Get all PSD-related columns (Delta, Theta, Alpha, Beta bands)
psd_columns = [col for col in df.columns if any(band in col for band in ['Delta', 'Theta', 'Alpha', 'Beta'])]
print(f"PSD columns: {len(psd_columns)}")

# Extract some key channels for plotting
# Let's plot frontal, central, parietal, and occipital regions
regions_to_plot = {
    'Frontal': [col for col in psd_columns if 'Frontal' in col and 'Midline' not in col],
    'Central': [col for col in psd_columns if 'Central' in col],
    'Parietal': [col for col in psd_columns if 'Parietal' in col and not col.startswith('Parietal_')],
    'Occipital': [col for col in psd_columns if 'Occipital' in col]
}

# Create figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('P26 EEG Power Spectral Density (PSD) Values', fontsize=16, fontweight='bold')

for idx, (region, cols) in enumerate(regions_to_plot.items()):
    ax = axes[idx // 2, idx % 2]
    
    # For each column in this region, plot the time series
    for col in cols[:5]:  # Limit to first 5 columns per region for clarity
        if col in p26_data.columns:
            ax.plot(p26_data['window_start'], p26_data[col], label=col.split('_')[-1], alpha=0.7)
    
    ax.set_xlabel('Time (s)', fontsize=11)
    ax.set_ylabel('PSD (log power)', fontsize=11)
    ax.set_title(f'{region} Region', fontsize=13, fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(r'c:\vr_tsst_2025\output\plots\p26_psd_time_series.png', dpi=300, bbox_inches='tight')
print("Saved: p26_psd_time_series.png")

# Create a second plot showing frequency bands across all regions
fig2, axes2 = plt.subplots(2, 2, figsize=(16, 12))
fig2.suptitle('P26 EEG Frequency Bands by Region', fontsize=16, fontweight='bold')

bands = ['Delta', 'Theta', 'Alpha', 'Beta']
colors = ['blue', 'green', 'orange', 'red']

for idx, band in enumerate(bands):
    ax = axes2[idx // 2, idx % 2]
    
    # Get all columns for this band
    band_cols = [col for col in psd_columns if band in col and not 'Low' in col and not 'High' in col]
    
    for col in band_cols[:8]:  # Plot up to 8 columns
        if col in p26_data.columns:
            region_name = col.split('_')[0]
            ax.plot(p26_data['window_start'], p26_data[col], label=region_name, alpha=0.7)
    
    ax.set_xlabel('Time (s)', fontsize=11)
    ax.set_ylabel(f'{band} Power (log)', fontsize=11)
    ax.set_title(f'{band} Band', fontsize=13, fontweight='bold', color=colors[idx])
    ax.legend(loc='best', fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(r'c:\vr_tsst_2025\output\plots\p26_psd_frequency_bands.png', dpi=300, bbox_inches='tight')
print("Saved: p26_psd_frequency_bands.png")

# Create heatmap of PSD values
fig3, ax3 = plt.subplots(figsize=(20, 8))

# Select a subset of columns for the heatmap (every 5th column to avoid clutter)
heatmap_cols = psd_columns[::5]
heatmap_data = p26_data[heatmap_cols].T

im = ax3.imshow(heatmap_data, aspect='auto', cmap='RdYlBu_r', interpolation='nearest')
ax3.set_xlabel('Window Index', fontsize=12)
ax3.set_ylabel('EEG Feature', fontsize=12)
ax3.set_title('P26 PSD Heatmap (All Features)', fontsize=14, fontweight='bold')
ax3.set_yticks(range(len(heatmap_cols)))
ax3.set_yticklabels([col.replace('_', ' ') for col in heatmap_cols], fontsize=8)

# Add colorbar
cbar = plt.colorbar(im, ax=ax3)
cbar.set_label('Log Power', fontsize=11)

plt.tight_layout()
plt.savefig(r'c:\vr_tsst_2025\output\plots\p26_psd_heatmap.png', dpi=300, bbox_inches='tight')
print("Saved: p26_psd_heatmap.png")

# Summary statistics
print("\nSummary Statistics for P26:")
print(f"Time range: {p26_data['window_start'].min():.2f}s to {p26_data['window_end'].max():.2f}s")
print(f"Number of windows: {len(p26_data)}")
print(f"\nMean PSD values by band:")
for band in bands:
    band_cols = [col for col in psd_columns if band in col and not 'Low' in col and not 'High' in col]
    if band_cols:
        mean_val = p26_data[band_cols].mean().mean()
        print(f"  {band}: {mean_val:.4f}")

plt.show()
