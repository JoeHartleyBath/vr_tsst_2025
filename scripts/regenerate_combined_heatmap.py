import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec
import os

# --- Configuration ---
RESULTS_PATH = r'results/classic_analyses/stratified_rmcorr/stratified_rmcorr_results.csv'
OUTPUT_FILE = r'manuscript/figures/heatmap_combined.png'

# Pretty Labels conforming to IEEE style (concise)
PRETTY_FEATURES = {
    "hrv_rmssd_precond": "HRV (RMSSD)",
    "hr_med_precond": "Heart Rate",
    "eda_tonic_mean_precond": "EDA Tonic Level",
    "eda_pkht_mean_precond": "EDA Peak Amp.",
    "pupil_full_pupil_med_precond": "Pupil Diameter",
    "eeg_fm_theta_power_precond": "EEG Frontal Theta",
    "eeg_f_beta_power_precond": "EEG Frontal Beta",
    "eeg_p_alpha_power_precond": "EEG Parietal Alpha",
    "eeg_faa_precond": "Frontal Alpha Asym.",
    "eeg_ab_ratio_precond": "Alpha/Beta Ratio",
    "eeg_tb_ratio_precond": "Theta/Beta Ratio"
}

# Logical Grouping: Autonomic first, then EEG
FEATURE_ORDER = [
     "HRV (RMSSD)", "Heart Rate", "EDA Tonic Level", "EDA Peak Amp.", "Pupil Diameter",
     "EEG Frontal Theta", "EEG Frontal Beta", "EEG Parietal Alpha", "Frontal Alpha Asym.",
     "Alpha/Beta Ratio", "Theta/Beta Ratio"
]

def load_data():
    if not os.path.exists(RESULTS_PATH):
        path = os.path.join(os.getcwd(), RESULTS_PATH)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Cannot find {RESULTS_PATH}")
        return pd.read_csv(path)
    return pd.read_csv(RESULTS_PATH)

def prepare_matrix(df, rating_type, subset_order):
    sub_df = df[df['rating'] == rating_type].copy()
    sub_df = sub_df[sub_df['subset'].isin(subset_order)]
    sub_df['feature_display'] = sub_df['feature'].map(PRETTY_FEATURES)
    
    # Pivot
    r_matrix = sub_df.pivot(index='feature_display', columns='subset', values='rmcorr_r')
    p_matrix = sub_df.pivot(index='feature_display', columns='subset', values='p_fdr')
    sig_matrix = sub_df.pivot(index='feature_display', columns='subset', values='sig').fillna('')
    
    # Reindex for consistent feature order
    r_matrix = r_matrix.reindex(FEATURE_ORDER)
    p_matrix = p_matrix.reindex(FEATURE_ORDER)
    sig_matrix = sig_matrix.reindex(FEATURE_ORDER)
    
    # Ensure correct column order
    return r_matrix[subset_order], p_matrix[subset_order], sig_matrix[subset_order]

def draw_heatmap(ax, r_mat, p_mat, sig_mat, title, cbar=False, y_labels=True):
    # Handle NaNs (missing features) to avoid errors
    if r_mat.isnull().all().all():
        return

    # Draw Heatmap
    # annot_kws color='black' forces numbers to be black
    sns.heatmap(r_mat, annot=True, fmt=".2f", 
                cmap="RdYlBu_r", vmin=-1, vmax=1, center=0,
                cbar=cbar, ax=ax,
                linewidths=0.5, linecolor='white',
                annot_kws={"size": 8, "color": "black"},
                cbar_kws={'label': 'Correlation ($r_{rm}$)', 'orientation': 'horizontal'})
    
    ax.set_title(title, fontsize=10, fontweight='bold', pad=10)
    ax.set_xlabel("")
    ax.set_ylabel("")
    
    if not y_labels:
        ax.set_yticks([])
    else:
        ax.tick_params(axis='y', labelsize=9)
        
    # Borders and Stars
    for y in range(r_mat.shape[0]):
        for x in range(r_mat.shape[1]):
            # FDR Significance Border
            p_val = p_mat.iloc[y, x]
            if pd.notna(p_val) and p_val < 0.05:
                ax.add_patch(patches.Rectangle((x, y), 1, 1, fill=False, edgecolor='black', lw=2, clip_on=False))
            
            # Significance Symbol
            sym = sig_mat.iloc[y, x]
            if isinstance(sym, str): sym = sym.strip()
            if sym:
                val = r_mat.iloc[y, x]
                # Use white text only for very dark cells (|r| > 0.8)
                t_col = 'white' if pd.notna(val) and abs(val) > 0.8 else 'black'
                
                # Offset star
                ax.text(x + 0.85, y + 0.2, sym, 
                        color=t_col, ha='center', va='center', 
                        fontsize=10, fontweight='bold')

def main():
    df = load_data()
    
    # Check if subsets exist in df
    print("Subsets found:", df['subset'].unique())
    
    r_stress, p_stress, s_stress = prepare_matrix(df, 'stress', ['Low workload', 'High workload'])
    r_mwl, p_mwl, s_mwl = prepare_matrix(df, 'workload', ['Low stress', 'High stress'])

    # Figure Layout: 1 Row, 2 Cols, shared Y-axis
    fig = plt.figure(figsize=(7.5, 6), dpi=300)
    # width_ratios=[1, 1] ensures equal visual width for the plot columns
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1], wspace=0.08) 
    
    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1])
    
    draw_heatmap(ax1, r_stress, p_stress, s_stress, "Stress Correlations", cbar=False, y_labels=True)
    draw_heatmap(ax2, r_mwl, p_mwl, s_mwl, "Workload Correlations", cbar=False, y_labels=False)
    
    # Shared Colorbar
    cbar_ax = fig.add_axes([0.3, 0.08, 0.4, 0.025]) 
    norm = plt.Normalize(-1, 1)
    sm = plt.cm.ScalarMappable(cmap="RdYlBu_r", norm=norm)
    sm.set_array([])
    cb = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
    cb.set_label(r'Repeated-Measures Correlation ($r_{rm}$)', fontsize=9)
    cb.ax.tick_params(labelsize=8)
    
    # Subplot labels (a), (b) - Using axis-relative positioning for alignment
    # (a) positioned at the top-left corner of the Stress plot (x=0) to reduce gap to title
    ax1.text(0.0, 1.05, '(a)', transform=ax1.transAxes, fontsize=12, fontweight='bold', va='bottom', ha='left')
    # (b) positioned at the top-left corner of the Workload plot (x=0) for consistency
    ax2.text(0.0, 1.05, '(b)', transform=ax2.transAxes, fontsize=12, fontweight='bold', va='bottom', ha='left')
    
    # Margins
    plt.subplots_adjust(bottom=0.20, top=0.88, right=0.95, left=0.25)
    
    if not os.path.exists(os.path.dirname(OUTPUT_FILE)):
        os.makedirs(os.path.dirname(OUTPUT_FILE))
        
    plt.savefig(OUTPUT_FILE)
    print(f"Comparison heatmap saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
