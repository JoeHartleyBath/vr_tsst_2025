# Manual Edit Ledger for Overleaf

Use this checklist to reconcile the Overleaf manuscript with the latest local results.

## 1. Methodology Section
- [ ] **ASR Removal**: Search for "Artifact Subspace Reconstruction" or "ASR". Remove these sentences. Replace with description of "Selective QC Filtering" (exclusion of noisy channels/segments without reconstruction).
- [ ] **Participant Count**: Verify the N number (N=44 vs N=47). Local analysis uses N=44 (QC filtered). Ensure text reflects this.

## 2. Results Section
- [ ] **ANOVA Stats**: Check the F-values and p-values in the "Physiological Results" section.
  - *Source*: See `results/classic_analyses/anovas/art_anova_results.csv` locally.
- [ ] **Correlation Values**: Update Pearson correlation coefficients if they differ.
  - *Source*: `output/correlations.csv`.

## 3. Figures
- [ ] **Figure 3 (Interaction)**: Check if the interaction plot matches `results/figures/interaction_plots.png`.
- [ ] **Figure 4 (Heatmaps)**: Ensure heatmaps match `results/figures/heatmap_stress_corr.png`.

## 4. Discussion
- [ ] **Limitations**: Add a note about the exclusion of 3 participants due to signal quality if not present.
