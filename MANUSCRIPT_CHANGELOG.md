# Manuscript Update Changelog (methods/results refresh)

This log records edits made to `manuscript.txt` to align Methods/Results with the latest aggregated pipeline outputs (ANOVAs, posthoc contrasts, correlations, SVM). No new analyses were added beyond recomputing the same subjective manipulation-check statistics from the current `output/final_data.csv`.

## Sample size / QC wording
- Updated abstract sample size from `(n=40)` to `(N=44)`.
- Updated Participants section to state 44 retained after QC + dataset validation (removed the previous “8 excluded (3 EEG; 5 physio)” breakdown).
- Updated Results → Participant Characteristics from “Forty participants … after exclusion of eight …” to “Forty-four participants … after QC + dataset validation”, and added a missingness note explaining that effective `n`/`n_pairs` varies by feature/condition.

## Subjective manipulation checks (Factorial Analysis of Subjective Measures)
Updated the narrative and Table 1 to match the current dataset:
- Stress ratings (High vs Low stress): mean diff = 0.82; paired t(43)=2.91; p=0.006; dz=0.44.
- NASA mental demand (High vs Low workload): mean diff = 2.07; paired t(43)=6.98; p<0.001; dz=1.05.
- Congruent vs incongruent stress–MWL rating correlations:
  - congruent: r=0.43, p<0.001 (n=88)
  - incongruent: r=0.30, p=0.005 (n=88)
  - Fisher z=0.97, p=0.33

**Source**: `output/final_data.csv` (recomputed using the same stats described in the manuscript).

## Subjective ANOVAs rerun + plots regenerated (Dec 2025)
- Reran the 2×2 within-subject ANOVAs for subjective stress ratings and NASA-TLX mental demand from the post-QC `output/final_data.csv` (N=44).
- Regenerated the subjective interaction plot figure used in the manuscript.

**Script**: `scripts/analysis/subjective/subjective_anovas_and_plots.R`

**Outputs**:
- `results/classic_analyses/subjective/subjective_anova_results.csv`
- `results/classic_analyses/subjective/subjective_paired_ttests.csv`
- `results/classic_analyses/subjective/stress_workload_rating_correlations.csv`
- `results/classic_analyses/subjective/sessionInfo.txt`
- `interaction_plots.png`

## ART-ANOVA tables (canonical physiology/EEG features)
Replaced all values in Tables “ART-ANOVA stress main effects”, “ART-ANOVA workload main effects”, and “ART-ANOVA interaction effects” to match the latest ART outputs.

**Source**: `results/classic_analyses/anovas/art_anova_results.csv`.

## Tonic EDA posthoc table
Updated the posthoc contrast and cell means/SEs for tonic EDA.

**Sources**:
- `results/classic_analyses/anovas/posthoc_eda_tonic_mean_precond_stress.csv`
- `results/classic_analyses/anovas/summary_eda_tonic_mean_precond.csv`

## Correlations (Results text)
Rewrote the Results → Correlations narrative to reflect:
- Stratified repeated-measures correlations (with modality-family BH-FDR): several stress–physiology associations are significant (q<0.05), while no MWL–physiology association is significant (lowest q≈0.061).
- Eight FDR-significant conditionwise Spearman correlations, with a short summary of where they occur.

**Sources**:
- `results/classic_analyses/stratified_rmcorr/stratified_rmcorr_results.csv`
- `results/classic_analyses/conditionwise_correlations/conditionwise_spearman_correlations.csv`

## SVM classification
- Replaced the “Table X” placeholder with a real reference to the SVM table.
- Updated the table values to match current LOSO results and corrected the narrative to reflect best mean AUC at k=5.

**Sources**:
- `results/svm/svm_progress_stress_label.csv`
- `results/svm/svm_progress_workload_label.csv`

## Discussion consistency refresh
Updated the Discussion to remove now-outdated claims and to align with the updated Results (no new analyses):
- Replaced claims of robust stratified associations with the correct statement that no stratified rmcorr survived FDR; clarified that conditionwise correlations identified a small set of significant, condition-specific associations.
- Updated the SVM performance statement to match the current LOSO results (Stress: Accuracy=0.55, AUC=0.58; MWL: Accuracy=0.61, AUC=0.66).
- Corrected the subjective MWL interaction statement (interaction not significant) while retaining a cautious interpretation.
- Updated limitations text to match the nested LOSO setup described in Methods.

## Double-blind + SVM-only cleanup (Dec 2025)
- Removed the internal “NOTES” drafting comment in the manuscript preamble.
- Updated the Contributions “Inference” bullet to be SVM-only (removed XGBoost/continuous prediction wording).
- Removed the Acknowledgments section for strict double-blind submission.
- Fixed the incomplete Conclusion fragment (“Practically, .”).

## Regenerated stratified rmcorr heatmaps (Dec 2025)
- Increased heatmap font sizes (moderate) and reran the R script to regenerate the stratified repeated-measures correlation heatmaps.
- Fixed an NA issue in the stratified rmcorr heatmaps caused by `eda_pkht_med` being constant (all zeros) in `output/final_data.csv`; updated the canonical feature to use `eda_pkht_mean` instead and regenerated the outputs.
- Updated the manuscript figure paths to point at the regenerated heatmap outputs under `results/classic_analyses/stratified_rmcorr/` (and also copied the PNGs to the repo root to match legacy filenames).
- Increased heatmap text sizes further and changed significance display to asterisks appended to the cell correlation values (FDR-adjusted p-values).
- Updated stratified rmcorr multiple-comparison correction to also compute BH-FDR within modality families (EEG / EDA / HR/HRV / Pupil) and switched the heatmap asterisks to reflect the modality-family FDR (`p_fdr_modality`).
- Updated the stratified rmcorr heatmaps for IEEE-style readability: significant cells are highlighted via thick borders (q<0.05) and trend cells via thinner borders (0.05≤q<0.10), and vector PDF exports are generated alongside PNGs.
- Restored in-cell significance markers (asterisks/†) while keeping the border encoding, and increased in-cell text size.
- Updated manuscript Methods/Results/Discussion and heatmap captions to treat modality-family FDR stratified rmcorr as the primary correlation result (and to state the significant cells in-text), while keeping the heatmaps included as PNG.

**Script**: `scripts/analysis/correlations/stratified_rmcorr_correlations.R`

**Outputs**:
- `results/classic_analyses/stratified_rmcorr/heatmap_stress_corr.png`
- `results/classic_analyses/stratified_rmcorr/heatmap_workload_corr.png`
