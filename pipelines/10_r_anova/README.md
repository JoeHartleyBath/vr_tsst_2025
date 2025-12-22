# Stage 10: R ANOVA Statistical Analysis

Repeated-measures ANOVA for experimental condition effects on physiological features.

## Entry Point
- **Main**: `anova_main.R`

## Inputs
- **File**: `output/anova_features_precond.rds` (from Stage 06)
- **Config**: `scripts/utils/config.yaml`

## Outputs
- **Directory**: `results/classic_analyses/`
- **Files**:
  - `anova_results_*.csv` (F-statistics, p-values, effect sizes)
  - `post_hoc_comparisons_*.csv` (pairwise comparisons with corrections)
  - `descriptive_stats_*.csv` (means, SDs by condition)
  - Plots: Interaction plots, box plots, effect size visualizations

## Analyses
- **Repeated-Measures ANOVA**: Within-subject effects of experimental condition
- **Factors**: 
  - Condition (baseline, math1, math2, math3, social, recovery, forest)
  - Optional: Stress level, Workload level as between-subject factors
- **Post-Hoc**: Tukey HSD or Bonferroni-corrected pairwise comparisons
- **Effect Sizes**: Partial eta-squared, Cohen's d

## Features Analyzed
- EEG spectral power (delta, theta, alpha, beta, gamma)
- EEG band ratios (theta/beta, alpha/theta)
- Heart rate and HRV metrics
- GSR amplitude and SCR frequency
- Pupil dilation metrics

## Statistical Tests
1. Test sphericity assumptions (Mauchly's test)
2. Apply Greenhouse-Geisser correction if needed
3. Compute main effects and interactions
4. Post-hoc pairwise comparisons
5. Effect size calculations

## Dependencies
- R 4.3+ with afex, emmeans, tidyverse, car

## Usage
```r
Rscript anova_main.R
```

## Performance
- Processing time: 10-30 minutes (depends on feature count)

## Notes
- Provides traditional statistical validation of experimental manipulation
- Complements machine learning results (Stages 7-9)
- Results inform feature selection for ML models
- Useful for publication: experimental validation of stress/workload induction
