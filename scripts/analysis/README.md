# Statistical Analysis Scripts

Classical statistical analyses for hypothesis testing and exploration.

## Structure

```
analysis/
├── anova/               # Analysis of variance
├── correlations/        # Correlation analyses
└── sanity_checks/       # Data quality checks
```

## Scripts

### ANOVA (`anova/`)
- **anova_main.R**: Main ANOVA script for condition effects
- Tests differences in features across experimental conditions
- Output: `results/classic_analyses/anovas/`

### Correlations (`correlations/`)
- **conditionwise_correlations.R**: Correlations within each condition
- **stratified_rmcorr_correlations.R**: Repeated measures correlations
- Output: `results/classic_analyses/conditionwise_correlations/`, `stratified_rmcorr/`

### Sanity Checks (`sanity_checks/`)
- **baseline_vs_tasks_ttests.R**: Validate baseline vs task differences
- Quality control for expected physiological changes

## Usage

All R scripts should be run from the project root:
```r
source("scripts/analysis/anova/anova_main.R")
```

## Dependencies

- R utilities: `scripts/utils/r/data_prep_helpers.R`
- Input data: `output/aggregated/all_data_aggregated.csv`
