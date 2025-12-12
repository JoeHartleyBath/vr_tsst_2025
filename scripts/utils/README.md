# Utility Scripts

Shared helper functions and libraries used across preprocessing, analysis, and modeling.

## Structure

```
utils/
├── matlab/            # MATLAB utility functions and packages
└── r/                 # R utility functions
```

## MATLAB Utilities (`matlab/`)

### YAML Package (`+yaml/`)
- MATLAB library for reading/writing YAML configuration files
- Used by EEG preprocessing scripts
- Usage: `config = yaml.ReadYaml('config/eeglab_template.yaml')`

### Other MATLAB Utilities
- Helper functions for EEGLAB processing
- Custom signal processing utilities

## R Utilities (`r/`)

### Key Files
- **data_prep_helpers.R**: Common data loading and preparation functions
  - Load aggregated data
  - Apply baseline corrections
  - Merge with counterbalance information
- **feature_naming.R**: Standardize feature column names

### Usage

Source utilities at the start of analysis/modeling scripts:
```r
source("scripts/utils/r/data_prep_helpers.R")
data <- load_aggregated_data()
```

## Guidelines

- **Reusable**: Functions should work across multiple scripts
- **Well-documented**: Clear docstrings and examples
- **Tested**: Unit tests in `scripts/tests/`
- **No side effects**: Avoid modifying global state
