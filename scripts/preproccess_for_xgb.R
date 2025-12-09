# ML data preparation script

library(tidyverse)
library(readxl)
library(rstatix)
library(rmcorr)
library(glue)
library(rlang)
library(yaml)

# Load config & helper scripts
config <- yaml::read_yaml("scripts/utils/config.yaml")
source("scripts/utils/feature_naming.R")
source("scripts/utils/transform_functions.R")
source("scripts/utils/data_prep_helpers.R")
source("scripts/utils/save_helpers.R")

# 1. Load and pre-process raw data
data <- load_and_prepare_data(config)

subjective_cols <- c("stress", "workload")

# 2. Compute full-window metrics and baseline-adjusted values
processed   <- prepare_full_window_data(data)
full_data   <- processed$full_data
full_change_long <- processed$full_change_long


# 3. Merge into wide-format dataset
final_data <- wide_transform_full_changes(full_data, full_change_long, subjective_cols)

# 4. Extract raw (non-normalised) ANOVA dataset
anova_data <- make_anova_dataset(final_data, subjective_cols, config)

# 5. Apply scaling/transforms
final_data_transformed <- apply_transformations(final_data, subjective_cols, full_data)

# 6. Remove redundant / duplicate feature variants
final_data_transformed <- clean_feature_duplicates(final_data_transformed)

# 7. Save final outputs
save_outputs(
  final_data_transformed = final_data_transformed,
  full_data = full_data,
  config = config
)
