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
source("utils/r/feature_naming.R")
source("utils/r/transform_functions.R")
source("utils/r/data_prep_helpers.R")
source("utils/r/save_helpers.R")

# 1. Load and pre-process raw data
data <- load_and_prepare_data(config)

subjective_cols <- c("stress", "workload")

# 2. Compute full-window metrics and baseline-adjusted values
processed   <- prepare_full_window_data(data)
full_data   <- processed$full_data
full_change_long <- processed$full_change_long


# 3. Merge into wide-format dataset
final_data <- wide_transform_full_changes(full_data, full_change_long, subjective_cols)

# Log sample sizes
message(sprintf(
  "[Dataset] Full dataset: N=%d participants (includes %d QC failures for non-EEG analyses)",
  n_distinct(final_data$participant_id),
  sum(final_data %>% distinct(participant_id, qc_failed) %>% pull(qc_failed))
))

# 4. Extract raw (non-normalised) ANOVA dataset
anova_data <- make_anova_dataset(final_data, subjective_cols, config)

# 5. Apply scaling/transforms (on full dataset)
final_data_transformed <- apply_transformations(final_data, subjective_cols, full_data)

# 6. Remove redundant / duplicate feature variants
final_data_transformed <- clean_feature_duplicates(final_data_transformed)

# 7. Create EEG-valid subset for ML pipelines
# Filter out QC failures ONLY for analyses involving EEG features
final_data_eeg_valid <- final_data_transformed %>%
  filter(!qc_failed)

message(sprintf(
  "[Dataset] EEG-valid subset: N=%d participants (QC failures excluded for EEG analyses)",
  n_distinct(final_data_eeg_valid$participant_id)
))

# 8. Save final outputs
save_outputs(
  final_data_transformed = final_data_transformed,
  final_data_eeg_valid = final_data_eeg_valid,
  full_data = full_data,
  config = config
)
