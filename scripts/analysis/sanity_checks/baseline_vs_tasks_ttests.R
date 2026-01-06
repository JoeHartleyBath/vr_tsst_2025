### Load libraries
library(dplyr)
library(purrr)
library(broom)

config <- yaml::read_yaml("scripts/utils/config.yaml")

### Load data
df <- readRDS(file.path(config$paths$output, "full_data_for_reference.rds"))

### Canonical features
canonical_feats <- config$canonical_features
features <- canonical_feats

### Extract baseline and task data
baseline_df <- df %>%
  filter(condition_type == "Pre_Baseline") %>%
  select(participant_id, all_of(features))

task_df <- df %>%
  filter(condition_type == "Task") %>%
  group_by(participant_id) %>%
  summarise(across(all_of(features), mean, na.rm = TRUE), .groups = "drop")

### Merge baseline + mean task values
paired_df <- baseline_df %>%
  inner_join(task_df, by = "participant_id", suffix = c("_baseline", "_task"))

### Run paired t-tests for each feature
results <- map_df(features, function(feat) {
  
  x <- paired_df[[paste0(feat, "_baseline")]]
  y <- paired_df[[paste0(feat, "_task")]]
  
  tt <- t.test(x, y, paired = TRUE)
  
  tibble(
    feature = feat,
    mean_baseline = mean(x, na.rm = TRUE),
    mean_task = mean(y, na.rm = TRUE),
    diff_mean = mean(y - x, na.rm = TRUE),
    t = tt$statistic,
    p = tt$p.value,
    df = tt$parameter
  )
})

### View results
results
