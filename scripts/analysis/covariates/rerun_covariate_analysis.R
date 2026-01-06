# scripts/analysis/covariates/rerun_covariate_analysis.R
# Rerun Condition Covariate Analysis (Response Rate) with new features

suppressPackageStartupMessages({
  library(tidyverse)
  library(lme4)
  library(lmerTest)
  library(yaml)
  library(broom.mixed)
})

cat("Starting Covariate Analysis Script...\n")

# 1. Load Config & Data
config <- yaml::read_yaml("scripts/utils/config.yaml")
out_dir <- config$paths$results
cov_dir <- file.path(out_dir, "covariate_analysis")
dir.create(cov_dir, showWarnings = FALSE, recursive = TRUE)

message("Loading data...")
final_data <- readRDS(file.path(config$paths$output, "final_data.rds"))
full_data  <- readRDS(file.path(config$paths$output, "full_data_for_reference.rds"))

# 2. Prepare Covariate (Response Rate)
# Extract from full_data (Task conditions only)
response_data <- full_data %>%
  filter(condition_type == "Task") %>%
  select(participant_id, round, response_rate = response_rate_per_min) %>%
  mutate(
    participant_id = as.numeric(participant_id),
    round = as.character(round)
  )

# Join to final_data
analysis_df <- final_data %>%
  mutate(
    participant_id = as.numeric(participant_id),
    round = as.character(round)
  ) %>%
  left_join(response_data, by = c("participant_id", "round")) %>%
  mutate(
    # Create factors from condition string
    stress_level = if_else(str_detect(condition, "High Stress"), "High", "Low"),
    workload_level = if_else(str_detect(condition, "High Cog"), "High", "Low"),
    
    stress_level = factor(stress_level, levels = c("Low", "High")),
    workload_level = factor(workload_level, levels = c("Low", "High")),
    
    # Standardize response rate for better convergence and interpretation
    response_rate_z = as.numeric(scale(response_rate))
  )

message("Merged Response Rate. N = ", nrow(analysis_df))

# 3. Define Features
canonical_feats <- config$canonical_features
features <- paste0(canonical_feats, "_precond")

# Verify features exist
available_features <- intersect(features, names(analysis_df))
message("Analyzing ", length(available_features), " features: ", paste(available_features, collapse = ", "))

# 4. Helper to extract model stats
get_model_stats <- function(model, feat_name, model_type) {
  tidy(model) %>%
    filter(effect == "fixed") %>%
    select(term, estimate, std.error, statistic, p.value) %>%
    mutate(
      feature = feat_name,
      model = model_type,
      sig = case_when(
        p.value < 0.001 ~ "***",
        p.value < 0.01  ~ "**",
        p.value < 0.05  ~ "*",
        p.value < 0.10  ~ "†",
        TRUE ~ ""
      )
    )
}

# 5. Run Comparison Loop
all_results <- list()

for (feat in available_features) {
  # Skip if all NA
  if (all(is.na(analysis_df[[feat]]))) next
  
  # Standardize outcome for comparable coefficients across features?
  # No, let's keep original units for consistency with ANOVA results.
  
  # Formulae
  f_base <- as.formula(paste(feat, "~ stress_level * workload_level + (1|participant_id)"))
  f_cov  <- as.formula(paste(feat, "~ stress_level * workload_level + response_rate_z + (1|participant_id)"))
  
  # Fit Models
  m_base <- try(lmer(f_base, data = analysis_df), silent = TRUE)
  m_cov  <- try(lmer(f_cov, data = analysis_df), silent = TRUE)
  
  # Process Base
  if (!inherits(m_base, "try-error")) {
    stats_base <- get_model_stats(m_base, feat, "baseline")
    all_results[[length(all_results) + 1]] <- stats_base
  }
  
  # Process Covariate
  if (!inherits(m_cov, "try-error")) {
    stats_cov <- get_model_stats(m_cov, feat, "with_response_rate")
    all_results[[length(all_results) + 1]] <- stats_cov
  }
}

# 6. Combine and Format Results
results_df <- bind_rows(all_results)

# Create Wide Comparison Table (Matching previous format broadly)
comparison_df <- results_df %>%
  select(feature, term, model, estimate, std.error, statistic, p.value, sig) %>%
  pivot_wider(
    names_from = model,
    values_from = c(estimate, std.error, statistic, p.value, sig),
    names_glue = "{.value}_{model}"
  ) %>%
  # Filter to terms of interest
  filter(term %in% c("(Intercept)", "stress_levelHigh", "workload_levelHigh", 
                     "response_rate_z", "stress_levelHigh:workload_levelHigh")) %>%
  arrange(feature, term)

# 7. Save
out_file <- file.path(cov_dir, "covariate_comparison_summary_rerun.csv")
write_csv(comparison_df, out_file)

message("Covariate analysis complete.")
message("Results saved to: ", out_file)
