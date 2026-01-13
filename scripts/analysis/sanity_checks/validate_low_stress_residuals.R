# =====================================================================
# validate_low_stress_residuals.R
# Purpose: Sanity check to ensure the "Low Stress" condition is not 
# significantly elevated compared to the pre-experiment baseline.
#
# Inputs: output/final_data.csv
# Outputs: results/classic_analyses/sanity_checks/low_stress_residual_tests_vs0.csv
# =====================================================================

suppressPackageStartupMessages({
  library(dplyr)
  library(readr)
  library(tidyr)
  library(purrr)
  library(stringr)
  library(yaml)
})

# --------------------------
# Configuration
# --------------------------
config <- yaml::read_yaml("scripts/utils/config.yaml")
infile <- file.path("output", "final_data.csv")
out_dir <- file.path("results", "classic_analyses", "sanity_checks")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

# --------------------------
# Load Data
# --------------------------
if (!file.exists(infile)) stop("Missing input: ", infile)

raw_df <- readr::read_csv(infile, show_col_types = FALSE)

# Preprocessing: Define factors and numeric types
df_all <- raw_df %>%
  mutate(
    participant_id = factor(participant_id),
    stress_level = factor(ifelse(grepl("^High Stress", condition), "High", "Low"), levels = c("Low", "High")),
    stress_rating = as.numeric(stress)
  ) %>%
  filter(is.finite(stress_rating))

# Select physio features (excluding EEG for simple check if desired, or assume canonical_features lists all)
canonical_feats <- config$canonical_features
# Usually we check all canonical features. Adjust if you only want non-EEG.
feature_cols <- paste0(canonical_feats, "_precond")
feature_cols <- feature_cols[feature_cols %in% names(df_all)]

# --------------------------
# Helper Function
# --------------------------
one_sample_t_on_participant_means <- function(df, value_col, label) {
  # Collapse to one mean per participant (in case of multiple low stress trials per participant)
  d <- df %>%
    group_by(participant_id) %>%
    summarize(v = mean(.data[[value_col]], na.rm = TRUE), .groups = "drop") %>%
    filter(is.finite(v))

  if (nrow(d) < 5) return(NULL)

  # t-test against 0
  tt <- stats::t.test(d$v, mu = 0)
  tibble(
    measure = label,
    n_participants = nrow(d),
    mean_val = mean(d$v),
    t_stat = unname(tt$statistic),
    p_val = unname(tt$p.value)
  )
}

# --------------------------
# Analysis: Low Stress vs 0
# --------------------------
low_stress_df <- df_all %>% filter(stress_level == "Low")

message("Testing Low Stress residuals against 0 (N=", n_distinct(low_stress_df$participant_id), ")")

# 1. Subjective Stress
results_subj <- one_sample_t_on_participant_means(
  low_stress_df, "stress_rating", "Subjective Stress (Low Condition)"
)

# 2. Physio Features
results_physio <- purrr::map_dfr(feature_cols, function(fc) {
  one_sample_t_on_participant_means(low_stress_df, fc, fc)
})

# 3. Combine and Save
final_tbl <- bind_rows(results_subj, results_physio) %>%
  mutate(p_fdr = p.adjust(p_val, method = "BH"))

out_file <- file.path(out_dir, "low_stress_residual_tests_vs0.csv")
readr::write_csv(final_tbl, out_file)
message("✓ Results saved to ", out_file)
