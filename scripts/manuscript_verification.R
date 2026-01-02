# manuscript_verification.R
# 1. Covariate Analysis (LMM)
# 2. ICC Computation

suppressPackageStartupMessages({
  library(tidyverse)
  library(lme4)
  library(lmerTest)
  library(psych)
})

# Paths
data_path <- "output/final_data.csv"
demo_path <- "results/demographics.csv"
out_dir <- "results/manuscript_verification"
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

# Load Data
cat("Loading data...\n")
df <- read.csv(data_path)
demo <- read.csv(demo_path)

# Merge Demographics
# Ensure participant_id is integer in both
df$participant_id <- as.integer(df$participant_id)
demo$participant_id <- as.integer(demo$participant_id)

df_merged <- df %>%
  left_join(demo, by = "participant_id")

cat("Merged data dimensions:", dim(df_merged), "\n")

# ============================================================
# 1. Covariate Analysis (LMM)
# ============================================================
cat("\n--- Covariate Analysis (LMM) ---\n")
sink(file.path(out_dir, "covariate_analysis.txt"))

# Models: Outcome ~ Condition + Age + Gender + (1|Participant)
# Outcomes: stress, workload
outcomes <- c("stress", "workload")

for (outcome in outcomes) {
  cat(paste("\nModel for:", outcome, "\n"))
  formula_str <- paste(outcome, "~ condition + age + gender + (1|participant_id)")
  
  tryCatch({
    model <- lmer(as.formula(formula_str), data = df_merged)
    print(summary(model))
    
    # Check significance of covariates
    anova_res <- anova(model)
    print(anova_res)
  }, error = function(e) {
    cat("Error fitting model:", e$message, "\n")
  })
}
sink()
cat("Covariate analysis saved to", file.path(out_dir, "covariate_analysis.txt"), "\n")

# ============================================================
# 2. ICC Computation
# ============================================================
cat("\n--- ICC Computation ---\n")
sink(file.path(out_dir, "icc_results.txt"))

# We need repeated measures for ICC.
# Usually ICC is calculated on the Baseline or Pre-Condition to check stability across time/blocks,
# OR on the Task conditions to check reliability.
# The original script looked at "precond_bl" across rounds.

# Let's check if we have 'round' or 'repetition' info.
# df has 'round' column.

# Filter for Pre-Condition (Relaxation) if available, or just check stability of features across task rounds.
# Let's look at 'condition' values to identify baseline/pre-condition.
conditions <- unique(df$condition)
cat("Conditions found:", paste(conditions, collapse=", "), "\n")

# Assuming we want to check ICC of physiological features across the task rounds (1, 2, 3...)
# or across different baseline periods.

# Let's try to calculate ICC for a few key features across task rounds.
features <- c("hr_mean_precond", "hrv_rmssd_precond", "eda_mean_precond", "eeg_fm_theta_power_precond")

# Check if features exist
available_features <- intersect(features, names(df))

for (feat in available_features) {
  cat(paste("\nICC for:", feat, "\n"))
  
  # Reshape to wide format: Rows=Participants, Cols=Rounds
  # We need to ensure we have unique rounds per participant for the same condition type?
  # Or just across all data points?
  
  # Let's assume we want to check reliability across the 3 rounds of the task.
  # Filter for Task conditions if possible.
  
  icc_data <- df_merged %>%
    select(participant_id, round, all_of(feat)) %>%
    pivot_wider(names_from = round, values_from = all_of(feat), values_fn = mean) %>%
    select(-participant_id)
  
  tryCatch({
    icc_res <- ICC(icc_data)
    print(icc_res)
  }, error = function(e) {
    cat("Error calculating ICC:", e$message, "\n")
  })
}
sink()
cat("ICC results saved to", file.path(out_dir, "icc_results.txt"), "\n")
