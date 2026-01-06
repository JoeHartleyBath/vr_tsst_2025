# check_feature_icc.R
# Calculate ICC for baseline-corrected features to assess between-subject variance

suppressPackageStartupMessages({
  library(tidyverse)
  library(psych)
  library(lme4)
})

# Load Data
df <- read.csv("output/final_data.csv")

# Features that showed some promise in ANOVA (from manuscript)
features <- c(
  "hrv_rmssd_precond",
  "hr_med_precond",
  "eda_tonic_mean_precond",
  "eda_pkht_med_precond",
  "eeg_fm_theta_power_precond",
  "eeg_f_beta_power_precond",
  "eeg_p_alpha_power_precond",
  "eeg_ab_ratio_precond",
  "eeg_tb_ratio_precond",
  "pupil_full_pupil_med_precond"
)

cat("--- ICC Analysis of Baseline-Corrected Features ---\n")
cat("High ICC indicates that individual differences (Between-Subject) dominate the variance,\n")
cat("which makes LOSO classification difficult.\n\n")

for (feat in features) {
  # Check if feature exists
  if (!feat %in% names(df)) next
  
  # Subset data: We want to see if 'participant' explains variance in the *response*
  # We use the baseline-corrected values (which are already in the CSV with _precond suffix)
  
  # Method 1: Psych package ICC (requires wide format)
  # We average across conditions to see if people have a "stable" response style?
  # Or we just look at the variance components directly using LMM.
  
  # LMM Approach: Value ~ (1|Participant)
  # This tells us how much variance in the *baseline-corrected* data is still attached to the ID.
  
  formula_str <- paste(feat, "~ (1|participant_id)")
  model <- lmer(as.formula(formula_str), data = df)
  
  vc <- VarCorr(model)
  var_between <- as.numeric(vc)
  var_residual <- attr(vc, "sc")^2
  icc <- var_between / (var_between + var_residual)
  
  cat(paste0("Feature: ", feat, "\n"))
  cat(paste0("  Between-Subject Variance: ", round(var_between, 4), "\n"))
  cat(paste0("  Within-Subject Variance:  ", round(var_residual, 4), "\n"))
  cat(paste0("  ICC (Proportion Between): ", round(icc, 3), "\n"))
  cat("------------------------------------------------\n")
}
