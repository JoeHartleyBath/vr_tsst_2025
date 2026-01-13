library(tidyverse)
library(lme4)
library(lmerTest)

# Load data
df <- readRDS("output/full_data_for_reference.rds")

# Check structure
# We expect condition_type == "Relaxation"
# We need to identifying the order (Block 1, 2, 3, 4).
# Check if distinct blocks are identifiable.

relax_df <- df %>% 
  filter(condition_type == "Relaxation")

# Assuming 'block' or 'round' or similar exists.
# If not, we might need to infer it from time or condition name.
# Let's inspect column names first.
print(colnames(relax_df))

# If we can identify blocks, we test for linear trend.
# Features: hr_med, eda_tonic_mean, eeg_fm_theta_power

features <- c("hr_med", "eda_tonic_mean", "eeg_fm_theta_power")

for (feat in features) {
  if (feat %in% colnames(relax_df)) {
    cat("\nTesting stability for:", feat, "\n")
    # We need a time/order variable.
    # If 'round' is 1,2,3,4
    if ("round" %in% colnames(relax_df)) {
       model <- lmer(as.formula(paste(feat, "~ round + (1|participant_id)")), data=relax_df)
       print(summary(model))
    } else {
       cat("Column 'round' not found. checking 'condition' or other columns.\n")
       print(unique(relax_df$condition))
    }
  }
}
