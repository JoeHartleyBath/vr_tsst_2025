library(tidyverse)
library(e1071)
library(doParallel)
library(yaml)
library(dplyr)
library(caret)
library(yardstick)

set.seed(42)

# =====================================================================
# 1. Config & Setup
# =====================================================================
config  <- yaml::read_yaml("config/general.yaml")
out_dir_svm <- file.path("results", "svm")
dir.create(out_dir_svm, recursive = TRUE, showWarnings = FALSE)

# Progress file
progress_file <- file.path(out_dir_svm, "svm_pooled_heuristic_stress_lowmwl_only.csv")

header <- c(
  "timestamp", "fold", "target", "accuracy", "f1", "auc", 
  "features_used"
)
write.table(t(header), file = progress_file, sep = ",", col.names = FALSE, row.names = FALSE)

# =====================================================================
# 2. Data Loading & Filtering
# =====================================================================
df <- readRDS(file.path("output", "final_data.rds"))

# Filter for Low Workload ONLY
# We want to compare Low Stress vs High Stress within the Low MWL block
df_subset <- df %>%
  filter(str_detect(condition, "Low Cog")) %>%
  mutate(
    target = factor(if_else(str_detect(condition, "High Stress"), 1L, 0L), 
                    levels = c(0, 1))
  )

# Select Specific Features
features <- c("eda_tonic_mean_precond", "hr_med_precond", 
              "pupil_full_pupil_med_precond", "eeg_faa_precond")

df_subset <- df_subset %>%
  select(participant_id, target, all_of(features)) %>%
  na.omit() # Drop rows with missing values in these key features

# Subject-level scaling (critical for pooled model)
# NOTE: Scaling with fewer points (2 per subject) is risky but necessary for pooled SVM
# If variance is 0 (e.g. constant values), scale returns NaNs. We must handle this.
scale_safe <- function(x) {
  # Handle all-NA or empty vectors
  if (all(is.na(x)) || length(x) < 2) return(rep(0, length(x)))
  
  s <- sd(x, na.rm = TRUE)
  # Handle zero variance or NA variance
  if (is.na(s) || s == 0) return(rep(0, length(x)))
  
  return(as.vector(scale(x)))
}

df_subset <- df_subset %>%
  group_by(participant_id) %>%
  mutate(across(all_of(features), scale_safe)) %>%
  ungroup()

# =====================================================================
# 3. Model Function
# =====================================================================
run_svm_fixed <- function(train, test, feats) {
  # Heuristic: C=1, Gamma = 1 / n_features
  model <- svm(
    reformulate(feats, "target"),
    data = train,
    kernel = "linear",
    cost = 1,
    gamma = 1 / length(feats),
    probability = TRUE,
    scale = FALSE
  )
  
  preds <- predict(model, test %>% select(all_of(feats)))
  acc <- mean(preds == pull(test, "target"))
  
  pred_prob <- predict(model, test %>% select(all_of(feats)), probability = TRUE)
  prob_vals <- attr(pred_prob, "probabilities")[, "1"]
  
  auc <- roc_auc_vec(truth = test$target, estimate = prob_vals, event_level = "second")
  f1  <- f_meas_vec(truth = test$target, estimate = preds, event_level = "second")
  
  list(acc = acc, f1 = f1, auc = auc)
}

# =====================================================================
# 4. 5-Fold Pooled CV
# =====================================================================
# Stratified by target
folds <- createFolds(df_subset$target, k = 5, list = TRUE, returnTrain = FALSE)

results <- list()

print(paste("Samples:", nrow(df_subset)))

for (i in seq_along(folds)) {
  test_idx <- folds[[i]]
  train_df <- df_subset[-test_idx, ]
  test_df  <- df_subset[test_idx, ]
  
  res <- run_svm_fixed(train_df, test_df, features)
  
  row <- data.frame(
    timestamp = Sys.time(),
    fold = names(folds)[i],
    target = "stress_low_mwl",
    accuracy = res$acc,
    f1 = res$f1,
    auc = res$auc,
    features_used = paste(features, collapse = ";")
  )
  
  write.table(row, file = progress_file, append = TRUE, sep = ",", col.names = FALSE, row.names = FALSE)
  results[[i]] <- row
}

# Summary
final_res <- bind_rows(results)
print(final_res %>% summarise(mean_acc = mean(accuracy), mean_auc = mean(auc)))