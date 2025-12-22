# XGBoost Quick Test - 5-Fold CV with Default Parameters
# ========================================================

library(dplyr)
library(stringr)
library(xgboost)
library(yardstick)
library(yaml)
library(caret)

set.seed(42)

# Load config and data
config <- yaml::read_yaml("scripts/utils/config.yaml")

df <- readRDS(file.path(config$paths$output, "final_data.rds")) %>%
  mutate(
    stress_label   = if_else(str_detect(condition, "High Stress"), 1L, 0L),
    workload_label = if_else(str_detect(condition, "High Cog"),    1L, 0L)
  )

# Get features
drop_pattern <- "response|eeg_o|(_min_|_max_)|head|totalscrs|_glob|_raw|_delta"
features <- names(df)[str_detect(names(df), "_precond$")]
features <- features[!str_detect(features, drop_pattern)]

df <- df %>%
  select(participant_id, stress_label, workload_label, all_of(features)) %>%
  group_by(participant_id) %>%
  mutate(across(all_of(features), ~ scale(.)[,1])) %>%
  ungroup()

cat("\nData loaded:", nrow(df), "samples,", length(features), "features\n")

# Run quick test for both targets
targets <- c("stress_label", "workload_label")

for (target in targets) {
  cat("\n", rep("=", 60), "\n")
  cat("Target:", target, "\n")
  cat(rep("=", 60), "\n")
  
  # Create 5 stratified folds
  folds <- createFolds(df[[target]], k = 5, list = TRUE, returnTrain = FALSE)
  
  all_preds <- numeric(nrow(df))
  all_truth <- df[[target]]
  
  for (i in seq_along(folds)) {
    test_idx <- folds[[i]]
    train_idx <- setdiff(1:nrow(df), test_idx)
    
    train_df <- df[train_idx, ]
    test_df <- df[test_idx, ]
    
    # Prepare matrices
    X_train <- as.matrix(train_df[, features])
    y_train <- train_df[[target]]
    X_test <- as.matrix(test_df[, features])
    y_test <- test_df[[target]]
    
    # Handle missing values
    X_train[is.na(X_train)] <- 0
    X_test[is.na(X_test)] <- 0
    
    dtrain <- xgb.DMatrix(X_train, label = y_train)
    dtest <- xgb.DMatrix(X_test, label = y_test)
    
    # Train XGBoost with default parameters
    model <- xgb.train(
      params = list(
        objective = "binary:logistic",
        eval_metric = "logloss",
        eta = 0.1,
        max_depth = 6,
        subsample = 0.8,
        colsample_bytree = 0.8
      ),
      data = dtrain,
      nrounds = 100,
      verbose = 0
    )
    
    # Predict
    preds <- predict(model, dtest)
    all_preds[test_idx] <- preds
    
    cat("  Fold", i, "complete\n")
  }
  
  # Evaluate
  truth_factor <- factor(all_truth, levels = c(0, 1))
  preds_class <- factor(as.integer(all_preds > 0.5), levels = c(0, 1))
  
  acc <- mean(preds_class == truth_factor)
  f1 <- f_meas_vec(truth = truth_factor, estimate = preds_class, event_level = "second")
  auc <- roc_auc_vec(truth = truth_factor, estimate = all_preds, event_level = "second")
  
  cat("\nResults:\n")
  cat("  Accuracy:", round(acc, 3), "\n")
  cat("  F1:      ", round(f1, 3), "\n")
  cat("  AUC:     ", round(auc, 3), "\n")
}

cat("\n", rep("=", 60), "\n")
cat("Quick test complete!\n")
cat("\nComparison with SVM (from overnight run):\n")
cat("  Stress:   SVM AUC = 0.585\n")
cat("  Workload: SVM AUC = 0.659\n")
cat("\nIf XGBoost AUC > SVM, run full LOSO with tuning.\n")
cat(rep("=", 60), "\n\n")
