# Logistic Regression LOSO Classification – Stress & Workload
# -------------------------------------------------------------
# Matches SVM pipeline structure: LOSO with nested CV for tuning

library(tidyverse)
library(glmnet)
library(doParallel)
library(yaml)
library(dplyr)
library(caret)
library(yardstick)

set.seed(42)

source("utils/r/prune_feats.R")
source("utils/r/feature_selection.R")

# =====================================================================
# 1. Load data
# =====================================================================
config  <- yaml::read_yaml("scripts/utils/config.yaml")

out_dir_logreg <- file.path(config$paths$results, "logreg")
dir.create(out_dir_logreg, recursive = TRUE, showWarnings = FALSE)

targets <- c("stress_label", "workload_label")

pf_list <- setNames(
  sapply(targets, function(t) file.path(out_dir_logreg, paste0("logreg_progress_", t, ".csv"))),
  targets
)

main_header <- c(
  "timestamp", "test_pid", "target", "k",
  "final_acc", "final_f1", "final_auc", "inner_best_acc",
  "lambda", "alpha", "feature_n", "feature_set", "features_used"
)

tuning_header <- c(
  "timestamp", "test_pid", "target", "k", "lambda", "alpha",
  "inner_acc", "inner_auc", "inner_f1", "feature_n", "feature_set", "features_used"
)

# Write headers
for (pf in pf_list) {
  write.table(t(main_header), file = pf, sep = ",",
              col.names = FALSE, row.names = FALSE, quote = FALSE)
}

for (t in targets) {
  write.table(t(tuning_header), file = paste0(pf_list[[t]], "_tuning.csv"), sep = ",",
              col.names = FALSE, row.names = FALSE, quote = FALSE)
}

# =====================================================================
# 2. Load and prepare data
# =====================================================================
df <- readRDS(file.path(config$paths$output, "final_data.rds")) %>%
  mutate(
    stress_label = factor(if_else(str_detect(condition, "High Stress"), 1L, 0L), levels = c(0, 1)),
    workload_label = factor(if_else(str_detect(condition, "High Cog"), 1L, 0L), levels = c(0, 1))
  )

# Feature filtering
drop_pattern <- paste0(
  "response|eeg_o|(_min_|_max_)|head|totalscrs|_glob|_raw|_delta|_assym"
)
features <- names(df)[str_detect(names(df), "_precond$")]
features <- features[!str_detect(features, drop_pattern)]

df <- df %>%
  dplyr::select(participant_id, stress_label, workload_label, all_of(features))

df <- df %>%
  group_by(participant_id) %>%
  mutate(across(all_of(features), ~ scale(.)[,1])) %>%
  ungroup()

participants <- unique(df$participant_id)

# =====================================================================
# 3. Hyperparameter grids
# =====================================================================
# Lambda (regularization strength) - log scale
lambda_grid <- 10^seq(-4, 1, length.out = 12)

# Alpha: 0 = Ridge, 1 = Lasso, 0.5 = Elastic Net
alpha_grid <- c(0, 0.25, 0.5, 0.75, 1.0)

# k values to try
k_grid <- c(20, 10, 5)

# =====================================================================
# 4. Core Logistic Regression evaluation
# =====================================================================
run_logreg <- function(train, test, target, feats, lambda, alpha) {
  X_train <- as.matrix(train[, feats])
  y_train <- as.numeric(as.character(train[[target]]))
  X_test <- as.matrix(test[, feats])
  y_test <- test[[target]]
  
  # Handle NAs
  X_train[is.na(X_train)] <- 0
  X_test[is.na(X_test)] <- 0
  
  # Train model
  model <- glmnet(
    x = X_train,
    y = y_train,
    family = "binomial",
    alpha = alpha,
    lambda = lambda
  )
  
  # Predict
  probs <- predict(model, newx = X_test, s = lambda, type = "response")[,1]
  preds <- factor(ifelse(probs > 0.5, 1, 0), levels = c(0, 1))
  
  # Metrics
  acc <- mean(preds == y_test)
  f1  <- f_meas_vec(truth = y_test, estimate = preds, event_level = "second")
  auc <- roc_auc_vec(truth = y_test, estimate = probs, event_level = "second")
  
  list(acc = acc, f1 = f1, auc = auc)
}

# =====================================================================
# 5. Nested LOSO
# =====================================================================
nested_loso <- function(df, target, progress_file) {
  outer_results <- list()
  
  for (test_pid in participants) {
    
    # -----------------------------
    # 1. Train/Test split
    # -----------------------------
    train_df <- df %>% filter(participant_id != test_pid)
    test_df  <- df %>% filter(participant_id == test_pid)
    
    # -----------------------------
    # 2. Prune features
    # -----------------------------
    pruned_feats <- prune_features(train_df, target = target)
    
    # Drop NAs using training medians
    train_df <- train_df %>% mutate(across(all_of(pruned_feats), ~ replace(., is.na(.), median(., na.rm = TRUE))))
    test_df  <- test_df  %>% mutate(across(all_of(pruned_feats), ~ replace(., is.na(.), median(train_df[[cur_column()]], na.rm = TRUE))))
    
    # -----------------------------
    # 3. Try different k values
    # -----------------------------
    for (k in k_grid) {
      
      # Select top k features
      if (length(pruned_feats) <= k) {
        feats_k <- pruned_feats
      } else {
        # Simple variance-based selection
        feat_vars <- sapply(train_df[pruned_feats], var, na.rm = TRUE)
        feats_k <- names(sort(feat_vars, decreasing = TRUE)[1:k])
      }
      
      # -----------------------------
      # 4. Inner CV for hyperparameter tuning
      # -----------------------------
      inner_pids <- setdiff(participants, test_pid)
      
      tune_results <- map_df(inner_pids, function(inner_pid) {
        inner_train <- train_df %>% filter(participant_id != inner_pid)
        inner_test  <- train_df %>% filter(participant_id == inner_pid)
        
        # Grid search
        grid_results <- expand.grid(lambda = lambda_grid, alpha = alpha_grid) %>%
          rowwise() %>%
          mutate(
            metrics = list(run_logreg(inner_train, inner_test, target, feats_k, lambda, alpha)),
            inner_acc = metrics$acc,
            inner_auc = metrics$auc,
            inner_f1  = metrics$f1
          ) %>%
          dplyr::select(-metrics)
        
        # Average across inner folds
        grid_results %>%
          group_by(lambda, alpha) %>%
          summarise(
            inner_acc = mean(inner_acc, na.rm = TRUE),
            inner_auc = mean(inner_auc, na.rm = TRUE),
            inner_f1  = mean(inner_f1, na.rm = TRUE),
            .groups = "drop"
          )
      })
      
      # Aggregate tuning results
      tune_summary <- tune_results %>%
        group_by(lambda, alpha) %>%
        summarise(
          inner_acc = mean(inner_acc, na.rm = TRUE),
          inner_auc = mean(inner_auc, na.rm = TRUE),
          inner_f1  = mean(inner_f1, na.rm = TRUE),
          .groups = "drop"
        ) %>%
        mutate(
          timestamp = Sys.time(),
          test_pid = test_pid,
          target = target,
          k = k,
          feature_n = length(feats_k),
          feature_set = paste0("top", k),
          features_used = paste(feats_k, collapse = ";")
        )
      
      best_row <- tune_summary[which.max(tune_summary$inner_auc), ]
      
      # Append tuning iteration detail log
      write.table(
        tune_summary,
        file = paste0(progress_file, "_tuning.csv"),
        append = TRUE,
        sep = ",",
        col.names = FALSE,
        row.names = FALSE
      )
      
      best_inner_acc <- best_row$inner_acc
      best_params <- list(lambda = best_row$lambda, alpha = best_row$alpha)
      
      # -----------------------------
      # 6. Final LOSO test
      # -----------------------------
      final_metrics <- run_logreg(train_df, test_df, target, feats_k, best_params$lambda, best_params$alpha)
      
      res_row <- tibble(
        timestamp      = Sys.time(),
        test_pid       = test_pid,
        target         = target,
        k              = k,
        final_acc      = final_metrics$acc,
        final_f1       = final_metrics$f1,
        final_auc      = final_metrics$auc,
        inner_best_acc = best_inner_acc,
        lambda         = best_params$lambda,
        alpha          = best_params$alpha,
        feature_n      = length(feats_k),
        feature_set    = paste0("top", k),
        features_used  = paste(feats_k, collapse = ";")
      )
      
      write.table(
        res_row,
        file = progress_file,
        append = TRUE,
        sep = ",",
        col.names = FALSE,
        row.names = FALSE
      )
      
      cat(sprintf("[%s] P%02d k=%d → AUC=%.3f (λ=%.4f, α=%.2f)\n",
                  target, test_pid, k, final_metrics$auc, best_params$lambda, best_params$alpha))
    }
  }
}

# =====================================================================
# 6. Run for both targets
# =====================================================================
cat("\n=== Starting Logistic Regression LOSO ===\n")
cat("Participants:", length(participants), "\n")
cat("Features:", length(features), "\n")
cat("k-values:", paste(k_grid, collapse = ", "), "\n\n")

for (target in targets) {
  cat("\n", rep("=", 60), "\n")
  cat("TARGET:", target, "\n")
  cat(rep("=", 60), "\n")
  
  nested_loso(df, target, pf_list[[target]])
  
  cat("\nCompleted:", target, "\n")
}

cat("\n=== Logistic Regression LOSO Complete ===\n")
cat("Results saved to:", out_dir_logreg, "\n")
cat("\nRun best model finder:\n")
cat("  Rscript pipelines/11_r_logreg/logreg_best_model_finder.R\n\n")
