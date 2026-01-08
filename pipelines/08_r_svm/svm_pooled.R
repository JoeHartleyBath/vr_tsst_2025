library(tidyverse)
library(e1071)
library(doParallel)
library(yaml)
library(dplyr)
library(caret)
library(yardstick)

set.seed(42)

# Load shared utilities
source("utils/r/prune_feats.R")
source("utils/r/feature_selection.R")

# =====================================================================
# 1. Config & Setup
# =====================================================================
config  <- yaml::read_yaml("config/general.yaml")

# Save to same results folder but distiguished filenames
out_dir_svm <- file.path("results", "svm")
dir.create(out_dir_svm, recursive = TRUE, showWarnings = FALSE)

targets <- c("stress_label", "workload_label")

# Progress files
pf_list <- setNames(
  sapply(targets, function(t) file.path(out_dir_svm, paste0("svm_pooled_5fold_progress_", t, ".csv"))),
  targets
)

main_header <- c(
  "timestamp", "fold", "target", "k",
  "final_acc", "final_f1", "final_auc", "inner_best_acc",
  "cost", "gamma", "feature_n", "feature_set", "features_used"
)

tuning_header <- c(
  "timestamp", "fold", "target", "k", "cost", "gamma",
  "inner_acc", "inner_auc", "inner_f1", "feature_n", "feature_set", "features_used"
)

# Write headers for new files
for (pf in pf_list) {
  write.table(t(main_header), file = pf, sep = ",",
              col.names = FALSE, row.names = FALSE, quote = FALSE)
  
  write.table(t(tuning_header), file = paste0(pf, "_tuning.csv"), sep = ",",
              col.names = FALSE, row.names = FALSE, quote = FALSE)
}

# =====================================================================
# 2. Data Loading & Prep (Exact same as LOSO)
# =====================================================================
df <- readRDS(file.path("output", "final_data.rds"))

df <- df %>%
  mutate(
    stress_label = factor(if_else(str_detect(condition, "High Stress"), 1L, 0L), 
                          levels = c(0, 1)),
    workload_label = factor(if_else(str_detect(condition, "High Cog"), 1L, 0L), 
                            levels = c(0, 1))
  )

features <- select_analysis_features(df, suffix = "_precond")

df <- df %>%
  select(participant_id, stress_label, workload_label, all_of(features))

# NOTE: We keep within-subject scaling. This is critical for the "idiosyncracy" argument.
# We want to show that even after subject-normalization, the PATTERNS (clusters) are personal.
df <- df %>%
  group_by(participant_id) %>%
  mutate(across(all_of(features), ~ scale(.)[,1])) %>%
  ungroup()

# =====================================================================
# 3. Hyperparameter grids
# =====================================================================
cost_grid <- c(2, 4, 8, 16, 32, 64, 128, 256)
gamma_grid <- c(2^-7, 2^-6, 2^-5, 2^-4, 2^-3, 0.03, 0.06, 0.1)
k_grid <- c(20, 10, 5)

# =====================================================================
# 4. Helper Functions
# =====================================================================
run_svm_single <- function(train, test, target, feats, cost, gamma) {
  model <- svm(
    reformulate(feats, target),
    data = train,
    kernel = "radial",
    cost = cost,
    gamma = gamma,
    probability = TRUE,
    scale = FALSE
  )
  
  preds <- predict(model, test %>% select(all_of(feats)))
  acc <- mean(preds == pull(test, target))
  
  pred_obj <- predict(
    model,
    test %>% select(all_of(feats)),
    probability = TRUE
  )
  
  prob_df <- attr(pred_obj, "probabilities")
  positive_class <- levels(train[[target]])[2]
  probs <- prob_df[, positive_class]
  
  f1  <- f_meas_vec(truth = test[[target]], estimate = preds, event_level = "second")
  auc <- roc_auc_vec(truth = test[[target]], estimate = probs, event_level = "second")
  
  list(acc = acc, f1 = f1, auc = auc)
}

# =====================================================================
# 5. Nested Pooled 5-Fold CV
# =====================================================================
run_pooled_cv <- function(df, target, progress_file) {
  
  # createFolds ensures stratification by target class
  # We use returnTrain = FALSE to get test indices
  folds <- createFolds(df[[target]], k = 5, list = TRUE, returnTrain = FALSE)
  
  outer_results <- list()
  
  for (f_idx in seq_along(folds)) {
    fold_name <- names(folds)[f_idx]
    test_indices <- folds[[f_idx]]
    
    # 1. Train/Test split (Pools all subjects)
    test_df  <- df[test_indices, ]
    train_df <- df[-test_indices, ]
    
    # 2. Prune features (on training set ONLY)
    pruned_feats <- prune_features(train_df, target = target)
    
    # Handle NAs
    train_df <- train_df %>% mutate(across(all_of(pruned_feats), ~ replace(., is.na(.), median(., na.rm = TRUE))))
    test_df  <- test_df  %>% mutate(across(all_of(pruned_feats), ~ replace(., is.na(.), median(train_df[[cur_column()]], na.rm = TRUE))))
    
    # 3. Rank features (Training set ONLY)
    r_ranked_feats <- pruned_feats[
      order(abs(cor(train_df[pruned_feats],  as.numeric(train_df[[target]]), use = "complete.obs")),
            decreasing = TRUE)
    ]
    
    # 4. Loop over K features
    for (k in k_grid) {
      feats_k <- head(r_ranked_feats, k)
      
      # 5. Nested Inner CV for Tuning (5-fold on TRAINING data)
      tune_grid <- expand.grid(cost = cost_grid, gamma = gamma_grid)
      
      # Create inner folds
      inner_folds <- createFolds(train_df[[target]], k = 5, list = TRUE, returnTrain = FALSE)
      
      # Parallel Grid Search
      tune_results <- foreach(i = 1:nrow(tune_grid), .combine = rbind, 
                              .packages = c("dplyr", "yardstick", "e1071", "caret")) %dopar% {
                                cost  <- tune_grid$cost[i]
                                gamma <- tune_grid$gamma[i]
                                
                                # Evaluate across 5 inner folds
                                inner_metrics <- vector("list", 5)
                                for(j in 1:5) {
                                  inner_test_idx <- inner_folds[[j]]
                                  inner_valid <- train_df[inner_test_idx, ]
                                  inner_train <- train_df[-inner_test_idx, ]
                                  
                                  res <- run_svm_single(inner_train, inner_valid, target, feats_k, cost, gamma)
                                  inner_metrics[[j]] <- data.frame(acc = res$acc, auc = res$auc, f1 = res$f1)
                                }
                                inner_metrics <- bind_rows(inner_metrics)
                                
                                data.frame(
                                  timestamp   = Sys.time(),
                                  fold        = fold_name,
                                  target      = target,
                                  k           = k,
                                  cost        = cost,
                                  gamma       = gamma,
                                  inner_acc   = mean(inner_metrics$acc, na.rm=TRUE),
                                  inner_auc   = mean(inner_metrics$auc, na.rm=TRUE),
                                  inner_f1    = mean(inner_metrics$f1, na.rm=TRUE),
                                  feature_n   = length(feats_k),
                                  feature_set = paste0("top", k),
                                  features_used = paste(feats_k, collapse = ";")
                                )
                              }
      
      best_row <- tune_results[which.max(tune_results$inner_auc), ]
      
      # Log tuning
      write.table(
        tune_results,
        file = paste0(progress_file, "_tuning.csv"),
        append = TRUE, sep = ",", col.names = FALSE, row.names = FALSE
      )
      
      # 6. Final Evaluation for this Fold
      final_metrics <- run_svm_single(train_df, test_df, target, feats_k, best_row$cost, best_row$gamma)
      
      res_row <- tibble(
        timestamp      = Sys.time(),
        fold           = fold_name,
        target         = target,
        k              = k,
        final_acc      = final_metrics$acc,
        final_f1       = final_metrics$f1,
        final_auc      = final_metrics$auc,
        inner_best_acc = best_row$inner_acc,
        cost           = best_row$cost,
        gamma          = best_row$gamma,
        feature_n      = length(feats_k),
        feature_set    = paste0("top", k),
        features_used  = paste(feats_k, collapse = ";")
      )
      
      write.table(res_row, file = progress_file, append = TRUE, sep = ",", col.names = FALSE, row.names = FALSE)
      outer_results <- append(outer_results, list(res_row))
    }
  }
  
  bind_rows(outer_results)
}

# =====================================================================
# 6. Execution
# =====================================================================
cores <- min(10L, parallel::detectCores() - 2)
cl <- makeCluster(cores)
registerDoParallel(cl)

clusterExport(cl, c("run_svm_single"))

print("Starting Pooled 5-Fold SVM...")

results_list <- list()
for (t in targets) {
  print(paste("Running target:", t))
  # Note: running sequentially per target to simplify logging, inner loop is parallel
  res <- run_pooled_cv(df, t, progress_file = pf_list[[t]])
  results_list[[t]] <- res
}

stopCluster(cl)

print("Formatting summary...")
all_res <- bind_rows(results_list)

summary_results <- all_res %>%
  group_by(target, k) %>%
  summarise(
    mean_acc = mean(final_acc),
    mean_auc = mean(final_auc),
    sd_auc = sd(final_auc),
    .groups = "drop"
  )

print(summary_results)