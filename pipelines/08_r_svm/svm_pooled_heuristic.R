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

# Save to same results folder but distiguished filenames for the heuristic run
out_dir_svm <- file.path("results", "svm")
dir.create(out_dir_svm, recursive = TRUE, showWarnings = FALSE)

targets <- c("stress_label", "workload_label")

# Progress files - updated suffix to indicate no tuning
pf_list <- setNames(
  sapply(targets, function(t) file.path(out_dir_svm, paste0("svm_pooled_5fold_heuristic_progress_", t, ".csv"))),
  targets
)

main_header <- c(
  "timestamp", "fold", "target", "k",
  "final_acc", "final_f1", "final_auc", "inner_best_acc",
  "cost", "gamma", "feature_n", "feature_set", "features_used"
)

# Write headers for new files
for (pf in pf_list) {
  write.table(t(main_header), file = pf, sep = ",",
              col.names = FALSE, row.names = FALSE, quote = FALSE)
}

# =====================================================================
# 2. Data Loading & Prep
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

# NOTE: We keep within-subject scaling.
df <- df %>%
  group_by(participant_id) %>%
  mutate(across(all_of(features), ~ scale(.)[,1])) %>%
  ungroup()

# =====================================================================
# 3. Parameters
# =====================================================================
# Heuristic Parameters (Fixed)
# Cost = 1 (Standard soft margin)
# Gamma = 1/k (Calculated dynamically per feature set size)
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
# 5. Pooled 5-Fold CV (No Inner Tuning)
# =====================================================================
run_pooled_cv_heuristic <- function(df, target, progress_file) {
  
  # createFolds ensures stratification by target class
  folds <- createFolds(df[[target]], k = 5, list = TRUE, returnTrain = FALSE)
  
  # Parallelize the folds loop directly
  outer_results <- foreach(f_idx = seq_along(folds), .combine = rbind,
                           .packages = c("dplyr", "yardstick", "e1071", "caret")) %dopar% {
    
    fold_name <- names(folds)[f_idx]
    test_indices <- folds[[f_idx]]
    
    # 1. Train/Test split
    test_df  <- df[test_indices, ]
    train_df <- df[-test_indices, ]
    
    # 2. Prune features (on training set ONLY) - Using sourced function
    # Need to make sure prune_features is exported or available. 
    # Since we are inside %dopar%, sourced functions need to be carefully handled.
    # But often foreach exports the global environment. We will assume prune_features is available.
    
    # Check for NA columns first as simple prune
    na_cols <- colSums(is.na(train_df)) == nrow(train_df)
    train_df <- train_df[, !na_cols]
    test_df <- test_df[, !na_cols] # assume same cols
    
    # Handle remaining NAs with median imputation
    numeric_cols <- names(train_df)[sapply(train_df, is.numeric)]
    for(col in numeric_cols) {
      if(any(is.na(train_df[[col]]))) {
        med_val <- median(train_df[[col]], na.rm = TRUE)
        train_df[[col]][is.na(train_df[[col]])] <- med_val
        test_df[[col]][is.na(test_df[[col]])] <- med_val
      }
    }
    
    # 3. Rank features (Training set ONLY)
    # Using correlation rank
    valid_feats <- intersect(names(train_df), features) 
    # Recalculate cor
    cor_vals <- sapply(valid_feats, function(f) {
      abs(cor(train_df[[f]], as.numeric(train_df[[target]]), use = "complete.obs"))
    })
    
    r_ranked_feats <- names(sort(cor_vals, decreasing = TRUE))
    
    # 4. Loop over K
    fold_k_results <- list()
    for (k in k_grid) {
      feats_k <- head(r_ranked_feats, k)
      
      # Heuristic Parameters
      cost_fixed <- 1
      gamma_fixed <- 1 / length(feats_k)
      
      # 5. Final Evaluation
      final_metrics <- run_svm_single(train_df, test_df, target, feats_k, cost_fixed, gamma_fixed)
      
      res_row <- data.frame(
        timestamp      = Sys.time(),
        fold           = fold_name,
        target         = target,
        k              = k,
        final_acc      = final_metrics$acc,
        final_f1       = final_metrics$f1,
        final_auc      = final_metrics$auc,
        inner_best_acc = NA, # No tuning
        cost           = cost_fixed,
        gamma          = gamma_fixed,
        feature_n      = length(feats_k),
        feature_set    = paste0("top", k),
        features_used  = paste(feats_k, collapse = ";")
      )
      
      fold_k_results[[length(fold_k_results) + 1]] <- res_row
    }
    bind_rows(fold_k_results)
  }
  
  # Log all results for this target
  write.table(outer_results, file = progress_file, append = TRUE, sep = ",", col.names = FALSE, row.names = FALSE)
  
  return(outer_results)
}

# =====================================================================
# 6. Execution
# =====================================================================
cores <- min(5L, parallel::detectCores() - 2)
cl <- makeCluster(cores)
registerDoParallel(cl)

clusterExport(cl, c("run_svm_single", "features", "k_grid"))

print("Starting Pooled 5-Fold SVM (Heuristic Parameters)...")

results_list <- list()
for (t in targets) {
  print(paste("Running target:", t))
  res <- run_pooled_cv_heuristic(df, t, progress_file = pf_list[[t]])
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
