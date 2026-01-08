library(tidyverse)
library(e1071)
library(yaml)
library(caret)
library(yardstick)

set.seed(42)

source("utils/r/prune_feats.R")
source("utils/r/feature_selection.R")

cat("=== QUICK TEST: Optimized LOSO SVM (3 participants) ===\n\n")

# =====================================================================
# 1. Load data
# =====================================================================
config  <- yaml::read_yaml("scripts/utils/config.yaml")

df <- readRDS(file.path(config$paths$output, "final_data.rds"))

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

df <- df %>%
  group_by(participant_id) %>%
  mutate(across(all_of(features), ~ scale(.)[,1])) %>%
  ungroup()

all_participants <- unique(df$participant_id)

# Test on first 3 participants only
participants <- head(all_participants, 3)

cat("Testing on participants:", paste(participants, collapse = ", "), "\n")
cat("Total observations:", nrow(df %>% filter(participant_id %in% participants)), "\n\n")

# =====================================================================
# 2. Hyperparameter Grids (same as optimized)
# =====================================================================
cost_grid <- c(1, 4, 16, 64)
gamma_grid <- c(0.01, 0.05, 0.1, 0.2)
kernel_grid <- c("linear", "radial", "polynomial")
degree_grid <- c(2, 3)
k_grid <- c(3, 7, 10, 15)
cor_threshold <- 0.90

# =====================================================================
# 3. Multi-Kernel SVM Function
# =====================================================================
run_svm_multikernel <- function(train, test, target, feats, cost, gamma, kernel = "radial", degree = 3) {
  
  if (kernel == "linear") {
    model <- svm(
      reformulate(feats, target),
      data = train,
      kernel = "linear",
      cost = cost,
      probability = TRUE,
      scale = FALSE
    )
  } else if (kernel == "polynomial") {
    model <- svm(
      reformulate(feats, target),
      data = train,
      kernel = "polynomial",
      cost = cost,
      degree = degree,
      gamma = gamma,
      probability = TRUE,
      scale = FALSE
    )
  } else {
    model <- svm(
      reformulate(feats, target),
      data = train,
      kernel = "radial",
      cost = cost,
      gamma = gamma,
      probability = TRUE,
      scale = FALSE
    )
  }
  
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
# 4. Quick Test LOSO
# =====================================================================
quick_test <- function(df, target) {
  results <- list()
  
  for (test_pid in participants) {
    cat(sprintf("Testing PID=%s, Target=%s\n", test_pid, target))
    
    train_df <- df %>% filter(participant_id != test_pid)
    test_df  <- df %>% filter(participant_id == test_pid)
    
    pruned_feats <- prune_features(train_df, target = target, 
                                   missing_thresh = 0.30, 
                                   cor_cutoff = cor_threshold)
    
    train_df <- train_df %>% mutate(across(all_of(pruned_feats), ~ replace(., is.na(.), median(., na.rm = TRUE))))
    test_df  <- test_df  %>% mutate(across(all_of(pruned_feats), ~ replace(., is.na(.), median(train_df[[cur_column()]], na.rm = TRUE))))
    
    r_ranked_feats <- pruned_feats[
      order(abs(cor(train_df[pruned_feats], as.numeric(train_df[[target]]), use = "complete.obs")),
            decreasing = TRUE)
    ]
    
    # Test only k=10 to save time
    k <- 10
    feats_k <- head(r_ranked_feats, k)
    
    best_inner_auc <- -Inf
    best_params <- NULL
    
    inner_pids <- setdiff(participants, test_pid)
    
    # Quick grid search (reduced)
    for (kern in c("radial", "polynomial")) {
      if (kern == "polynomial") {
        for (cost_val in c(4, 16)) {
          for (gamma_val in c(0.1, 0.2)) {
            for (deg_val in c(2, 3)) {
              inner_metrics <- map_dfr(
                inner_pids,
                function(inner_pid) {
                  inner_train <- train_df %>% filter(participant_id != inner_pid)
                  inner_test  <- train_df %>% filter(participant_id == inner_pid)
                  
                  res <- run_svm_multikernel(inner_train, inner_test, target, feats_k, 
                                            cost_val, gamma_val, kernel = "polynomial", degree = deg_val)
                  tibble(acc = res$acc, auc = res$auc, f1 = res$f1)
                }
              )
              
              mean_inner_auc <- mean(inner_metrics$auc, na.rm = TRUE)
              
              if (mean_inner_auc > best_inner_auc) {
                best_inner_auc <- mean_inner_auc
                best_params <- list(cost = cost_val, gamma = gamma_val, kernel = "polynomial", degree = deg_val)
              }
            }
          }
        }
      } else {
        for (cost_val in c(1, 4)) {
          for (gamma_val in c(0.01, 0.05)) {
            inner_metrics <- map_dfr(
              inner_pids,
              function(inner_pid) {
                inner_train <- train_df %>% filter(participant_id != inner_pid)
                inner_test  <- train_df %>% filter(participant_id == inner_pid)
                
                res <- run_svm_multikernel(inner_train, inner_test, target, feats_k, 
                                          cost_val, gamma_val, kernel = "radial")
                tibble(acc = res$acc, auc = res$auc, f1 = res$f1)
              }
            )
            
            mean_inner_auc <- mean(inner_metrics$auc, na.rm = TRUE)
            
            if (mean_inner_auc > best_inner_auc) {
              best_inner_auc <- mean_inner_auc
              best_params <- list(cost = cost_val, gamma = gamma_val, kernel = "radial", degree = 3)
            }
          }
        }
      }
    }
    
    cat(sprintf("  Best: kernel=%s, cost=%.1f, gamma=%.2f, inner AUC=%.3f\n",
                best_params$kernel, best_params$cost, best_params$gamma, best_inner_auc))
    
    final_metrics <- run_svm_multikernel(train_df, test_df, target, feats_k, 
                                        best_params$cost, best_params$gamma,
                                        best_params$kernel, best_params$degree)
    
    cat(sprintf("  Final: AUC=%.3f, Acc=%.3f\n\n", final_metrics$auc, final_metrics$acc))
    
    results[[length(results) + 1]] <- tibble(
      test_pid = test_pid,
      target = target,
      kernel = best_params$kernel,
      auc = final_metrics$auc,
      acc = final_metrics$acc
    )
  }
  
  bind_rows(results)
}

# =====================================================================
# 5. Run quick test
# =====================================================================
cat("\n=== Testing Stress Label ===\n")
stress_results <- quick_test(df, "stress_label")

cat("\n=== Testing Workload Label ===\n")
workload_results <- quick_test(df, "workload_label")

# =====================================================================
# 6. Summary
# =====================================================================
cat("\n=== SUMMARY ===\n")
cat("\nStress Label:\n")
print(stress_results)
cat(sprintf("  Mean AUC: %.3f\n", mean(stress_results$auc)))

cat("\nWorkload Label:\n")
print(workload_results)
cat(sprintf("  Mean AUC: %.3f\n", mean(workload_results$auc)))

cat("\nQuick test complete! If results look good, run full svm_loso_optimized.R\n")
