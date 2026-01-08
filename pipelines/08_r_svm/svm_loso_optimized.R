library(tidyverse)
library(e1071)
library(doParallel)
library(yaml)
library(dplyr)
library(caret)
library(yardstick)

set.seed(42)

source("utils/r/prune_feats.R")
source("utils/r/feature_selection.R")

cat("=== OPTIMIZED LOSO SVM ===\n")
cat("Optimizations applied:\n")
cat("  1. Multi-kernel (linear, radial, polynomial)\n")
cat("  2. Reduced hyperparameter grid (cost, gamma)\n")
cat("  3. Relaxed correlation threshold (0.90 vs 0.75)\n")
cat("  4. Refined k grid [3, 7, 10, 15]\n")
cat("  5. Progress tracking and ETA\n\n")

# =====================================================================
# 1. Load data
# =====================================================================
config  <- yaml::read_yaml("scripts/utils/config.yaml")

out_dir_svm <- file.path(config$paths$results, "svm")
dir.create(out_dir_svm, recursive = TRUE, showWarnings = FALSE)

targets <- c("stress_label", "workload_label")

pf_list <- setNames(
  sapply(targets, function(t) file.path(out_dir_svm, paste0("svm_loso_optimized_", t, ".csv"))),
  targets
)

main_header <- c(
  "timestamp", "test_pid", "target", "k", "kernel", "degree",
  "final_acc", "final_f1", "final_auc", "inner_best_auc",
  "cost", "gamma", "feature_n", "feature_set", "features_used"
)

tuning_header <- c(
  "timestamp", "test_pid", "target", "k", "kernel", "degree", "cost", "gamma",
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

participants <- unique(df$participant_id)

cat("Total participants:", length(participants), "\n")
cat("Total observations:", nrow(df), "\n")
cat("Total features:", length(features), "\n\n")

# =====================================================================
# 2. Optimized Hyperparameter Grids
# =====================================================================
# Reduced from 8×8=64 to 4×4=16 combinations per kernel (-75% compute)
cost_grid <- c(1, 4, 16, 64)
gamma_grid <- c(0.01, 0.05, 0.1, 0.2)
kernel_grid <- c("linear", "radial", "polynomial")
degree_grid <- c(2, 3)  # for polynomial only
k_grid <- c(3, 7, 10, 15)  # Refined from [20, 10, 5]
cor_threshold <- 0.90  # Relaxed from 0.75

cat("Hyperparameter grid sizes:\n")
cat("  Linear kernel:     4 cost × 4 k = 16 combos\n")
cat("  Radial kernel:     4 cost × 4 gamma × 4 k = 64 combos\n")
cat("  Polynomial kernel: 4 cost × 4 gamma × 2 degree × 4 k = 128 combos\n")
cat("  Total per fold:    208 combinations\n")
cat("  (vs 192 in original, but with multi-kernel capability)\n\n")

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
    # radial (RBF)
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
# 4. Optimized Nested LOSO
# =====================================================================
nested_loso_optimized <- function(df, target, progress_file) {
  outer_results <- list()
  
  start_time <- Sys.time()
  total_pids <- length(participants)
  
  for (pid_idx in seq_along(participants)) {
    test_pid <- participants[pid_idx]
    
    cat(sprintf("[%s] Fold %d/%d (PID=%s) - Target: %s\n", 
                Sys.time(), pid_idx, total_pids, test_pid, target))
    
    # Train/Test split
    train_df <- df %>% filter(participant_id != test_pid)
    test_df  <- df %>% filter(participant_id == test_pid)
    
    # Prune features with RELAXED correlation threshold
    pruned_feats <- prune_features(train_df, target = target, 
                                   missing_thresh = 0.30, 
                                   cor_cutoff = cor_threshold)
    
    # Impute NAs
    train_df <- train_df %>% mutate(across(all_of(pruned_feats), ~ replace(., is.na(.), median(., na.rm = TRUE))))
    test_df  <- test_df  %>% mutate(across(all_of(pruned_feats), ~ replace(., is.na(.), median(train_df[[cur_column()]], na.rm = TRUE))))
    
    # Rank features by correlation
    r_ranked_feats <- pruned_feats[
      order(abs(cor(train_df[pruned_feats], as.numeric(train_df[[target]]), use = "complete.obs")),
            decreasing = TRUE)
    ]
    
    # Test over k-grid
    for (k in k_grid) {
      feats_k <- head(r_ranked_feats, k)
      
      best_inner_auc <- -Inf
      best_params <- NULL
      
      # Inner LOSO for hyperparameter tuning
      inner_pids <- setdiff(participants, test_pid)
      
      # Loop over kernels
      for (kern in kernel_grid) {
        
        if (kern == "linear") {
          # Linear: only cost matters
          for (cost_val in cost_grid) {
            inner_metrics <- map_dfr(
              inner_pids,
              function(inner_pid) {
                inner_train <- train_df %>% filter(participant_id != inner_pid)
                inner_test  <- train_df %>% filter(participant_id == inner_pid)
                
                res <- run_svm_multikernel(inner_train, inner_test, target, feats_k, 
                                          cost_val, gamma = 0.01, kernel = "linear")
                tibble(acc = res$acc, auc = res$auc, f1 = res$f1)
              }
            )
            
            mean_inner_auc <- mean(inner_metrics$auc, na.rm = TRUE)
            
            # Log tuning
            tune_row <- data.frame(
              timestamp   = Sys.time(),
              test_pid    = test_pid,
              target      = target,
              k           = k,
              kernel      = "linear",
              degree      = NA,
              cost        = cost_val,
              gamma       = NA,
              inner_acc   = mean(inner_metrics$acc, na.rm = TRUE),
              inner_auc   = mean_inner_auc,
              inner_f1    = mean(inner_metrics$f1, na.rm = TRUE),
              feature_n   = length(feats_k),
              feature_set = paste0("top", k),
              features_used = paste(feats_k, collapse = ";")
            )
            
            write.table(tune_row, file = paste0(progress_file, "_tuning.csv"),
                       append = TRUE, sep = ",", col.names = FALSE, row.names = FALSE)
            
            if (mean_inner_auc > best_inner_auc) {
              best_inner_auc <- mean_inner_auc
              best_params <- list(cost = cost_val, gamma = 0.01, kernel = "linear", degree = 3)
            }
          }
          
        } else if (kern == "polynomial") {
          # Polynomial: cost, gamma, degree
          for (cost_val in cost_grid) {
            for (gamma_val in gamma_grid) {
              for (deg_val in degree_grid) {
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
                
                # Log tuning
                tune_row <- data.frame(
                  timestamp   = Sys.time(),
                  test_pid    = test_pid,
                  target      = target,
                  k           = k,
                  kernel      = "polynomial",
                  degree      = deg_val,
                  cost        = cost_val,
                  gamma       = gamma_val,
                  inner_acc   = mean(inner_metrics$acc, na.rm = TRUE),
                  inner_auc   = mean_inner_auc,
                  inner_f1    = mean(inner_metrics$f1, na.rm = TRUE),
                  feature_n   = length(feats_k),
                  feature_set = paste0("top", k),
                  features_used = paste(feats_k, collapse = ";")
                )
                
                write.table(tune_row, file = paste0(progress_file, "_tuning.csv"),
                           append = TRUE, sep = ",", col.names = FALSE, row.names = FALSE)
                
                if (mean_inner_auc > best_inner_auc) {
                  best_inner_auc <- mean_inner_auc
                  best_params <- list(cost = cost_val, gamma = gamma_val, kernel = "polynomial", degree = deg_val)
                }
              }
            }
          }
          
        } else {
          # Radial: cost, gamma
          for (cost_val in cost_grid) {
            for (gamma_val in gamma_grid) {
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
              
              # Log tuning
              tune_row <- data.frame(
                timestamp   = Sys.time(),
                test_pid    = test_pid,
                target      = target,
                k           = k,
                kernel      = "radial",
                degree      = NA,
                cost        = cost_val,
                gamma       = gamma_val,
                inner_acc   = mean(inner_metrics$acc, na.rm = TRUE),
                inner_auc   = mean_inner_auc,
                inner_f1    = mean(inner_metrics$f1, na.rm = TRUE),
                feature_n   = length(feats_k),
                feature_set = paste0("top", k),
                features_used = paste(feats_k, collapse = ";")
              )
              
              write.table(tune_row, file = paste0(progress_file, "_tuning.csv"),
                         append = TRUE, sep = ",", col.names = FALSE, row.names = FALSE)
              
              if (mean_inner_auc > best_inner_auc) {
                best_inner_auc <- mean_inner_auc
                best_params <- list(cost = cost_val, gamma = gamma_val, kernel = "radial", degree = 3)
              }
            }
          }
        }
      }
      
      cat(sprintf("  k=%d: Best kernel=%s, cost=%.2f, gamma=%.3f, inner AUC=%.3f\n",
                  k, best_params$kernel, best_params$cost, best_params$gamma, best_inner_auc))
      
      # Final LOSO test with best params
      final_metrics <- run_svm_multikernel(train_df, test_df, target, feats_k, 
                                          best_params$cost, best_params$gamma,
                                          best_params$kernel, best_params$degree)
      
      res_row <- tibble(
        timestamp      = Sys.time(),
        test_pid       = test_pid,
        target         = target,
        k              = k,
        kernel         = best_params$kernel,
        degree         = best_params$degree,
        final_acc      = final_metrics$acc,
        final_f1       = final_metrics$f1,
        final_auc      = final_metrics$auc,
        inner_best_auc = best_inner_auc,
        cost           = best_params$cost,
        gamma          = best_params$gamma,
        feature_n      = length(feats_k),
        feature_set    = paste0("top", k),
        features_used  = paste(feats_k, collapse = ";")
      )
      
      write.table(res_row, file = progress_file, append = TRUE, sep = ",", 
                 col.names = FALSE, row.names = FALSE)
      outer_results <- append(outer_results, list(res_row))
    }
    
    # Progress tracking
    elapsed <- as.numeric(difftime(Sys.time(), start_time, units = "secs"))
    avg_time_per_pid <- elapsed / pid_idx
    remaining_pids <- total_pids - pid_idx
    eta_secs <- avg_time_per_pid * remaining_pids
    
    cat(sprintf("  Progress: %d/%d (%.1f%%), Elapsed: %.1f min, ETA: %.1f min\n\n",
                pid_idx, total_pids, 100 * pid_idx / total_pids,
                elapsed / 60, eta_secs / 60))
  }
  
  bind_rows(outer_results)
}

# =====================================================================
# 5. Run for both targets (sequential to avoid memory issues)
# =====================================================================
all_results <- list()

for (t in targets) {
  cat(sprintf("\n========================================\n"))
  cat(sprintf("Starting LOSO for target: %s\n", t))
  cat(sprintf("========================================\n\n"))
  
  results <- nested_loso_optimized(df, t, progress_file = pf_list[[t]])
  all_results[[t]] <- results
}

# =====================================================================
# 6. Summarize results
# =====================================================================
cat("\n========================================\n")
cat("FINAL RESULTS SUMMARY\n")
cat("========================================\n\n")

for (t in targets) {
  cat(sprintf("Target: %s\n", t))
  cat(sprintf("%-50s\n", paste(rep("-", 50), collapse = "")))
  
  res <- all_results[[t]]
  
  # Overall summary
  overall <- res %>%
    summarise(
      mean_acc = mean(final_acc, na.rm = TRUE),
      mean_f1  = mean(final_f1, na.rm = TRUE),
      mean_auc = mean(final_auc, na.rm = TRUE),
      best_auc = max(final_auc, na.rm = TRUE),
      worst_auc = min(final_auc, na.rm = TRUE)
    )
  
  cat(sprintf("  Mean AUC:     %.3f\n", overall$mean_auc))
  cat(sprintf("  Mean Accuracy: %.3f\n", overall$mean_acc))
  cat(sprintf("  Mean F1:      %.3f\n", overall$mean_f1))
  cat(sprintf("  Best fold AUC: %.3f\n", overall$best_auc))
  cat(sprintf("  Worst fold AUC: %.3f\n\n", overall$worst_auc))
  
  # By k value
  by_k <- res %>%
    group_by(k) %>%
    summarise(
      mean_auc = mean(final_auc, na.rm = TRUE),
      mean_acc = mean(final_acc, na.rm = TRUE),
      .groups = "drop"
    ) %>%
    arrange(desc(mean_auc))
  
  cat("  Performance by k:\n")
  print(by_k)
  cat("\n")
  
  # By kernel
  by_kernel <- res %>%
    group_by(kernel) %>%
    summarise(
      count = n(),
      mean_auc = mean(final_auc, na.rm = TRUE),
      .groups = "drop"
    ) %>%
    arrange(desc(mean_auc))
  
  cat("  Performance by kernel:\n")
  print(by_kernel)
  cat("\n")
}

cat("Results saved to:\n")
for (t in targets) {
  cat(sprintf("  %s\n", pf_list[[t]]))
}

cat("\nOptimization complete!\n")
