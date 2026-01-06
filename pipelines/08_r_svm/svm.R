library(tidyverse)
library(e1071)
library(doParallel)
library(yaml)
library(dplyr)
library(caret)
library(yardstick)  # Added for F1 and AUC

set.seed(42)

source("utils/r/prune_feats.R")
source("utils/r/feature_selection.R")

# =====================================================================
# 1. Load data
# =====================================================================
config  <- yaml::read_yaml("scripts/utils/config.yaml")

out_dir_svm <- file.path(config$paths$results, "svm")
dir.create(out_dir_svm, recursive = TRUE, showWarnings = FALSE)

targets <- c("stress_label", "workload_label")

pf_list <- setNames(
  sapply(targets, function(t) file.path(out_dir_svm, paste0("svm_progress_", t, ".csv"))),
  targets
)

main_header <- c(
  "timestamp", "test_pid", "target", "k",
  "final_acc", "final_f1", "final_auc", "inner_best_acc",
  "cost", "gamma", "feature_n", "feature_set", "features_used"
)

tuning_header <- c(
  "timestamp", "test_pid", "target", "k", "cost", "gamma",
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



# Feature filtering using centralized selection
features <- select_analysis_features(df, suffix = "_precond")

df <- df %>%
  select(participant_id, stress_label, workload_label, all_of(features))

df <- df %>%
  group_by(participant_id) %>%
  mutate(across(all_of(features), ~ scale(.)[,1])) %>%
  ungroup()

participants <- unique(df$participant_id)

# =====================================================================
# 2. Hyperparameter grids
# =====================================================================
cost_grid <- c(2, 4, 8, 16, 32, 64, 128, 256)
gamma_grid <- c(2^-7, 2^-6, 2^-5, 2^-4, 2^-3, 0.03, 0.06, 0.1)
kernel     <- "radial"
# k values to try
k_grid <- c(20, 10, 5)


# =====================================================================
# 4. Core SVM evaluation (updated for F1 and AUC)
# =====================================================================
run_svm <- function(train, test, target, feats, cost, gamma) {
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
  
  # Use the positive class based on the factor level definition
  positive_class <- levels(train[[target]])[2]
  probs <- prob_df[, positive_class]
  
  
  f1  <- f_meas_vec(truth = test[[target]], estimate = preds, event_level = "second")
  auc <- roc_auc_vec(truth = test[[target]], estimate = probs, event_level = "second")
  
  
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
    # 2. Prune features (NZV + missing + correlation)
    # -----------------------------
    pruned_feats <- prune_features(train_df, target = target)
    
    # Drop NAs using training medians
    train_df <- train_df %>% mutate(across(all_of(pruned_feats), ~ replace(., is.na(.), median(., na.rm = TRUE))))
    test_df  <- test_df  %>% mutate(across(all_of(pruned_feats), ~ replace(., is.na(.), median(train_df[[cur_column()]], na.rm = TRUE))))
    
    # -----------------------------

    # Rank pruned features by absolute correlation strength (descending)
    r_ranked_feats <- pruned_feats[
      order(abs(cor(train_df[pruned_feats],  as.numeric(train_df[[target]]), use = "complete.obs")),
            decreasing = TRUE)
    ]
    
    # -----------------------------
    # 4. Test over k-grid
    # -----------------------------
    for (k in k_grid) {
      feats_k <- head(r_ranked_feats, k)
      
      best_inner_acc <- -Inf
      best_params <- NULL
      
      # -----------------------------
      # 5. Nested tuning
      # -----------------------------
      tune_grid <- expand.grid(cost = cost_grid, gamma = gamma_grid)
      
      # Save tuning iteration stats
      tune_results <- foreach(i = 1:nrow(tune_grid), .combine = rbind, 
                              .packages = c("dplyr", "yardstick")) %dopar% {
                                cost  <- tune_grid$cost[i]
                                gamma <- tune_grid$gamma[i]
                                
                                inner_metrics <- map_dfr(
                                  setdiff(participants, test_pid),
                                  function(inner_pid) {
                                    inner_train <- train_df %>% filter(participant_id != inner_pid)
                                    inner_test  <- train_df %>% filter(participant_id == inner_pid)
                                    
                                    res <- run_svm(inner_train, inner_test, target, feats_k, cost, gamma)
                                    tibble(acc = res$acc, auc = res$auc, f1 = res$f1)
                                  }
                                )
                                
                                data.frame(
                                  timestamp   = Sys.time(),
                                  test_pid    = test_pid,
                                  target      = target,
                                  k           = k,
                                  cost        = cost,
                                  gamma       = gamma,
                                  inner_acc   = mean(inner_metrics$acc),
                                  inner_auc   = mean(inner_metrics$auc),
                                  inner_f1    = mean(inner_metrics$f1),
                                  feature_n   = length(feats_k),
                                  feature_set = paste0("top", k),
                                  features_used = paste(feats_k, collapse = ";")
                                )
                                
                              }
      
      best_row <- tune_results[which.max(tune_results$inner_auc), ]
      
      # Append tuning iteration detail log
      write.table(
        tune_results,
        file = paste0(progress_file, "_tuning.csv"),
        append = TRUE,
        sep = ",",
        col.names = FALSE,
        row.names = FALSE
      )
      
      best_inner_acc <- best_row$inner_acc
      best_params <- list(cost = best_row$cost, gamma = best_row$gamma)

      
      # -----------------------------
      # 6. Final LOSO test
      # -----------------------------
      final_metrics <- run_svm(train_df, test_df, target, feats_k, best_params$cost, best_params$gamma)
      
      res_row <- tibble(
        timestamp      = Sys.time(),
        test_pid       = test_pid,
        target         = target,
        k              = k,
        final_acc      = final_metrics$acc,
        final_f1       = final_metrics$f1,
        final_auc      = final_metrics$auc,
        inner_best_acc = best_inner_acc,
        cost           = best_params$cost,
        gamma          = best_params$gamma,
        feature_n      = length(feats_k),
        feature_set    = paste0("top", k),
        features_used  = paste(feats_k, collapse = ";")
      )
      
      # Save to file and append to results
      write.table(res_row, file = progress_file, append = TRUE, sep = ",", col.names = FALSE, row.names = FALSE)
      outer_results <- append(outer_results, list(res_row))
    }
  }
  
  bind_rows(outer_results)
}


# =====================================================================
# 6. Parallel execution per target
# =====================================================================
cores <- min(14L, parallel::detectCores() - 2)  # CHANGED: Use 14 cores (leave 2 for system)
cl <- makeCluster(cores)
registerDoParallel(cl)

clusterExport(cl, c("df", "participants", "targets", "cost_grid", "gamma_grid",
                     "run_svm", "nested_loso", "k_grid", "prune_features"))

clusterEvalQ(cl, {
  library(tidyverse)
  library(e1071)
  library(doParallel)
  library(yaml)
  library(dplyr)
  library(caret)
  library(yardstick)
})

results_nested <- foreach(t = targets, .combine = bind_rows) %dopar% {
  nested_loso(df, t, progress_file = pf_list[[t]])
}

stopCluster(cl)

# =====================================================================
# 7. Summarise results
# =====================================================================
summary_results <- results_nested %>%
  group_by(target) %>%
  summarise(
    loso_acc = mean(final_acc),
    loso_f1  = mean(final_f1),
    loso_auc = mean(final_auc),
    inner_acc = mean(inner_best_acc),
    .groups = "drop"
  )

print(summary_results)
results_nested
