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
progress_file <- file.path(out_dir_svm, "svm_pooled_heuristic_max_contrast.csv")

header <- c(
  "timestamp", "fold", "target", "accuracy", "f1", "auc", 
  "features_used"
)
write.table(t(header), file = progress_file, sep = ",", col.names = FALSE, row.names = FALSE)

# =====================================================================
# 2. Data Loading & Filtering (Max Contrast)
# =====================================================================
df <- readRDS(file.path("output", "final_data.rds"))

# Filter for Low/Low vs High/High
# Control: Low Stress + Low Cog (Target 0)
# Stress: High Stress + High Cog (Target 1)
df_subset <- df %>%
  filter((str_detect(condition, "Low Stress") & str_detect(condition, "Low Cog")) |
         (str_detect(condition, "High Stress") & str_detect(condition, "High Cog"))) %>%
  mutate(
    target = factor(if_else(str_detect(condition, "High Stress"), 1L, 0L), 
                    levels = c(0, 1))
  )

# Use ALL analysis features initially
# We will use the utils/r/feature_selection.R script to get the features
source("utils/r/feature_selection.R")
source("utils/r/prune_feats.R")
# Re-read if source overrides df
features_all <- select_analysis_features(df_subset, suffix = "_precond")

df_subset <- df_subset %>%
  select(participant_id, target, all_of(features_all))

# Subject-level scaling
scale_safe <- function(x) {
  if (all(is.na(x)) || length(x) < 2) return(rep(0, length(x)))
  s <- sd(x, na.rm = TRUE)
  if (is.na(s) || s == 0) return(rep(0, length(x)))
  return(as.vector(scale(x)))
}

df_subset <- df_subset %>%
  group_by(participant_id) %>%
  mutate(across(all_of(features_all), scale_safe)) %>%
  ungroup()

# =====================================================================
# 3. Hyperparameter Grids
# =====================================================================
cost_grid <- c(1, 4, 16, 64)
gamma_grid <- c(0.01, 0.05, 0.1, 0.2)
kernel_grid <- c("linear", "radial", "polynomial")
degree_grid <- c(2, 3)  # for polynomial only
k_grid <- c(3, 7, 10, 15)
cor_threshold <- 0.90  # Relaxed from 0.75

# =====================================================================
# 4. Model Function with Hyperparameter Tuning
# =====================================================================
run_svm_adaptive <- function(train, test, feats_all, k=10, cost=1, gamma=0.1, kernel="radial", degree=3) {
  
  # 1. Prune features (remove high-missing, NZV, highly-correlated)
  pruned_feats <- prune_features(train, target = "target", missing_thresh = 0.30, cor_cutoff = cor_threshold)
  
  # 2. Feature Selection on TRAINING set - Rank by correlation
  cor_vals <- sapply(pruned_feats, function(f) {
    # handling potential NAs in correlation
    clean_dat <- na.omit(data.frame(x = train[[f]], y = as.numeric(train$target)))
    # Need enough points and variance
    if(nrow(clean_dat) < 3 || sd(clean_dat$x)==0) return(0)
    abs(cor(clean_dat$x, clean_dat$y))
  })
  
  ranked_feats <- names(sort(cor_vals, decreasing = TRUE))
  selected_feats <- head(ranked_feats, k)
  
  # Impute NAs with median (heuristic) for selected features
  for(f in selected_feats) {
    med_val <- median(train[[f]], na.rm=TRUE)
    if(is.na(med_val)) med_val <- 0
    train[[f]][is.na(train[[f]])] <- med_val
    test[[f]][is.na(test[[f]])]   <- med_val
  }
  
  # 3. Train Model (Multi-kernel, Tuned)
  if (kernel == "linear") {
    model <- svm(
      reformulate(selected_feats, "target"),
      data = train,
      kernel = "linear",
      cost = cost,
      probability = TRUE,
      scale = FALSE
    )
  } else if (kernel == "polynomial") {
    model <- svm(
      reformulate(selected_feats, "target"),
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
      reformulate(selected_feats, "target"),
      data = train,
      kernel = "radial",
      cost = cost,
      gamma = gamma,
      probability = TRUE,
      scale = FALSE
    )
  }
  
  preds <- predict(model, test %>% select(all_of(selected_feats)))
  acc <- mean(preds == pull(test, "target"))
  
  pred_prob <- predict(model, test %>% select(all_of(selected_feats)), probability = TRUE)
  prob_vals <- attr(pred_prob, "probabilities")[, "1"]
  
  auc <- roc_auc_vec(truth = test$target, estimate = prob_vals, event_level = "second")
  f1  <- f_meas_vec(truth = test$target, estimate = preds, event_level = "second")
  
  list(acc = acc, f1 = f1, auc = auc, feats = paste(selected_feats, collapse=";"))
}

# =====================================================================
# 5. Nested 5-Fold CV with Hyperparameter Tuning
# =====================================================================
# Stratified by target
folds <- createFolds(df_subset$target, k = 5, list = TRUE, returnTrain = FALSE)
results <- list()

print(paste("Samples:", nrow(df_subset)))
print("Class balance per fold:")
for (i in seq_along(folds)) {
  print(paste("Fold", i, ":", table(df_subset$target[folds[[i]]])))
}

for (i in seq_along(folds)) {
  test_idx <- folds[[i]]
  train_df <- df_subset[-test_idx, ]
  test_df  <- df_subset[test_idx, ]
  
  cat("\n=== Outer Fold", i, "===", "\n")
  
  # Loop over k values
  for (k in k_grid) {
    cat("Testing k =", k, "features...\n")
    
    # Inner CV for hyperparameter tuning
    inner_folds <- createFolds(train_df$target, k = 5, list = TRUE, returnTrain = FALSE)
    
    best_auc <- 0
    best_params <- list(cost = 1, gamma = 0.1, kernel = "radial", degree = 3)
    
    # Loop over kernels
    for (kern in kernel_grid) {
      if (kern == "linear") {
        # Linear doesn't use gamma
        for (cost_val in cost_grid) {
          inner_aucs <- numeric(5)
          for (inner_i in 1:5) {
            inner_test_idx <- inner_folds[[inner_i]]
            inner_train <- train_df[-inner_test_idx, ]
            inner_valid <- train_df[inner_test_idx, ]
            
            res <- run_svm_adaptive(inner_train, inner_valid, features_all, k=k, 
                                    cost=cost_val, gamma=0.1, kernel="linear")
            inner_aucs[inner_i] <- res$auc
          }
          
          mean_inner_auc <- mean(inner_aucs, na.rm=TRUE)
          if (mean_inner_auc > best_auc) {
            best_auc <- mean_inner_auc
            best_params <- list(cost = cost_val, gamma = 0.1, kernel = "linear", degree = 3)
          }
        }
      } else if (kern == "polynomial") {
        # Polynomial uses cost, gamma, and degree
        for (cost_val in cost_grid) {
          for (gamma_val in gamma_grid) {
            for (deg_val in degree_grid) {
              inner_aucs <- numeric(5)
              for (inner_i in 1:5) {
                inner_test_idx <- inner_folds[[inner_i]]
                inner_train <- train_df[-inner_test_idx, ]
                inner_valid <- train_df[inner_test_idx, ]
                
                res <- run_svm_adaptive(inner_train, inner_valid, features_all, k=k, 
                                        cost=cost_val, gamma=gamma_val, kernel="polynomial", degree=deg_val)
                inner_aucs[inner_i] <- res$auc
              }
              
              mean_inner_auc <- mean(inner_aucs, na.rm=TRUE)
              if (mean_inner_auc > best_auc) {
                best_auc <- mean_inner_auc
                best_params <- list(cost = cost_val, gamma = gamma_val, kernel = "polynomial", degree = deg_val)
              }
            }
          }
        }
      } else {
        # Radial uses cost and gamma
        for (cost_val in cost_grid) {
          for (gamma_val in gamma_grid) {
            inner_aucs <- numeric(5)
            for (inner_i in 1:5) {
              inner_test_idx <- inner_folds[[inner_i]]
              inner_train <- train_df[-inner_test_idx, ]
              inner_valid <- train_df[inner_test_idx, ]
              
              res <- run_svm_adaptive(inner_train, inner_valid, features_all, k=k, 
                                      cost=cost_val, gamma=gamma_val, kernel="radial")
              inner_aucs[inner_i] <- res$auc
            }
            
            mean_inner_auc <- mean(inner_aucs, na.rm=TRUE)
            if (mean_inner_auc > best_auc) {
              best_auc <- mean_inner_auc
              best_params <- list(cost = cost_val, gamma = gamma_val, kernel = "radial", degree = 3)
            }
          }
        }
      }
    }
    
    cat("  Best params: kernel=", best_params$kernel, ", cost=", best_params$cost, 
        ", gamma=", best_params$gamma, ", degree=", best_params$degree,
        ", inner AUC=", round(best_auc, 3), "\n")
    
    # Evaluate on outer test set with best params
    res <- run_svm_adaptive(train_df, test_df, features_all, k=k, 
                           cost=best_params$cost, gamma=best_params$gamma,
                           kernel=best_params$kernel, degree=best_params$degree)
    
    row <- data.frame(
      timestamp = Sys.time(),
      fold = names(folds)[i],
      k = k,
      kernel = best_params$kernel,
      cost = best_params$cost,
      gamma = best_params$gamma,
      degree = best_params$degree,
      target = "stress_mixed_max_contrast",
      accuracy = res$acc,
      f1 = res$f1,
      auc = res$auc,
      features_used = res$feats
    )
    
    write.table(row, file = progress_file, append = TRUE, sep = ",", col.names = FALSE, row.names = FALSE)
    results[[length(results) + 1]] <- row
  }
}

# Summary
final_res <- bind_rows(results)
cat("\n=== Final Results ===", "\n")
print(final_res %>% 
  group_by(k) %>% 
  summarise(mean_acc = mean(accuracy), mean_auc = mean(auc), .groups = "drop") %>%
  arrange(desc(mean_auc)))

cat("\nBest overall AUC:", max(final_res$auc), "\n")
cat("Mean AUC across all folds:", mean(final_res$auc), "\n")

cat("\n=== Kernel Performance ===", "\n")
print(final_res %>% 
  group_by(kernel) %>% 
  summarise(mean_auc = mean(auc), count = n(), .groups = "drop") %>%
  arrange(desc(mean_auc)))

cat("\n=== Best Configuration by Fold ===", "\n")
best_per_fold <- final_res %>%
  group_by(fold) %>%
  filter(auc == max(auc)) %>%
  select(fold, k, kernel, cost, gamma, degree, auc)
print(best_per_fold)