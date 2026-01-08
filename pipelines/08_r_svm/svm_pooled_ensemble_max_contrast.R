library(tidyverse)
library(e1071)
library(yaml)
library(caret)
library(yardstick)

set.seed(42)

# =====================================================================
# 1. Config & Setup
# =====================================================================
config  <- yaml::read_yaml("config/general.yaml")
out_dir_svm <- file.path("results", "svm")
dir.create(out_dir_svm, recursive = TRUE, showWarnings = FALSE)

progress_file <- file.path(out_dir_svm, "svm_pooled_ensemble_max_contrast.csv")

# =====================================================================
# 2. Data Loading & Filtering (Max Contrast)
# =====================================================================
df <- readRDS(file.path("output", "final_data.rds"))

df_subset <- df %>%
  filter((str_detect(condition, "Low Stress") & str_detect(condition, "Low Cog")) |
         (str_detect(condition, "High Stress") & str_detect(condition, "High Cog"))) %>%
  mutate(
    target = factor(if_else(str_detect(condition, "High Stress"), 1L, 0L), 
                    levels = c(0, 1))
  )

source("utils/r/feature_selection.R")
source("utils/r/prune_feats.R")
features_all <- select_analysis_features(df_subset, suffix = "_precond")

df_subset <- df_subset %>%
  select(participant_id, target, all_of(features_all))

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
# 3. Ensemble Configuration (Best from previous runs)
# =====================================================================
ensemble_configs <- list(
  list(k = 7, kernel = "polynomial", cost = 64, gamma = 0.2, degree = 3),
  list(k = 10, kernel = "polynomial", cost = 4, gamma = 0.2, degree = 2),
  list(k = 15, kernel = "polynomial", cost = 4, gamma = 0.2, degree = 2),
  list(k = 10, kernel = "radial", cost = 1, gamma = 0.01, degree = 3)
)

cor_threshold <- 0.90

# =====================================================================
# 4. Model Function
# =====================================================================
run_svm_adaptive <- function(train, test, feats_all, k=10, cost=1, gamma=0.1, kernel="radial", degree=3) {
  
  pruned_feats <- prune_features(train, target = "target", missing_thresh = 0.30, cor_cutoff = cor_threshold)
  
  cor_vals <- sapply(pruned_feats, function(f) {
    clean_dat <- na.omit(data.frame(x = train[[f]], y = as.numeric(train$target)))
    if(nrow(clean_dat) < 3 || sd(clean_dat$x)==0) return(0)
    abs(cor(clean_dat$x, clean_dat$y))
  })
  
  ranked_feats <- names(sort(cor_vals, decreasing = TRUE))
  selected_feats <- head(ranked_feats, k)
  
  for(f in selected_feats) {
    med_val <- median(train[[f]], na.rm=TRUE)
    if(is.na(med_val)) med_val <- 0
    train[[f]][is.na(train[[f]])] <- med_val
    test[[f]][is.na(test[[f]])]   <- med_val
  }
  
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
  
  pred_prob <- predict(model, test %>% select(all_of(selected_feats)), probability = TRUE)
  prob_vals <- attr(pred_prob, "probabilities")[, "1"]
  
  list(probs = prob_vals, model = model, feats = selected_feats)
}

# =====================================================================
# 5. Ensemble 5-Fold CV
# =====================================================================
folds <- createFolds(df_subset$target, k = 5, list = TRUE, returnTrain = FALSE)
results <- list()

cat("Samples:", nrow(df_subset), "\n")
cat("Ensemble models:", length(ensemble_configs), "\n\n")

for (i in seq_along(folds)) {
  test_idx <- folds[[i]]
  train_df <- df_subset[-test_idx, ]
  test_df  <- df_subset[test_idx, ]
  
  cat("=== Fold", i, "===\n")
  
  # Get predictions from each ensemble member
  all_probs <- matrix(NA, nrow = nrow(test_df), ncol = length(ensemble_configs))
  
  for (j in seq_along(ensemble_configs)) {
    cfg <- ensemble_configs[[j]]
    res <- run_svm_adaptive(train_df, test_df, features_all, 
                           k = cfg$k, cost = cfg$cost, gamma = cfg$gamma,
                           kernel = cfg$kernel, degree = cfg$degree)
    all_probs[, j] <- res$probs
  }
  
  # Average probabilities (soft voting)
  ensemble_probs <- rowMeans(all_probs, na.rm = TRUE)
  ensemble_preds <- factor(ifelse(ensemble_probs > 0.5, 1, 0), levels = c(0, 1))
  
  # Calculate metrics
  acc <- mean(ensemble_preds == test_df$target)
  f1  <- f_meas_vec(truth = test_df$target, estimate = ensemble_preds, event_level = "second")
  auc <- roc_auc_vec(truth = test_df$target, estimate = ensemble_probs, event_level = "second")
  
  cat("  Ensemble AUC:", round(auc, 3), "\n")
  cat("  Individual model AUCs:", paste(round(apply(all_probs, 2, function(p) {
    roc_auc_vec(truth = test_df$target, estimate = p, event_level = "second")
  }), 3), collapse = ", "), "\n\n")
  
  row <- data.frame(
    timestamp = Sys.time(),
    fold = names(folds)[i],
    method = "ensemble_soft_voting",
    n_models = length(ensemble_configs),
    accuracy = acc,
    f1 = f1,
    auc = auc
  )
  
  write.table(row, file = progress_file, append = TRUE, sep = ",", col.names = !file.exists(progress_file), row.names = FALSE)
  results[[i]] <- row
}

# Summary
final_res <- bind_rows(results)
cat("=== Ensemble Final Results ===\n")
cat("Mean AUC:", round(mean(final_res$auc), 3), "\n")
cat("Mean Accuracy:", round(mean(final_res$accuracy), 3), "\n")
cat("Best Fold AUC:", round(max(final_res$auc), 3), "\n")
cat("Worst Fold AUC:", round(min(final_res$auc), 3), "\n")
