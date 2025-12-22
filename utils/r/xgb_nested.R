# XGBoost LOSO Classification – Stress & Workload
# ------------------------------------------------
# One run per target. Full leakage-free LOSO.
# Pruning and tuning done inside each fold.

library(dplyr)
library(purrr)
library(readr)
library(glue)
library(tidyr)
library(magrittr)
library(xgboost)
library(yardstick)
library(stringr)
library(yaml)
library(caret)
library(ParBayesianOptimization)
library(rvif)
library(tibble)
library(doParallel)
library(glue) 


# -----------------------------------------------
# Source external pruning and tuning functions
# -----------------------------------------------
source("scripts/prune_feats.R")   
source("scripts/bayes_opt.R")     

set.seed(42)

# -----------------------------------------------
# Tuning bounds (log scale where appropriate)
# -----------------------------------------------
dbounds <- list(
  eta               = c(log(0.03),  log(0.5)),
  max_depthL        = c(4L,         10L),
  subsample         = c(0.60,       0.95),
  colsample_bytree  = c(0.60,       0.95),
  min_child_weightL = c(1L,         8L),
  lambda            = c(log(0.1),   log(10)),
  alpha             = c(log(0.1),   log(5))
)


# -----------------------------------------------
# Config + paths
# -----------------------------------------------
config <- yaml::read_yaml("scripts/utils/config.yaml")

results_dir <- file.path(config$paths$xgb_results, "classification")
bayes_dir       <- file.path(config$paths$xgb_bayes,"classification")
feats_dir<- file.path(config$paths$init_pruned,  "classification")

dir.create(results_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(bayes_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(feats_dir, recursive = TRUE, showWarnings = FALSE)

# -----------------------------------------------
# Drop pattern for pruning
# (must be used explicitly inside prune_features)
# -----------------------------------------------
drop_pattern <- paste0(
  "response|",
  "eeg_o|",
  "(_min_|_max_)|",
  "head|",
  "totalscrs|",
  "_glob|",
  "_raw|",
  "_delta"
)


# ---------------------------------------------------------
# Load data
# ---------------------------------------------------------
df <- readRDS(file.path(config$paths$output, "final_data.rds")) %>%
  mutate(
    # Binary labels used as classification targets
    stress_label   = if_else(str_detect(condition, "High Stress"), 1L, 0L),
    workload_label = if_else(str_detect(condition, "High Cog"),    1L, 0L),
    
    # Subjective ratings (used only for pruning features)
    stress_rating   = stress,
    workload_rating = workload
  )

features <- names(df)[str_detect(names(df), "_precond$")]
features <- features[!str_detect(features, drop_pattern)]

df <- df %>%
  select(
    participant_id,
    stress_rating, workload_rating,   
    stress_label, workload_label,   
    all_of(features)
  )

df <- df %>%
  group_by(participant_id) %>%
  mutate(across(all_of(features), ~ scale(.)[,1])) %>%
  ungroup()

# Participant list and modelling targets
participants <- unique(df$participant_id)
targets <- c( "stress_label", "workload_label")

# -----------------------------------------------
# Helper
# -----------------------------------------------
make_pid_folds <- function(train_df, k_max = 3L, seed = 42L) {
  set.seed(seed)  # ensures reproducibility
  
  pids <- unique(train_df$participant_id)
  K <- max(2L, min(k_max, length(pids)))
  
  # Check class balance per PID
  pid_classes <- train_df %>%
    group_by(participant_id) %>%
    summarise(
      stress = if ("stress_label" %in% names(.)) mean(stress_label),
      workload = if ("workload_label" %in% names(.)) mean(workload_label),
      .groups = "drop"
    )
  
  # Shuffle by target mean → spreads signal more evenly across folds
  pid_order <- pid_classes %>%
    arrange(desc(stress), desc(workload)) %>%  # Replace target as appropriate
    pull(participant_id)
  
  buckets <- split(pid_order, rep(1:K, length.out = length(pid_order)))
  
  lapply(buckets, function(ps) which(train_df$participant_id %in% ps))
}



run_single_fold <- function(train_df, test_df, target, pid) {
  
  # Assign regression labels (kept from original)
  train_df <- train_df %>%
    mutate(subj_label = if (target == "stress_label") stress_rating else workload_rating)
  
  # 1. Feature pruning (unchanged)
  feats <- prune_features(
    df = train_df,
    missing_thresh = 0.30,
    cor_cutoff = 0.99
  )
  
  # 2. Nested tuning: now matches SVM logic
  inner_pids <- setdiff(participants, pid)
  
  tune_results <- map_dfr(inner_pids, function(inner_pid) {
    
    inner_train <- train_df %>% filter(participant_id != inner_pid)
    inner_val   <- train_df %>% filter(participant_id == inner_pid)
    
    tuned <- run_xgb_tuning(
      df              = inner_train,
      feats           = feats,
      target          = target,
      bounds          = dbounds,
      iters           = 32,
      bayes_save_path = bayes_dir
    )
    
    best <- getBestPars(tuned)
    
    params <- list(
      objective        = "binary:logistic",
      eval_metric      = "logloss",
      tree_method      = "hist",
      eta              = exp(best$eta),
      max_depth        = best$max_depthL,
      subsample        = best$subsample,
      colsample_bytree = best$colsample_bytree,
      min_child_weight = best$min_child_weightL,
      lambda           = exp(best$lambda),
      alpha            = exp(best$alpha)
    )
    
    # Early stopping using PID folds from inner_train
    fold_idx <- make_pid_folds(inner_train, k = 3)
    
    cv <- xgb.cv(
      params                = params,
      data                  = xgb.DMatrix(as.matrix(inner_train[, feats, drop = FALSE]), label = inner_train[[target]]),
      nrounds               = 4000L,
      folds                 = fold_idx,
      early_stopping_rounds = 50L,
      verbose               = FALSE
    )
    
    nrounds <- cv$best_iteration
    
    model <- xgb.train(
      params  = params,
      data    = xgb.DMatrix(as.matrix(inner_train[, feats, drop = FALSE]), label = inner_train[[target]]),
      nrounds = nrounds,
      verbose = 0
    )
    
    prob <- predict(model, xgb.DMatrix(as.matrix(inner_val[, feats, drop = FALSE])))
    class <- as.integer(prob > 0.5)
    
    tibble(
      pid = inner_pid,
      auc = roc_auc_vec(
        truth = factor(inner_val[[target]], levels = c(0, 1)),
        estimate = prob,
        event_level = "second"
      ),
      acc = mean(class == inner_val[[target]]),
      f1  = f_meas_vec(
        truth = factor(inner_val[[target]], levels = c(0, 1)),
        estimate = factor(class, levels = c(0, 1)),
        event_level = "second"
      ),
      eta = params$eta,
      max_depth = params$max_depth,
      subsample = params$subsample,
      colsample_bytree = params$colsample_bytree,
      min_child_weight = params$min_child_weight,
      lambda = params$lambda,
      alpha = params$alpha,
      nrounds = nrounds
    )
    
  })
  
  # 3. Extract best settings from inner loop
  best_row <- tune_results[which.max(tune_results$auc), ]
  best_params <- as.list(best_row[c("eta","max_depth","subsample","colsample_bytree","min_child_weight","lambda","alpha")])
  best_params$nrounds <- best_row$nrounds
  
  # 4. Final model using full train_df, tested on test_df
  final_model <- xgb.train(
    params  = best_params,
    data    = xgb.DMatrix(as.matrix(train_df[, feats, drop = FALSE]), label = train_df[[target]]),
    nrounds = best_params$nrounds,
    verbose = 0
  )
  
  prob_test <- predict(final_model, xgb.DMatrix(as.matrix(test_df[, feats, drop = FALSE])))
  class_test <- as.integer(prob_test > 0.5)
  
  tibble(
    participant_id = unique(test_df$participant_id),
    actual         = test_df[[target]],
    prob           = prob_test,
    class          = class_test,
    final_auc      = roc_auc_vec(truth = test_df[[target]], estimate = prob_test, event_level = "second"),
    final_acc      = mean(class_test == test_df[[target]]),
    final_f1       = f_meas_vec(truth = test_df[[target]], estimate = class_test, event_level = "second"),
    inner_best_auc = max(tune_results$auc, na.rm = TRUE),
    eta            = best_params$eta,
    max_depth      = best_params$max_depth,
    subsample      = best_params$subsample,
    colsample_bytree = best_params$colsample_bytree,
    min_child_weight = best_params$min_child_weight,
    lambda         = best_params$lambda,
    alpha          = best_params$alpha,
    nrounds_used   = best_params$nrounds
  )
}


# =====================================================
# Parallel backend
# =====================================================
closeAllConnections()                 # Clear stale connections

cores <- min(16L, parallel::detectCores() - 1)  # Limit workers
cl    <- parallel::makeCluster(cores)          # Create fresh cluster
doParallel::registerDoParallel(cl)             # Register freshly created cluster


clusterEvalQ(cl, {
  library(xgboost)
  library(yardstick)
  library(dplyr)
})
clusterExport(cl, c("make_pid_folds"), envir = environment())


# -----------------------------------------------
# Main LOSO loop
# -----------------------------------------------
all_results <- list()

for (target in targets) {
  message("\n--- Running LOSO classification: ", target, " ---")
  
  fold_results <- map_df(participants, function(pid) {
    train_df <- df %>% filter(participant_id != pid)
    test_df  <- df %>% filter(participant_id == pid)
    run_single_fold(train_df, test_df, target, pid)
  }) %>% mutate(target = target)
  
  all_results [[target]] <- fold_results
}

all_results <- bind_rows(all_results)

# -----------------------------------------------
# Performance summary
# -----------------------------------------------
summary_out <- all_results %>%
  mutate(
    actual_f = factor(actual, levels = c(0,1)),
    class_f  = factor(class, levels = c(0,1))
  ) %>%
  group_by(target) %>%
  summarise(
    f1       = yardstick::f_meas_vec(truth = actual_f, estimate = class_f, event_level = "second"),
    accuracy = mean(class == actual),
    auc      = yardstick::roc_auc_vec(truth = actual_f, estimate = prob, event_level = "second"),
    logloss  = yardstick::mn_log_loss_vec(truth = actual_f, estimate = prob),
    .groups  = "drop"
  )

hyperparam_summary <- all_results %>%
  group_by(target) %>%
  summarise(
    eta_mean      = mean(eta),
    max_depth_md  = median(max_depth),
    subsample_md  = median(subsample),
    colsample_md  = median(colsample_bytree),
    min_child_md  = median(min_child_weight),
    lambda_md     = median(lambda),
    alpha_md      = median(alpha),
    nrounds_md    = median(nrounds_used),
    .groups = "drop"
  )


write_csv(hyperparam_summary, file.path(results_dir, "hyperparams_LOSO_summary.csv"))
write_rds(all_results, file.path(results_dir, "predictions_LOSO_fold_tuning.rds"))
write_csv(summary_out, file.path(results_dir, "summary_LOSO_fold_tuning.csv"))

summary_out


# Shut down cluster

stopCluster(cl)
















