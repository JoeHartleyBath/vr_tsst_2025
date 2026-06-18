library(tidyverse)
library(e1071)
library(doParallel)
library(foreach)
library(iterators)
library(parallel)
library(yaml)
library(dplyr)
library(caret)
library(yardstick)

source("pipelines/08_r_svm/svm_ablation_lib.R")
source("utils/r/prune_feats.R")
source("utils/r/feature_selection.R")

set.seed(42)

# ---------------------------------------------------------------------
# Env/config
# ---------------------------------------------------------------------
config <- yaml::read_yaml("scripts/utils/config.yaml")

run_dir <- get_env_str("SVM_RUN_DIR", NA_character_)
domain <- get_env_str("SVM_DOMAIN", "all")
k <- get_env_int("SVM_K", 5)
overwrite <- get_env_bool("SVM_OVERWRITE", FALSE)

if (is.na(run_dir) || run_dir == "") {
  stop("Missing required env var SVM_RUN_DIR")
}

dir.create(run_dir, recursive = TRUE, showWarnings = FALSE)

msg("Run output dir: %s", normalizePath(run_dir, winslash = "/", mustWork = FALSE))

# ---------------------------------------------------------------------
# Load dataset
# ---------------------------------------------------------------------
path_eeg_valid <- file.path(config$paths$output, "final_data_eeg_valid.rds")
path_final <- file.path(config$paths$output, "final_data.rds")

if (file.exists(path_eeg_valid)) {
  df_raw <- readRDS(path_eeg_valid)
  msg("Loaded EEG dataset %s (%d rows, %d columns)", path_eeg_valid, nrow(df_raw), ncol(df_raw))
} else {
  df_raw <- readRDS(path_final)
  msg("Loaded dataset %s (%d rows, %d columns)", path_final, nrow(df_raw), ncol(df_raw))
}

if (!all(c("participant_id", "condition") %in% names(df_raw))) {
  stop("Dataset must include participant_id and condition")
}

# Derive binary targets (match canonical svm.R)
df_raw <- derive_binary_targets_from_condition(df_raw)

targets <- c("stress_label", "workload_label")

# ---------------------------------------------------------------------
# Select domain features
# ---------------------------------------------------------------------
features_all <- select_analysis_features(df_raw, suffix = "_precond")

# Apply domain allowlist from config/svm_ablation.yaml
cfg_domain <- read_domain_config("config/svm_ablation.yaml")
features_domain <- filter_features_for_domain(features_all, cfg_domain, domain)

msg("Domain=%s; candidate features=%d", domain, length(features_domain))

if (length(features_domain) < k) {
  stop(sprintf("Not enough features for k=%d (domain '%s' has %d)", k, domain, length(features_domain)))
}

# Keep only needed columns
# Note: keep condition in the modeling DF so it can be written to predictions.
df <- df_raw %>%
  select(participant_id, condition, all_of(targets), all_of(features_domain))

# Within-participant scaling
df <- df %>%
  group_by(participant_id) %>%
  mutate(across(all_of(features_domain), ~ as.numeric(scale(.)[, 1]))) %>%
  ungroup()

participants <- sort(unique(df$participant_id))
msg("Participants detected: %d", length(participants))

# Optional participant sub-sampling from orchestrator.
participant_ids_raw <- get_env_str("SVM_PARTICIPANT_IDS", "")
sample_n_env <- get_env_int("SVM_SAMPLE_N", NA_integer_)
sample_rep_env <- get_env_int("SVM_SAMPLE_REP", NA_integer_)
sample_seed_env <- get_env_int("SVM_SAMPLE_SEED", NA_integer_)

if (participant_ids_raw != "") {
  selected_ids <- participant_ids_raw %>%
    str_split(",") %>%
    .[[1]] %>%
    trimws() %>%
    as.integer() %>%
    unique()

  selected_ids <- selected_ids[!is.na(selected_ids)]

  if (length(selected_ids) < 3) {
    stop("SVM_PARTICIPANT_IDS must contain at least 3 valid participant IDs")
  }

  missing_ids <- setdiff(selected_ids, participants)
  if (length(missing_ids) > 0) {
    stop(sprintf("SVM_PARTICIPANT_IDS contains IDs not in dataset: %s", paste(missing_ids, collapse = ",")))
  }

  participants <- sort(selected_ids)
  df <- df %>% filter(participant_id %in% participants)
  msg("Using participant subset from env: n=%d", length(participants))
}

sample_meta <- tibble(
  sample_n = ifelse(is.na(sample_n_env), length(participants), sample_n_env),
  sample_rep = ifelse(is.na(sample_rep_env), NA_integer_, sample_rep_env),
  sample_seed = ifelse(is.na(sample_seed_env), NA_integer_, sample_seed_env),
  n_participants_used = length(participants),
  participant_ids = paste(participants, collapse = ",")
)

write_csv(sample_meta, file.path(run_dir, "svm_sample_metadata.csv"))

# ---------------------------------------------------------------------
# Hyperparameter grid (match pipelines/08_r_svm/svm.R)
# ---------------------------------------------------------------------
cost_grid <- c(2, 4, 8, 16, 32, 64, 128, 256)
gamma_grid <- c(2^-7, 2^-6, 2^-5, 2^-4, 2^-3, 0.03, 0.06, 0.1)

# ---------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------
run_svm_predict <- function(train, test, target, feats, cost, gamma) {
  model <- svm(
    reformulate(feats, target),
    data = train,
    kernel = "radial",
    cost = cost,
    gamma = gamma,
    probability = TRUE,
    scale = FALSE
  )

  pred_obj <- predict(model, test %>% select(all_of(feats)), probability = TRUE)
  prob_df <- attr(pred_obj, "probabilities")

  positive_class <- levels(train[[target]])[2]
  probs <- prob_df[, positive_class]

  list(preds = pred_obj, probs = probs)
}

compute_metrics <- function(truth, preds, probs) {
  acc <- mean(preds == truth)
  f1 <- f_meas_vec(truth = truth, estimate = preds, event_level = "second")
  auc <- roc_auc_vec(truth = truth, estimate = probs, event_level = "second")
  list(acc = acc, f1 = f1, auc = auc)
}

# ---------------------------------------------------------------------
# Output file helpers
# ---------------------------------------------------------------------
progress_header <- c(
  "timestamp", "test_pid", "target", "k",
  "final_acc", "final_f1", "final_auc", "inner_best_acc",
  "cost", "gamma", "feature_n", "feature_set", "features_used"
)

tuning_header <- c(
  "timestamp", "test_pid", "target", "k", "cost", "gamma",
  "inner_acc", "inner_auc", "inner_f1", "feature_n", "feature_set", "features_used"
)

pred_header <- c("participant_id", "condition", "target", "k", "y_true", "y_pred", "y_prob")

write_csv_header <- function(path, header) {
  write.table(t(header), file = path, sep = ",", col.names = FALSE, row.names = FALSE, quote = FALSE)
}

# ---------------------------------------------------------------------
# Parallel backend (for tuning grid)
# ---------------------------------------------------------------------
cores_env <- get_env_int("SVM_CORES", NA_integer_)
cores_auto <- max(1L, parallel::detectCores() - 2L)
cores <- if (is.na(cores_env)) min(14L, cores_auto) else cores_env
cores <- max(1L, min(cores, cores_auto))

msg("Spawning PSOCK cluster with %d workers", cores)
cl <- makeCluster(cores)
registerDoParallel(cl)

clusterEvalQ(cl, {
  library(tidyverse)
  library(e1071)
  library(yardstick)
  library(dplyr)
})

clusterExport(
  cl,
  varlist = c("participants", "k", "cost_grid", "gamma_grid", "run_svm_predict", "compute_metrics"),
  envir = environment()
)

on.exit({
  try(stopCluster(cl), silent = TRUE)
}, add = TRUE)

# ---------------------------------------------------------------------
# Nested LOSO (observed)
# ---------------------------------------------------------------------
run_target_observed <- function(target) {
  progress_file <- file.path(run_dir, paste0("svm_progress_", target, ".csv"))
  tuning_file <- paste0(progress_file, "_tuning.csv")
  pred_file <- file.path(run_dir, paste0("svm_predictions_", target, ".csv"))
  fold_plan_file <- file.path(run_dir, paste0("svm_fold_plan_", target, ".rds"))

  stop_if_paths_exist(c(progress_file, tuning_file, pred_file, fold_plan_file), overwrite = overwrite)

  write_csv_header(progress_file, progress_header)
  write_csv_header(tuning_file, tuning_header)
  write_csv_header(pred_file, pred_header)

  outer_folds <- list()

  for (test_pid in participants) {
    msg("[%s] Outer fold test_pid=%s", target, as.character(test_pid))
    train_df <- df %>% filter(participant_id != test_pid)
    test_df <- df %>% filter(participant_id == test_pid)

    pruned_feats <- prune_features(train_df, target = target)

    if (length(pruned_feats) < k) {
      stop(sprintf("After pruning, not enough features for pid=%s target=%s: %d", test_pid, target, length(pruned_feats)))
    }

    # Median impute using training medians
    train_df <- train_df %>%
      mutate(across(all_of(pruned_feats), ~ replace(., is.na(.), median(., na.rm = TRUE))))

    test_df <- test_df %>%
      mutate(across(all_of(pruned_feats), ~ replace(., is.na(.), median(train_df[[cur_column()]], na.rm = TRUE))))

    # Rank by absolute correlation strength (match svm.R)
    r_ranked_feats <- pruned_feats[
      order(
        abs(cor(train_df[pruned_feats], as.numeric(train_df[[target]]), use = "complete.obs")),
        decreasing = TRUE
      )
    ]

    feats_k <- head(r_ranked_feats, k)

    tune_grid <- expand.grid(cost = cost_grid, gamma = gamma_grid)

    tune_results <- foreach(i = seq_len(nrow(tune_grid)), .combine = rbind, .packages = c("dplyr", "yardstick", "e1071")) %dopar% {
      cost <- tune_grid$cost[i]
      gamma <- tune_grid$gamma[i]

      inner_metrics <- map_dfr(
        setdiff(participants, test_pid),
        function(inner_pid) {
          inner_train <- train_df %>% filter(participant_id != inner_pid)
          inner_test <- train_df %>% filter(participant_id == inner_pid)

          pred <- run_svm_predict(inner_train, inner_test, target, feats_k, cost, gamma)
          m <- compute_metrics(inner_test[[target]], pred$preds, pred$probs)
          tibble(acc = m$acc, auc = m$auc, f1 = m$f1)
        }
      )

      data.frame(
        timestamp = Sys.time(),
        test_pid = test_pid,
        target = target,
        k = k,
        cost = cost,
        gamma = gamma,
        inner_acc = mean(inner_metrics$acc),
        inner_auc = mean(inner_metrics$auc),
        inner_f1 = mean(inner_metrics$f1),
        feature_n = length(feats_k),
        feature_set = paste0("top", k),
        features_used = paste(feats_k, collapse = ";")
      )
    }

    best_row <- tune_results[which.max(tune_results$inner_auc), ]

    # Append tuning detail log
    write.table(
      tune_results,
      file = tuning_file,
      append = TRUE,
      sep = ",",
      col.names = FALSE,
      row.names = FALSE
    )

    best_params <- list(cost = best_row$cost, gamma = best_row$gamma)

    final_pred <- run_svm_predict(train_df, test_df, target, feats_k, best_params$cost, best_params$gamma)
    final_metrics <- compute_metrics(test_df[[target]], final_pred$preds, final_pred$probs)

    res_row <- tibble(
      timestamp = Sys.time(),
      test_pid = test_pid,
      target = target,
      k = k,
      final_acc = final_metrics$acc,
      final_f1 = final_metrics$f1,
      final_auc = final_metrics$auc,
      inner_best_acc = best_row$inner_acc,
      cost = best_params$cost,
      gamma = best_params$gamma,
      feature_n = length(feats_k),
      feature_set = paste0("top", k),
      features_used = paste(feats_k, collapse = ";")
    )

    write.table(res_row, file = progress_file, append = TRUE, sep = ",", col.names = FALSE, row.names = FALSE)

    target_name <- target
    pred_rows <- tibble(
      participant_id = test_df$participant_id,
      condition = test_df$condition,
      target = target_name,
      k = k,
      y_true = as.integer(as.character(test_df[[target_name]])),
      y_pred = as.integer(as.character(final_pred$preds)),
      y_prob = as.numeric(final_pred$probs)
    )

    write.table(pred_rows, file = pred_file, append = TRUE, sep = ",", col.names = FALSE, row.names = FALSE)

    outer_folds[[as.character(test_pid)]] <- list(
      test_pid = test_pid,
      train_pids = setdiff(participants, test_pid),
      n_test_rows = nrow(test_df)
    )
  }

  fold_plan <- list(
    created = Sys.time(),
    domain = domain,
    target = target,
    k = k,
    n_participants = length(participants),
    n_rows = nrow(df),
    participants = participants,
    outer_folds = outer_folds
  )

  saveRDS(fold_plan, fold_plan_file)

  msg("Observed run complete for target=%s", target)
}

for (t in targets) {
  msg("Running observed LOSO for target=%s", t)
  run_target_observed(t)
}

msg("All observed runs complete")
