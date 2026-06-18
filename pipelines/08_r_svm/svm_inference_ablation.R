library(tidyverse)
library(yardstick)
library(yaml)

source("pipelines/08_r_svm/svm_ablation_lib.R")

seed_main <- get_env_int("SVM_SEED", 42)
rng_mode <- get_env_str("SVM_RNG_MODE", "independent")

if (!rng_mode %in% c("independent", "legacy")) {
  stop(sprintf("Unknown SVM_RNG_MODE='%s' (expected 'independent' or 'legacy')", rng_mode))
}

set.seed(seed_main)

run_dir <- get_env_str("SVM_RUN_DIR", NA_character_)
domain <- get_env_str("SVM_DOMAIN", "all")
k <- get_env_int("SVM_K", 5)
perm_P <- get_env_int("SVM_PERM_P", 1000)
boot_R <- get_env_int("SVM_BOOT_R", 1000)
overwrite <- get_env_bool("SVM_OVERWRITE", FALSE)
sample_n_env <- get_env_int("SVM_SAMPLE_N", NA_integer_)
sample_rep_env <- get_env_int("SVM_SAMPLE_REP", NA_integer_)
sample_seed_env <- get_env_int("SVM_SAMPLE_SEED", NA_integer_)

if (is.na(run_dir) || run_dir == "") {
  stop("Missing required env var SVM_RUN_DIR")
}

msg("Inference run dir: %s", normalizePath(run_dir, winslash = "/", mustWork = FALSE))
msg("domain=%s k=%d perm_P=%d boot_R=%d", domain, k, perm_P, boot_R)
msg("rng_mode=%s seed_main=%d", rng_mode, seed_main)

targets <- c("stress_label", "workload_label")

metric_from_predictions <- function(pred_df) {
  truth <- factor(pred_df$y_true, levels = c(0, 1))
  estimate <- factor(pred_df$y_pred, levels = c(0, 1))

  list(
    auc = roc_auc_vec(truth = truth, estimate = pred_df$y_prob, event_level = "second"),
    acc = mean(pred_df$y_pred == pred_df$y_true),
    f1 = f_meas_vec(truth = truth, estimate = estimate, event_level = "second")
  )
}

bootstrap_ci <- function(pred_df, R = 1000, seed = 123, rng_mode = "independent") {
  if (rng_mode == "independent") {
    set.seed(seed)
  }
  ids <- unique(pred_df$participant_id)
  n_ids <- length(ids)

  if (n_ids < 3) {
    return(list(
      auc_ci = c(NA_real_, NA_real_),
      acc_ci = c(NA_real_, NA_real_),
      f1_ci = c(NA_real_, NA_real_)
    ))
  }

  auc_vals <- rep(NA_real_, R)
  acc_vals <- rep(NA_real_, R)
  f1_vals <- rep(NA_real_, R)

  for (b in seq_len(R)) {
    sampled_ids <- sample(ids, size = n_ids, replace = TRUE)
    boot_df <- bind_rows(lapply(sampled_ids, function(pid) pred_df[pred_df$participant_id == pid, , drop = FALSE]))

    m <- metric_from_predictions(boot_df)
    auc_vals[[b]] <- m$auc
    acc_vals[[b]] <- m$acc
    f1_vals[[b]] <- m$f1
  }

  list(
    auc_ci = unname(quantile(auc_vals, probs = c(0.025, 0.975), na.rm = TRUE)),
    acc_ci = unname(quantile(acc_vals, probs = c(0.025, 0.975), na.rm = TRUE)),
    f1_ci = unname(quantile(f1_vals, probs = c(0.025, 0.975), na.rm = TRUE))
  )
}

permute_auc_within_participant <- function(pred_df, P = 1000, seed = 456) {
  if (rng_mode == "independent") {
    set.seed(seed)
  }
  null_auc <- rep(NA_real_, P)

  progress_every <- get_env_int("SVM_PERM_PROGRESS_EVERY", 1000)
  if (progress_every < 0) progress_every <- 0

  for (i in seq_len(P)) {
    if (i == 1 || (progress_every > 0 && (i %% progress_every) == 0)) {
      msg("Permutation progress: %d/%d", i, P)
    }
    perm_df <- pred_df %>%
      group_by(participant_id) %>%
      mutate(y_true_perm = sample(y_true, size = n(), replace = FALSE)) %>%
      ungroup()

    truth <- factor(perm_df$y_true_perm, levels = c(0, 1))
    null_auc[[i]] <- roc_auc_vec(truth = truth, estimate = perm_df$y_prob, event_level = "second")
  }

  null_auc
}

summary_rows <- list()

summary_file <- file.path(run_dir, "svm_inference_summary.csv")

# Guard all outputs we might write
perm_paths <- file.path(run_dir, paste0("svm_perm_auc_", targets, ".rds"))
stop_if_paths_exist(c(summary_file, perm_paths), overwrite = overwrite)

for (target in targets) {
  pred_path <- file.path(run_dir, paste0("svm_predictions_", target, ".csv"))
  if (!file.exists(pred_path)) {
    stop(sprintf("Missing predictions file: %s", pred_path))
  }

  pred_df <- read_csv(pred_path, show_col_types = FALSE) %>%
    mutate(
      participant_id = as.integer(participant_id),
      y_true = as.integer(y_true),
      y_pred = as.integer(y_pred),
      y_prob = as.numeric(y_prob)
    )

  n_participants <- length(unique(pred_df$participant_id))
  n_rows <- nrow(pred_df)

  obs <- metric_from_predictions(pred_df)

  # In legacy mode, we intentionally do NOT reseed between bootstrap and permutation.
  # This matches scripts that call set.seed() once and then let RNG state advance.
  if (rng_mode == "independent") {
    cis <- bootstrap_ci(pred_df, R = boot_R, seed = 123, rng_mode = rng_mode)
  } else {
    cis <- bootstrap_ci(pred_df, R = boot_R, rng_mode = rng_mode)
  }

  msg("Computing permutation null for %s (P=%d)", target, perm_P)
  if (rng_mode == "independent") {
    null_auc <- permute_auc_within_participant(pred_df, P = perm_P, seed = 456)
  } else {
    null_auc <- permute_auc_within_participant(pred_df, P = perm_P)
  }

  perm_path <- file.path(run_dir, paste0("svm_perm_auc_", target, ".rds"))
  saveRDS(null_auc, perm_path)

  exceed <- sum(null_auc >= obs$auc)
  perm_p_value <- (1 + exceed) / (perm_P + 1)

  summary_rows[[target]] <- tibble(
    domain = domain,
    target = target,
    k = k,
    sample_n = ifelse(is.na(sample_n_env), n_participants, sample_n_env),
    sample_rep = ifelse(is.na(sample_rep_env), NA_integer_, sample_rep_env),
    sample_seed = ifelse(is.na(sample_seed_env), NA_integer_, sample_seed_env),
    n_participants = n_participants,
    n_rows = n_rows,
    auc = obs$auc,
    acc = obs$acc,
    f1 = obs$f1,
    auc_ci_lo = cis$auc_ci[[1]],
    auc_ci_hi = cis$auc_ci[[2]],
    acc_ci_lo = cis$acc_ci[[1]],
    acc_ci_hi = cis$acc_ci[[2]],
    f1_ci_lo = cis$f1_ci[[1]],
    f1_ci_hi = cis$f1_ci[[2]],
    perm_P = perm_P,
    perm_p_value = perm_p_value
  )
}

summary_df <- bind_rows(summary_rows)
write_csv(summary_df, summary_file)

print(summary_df)
msg("Inference complete")
