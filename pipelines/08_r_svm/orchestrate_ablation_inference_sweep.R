# Orchestrate SVM ablation + inference sweeps across domain/k combinations
# and participant sample sizes. Runs all combinations, continues on failures,
# and writes summaries suitable for underpower checks.

suppressPackageStartupMessages({
  library(tidyverse)
  library(yaml)
})

# ------------------------------
# User-editable paths
# ------------------------------
ablation_script  <- "pipelines/08_r_svm/svm_ablation_observed.R"
inference_script <- "pipelines/08_r_svm/svm_inference_ablation.R"
results_root     <- "results/svm"

# ------------------------------
# Sweep configuration
# ------------------------------
domains <- c("all", "eeg", "peripheral")
k_values <- c(5, 10, 20)

# Participant sample-size sweep settings.
sample_n_values <- c(10, 20, 30)
repeats_per_n <- 5
sample_seed <- 20260618L
include_full_n_baseline <- TRUE

# Use one shared timestamp for the whole sweep so run folders group together.
run_timestamp <- format(Sys.time(), "%Y-%m-%d_%H%M%S")

# Resolve Rscript path from the current R installation (Windows-safe even if PATH is missing Rscript).
rscript_bin <- file.path(R.home("bin"), "Rscript.exe")
if (!file.exists(rscript_bin)) {
  alt <- file.path(R.home("bin"), "Rscript")
  if (file.exists(alt)) {
    rscript_bin <- alt
  } else {
    stop("Could not locate Rscript in this R installation.")
  }
}

# Forward-slash path helper (Windows-compatible for env vars and logs).
to_slash <- function(path) gsub("\\\\", "/", path)

restore_env_vars <- function(old_values) {
  to_unset <- names(old_values)[is.na(old_values)]
  to_set <- old_values[!is.na(old_values)]

  if (length(to_set) > 0) {
    do.call(Sys.setenv, as.list(to_set))
  }
  if (length(to_unset) > 0) {
    Sys.unsetenv(to_unset)
  }
}

run_with_env <- function(command, args, env_named) {
  old_values <- Sys.getenv(names(env_named), unset = NA_character_)
  on.exit(restore_env_vars(old_values), add = TRUE)

  do.call(Sys.setenv, as.list(env_named))

  system2(
    command = command,
    args = args,
    stdout = "",
    stderr = ""
  )
}

read_available_participants <- function() {
  config <- yaml::read_yaml("scripts/utils/config.yaml")

  path_eeg_valid <- file.path(config$paths$output, "final_data_eeg_valid.rds")
  path_final <- file.path(config$paths$output, "final_data.rds")

  if (file.exists(path_eeg_valid)) {
    df_raw <- readRDS(path_eeg_valid)
  } else {
    df_raw <- readRDS(path_final)
  }

  if (!"participant_id" %in% names(df_raw)) {
    stop("Dataset must include participant_id")
  }

  sort(unique(as.integer(df_raw$participant_id)))
}

make_sampling_plan <- function(all_ids, n_values, repeats, seed, include_full) {
  plan_rows <- list()
  idx <- 1L

  for (n in n_values) {
    if (n > length(all_ids)) {
      stop(sprintf("Requested sample N=%d but only %d participants are available", n, length(all_ids)))
    }

    for (rep_idx in seq_len(repeats)) {
      draw_seed <- as.integer(seed + n * 1000L + rep_idx)
      set.seed(draw_seed)
      sampled_ids <- sort(sample(all_ids, size = n, replace = FALSE))

      plan_rows[[idx]] <- tibble(
        sample_n = n,
        sample_rep = rep_idx,
        sample_seed = draw_seed,
        participant_ids_csv = paste(sampled_ids, collapse = ",")
      )
      idx <- idx + 1L
    }
  }

  if (isTRUE(include_full)) {
    plan_rows[[idx]] <- tibble(
      sample_n = length(all_ids),
      sample_rep = 1L,
      sample_seed = NA_integer_,
      participant_ids_csv = paste(all_ids, collapse = ",")
    )
  }

  bind_rows(plan_rows)
}

ablation_script_abs  <- to_slash(normalizePath(ablation_script, winslash = "/", mustWork = TRUE))
inference_script_abs <- to_slash(normalizePath(inference_script, winslash = "/", mustWork = TRUE))
results_root_abs     <- to_slash(normalizePath(results_root, winslash = "/", mustWork = FALSE))
rscript_bin_abs      <- to_slash(normalizePath(rscript_bin, winslash = "/", mustWork = TRUE))

if (!dir.exists(results_root_abs)) {
  dir.create(results_root_abs, recursive = TRUE, showWarnings = FALSE)
}

available_participants <- read_available_participants()
sampling_plan <- make_sampling_plan(
  all_ids = available_participants,
  n_values = sample_n_values,
  repeats = repeats_per_n,
  seed = sample_seed,
  include_full = include_full_n_baseline
)

sampling_plan <- sampling_plan %>% mutate(plan_id = row_number())

combinations <- expand.grid(
  domain = domains,
  k = k_values,
  plan_id = sampling_plan$plan_id,
  stringsAsFactors = FALSE
) %>%
  left_join(sampling_plan, by = "plan_id")

n_total <- nrow(combinations)
results <- vector("list", n_total)

cat("=== SVM Sweep Start ===\n")
cat("Timestamp group:", run_timestamp, "\n")
cat("Ablation script:", ablation_script_abs, "\n")
cat("Inference script:", inference_script_abs, "\n")
cat("Rscript:", rscript_bin_abs, "\n")
cat("Available participants:", length(available_participants), "\n")
cat("Sample N values:", paste(sample_n_values, collapse = ","), "\n")
cat("Repeats per N:", repeats_per_n, "\n")
cat("Sample seed:", sample_seed, "\n")
cat("Include full-N baseline:", include_full_n_baseline, "\n")
cat("Total combinations:", n_total, "\n\n")

sampling_plan_out <- sprintf("%s/sampling_plan_%s.csv", results_root_abs, run_timestamp)
write_csv(sampling_plan, sampling_plan_out)
cat("Sampling plan CSV:", sampling_plan_out, "\n\n")

for (i in seq_len(n_total)) {
  domain <- combinations$domain[i]
  k <- combinations$k[i]
  sample_n <- as.integer(combinations$sample_n[i])
  sample_rep <- as.integer(combinations$sample_rep[i])
  sample_seed_i <- as.integer(combinations$sample_seed[i])
  participant_ids_csv <- combinations$participant_ids_csv[i]

  run_dir <- sprintf(
    "%s/%s_k%d_n%d_r%d_%s",
    results_root_abs,
    domain,
    k,
    sample_n,
    sample_rep,
    run_timestamp
  )
  dir.create(run_dir, recursive = TRUE, showWarnings = FALSE)

  combo_label <- sprintf("domain=%s, k=%d, n=%d, rep=%d", domain, k, sample_n, sample_rep)
  combo_start <- Sys.time()

  cat(sprintf("[%d/%d] START %s\n", i, n_total, combo_label))
  cat("  Start time:", format(combo_start, "%Y-%m-%d %H:%M:%S"), "\n")
  cat("  SVM_RUN_DIR:", run_dir, "\n")

  env_vars <- c(
    SVM_DOMAIN = domain,
    SVM_K = as.character(k),
    SVM_RUN_DIR = run_dir,
    SVM_OVERWRITE = "FALSE",
    SVM_SAMPLE_N = as.character(sample_n),
    SVM_SAMPLE_REP = as.character(sample_rep),
    SVM_PARTICIPANT_IDS = participant_ids_csv
  )

  if (!is.na(sample_seed_i)) {
    env_vars <- c(env_vars, SVM_SAMPLE_SEED = as.character(sample_seed_i))
  }

  write_csv(
    tibble(
      domain = domain,
      k = k,
      sample_n = sample_n,
      sample_rep = sample_rep,
      sample_seed = sample_seed_i,
      participant_ids = participant_ids_csv
    ),
    file.path(run_dir, "svm_run_design.csv")
  )

  # 1) Run ablation
  ablation_exit <- tryCatch(
    run_with_env(
      command = rscript_bin_abs,
      args = c("--vanilla", ablation_script_abs),
      env_named = env_vars
    ),
    error = function(e) {
      cat("  ERROR launching ablation:", conditionMessage(e), "\n")
      999L
    }
  )

  inference_exit <- NA_integer_
  status <- "failed"
  fail_stage <- "ablation"

  if (identical(ablation_exit, 0L)) {
    # 2) Run inference only if ablation succeeded.
    inference_exit <- tryCatch(
      run_with_env(
        command = rscript_bin_abs,
        args = c("--vanilla", inference_script_abs),
        env_named = env_vars
      ),
      error = function(e) {
        cat("  ERROR launching inference:", conditionMessage(e), "\n")
        999L
      }
    )

    if (identical(inference_exit, 0L)) {
      status <- "success"
      fail_stage <- ""
    } else {
      status <- "failed"
      fail_stage <- "inference"
    }
  }

  combo_end <- Sys.time()
  elapsed_sec <- as.numeric(difftime(combo_end, combo_start, units = "secs"))

  cat("  End time:", format(combo_end, "%Y-%m-%d %H:%M:%S"), "\n")
  cat(sprintf("  Duration: %.1f sec\n", elapsed_sec))
  cat(sprintf("  Result: %s", toupper(status)))
  if (status == "failed") {
    cat(sprintf(" (stage=%s, ablation_exit=%s, inference_exit=%s)",
                fail_stage,
                as.character(ablation_exit),
                as.character(inference_exit)))
  }
  cat("\n\n")

  results[[i]] <- data.frame(
    domain = domain,
    k = k,
    sample_n = sample_n,
    sample_rep = sample_rep,
    sample_seed = sample_seed_i,
    run_dir = run_dir,
    status = status,
    fail_stage = fail_stage,
    ablation_exit = as.integer(ablation_exit),
    inference_exit = as.integer(inference_exit),
    start_time = format(combo_start, "%Y-%m-%d %H:%M:%S"),
    end_time = format(combo_end, "%Y-%m-%d %H:%M:%S"),
    elapsed_sec = elapsed_sec,
    stringsAsFactors = FALSE
  )
}

summary_df <- do.call(rbind, results)

cat("=== SVM Sweep Summary ===\n")
cat(sprintf("Succeeded: %d / %d\n", sum(summary_df$status == "success"), n_total))
cat(sprintf("Failed:    %d / %d\n\n", sum(summary_df$status == "failed"), n_total))

if (any(summary_df$status == "success")) {
  cat("Successful combinations:\n")
  ok <- summary_df[summary_df$status == "success", c("domain", "k", "run_dir")]
  apply(ok, 1, function(r) cat(sprintf("  - domain=%s, k=%s, run_dir=%s\n", r[[1]], r[[2]], r[[3]])))
  cat("\n")
}

if (any(summary_df$status == "failed")) {
  cat("Failed combinations:\n")
  bad <- summary_df[summary_df$status == "failed", c("domain", "k", "fail_stage", "ablation_exit", "inference_exit", "run_dir")]
  apply(
    bad,
    1,
    function(r) {
      cat(sprintf(
        "  - domain=%s, k=%s, stage=%s, ablation_exit=%s, inference_exit=%s, run_dir=%s\n",
        r[[1]], r[[2]], r[[3]], r[[4]], r[[5]], r[[6]]
      ))
    }
  )
  cat("\n")
}

summary_out <- sprintf("%s/sweep_summary_%s.csv", results_root_abs, run_timestamp)
write.csv(summary_df, summary_out, row.names = FALSE)
cat("Summary CSV:", summary_out, "\n")

# Collect successful inference outputs for power analysis.
success_df <- summary_df[summary_df$status == "success", , drop = FALSE]
if (nrow(success_df) > 0) {
  inference_rows <- list()

  for (j in seq_len(nrow(success_df))) {
    run_dir_j <- success_df$run_dir[j]
    inf_path <- file.path(run_dir_j, "svm_inference_summary.csv")
    if (!file.exists(inf_path)) {
      next
    }

    inf_df <- read_csv(inf_path, show_col_types = FALSE)
    inf_df$run_dir <- run_dir_j
    inference_rows[[length(inference_rows) + 1L]] <- inf_df
  }

  if (length(inference_rows) > 0) {
    inference_all <- bind_rows(inference_rows)
    inf_out <- sprintf("%s/sweep_inference_rows_%s.csv", results_root_abs, run_timestamp)
    write_csv(inference_all, inf_out)
    cat("Inference rows CSV:", inf_out, "\n")

    full_n <- length(available_participants)

    baseline <- inference_all %>%
      filter(sample_n == full_n) %>%
      group_by(domain, k, target) %>%
      summarise(
        baseline_auc = mean(auc, na.rm = TRUE),
        baseline_perm_p = mean(perm_p_value, na.rm = TRUE),
        .groups = "drop"
      )

    power_assessment <- inference_all %>%
      filter(sample_n %in% sample_n_values) %>%
      group_by(domain, k, target, sample_n) %>%
      summarise(
        runs = n(),
        mean_auc = mean(auc, na.rm = TRUE),
        sd_auc = sd(auc, na.rm = TRUE),
        mean_perm_p = mean(perm_p_value, na.rm = TRUE),
        sig_rate_p05 = mean(perm_p_value < 0.05, na.rm = TRUE),
        mean_auc_ci_width = mean(auc_ci_hi - auc_ci_lo, na.rm = TRUE),
        .groups = "drop"
      ) %>%
      left_join(baseline, by = c("domain", "k", "target")) %>%
      mutate(
        delta_auc_vs_full_n = mean_auc - baseline_auc,
        delta_perm_p_vs_full_n = mean_perm_p - baseline_perm_p
      )

    power_out <- sprintf("%s/power_assessment_%s.csv", results_root_abs, run_timestamp)
    write_csv(power_assessment, power_out)
    cat("Power assessment CSV:", power_out, "\n")
  }
}

cat("=== SVM Sweep End ===\n")
