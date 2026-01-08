# =====================================================================
# masking_moderation_lmm.R
# Moderation ("masking") analysis: test whether the within-participant
# association between subjective ratings and physiological features
# changes as a function of the other factor (Stress ↔ MWL).
#
# Mirrors data loading + canonical feature handling from:
#   scripts/analysis/correlations/stratified_rmcorr_correlations.R
#
# Outputs:
#   results/classic_analyses/masking_moderation_lmm/
#     masking_moderation_summary.csv
#     masking_moderation_summary.rds
#     masking_moderation_diagnostics.csv
#     masking_moderation_log.txt
#
# Key design choices:
# - Uses canonical pre-condition baseline-adjusted features (suffix: _precond)
#   and applies within-participant mean/SD z-scoring inside this script.
# - Uses within-participant centered predictors (rating_w) to match the
#   within-subject focus of repeated-measures correlations.
# - Primary model: LMM with random intercept for participant.
# - If assumptions/convergence break: retries optimizers; falls back to
#   participant fixed-effects LM; optionally runs bootstrap inference.
# =====================================================================

suppressPackageStartupMessages({
  library(tidyverse)
  library(yaml)
  library(lme4)
  library(lmerTest)
  library(boot)
})

# =====================================================================
# 0. CONFIG
# =====================================================================

config <- yaml::read_yaml("scripts/utils/config.yaml")

out_dir <- file.path(config$paths$results, "classic_analyses", "masking_moderation_lmm")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

log_path <- file.path(out_dir, "masking_moderation_log.txt")

# Parameters (safe defaults; adjust as needed)
params <- list(
  min_participants = 10,
  shapiro_max_n = 5000,
  hetero_abs_resid_rho_thresh = 0.30,
  bootstrap_if_non_normal = TRUE,
  bootstrap_if_hetero = TRUE,
  n_boot = 1000,           # used for FE-LM cluster bootstrap
  n_boot_lmm = 500,        # used for bootMer when needed (keep moderate for speed)
  seed = 123
)

set.seed(params$seed)

# =====================================================================
# 1. HELPERS
# =====================================================================

load_obj <- function(stem, dir_path = config$paths$output) {
  rds_path <- file.path(dir_path, paste0(stem, ".rds"))
  csv_path <- file.path(dir_path, paste0(stem, ".csv"))
  if (file.exists(rds_path)) {
    readRDS(rds_path)
  } else if (file.exists(csv_path)) {
    readr::read_csv(csv_path, show_col_types = FALSE)
  } else {
    stop("No file found for: ", stem)
  }
}

msg <- function(...) {
  line <- paste0(format(Sys.time(), "%Y-%m-%d %H:%M:%S"), " | ", paste0(..., collapse = ""))
  cat(line, "\n")
  cat(line, "\n", file = log_path, append = TRUE)
}

stop_if_missing_cols <- function(df, cols, df_name = "data") {
  missing <- setdiff(cols, names(df))
  if (length(missing) > 0) {
    stop(sprintf("%s is missing required columns: %s", df_name, paste(missing, collapse = ", ")))
  }
}

safe_factor <- function(x, levels = NULL) {
  if (!is.null(levels)) {
    factor(x, levels = levels)
  } else {
    factor(x)
  }
}

model_family <- function(feature_name) {
  case_when(
    str_detect(feature_name, "^eeg_") ~ "EEG",
    str_detect(feature_name, "^(hrv|hr)_") ~ "Physiological",
    str_detect(feature_name, "^eda_") ~ "Physiological",
    str_detect(feature_name, "^pupil_") ~ "Physiological",
    TRUE ~ "Other"
  )
}

# Within-person centering for predictor, plus between-person mean
add_within_between <- function(df, id_col, x_col) {
  id_sym <- rlang::ensym(id_col)
  x_sym <- rlang::ensym(x_col)
  x_name <- rlang::as_string(x_sym)

  df %>%
    group_by(!!id_sym) %>%
    mutate(
      !!paste0(x_name, "_b") := mean(!!x_sym, na.rm = TRUE),
      !!paste0(x_name, "_w") := (!!x_sym) - mean(!!x_sym, na.rm = TRUE)
    ) %>%
    ungroup()
}

# Residual diagnostics: normality + a simple heteroscedasticity proxy
compute_diag <- function(resid, fitted, shapiro_max_n = 5000) {
  out <- list(shapiro_p = NA_real_, hetero_rho = NA_real_, hetero_p = NA_real_)

  if (length(resid) >= 3 && length(resid) <= shapiro_max_n) {
    out$shapiro_p <- tryCatch(stats::shapiro.test(resid)$p.value, error = function(e) NA_real_)
  }

  # Simple heteroscedasticity proxy: Spearman corr(|resid|, fitted)
  if (length(resid) >= 5) {
    ct <- tryCatch(
      suppressWarnings(stats::cor.test(abs(resid), fitted, method = "spearman")),
      error = function(e) NULL
    )
    if (!is.null(ct)) {
      out$hetero_rho <- unname(ct$estimate)
      out$hetero_p <- ct$p.value
    }
  }

  out
}

# Try multiple optimizers for lmer
fit_lmer_retry <- function(formula, data) {
  opts <- list(
    lme4::lmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 2e5)),
    lme4::lmerControl(optimizer = "nloptwrap", optCtrl = list(maxeval = 2e5))
  )

  for (ctrl in opts) {
    fit <- tryCatch(
      suppressWarnings(lmerTest::lmer(formula, data = data, control = ctrl)),
      error = function(e) NULL
    )
    if (!is.null(fit)) {
      return(fit)
    }
  }

  NULL
}

# Extract term summary from lmerTest / lmer
extract_term_lmer <- function(fit, term_name) {
  sm <- summary(fit)
  coefs <- as.data.frame(sm$coefficients)
  if (!term_name %in% rownames(coefs)) {
    return(list(est = NA_real_, se = NA_real_, t = NA_real_, df = NA_real_, p = NA_real_))
  }
  row <- coefs[term_name, , drop = FALSE]
  # lmerTest names columns: Estimate, Std. Error, df, t value, Pr(>|t|)
  list(
    est = unname(row[["Estimate"]]),
    se = unname(row[["Std. Error"]]),
    df = if ("df" %in% names(row)) unname(row[["df"]]) else NA_real_,
    t  = unname(row[["t value"]]),
    p  = if ("Pr(>|t|)" %in% names(row)) unname(row[["Pr(>|t|)"]]) else NA_real_
  )
}

# Participant fixed-effects fallback (cluster bootstrap by participant)
fit_fe_lm <- function(formula, data, id_col = "participant_id") {
  id_sym <- rlang::ensym(id_col)
  data <- data %>% mutate(.pid = as.factor(!!id_sym))
  # Add participant as fixed effect
  fe_formula <- stats::update.formula(formula, . ~ . + .pid)
  stats::lm(fe_formula, data = data)
}

cluster_boot_coef <- function(data, formula, coef_name, id_col = "participant_id", R = 1000, seed = 123) {
  set.seed(seed)
  ids <- unique(data[[id_col]])
  if (length(ids) < 3) {
    return(list(p_boot = NA_real_, ci_low = NA_real_, ci_high = NA_real_, n_eff = 0L))
  }

  vals <- rep(NA_real_, R)
  for (b in seq_len(R)) {
    sampled_ids <- sample(ids, size = length(ids), replace = TRUE)
    boot_df <- dplyr::bind_rows(lapply(sampled_ids, function(pid) data[data[[id_col]] == pid, , drop = FALSE]))

    fit <- tryCatch(fit_fe_lm(formula, boot_df, id_col = id_col), error = function(e) NULL)
    if (is.null(fit)) next

    cf <- coef(fit)
    if (!coef_name %in% names(cf)) next
    vals[[b]] <- unname(cf[[coef_name]])
  }

  vals <- vals[is.finite(vals)]

  if (length(vals) < max(50, 0.1 * R)) {
    return(list(p_boot = NA_real_, ci_low = NA_real_, ci_high = NA_real_, n_eff = length(vals)))
  }

  # Two-sided bootstrap p-value around 0
  p_boot <- 2 * min(mean(vals >= 0), mean(vals <= 0))
  ci <- stats::quantile(vals, probs = c(0.025, 0.975), na.rm = TRUE)

  list(
    p_boot = p_boot,
    ci_low = unname(ci[[1]]),
    ci_high = unname(ci[[2]]),
    n_eff = length(vals)
  )
}

# =====================================================================
# 2. LOAD + PREP DATA
# =====================================================================

msg("[Masking LMM] Loading final_data")
final_data <- load_obj("final_data")

stop_if_missing_cols(final_data, c("participant_id", "condition", "stress", "workload"), df_name = "final_data")

# Derive binary stress/workload factors from condition (match stratified script)
df_full <- final_data %>%
  filter(!is.na(stress), !is.na(workload)) %>%
  mutate(
    stress_level = if_else(condition %in% c("High Stress - High Cog", "High Stress - Low Cog"), "High", "Low"),
    workload_level = if_else(condition %in% c("High Stress - High Cog", "Low Stress - High Cog"), "High", "Low")
  ) %>%
  mutate(
    stress_level = safe_factor(stress_level, levels = c("Low", "High")),
    workload_level = safe_factor(workload_level, levels = c("Low", "High"))
  )

has_qc_flag <- "qc_failed" %in% names(df_full)

# Canonical features (precond)
canonical_feats <- config$canonical_features
suffix <- "_precond"
features <- paste0(canonical_feats, suffix)

missing_feats <- setdiff(features, names(df_full))
if (length(missing_feats) > 0) {
  stop(
    "Missing expected canonical feature columns in final_data: ",
    paste(missing_feats, collapse = ", "),
    "\nExpected suffix is ", suffix, "."
  )
}

# Apply simple within-subject z-score to match SVM preprocessing
zscore_safe <- function(x) {
  m <- mean(x, na.rm = TRUE)
  s <- stats::sd(x, na.rm = TRUE)
  if (!is.finite(s) || s <= 1e-8) return(rep(0, length(x)))
  (x - m) / s
}

df_full <- df_full %>%
  group_by(participant_id) %>%
  mutate(across(all_of(features), zscore_safe)) %>%
  ungroup()

# EEG validity handling (same logic as rmcorr script)
eeg_features <- features[str_detect(features, regex("^eeg_", ignore_case = TRUE))]

msg(sprintf("[Masking LMM] Participants (all modalities): N=%d", n_distinct(df_full$participant_id)))
if (has_qc_flag) {
  msg(sprintf("[Masking LMM] Participants (EEG-valid, qc_failed==FALSE): N=%d", n_distinct(df_full %>% filter(!qc_failed) %>% pull(participant_id))))
}

# Add within/between predictors
# Note: within-person centering aligns with repeated-measures correlation intent.
df_full <- df_full %>%
  add_within_between(participant_id, stress) %>%
  add_within_between(participant_id, workload)

# =====================================================================
# 3. MODEL RUNNER
# =====================================================================

run_one_feature <- function(df_in, feature_col, analysis_name, rating_base, moderator_col) {
  # rating_base: "stress" or "workload" (predictor)
  # moderator_col: "workload_level" or "stress_level" (factor)

  # Selective filtering: use EEG-valid only for EEG features
  df <- df_in
  if (has_qc_flag && feature_col %in% eeg_features) {
    df <- df %>% filter(!qc_failed)
  }

  # Required columns for modeling
  rating_w <- paste0(rating_base, "_w")
  rating_b <- paste0(rating_base, "_b")

  keep_cols <- c("participant_id", feature_col, moderator_col, rating_w, rating_b)
  stop_if_missing_cols(df, keep_cols, df_name = "prepared df")

  d <- df %>%
    select(participant_id, feature = all_of(feature_col), moderator = all_of(moderator_col),
           rating_w = all_of(rating_w), rating_b = all_of(rating_b)) %>%
    drop_na()

  n_participants <- n_distinct(d$participant_id)
  n_obs <- nrow(d)

  # Basic data adequacy checks
  if (n_participants < params$min_participants || n_obs < 3 * params$min_participants) {
    return(tibble(
      analysis = analysis_name,
      feature = feature_col,
      modality = model_family(feature_col),
      n_participants = n_participants,
      n_obs = n_obs,
      model_type = NA_character_,
      converged = NA,
      singular = NA,
      shapiro_p = NA_real_,
      hetero_rho = NA_real_,
      hetero_p = NA_real_,
      interaction_term = NA_character_,
      interaction_est = NA_real_,
      interaction_se = NA_real_,
      interaction_df = NA_real_,
      interaction_t = NA_real_,
      interaction_p = NA_real_,
      interaction_p_boot = NA_real_,
      interaction_ci_low = NA_real_,
      interaction_ci_high = NA_real_,
      slope_low = NA_real_,
      slope_high = NA_real_
    ))
  }

  # Ensure moderator has both levels
  if (n_distinct(d$moderator) < 2) {
    return(tibble(
      analysis = analysis_name,
      feature = feature_col,
      modality = model_family(feature_col),
      n_participants = n_participants,
      n_obs = n_obs,
      model_type = NA_character_,
      converged = NA,
      singular = NA,
      shapiro_p = NA_real_,
      hetero_rho = NA_real_,
      hetero_p = NA_real_,
      interaction_term = NA_character_,
      interaction_est = NA_real_,
      interaction_se = NA_real_,
      interaction_df = NA_real_,
      interaction_t = NA_real_,
      interaction_p = NA_real_,
      interaction_p_boot = NA_real_,
      interaction_ci_low = NA_real_,
      interaction_ci_high = NA_real_,
      slope_low = NA_real_,
      slope_high = NA_real_
    ))
  }

  # Ensure some within-person variance in rating_w
  if (sd(d$rating_w, na.rm = TRUE) < 1e-8) {
    return(tibble(
      analysis = analysis_name,
      feature = feature_col,
      modality = model_family(feature_col),
      n_participants = n_participants,
      n_obs = n_obs,
      model_type = NA_character_,
      converged = NA,
      singular = NA,
      shapiro_p = NA_real_,
      hetero_rho = NA_real_,
      hetero_p = NA_real_,
      interaction_term = NA_character_,
      interaction_est = NA_real_,
      interaction_se = NA_real_,
      interaction_df = NA_real_,
      interaction_t = NA_real_,
      interaction_p = NA_real_,
      interaction_p_boot = NA_real_,
      interaction_ci_low = NA_real_,
      interaction_ci_high = NA_real_,
      slope_low = NA_real_,
      slope_high = NA_real_
    ))
  }

  # Model specification
  # Include rating_b to avoid conflating between-subject differences with within-subject slope.
  formula <- feature ~ rating_w * moderator + rating_b + (1 | participant_id)

  # Fit LMM
  fit <- fit_lmer_retry(formula, d)

  used_model <- "lmer"
  converged <- NA
  singular <- NA

  if (!is.null(fit)) {
    singular <- lme4::isSingular(fit, tol = 1e-4)
    converged <- is.null(fit@optinfo$conv$lme4$messages) && is.null(fit@optinfo$conv$messages)
  }

  # Fallback conditions: fit failed OR fit is singular/non-converged
  needs_fallback <- is.null(fit) || isFALSE(converged) || isTRUE(singular)
  if (needs_fallback) {
    used_model <- "fe_lm"
    fe_fit <- tryCatch(fit_fe_lm(feature ~ rating_w * moderator + rating_b, d), error = function(e) NULL)
    if (!is.null(fe_fit)) {
      fit <- fe_fit
      converged <- TRUE
      # Keep singular flag from LMM if available; otherwise NA
      if (is.null(singular)) singular <- NA
    } else if (is.null(fit)) {
      return(tibble(
        analysis = analysis_name,
        feature = feature_col,
        modality = model_family(feature_col),
        n_participants = n_participants,
        n_obs = n_obs,
        model_type = "failed",
        converged = FALSE,
        singular = singular,
        shapiro_p = NA_real_,
        hetero_rho = NA_real_,
        hetero_p = NA_real_,
        interaction_term = NA_character_,
        interaction_est = NA_real_,
        interaction_se = NA_real_,
        interaction_df = NA_real_,
        interaction_t = NA_real_,
        interaction_p = NA_real_,
        interaction_p_boot = NA_real_,
        interaction_ci_low = NA_real_,
        interaction_ci_high = NA_real_,
        slope_low = NA_real_,
        slope_high = NA_real_
      ))
    }
  }

  # Interaction term name depends on factor coding; moderator uses Low as baseline.
  # With formula: rating_w * moderator
  # interaction term: rating_w:moderatorHigh
  interaction_term <- "rating_w:moderatorHigh"

  # Diagnostics
  if (used_model == "lmer") {
    resid <- residuals(fit)
    fitted <- fitted(fit)
  } else {
    resid <- residuals(fit)
    fitted <- fitted.values(fit)
  }

  diag <- compute_diag(resid, fitted, shapiro_max_n = params$shapiro_max_n)

  non_normal <- !is.na(diag$shapiro_p) && diag$shapiro_p < 0.05
  hetero <- !is.na(diag$hetero_p) && diag$hetero_p < 0.05 && abs(diag$hetero_rho) >= params$hetero_abs_resid_rho_thresh

  # Extract inference for interaction + simple slopes
  interaction <- list(est = NA_real_, se = NA_real_, df = NA_real_, t = NA_real_, p = NA_real_)
  slope_low <- NA_real_
  slope_high <- NA_real_

  if (used_model == "lmer") {
    interaction <- extract_term_lmer(fit, interaction_term)

    # Simple slopes from fixed effects
    beta <- lme4::fixef(fit)
    slope_low <- unname(beta[["rating_w"]])
    slope_high <- unname(beta[["rating_w"]] + beta[[interaction_term]])
  } else {
    sm <- summary(fit)$coefficients
    if (interaction_term %in% rownames(sm)) {
      interaction$est <- unname(sm[interaction_term, "Estimate"])
      interaction$se <- unname(sm[interaction_term, "Std. Error"])
      interaction$t <- unname(sm[interaction_term, "t value"])
      interaction$p <- unname(sm[interaction_term, "Pr(>|t|)"])
      interaction$df <- unname(fit$df.residual)
    }
    beta <- coef(fit)
    slope_low <- if ("rating_w" %in% names(beta)) unname(beta[["rating_w"]]) else NA_real_
    slope_high <- if (interaction_term %in% names(beta) && "rating_w" %in% names(beta)) unname(beta[["rating_w"]] + beta[[interaction_term]]) else NA_real_
  }

  # Bootstrap inference if diagnostics indicate broken assumptions
  p_boot <- NA_real_
  ci_low <- NA_real_
  ci_high <- NA_real_

  needs_boot <- (non_normal && params$bootstrap_if_non_normal) || (hetero && params$bootstrap_if_hetero)

  if (needs_boot) {
    if (used_model == "lmer") {
      msg(sprintf("[Boot] %s | %s | bootMer(%d)", analysis_name, feature_col, params$n_boot_lmm))
      boot_vals <- tryCatch({
        b <- lme4::bootMer(
          fit,
          FUN = function(f) lme4::fixef(f)[[interaction_term]],
          nsim = params$n_boot_lmm,
          type = "parametric",
          seed = params$seed
        )
        as.numeric(b$t)
      }, error = function(e) NULL)

      if (!is.null(boot_vals)) {
        boot_vals <- boot_vals[is.finite(boot_vals)]
        if (length(boot_vals) >= 50) {
          p_boot <- 2 * min(mean(boot_vals >= 0), mean(boot_vals <= 0))
          ci <- stats::quantile(boot_vals, probs = c(0.025, 0.975), na.rm = TRUE)
          ci_low <- unname(ci[[1]])
          ci_high <- unname(ci[[2]])
        }
      }
    } else {
      msg(sprintf("[Boot] %s | %s | cluster bootstrap FE-LM (R=%d)", analysis_name, feature_col, params$n_boot))
      b <- cluster_boot_coef(
        data = d,
        formula = feature ~ rating_w * moderator + rating_b,
        coef_name = interaction_term,
        id_col = "participant_id",
        R = params$n_boot,
        seed = params$seed
      )
      p_boot <- b$p_boot
      ci_low <- b$ci_low
      ci_high <- b$ci_high
    }
  }

  tibble(
    analysis = analysis_name,
    feature = feature_col,
    modality = model_family(feature_col),
    n_participants = n_participants,
    n_obs = n_obs,
    model_type = used_model,
    converged = converged,
    singular = singular,
    shapiro_p = diag$shapiro_p,
    hetero_rho = diag$hetero_rho,
    hetero_p = diag$hetero_p,
    interaction_term = interaction_term,
    interaction_est = interaction$est,
    interaction_se = interaction$se,
    interaction_df = interaction$df,
    interaction_t = interaction$t,
    interaction_p = interaction$p,
    interaction_p_boot = p_boot,
    interaction_ci_low = ci_low,
    interaction_ci_high = ci_high,
    slope_low = slope_low,
    slope_high = slope_high
  )
}

# =====================================================================
# 4. RUN ANALYSES
# =====================================================================

msg("[Masking LMM] Running moderation models")

# A) Stress rating association moderated by Workload level
res_stress_by_workload <- map_dfr(
  features,
  ~ run_one_feature(
    df_in = df_full,
    feature_col = .x,
    analysis_name = "stress_by_workload",
    rating_base = "stress",
    moderator_col = "workload_level"
  )
)

# B) Workload rating association moderated by Stress level
res_workload_by_stress <- map_dfr(
  features,
  ~ run_one_feature(
    df_in = df_full,
    feature_col = .x,
    analysis_name = "workload_by_stress",
    rating_base = "workload",
    moderator_col = "stress_level"
  )
)

all_res <- bind_rows(res_stress_by_workload, res_workload_by_stress)

# FDR correction within analysis and modality family
all_res <- all_res %>%
  group_by(analysis, modality) %>%
  mutate(
    interaction_q = p.adjust(interaction_p, method = "BH"),
    interaction_q_boot = if_else(!is.na(interaction_p_boot), p.adjust(interaction_p_boot, method = "BH"), NA_real_)
  ) %>%
  ungroup()

write_csv(all_res, file.path(out_dir, "masking_moderation_summary.csv"))
saveRDS(all_res, file.path(out_dir, "masking_moderation_summary.rds"))

# Separate diagnostics table for quick filtering
all_diag <- all_res %>%
  select(analysis, feature, modality, n_participants, n_obs, model_type, converged, singular,
         shapiro_p, hetero_rho, hetero_p, interaction_p, interaction_q, interaction_p_boot, interaction_q_boot)

write_csv(all_diag, file.path(out_dir, "masking_moderation_diagnostics.csv"))

msg("✓ Saved masking moderation outputs to: ", out_dir)

# Print a compact console summary of interaction terms
pretty_labels <- unlist(config$pretty_features)

summary_print <- all_res %>%
  mutate(feature_base = sub(paste0(suffix, "$"), "", feature)) %>%
  mutate(feature_pretty = if_else(feature_base %in% names(pretty_labels), pretty_labels[feature_base], feature_base)) %>%
  arrange(analysis, modality, interaction_p) %>%
  select(analysis, modality, feature_pretty, interaction_est, interaction_p, interaction_q, interaction_p_boot, interaction_q_boot,
         slope_low, slope_high, model_type, converged, singular)

msg("[Masking LMM] Top interactions (by raw p)")
print(head(summary_print, 12))
