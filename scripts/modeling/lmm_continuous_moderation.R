# =====================================================================
# lmm_continuous_moderation.R
# Purpose: Test whether continuous stress moderates workload-feature associations using LMM.
# Hypothesis: Workload-feature slope weakens as stress increases.
# Model: feature ~ workload_w * stress_w + workload_b + stress_b + (1 | participant_id)
#
# Inputs: output/final_data.csv
# Outputs: results/classic_analyses/continuous_moderation/
# =====================================================================

suppressPackageStartupMessages({
  library(dplyr)
  library(readr)
  library(tidyr)
  library(yaml)
  library(purrr)
  library(stringr)
  library(lme4)
  library(lmerTest)
  library(tibble)
})

# --------------------------
# Helpers
# --------------------------

zscore_safe <- function(x) {
  x <- as.numeric(x)
  mu <- mean(x, na.rm = TRUE)
  s <- stats::sd(x, na.rm = TRUE)
  if (!is.finite(s) || s == 0) return(x * NA_real_)
  (x - mu) / s
}

center_safe <- function(x) {
  x <- as.numeric(x)
  mu <- mean(x, na.rm = TRUE)
  if (!is.finite(mu)) return(x * NA_real_)
  x - mu
}

# Robust LMM fitter handling singular models by falling back to Fixed Effects
fit_lmer_or_felm <- function(dat, formula, id_col = "participant_id") {
  # Returns a list with model_type and coefficient table for fixed effects.
  out <- list(model_type = NA_character_, converged = NA, singular = NA, coef = NULL)

  # Need enough participants for either model to be meaningful
  if (dplyr::n_distinct(dat[[id_col]]) < 2) {
    out$model_type <- "insufficient_participants"
    out$converged <- NA
    out$singular <- NA
    out$coef <- tibble::tibble(
      term = c("(Intercept)", "workload_w", "stress_w", "workload_w:stress_w"),
      estimate = NA_real_,
      std_error = NA_real_,
      df = NA_real_,
      t_value = NA_real_,
      p_value = NA_real_
    )
    return(out)
  }

  fit <- tryCatch(
    lmerTest::lmer(formula, data = dat, REML = TRUE,
                   control = lme4::lmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 2e5))),
    error = function(e) NULL
  )

  if (!is.null(fit)) {
    out$model_type <- "lmm"
    out$converged <- NA
    out$singular <- lme4::isSingular(fit, tol = 1e-4)

    if (!out$singular) {
      sm <- summary(fit)
      co <- as.data.frame(sm$coefficients)
      co$term <- rownames(co)
      rownames(co) <- NULL
      out$coef <- co %>%
        rename(
          estimate = Estimate,
          std_error = `Std. Error`,
          df = df,
          t_value = `t value`,
          p_value = `Pr(>|t|)`
        ) %>%
        select(term, estimate, std_error, df, t_value, p_value)
      return(out)
    }
  }

  # Fallback: participant fixed-effects linear model
  out$model_type <- "fe_lm"
  out$converged <- NA
  out$singular <- NA

  # Strip random-effects terms (e.g., (1|participant_id)) before passing to lm()
  fixed_formula <- lme4::nobars(formula)
  fixed_formula_str <- paste(deparse(fixed_formula), collapse = " ")
  fe_formula <- stats::as.formula(paste0(fixed_formula_str, " + factor(", id_col, ")"))
  
  fit2 <- tryCatch(stats::lm(fe_formula, data = dat), error = function(e) NULL)
  
  if (is.null(fit2)) return(out)

  sm2 <- summary(fit2)
  co2 <- as.data.frame(sm2$coefficients)
  co2$term <- rownames(co2)
  rownames(co2) <- NULL
  
  # Only keep relevant terms (not the participant factors)
  out$coef <- co2 %>%
    mutate(term = as.character(term)) %>%
    rename(
      estimate = Estimate,
      std_error = `Std. Error`,
      t_value = `t value`,
      p_value = `Pr(>|t|)`
    ) %>%
    mutate(df = unname(sm2$df[2])) %>%
    select(term, estimate, std_error, df, t_value, p_value) %>%
    filter(!grepl("^factor\\(", term))

  out
}

# --------------------------
# Load Data & Config
# --------------------------
config <- yaml::read_yaml("scripts/utils/config.yaml")
infile <- file.path("output", "final_data.csv")
out_dir <- file.path("results", "classic_analyses", "continuous_moderation")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

if (!file.exists(infile)) stop("Missing input: ", infile)
raw_df <- readr::read_csv(infile, show_col_types = FALSE)

# Preprocessing
df_all <- raw_df %>%
  mutate(
    participant_id = factor(participant_id),
    stress_level = factor(ifelse(grepl("^High Stress", condition), "High", "Low"), levels = c("Low", "High")),
    mwl_level = factor(ifelse(grepl("High Cog$", condition), "High", "Low"), levels = c("Low", "High")),
    stress_rating = as.numeric(stress),
    workload_rating = as.numeric(workload)
  ) %>%
  filter(is.finite(stress_rating), is.finite(workload_rating))

# Select features (physio only typically, but let's use all canonical non-eeg for now)
canonical_feats <- config$canonical_features
# Exclude EEG if desired, matching original script behavior
canonical_physio <- canonical_feats[!str_starts(canonical_feats, "eeg_")]
feature_cols <- paste0(canonical_physio, "_precond")
feature_cols <- feature_cols[feature_cols %in% names(df_all)]

if (length(feature_cols) == 0) message("Warning: No feature columns found.")

# --------------------------
# Analysis Loop
# --------------------------

prep_for_model <- function(df_in, feature_col) {
  df_in %>%
    select(participant_id, stress_level, mwl_level, stress_rating, workload_rating, y = all_of(feature_col)) %>%
    filter(is.finite(y)) %>%
    group_by(participant_id) %>%
    mutate(
      stress_w = zscore_safe(stress_rating),
      workload_w = zscore_safe(workload_rating),
      stress_b = mean(stress_rating, na.rm = TRUE),
      workload_b = mean(workload_rating, na.rm = TRUE),
      y_z = zscore_safe(y)
    ) %>%
    ungroup() %>%
    filter(is.finite(stress_w), is.finite(workload_w), is.finite(y_z), is.finite(stress_b), is.finite(workload_b))
}

run_models_for_feature <- function(feature_col) {
  # Full dataset
  d_all <- prep_for_model(df_all, feature_col)
  
  # Also run on Low-Stress subset if desired?
  # The original script did "All Conditions" and "Low Stress Only".
  # We will preserve this.
  d_low <- df_all %>% filter(stress_level == "Low") %>% prep_for_model(feature_col)

  fml <- y_z ~ workload_w * stress_w + workload_b + stress_b + (1 | participant_id)

  fit_all <- fit_lmer_or_felm(d_all, fml)
  fit_low <- fit_lmer_or_felm(d_low, fml)

  extract_terms <- function(fit_obj, tag) {
    if (is.null(fit_obj$coef)) return(NULL)
    fit_obj$coef %>%
      filter(term %in% c("(Intercept)", "workload_w", "stress_w", "workload_w:stress_w")) %>%
      mutate(
        dataset = tag,
        model_type = fit_obj$model_type,
        singular = fit_obj$singular
      )
  }

  bind_rows(
    extract_terms(fit_all, "all_conditions"),
    extract_terms(fit_low, "low_stress_only")
  ) %>%
    mutate(feature = feature_col)
}

message("Running LMMs for ", length(feature_cols), " features...")
model_coef_tbl <- purrr::map_dfr(feature_cols, run_models_for_feature)

# FDR Correction on interaction term
if (nrow(model_coef_tbl) > 0) {
  model_coef_tbl <- model_coef_tbl %>%
    group_by(dataset) %>%
    mutate(
      p_fdr = if_else(term == "workload_w:stress_w", p.adjust(p_value, method = "BH"), NA_real_)
    ) %>%
    ungroup()

  out_coef <- file.path(out_dir, "continuous_moderation_models_coefficients.csv")
  readr::write_csv(model_coef_tbl, out_coef)
  
  # Summary of interactions
  interaction_tbl <- model_coef_tbl %>%
    filter(term == "workload_w:stress_w") %>%
    select(dataset, feature, estimate, std_error, df, t_value, p_value, p_fdr, model_type, singular)
  
  out_int <- file.path(out_dir, "continuous_moderation_interactions_summary.csv")
  readr::write_csv(interaction_tbl, out_int)
  
  message("✓ Results saved to ", out_dir)
} else {
  message("! No valid models converged.")
}
