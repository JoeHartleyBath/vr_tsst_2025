# =====================================================================
# mixed_moderation_lmm.R
# Mixed-effects moderation models as an alternative to stratified rmcor.
# Tests whether the feature–rating association differs by the other factor
# without splitting the data.
#
# Models (random intercept per participant):
#   stress   ~ feature * workload_level + (1|participant_id)
#   workload ~ feature * stress_level   + (1|participant_id)
#
# Output:
#   results/classic_analyses/mixed_moderation/
#     lmm_moderation_stress_by_workload.csv
#     lmm_moderation_workload_by_stress.csv
#     heatmap_interaction_stress_by_workload.png
#     heatmap_interaction_workload_by_stress.png
#     sessionInfo.txt
# =====================================================================

suppressPackageStartupMessages({
  library(tidyverse)
  library(yaml)
  library(ggplot2)

  if (!requireNamespace("lme4", quietly = TRUE)) {
    stop("Package 'lme4' is required. Install with install.packages('lme4').")
  }
  if (!requireNamespace("lmerTest", quietly = TRUE)) {
    stop("Package 'lmerTest' is required. Install with install.packages('lmerTest').")
  }
  if (!requireNamespace("broom.mixed", quietly = TRUE)) {
    stop("Package 'broom.mixed' is required. Install with install.packages('broom.mixed').")
  }
})

set.seed(1)

# =====================================================================
# 0) CONFIG + HELPERS
# =====================================================================

config <- yaml::read_yaml("scripts/utils/config.yaml")

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

signif_star <- function(p) {
  dplyr::case_when(
    is.na(p)  ~ "",
    p < 0.001 ~ "***",
    p < 0.01  ~ "**",
    p < 0.05  ~ "*",
    TRUE      ~ ""
  )
}

# =====================================================================
# 1) LOAD + PREP DATA
# =====================================================================

final_data <- load_obj("final_data")

df <- final_data %>%
  filter(!is.na(stress), !is.na(workload)) %>%
  mutate(
    stress_level   = if_else(condition %in% c("High Stress - High Cog", "High Stress - Low Cog"),
                             "High", "Low"),
    workload_level = if_else(condition %in% c("High Stress - High Cog", "Low Stress - High Cog"),
                             "High", "Low"),
    stress_level   = factor(stress_level, levels = c("Low", "High")),
    workload_level = factor(workload_level, levels = c("Low", "High"))
  )

canonical_feats <- config$canonical_features
suffix <- "_precond_Z"
features <- paste0(canonical_feats, suffix)

# Pretty label mapping (normalize config keys: *_precond[_Z] -> base)
feat_labels <- unlist(config$pretty_features)
if (!is.null(names(feat_labels))) {
  names(feat_labels) <- names(feat_labels) %>%
    sub("_precond_Z$", "", .) %>%
    sub("_precond$", "", .)
}

feature_display <- function(feature_col) {
  base <- sub(paste0(suffix, "$"), "", feature_col)
  lbl <- feat_labels[[base]]
  if (is.null(lbl) || is.na(lbl) || lbl == "") base else lbl
}

# Validate required columns exist up-front
required_cols <- c("participant_id", "condition", "stress", "workload", "stress_level", "workload_level", features)
missing_cols <- setdiff(required_cols, names(df))
if (length(missing_cols) > 0) {
  stop("Missing required columns in final_data: ", paste(missing_cols, collapse = ", "))
}

out_dir <- file.path(config$paths$results, "classic_analyses", "mixed_moderation")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

# =====================================================================
# 2) MODEL FITTING
# =====================================================================

fit_one_feature <- function(data, outcome, moderator, feature_col) {
  # outcome: "stress" or "workload"
  # moderator: "workload_level" or "stress_level"

  tmp <- data %>%
    select(participant_id, all_of(outcome), all_of(moderator), all_of(feature_col)) %>%
    drop_na()

  n_subjects <- n_distinct(tmp$participant_id)
  n_obs <- nrow(tmp)

  if (n_subjects < 5 || n_obs < 10) {
    return(tibble(
      outcome = outcome,
      moderator = moderator,
      feature = feature_col,
      feature_display = feature_display(feature_col),
      n_subjects = n_subjects,
      n_obs = n_obs,
      beta_feature = NA_real_,
      beta_interaction = NA_real_,
      se_feature = NA_real_,
      se_interaction = NA_real_,
      t_feature = NA_real_,
      t_interaction = NA_real_,
      p_feature = NA_real_,
      p_interaction = NA_real_,
      slope_low = NA_real_,
      slope_high = NA_real_,
      se_slope_high = NA_real_,
      t_slope_high = NA_real_,
      p_slope_high = NA_real_
    ))
  }

  # Model: outcome ~ feature * moderator + (1|participant_id)
  fml <- as.formula(paste0(outcome, " ~ ", feature_col, " * ", moderator, " + (1|participant_id)"))
  m <- lmerTest::lmer(fml, data = tmp, REML = TRUE)

  fx <- broom.mixed::tidy(m, effects = "fixed") %>%
    select(term, estimate, std.error, statistic, p.value)

  # Under treatment coding, interaction term will include moderatorHigh
  int_term_1 <- paste0(feature_col, ":", moderator, "High")
  int_term_2 <- paste0(moderator, "High:", feature_col)

  row_feature <- fx %>% filter(term == feature_col)
  row_int <- fx %>% filter(term %in% c(int_term_1, int_term_2))

  if (nrow(row_feature) != 1 || nrow(row_int) != 1) {
    return(tibble(
      outcome = outcome,
      moderator = moderator,
      feature = feature_col,
      feature_display = feature_display(feature_col),
      n_subjects = n_subjects,
      n_obs = n_obs,
      beta_feature = NA_real_,
      beta_interaction = NA_real_,
      se_feature = NA_real_,
      se_interaction = NA_real_,
      t_feature = NA_real_,
      t_interaction = NA_real_,
      p_feature = NA_real_,
      p_interaction = NA_real_,
      slope_low = NA_real_,
      slope_high = NA_real_,
      se_slope_high = NA_real_,
      t_slope_high = NA_real_,
      p_slope_high = NA_real_
    ))
  }

  b_f <- row_feature$estimate
  se_f <- row_feature$std.error
  t_f <- row_feature$statistic
  p_f <- row_feature$p.value

  b_i <- row_int$estimate
  se_i <- row_int$std.error
  t_i <- row_int$statistic
  p_i <- row_int$p.value

  # Simple slopes for the feature effect:
  # Low moderator (reference): slope_low = b_f
  # High moderator: slope_high = b_f + b_i
  vc <- as.matrix(stats::vcov(m))
  v_f <- vc[feature_col, feature_col]
  v_i <- vc[row_int$term, row_int$term]
  cov_fi <- vc[feature_col, row_int$term]

  slope_low <- b_f
  slope_high <- b_f + b_i
  se_high <- sqrt(v_f + v_i + 2 * cov_fi)

  t_high <- slope_high / se_high
  # Approximate p for simple slope (normal approx)
  p_high <- 2 * stats::pnorm(abs(t_high), lower.tail = FALSE)

  tibble(
    outcome = outcome,
    moderator = moderator,
    feature = feature_col,
    feature_display = feature_display(feature_col),
    n_subjects = n_subjects,
    n_obs = n_obs,
    beta_feature = b_f,
    beta_interaction = b_i,
    se_feature = se_f,
    se_interaction = se_i,
    t_feature = t_f,
    t_interaction = t_i,
    p_feature = p_f,
    p_interaction = p_i,
    slope_low = slope_low,
    slope_high = slope_high,
    se_slope_high = se_high,
    t_slope_high = t_high,
    p_slope_high = p_high
  )
}

run_family <- function(outcome, moderator) {
  res <- purrr::map_dfr(features, ~ fit_one_feature(df, outcome, moderator, .x))

  res %>%
    mutate(
      p_interaction_fdr = p.adjust(p_interaction, method = "BH"),
      sig = signif_star(p_interaction_fdr)
    ) %>%
    arrange(p_interaction_fdr)
}

res_stress_by_workload <- run_family(outcome = "stress", moderator = "workload_level")
res_workload_by_stress <- run_family(outcome = "workload", moderator = "stress_level")

write_csv(res_stress_by_workload, file.path(out_dir, "lmm_moderation_stress_by_workload.csv"))
write_csv(res_workload_by_stress, file.path(out_dir, "lmm_moderation_workload_by_stress.csv"))

# =====================================================================
# 3) PLOTS (interaction term as heatmap)
# =====================================================================

plot_heat <- function(res, title, out_file) {
  p <- ggplot(
    res %>% mutate(feature_display = fct_reorder(feature_display, beta_interaction)),
    aes(x = 1, y = feature_display, fill = beta_interaction)
  ) +
    geom_tile(colour = "grey20", linewidth = 0.25) +
    geom_text(
      aes(label = ifelse(is.na(beta_interaction), "", sprintf("%.2f%s", beta_interaction, sig))),
      size = 5.0,
      colour = "black"
    ) +
    scale_fill_distiller(palette = "RdYlBu", direction = -1, name = "Interaction\n(beta)") +
    scale_x_continuous(breaks = NULL) +
    labs(title = title, x = NULL, y = NULL) +
    theme_minimal(base_size = 18) +
    theme(
      axis.text.y = element_text(size = 13),
      plot.title = element_text(size = 18, face = "bold"),
      legend.title = element_text(size = 14),
      legend.text = element_text(size = 12),
      panel.grid = element_blank()
    )

  ggsave(file.path(out_dir, out_file), p, width = 8.5, height = 10, dpi = 300)
  p
}

plot_heat(
  res_stress_by_workload,
  "Moderation of feature–stress association by workload level\nInteraction: feature × workload_level (BH-FDR across features)",
  "heatmap_interaction_stress_by_workload.png"
)

plot_heat(
  res_workload_by_stress,
  "Moderation of feature–workload association by stress level\nInteraction: feature × stress_level (BH-FDR across features)",
  "heatmap_interaction_workload_by_stress.png"
)

writeLines(capture.output(sessionInfo()), file.path(out_dir, "sessionInfo.txt"))

cat("\n✓ Saved mixed moderation results to:\n", out_dir, "\n\n")
cat("Top interaction results (stress by workload):\n")
print(
  res_stress_by_workload %>%
    select(feature, feature_display, beta_interaction, p_interaction, p_interaction_fdr) %>%
    head(10)
)
cat("\nTop interaction results (workload by stress):\n")
print(
  res_workload_by_stress %>%
    select(feature, feature_display, beta_interaction, p_interaction, p_interaction_fdr) %>%
    head(10)
)
