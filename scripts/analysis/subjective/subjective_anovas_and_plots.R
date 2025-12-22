suppressPackageStartupMessages({
  library(ggplot2)
})

infile <- file.path("output", "final_data.csv")
if (!file.exists(infile)) {
  stop("Missing input: ", infile)
}

out_dir <- file.path("results", "classic_analyses", "subjective")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

# Optional packages (installed on-demand)
need_pkg <- function(pkg) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    message("Installing missing R package: ", pkg)
    install.packages(pkg, repos = "https://cloud.r-project.org")
  }
}

need_pkg("dplyr")
need_pkg("tidyr")
need_pkg("readr")
need_pkg("afex")
need_pkg("patchwork")

suppressPackageStartupMessages({
  library(dplyr)
  library(tidyr)
  library(readr)
  library(afex)
  library(patchwork)
})

set.seed(1)
afex::afex_options(type = 3, method_mixed = "S")

raw_df <- readr::read_csv(infile, show_col_types = FALSE)

required_cols <- c("participant_id", "condition", "stress", "workload")
missing_cols <- setdiff(required_cols, names(raw_df))
if (length(missing_cols) > 0) {
  stop("Missing required columns in final_data.csv: ", paste(missing_cols, collapse = ", "))
}

# SUBJECTIVE ANOVA: Use FULL dataset (all 47 participants)
# Rationale: Stress/workload ratings don't depend on EEG quality
message("[Subjective ANOVA] Using full dataset (N=", n_distinct(raw_df$participant_id), " participants, includes EEG QC failures)")

# Derive within-subject factors from condition label (consistent with pipeline naming)
df <- raw_df %>%
  mutate(
    participant_id = factor(participant_id),
    stress_level = factor(ifelse(grepl("^High Stress", condition), "High", "Low"), levels = c("Low", "High")),
    mwl_level = factor(ifelse(grepl("High Cog$", condition), "High", "Low"), levels = c("Low", "High")),
    stress_rating = as.numeric(stress),
    mental_demand = as.numeric(workload)
  )

# Sanity checks
n_participants <- n_distinct(df$participant_id)
if (n_participants < 2) stop("Too few participants in final_data.csv")

# --- 2x2 repeated-measures ANOVAs (afex provides generalized eta squared, 'ges') ---
a_stress <- afex::aov_ez(
  id = "participant_id",
  dv = "stress_rating",
  within = c("stress_level", "mwl_level"),
  data = df,
  anova_table = list(es = "ges")
)

a_demand <- afex::aov_ez(
  id = "participant_id",
  dv = "mental_demand",
  within = c("stress_level", "mwl_level"),
  data = df,
  anova_table = list(es = "ges")
)

anova_stress_tbl <- as.data.frame(a_stress$anova_table) %>%
  tibble::rownames_to_column("effect") %>%
  mutate(dv = "stress_rating")

anova_demand_tbl <- as.data.frame(a_demand$anova_table) %>%
  tibble::rownames_to_column("effect") %>%
  mutate(dv = "mental_demand")

# Normalize column names for stable CSV output across afex versions
anova_tbl <- bind_rows(anova_stress_tbl, anova_demand_tbl) %>%
  rename_with(~ make.names(.x))

readr::write_csv(anova_tbl, file.path(out_dir, "subjective_anova_results.csv"))

# --- Collapsed cell summaries for manuscript table and plot CIs ---
summarize_mean_ci <- function(x) {
  x <- x[is.finite(x)]
  n <- length(x)
  m <- mean(x)
  s <- stats::sd(x)
  se <- s / sqrt(n)
  tcrit <- stats::qt(0.975, df = n - 1)
  dplyr::tibble(
    mean = m,
    sd = s,
    n = n,
    ci_low = m - tcrit * se,
    ci_high = m + tcrit * se
  )
}

# Collapse within-participant first, then summarize across participants
collapsed_stress_by_stress <- df %>%
  group_by(participant_id, stress_level) %>%
  summarize(stress_rating = mean(stress_rating, na.rm = TRUE), .groups = "drop")

collapsed_demand_by_mwl <- df %>%
  group_by(participant_id, mwl_level) %>%
  summarize(mental_demand = mean(mental_demand, na.rm = TRUE), .groups = "drop")

stress_level_summary <- collapsed_stress_by_stress %>%
  group_by(stress_level) %>%
  group_modify(~ summarize_mean_ci(.x$stress_rating)) %>%
  ungroup() %>%
  mutate(measure = "Subjective stress")

mwl_level_summary <- collapsed_demand_by_mwl %>%
  group_by(mwl_level) %>%
  group_modify(~ summarize_mean_ci(.x$mental_demand)) %>%
  ungroup() %>%
  mutate(measure = "Mental demand (NASA-TLX)")

readr::write_csv(stress_level_summary, file.path(out_dir, "subjective_stress_by_stresslevel_summary.csv"))
readr::write_csv(mwl_level_summary, file.path(out_dir, "subjective_demand_by_mwllevel_summary.csv"))

# --- Paired t-tests + dz for narrative ---
paired_t_and_dz <- function(wide_df, high_col, low_col) {
  d <- wide_df[[high_col]] - wide_df[[low_col]]
  d <- d[is.finite(d)]
  tt <- stats::t.test(d, mu = 0)
  dz <- mean(d) / stats::sd(d)
  list(
    mean_diff = mean(d),
    t = unname(tt$statistic),
    df = unname(tt$parameter),
    p = unname(tt$p.value),
    dz = dz
  )
}

stress_wide <- collapsed_stress_by_stress %>%
  tidyr::pivot_wider(names_from = stress_level, values_from = stress_rating)

demand_wide <- collapsed_demand_by_mwl %>%
  tidyr::pivot_wider(names_from = mwl_level, values_from = mental_demand)

stress_tt <- paired_t_and_dz(stress_wide, high_col = "High", low_col = "Low")
demand_tt <- paired_t_and_dz(demand_wide, high_col = "High", low_col = "Low")

narrative_tbl <- tibble::tibble(
  contrast = c("stress_high_minus_low", "mwl_high_minus_low"),
  mean_diff = c(stress_tt$mean_diff, demand_tt$mean_diff),
  t = c(stress_tt$t, demand_tt$t),
  df = c(stress_tt$df, demand_tt$df),
  p = c(stress_tt$p, demand_tt$p),
  dz = c(stress_tt$dz, demand_tt$dz)
)
readr::write_csv(narrative_tbl, file.path(out_dir, "subjective_paired_ttests.csv"))

# --- Stress–MWL rating correlations (congruent vs incongruent) ---
congruent <- c("High Stress - High Cog", "Low Stress - Low Cog")
incongruent <- c("High Stress - Low Cog", "Low Stress - High Cog")

corr_stats <- function(d) {
  d <- d %>% filter(is.finite(stress_rating), is.finite(mental_demand))
  ct <- stats::cor.test(d$stress_rating, d$mental_demand, method = "pearson")
  list(r = unname(ct$estimate), p = unname(ct$p.value), n = nrow(d))
}

c1 <- corr_stats(df %>% filter(condition %in% congruent))
c2 <- corr_stats(df %>% filter(condition %in% incongruent))

# Fisher z test for difference between independent correlations
z1 <- atanh(c1$r)
z2 <- atanh(c2$r)
z <- (z1 - z2) / sqrt(1 / (c1$n - 3) + 1 / (c2$n - 3))
p_z <- 2 * (1 - stats::pnorm(abs(z)))

corr_tbl <- tibble::tibble(
  subset = c("congruent", "incongruent", "difference"),
  r = c(c1$r, c2$r, NA_real_),
  p = c(c1$p, c2$p, p_z),
  n = c(c1$n, c2$n, NA_integer_),
  fisher_z = c(NA_real_, NA_real_, z)
)
readr::write_csv(corr_tbl, file.path(out_dir, "stress_workload_rating_correlations.csv"))

# --- Interaction plots (single PNG with two panels) ---
# Means/CI per condition with between-subject CI (for readability; consistent with manuscript caption).
cond_summary <- df %>%
  group_by(stress_level, mwl_level) %>%
  summarize(
    stress_mean = mean(stress_rating, na.rm = TRUE),
    stress_sd = sd(stress_rating, na.rm = TRUE),
    stress_n = sum(is.finite(stress_rating)),
    demand_mean = mean(mental_demand, na.rm = TRUE),
    demand_sd = sd(mental_demand, na.rm = TRUE),
    demand_n = sum(is.finite(mental_demand)),
    .groups = "drop"
  ) %>%
  mutate(
    stress_se = stress_sd / sqrt(stress_n),
    demand_se = demand_sd / sqrt(demand_n),
    stress_ci = stats::qt(0.975, df = stress_n - 1) * stress_se,
    demand_ci = stats::qt(0.975, df = demand_n - 1) * demand_se
  )

p1 <- ggplot(cond_summary, aes(x = stress_level, y = stress_mean, group = mwl_level, color = mwl_level)) +
  geom_line(linewidth = 0.8) +
  geom_point(size = 2) +
  geom_errorbar(aes(ymin = stress_mean - stress_ci, ymax = stress_mean + stress_ci), width = 0.1, linewidth = 0.6) +
  labs(x = "Stress condition", y = "Subjective stress rating", color = "MWL") +
  theme_minimal(base_size = 11) +
  theme(legend.position = "bottom")

p2 <- ggplot(cond_summary, aes(x = mwl_level, y = demand_mean, group = stress_level, color = stress_level)) +
  geom_line(linewidth = 0.8) +
  geom_point(size = 2) +
  geom_errorbar(aes(ymin = demand_mean - demand_ci, ymax = demand_mean + demand_ci), width = 0.1, linewidth = 0.6) +
  labs(x = "MWL condition", y = "NASA-TLX mental demand", color = "Stress") +
  theme_minimal(base_size = 11) +
  theme(legend.position = "bottom")

combined <- p1 + p2 + patchwork::plot_layout(ncol = 2)

ggsave(filename = "interaction_plots.png", plot = combined, width = 10.5, height = 4.2, dpi = 300)

# Session info for reproducibility
sink(file.path(out_dir, "sessionInfo.txt"))
print(sessionInfo())
sink()

message("Done. Wrote outputs to: ", out_dir)
message("Updated plot: interaction_plots.png")
