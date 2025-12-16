###############################################################################
# anova_main.R — Baseline-adjusted 2×2 RM ANOVAs on selected features
###############################################################################

suppressPackageStartupMessages({
  library(tidyverse)
  library(afex)
  library(emmeans)
  library(effectsize)
  library(ARTool)
  library(yaml)
  library(performance)
  library(stringr)
  library(ggplot2)
})

afex::afex_options(type = 3)

# ------------------------------------------------------------
# Load config + data
# ------------------------------------------------------------
config <- yaml::read_yaml("scripts/utils/config.yaml")
out_dir_classic <- file.path(config$paths$results, "classic_analyses")
dir.create(out_dir_classic, showWarnings = FALSE, recursive = TRUE)

df <- readRDS(
  "D:/phd_projects/vr_tsst_2025/output/anova_features_precond.rds"
)

df <- df %>%
  mutate(
    participant_id = factor(participant_id),
    stress_level   = factor(stress_level, levels = c("Low", "High")),
    workload_level = factor(workload_level, levels = c("Low", "High"))
  )


stopifnot(all(c("participant_id", "stress_level", "workload_level") %in% names(df)))

# ------------------------------------------------------------
# Feature list 
# ------------------------------------------------------------

canonical_feats <- config$canonical_features
suffix <- "_full_change_precond"  #baseline adjusted using precondition relaxation scene
features <- paste0(canonical_feats, suffix)

feat_labels <- config$pretty_features
# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
sig_marks <- function(p) {
  case_when(
    p < 0.001 ~ "***",
    p < 0.01  ~ "**",
    p < 0.05  ~ "*",
    p < 0.10  ~ "†",
    TRUE      ~ ""
  )
}

############

normality_results <- map_dfr(features, function(feat) {
  
  m <- afex::aov_car(
    as.formula(paste0(feat, " ~ stress_level * workload_level + Error(participant_id/(stress_level * workload_level))")),
    data = df,
    factorize = FALSE
  )
  
  norm_check <- check_normality(m)
  
  # Save QQ plot
  p <- plot(norm_check, type = "qq", detrend = TRUE)
  ggsave(
    filename = file.path(out_dir_classic, "anovas", paste0("qqplot_", feat, ".png")),
    plot = p,
    width = 6, height = 6, dpi = 300
  )
  
  
  # Extract p-value
  p_value <- as.numeric(norm_check)
  
  
  tibble(
    feature = feat,
    p_value = p_value
  )
})

print(normality_results)


# Term labels in correct order
term_labels <- c("Stress", "Workload", "Stress × Workload")

art_results <- map_dfr(features, function(feat) {
  
  dat <- df %>%
    select(participant_id, stress_level, workload_level, all_of(feat)) %>%
    drop_na()
  
  if (nrow(dat) == 0) return(NULL)
  
  m_art <- art(
    as.formula(
      paste0(feat, " ~ stress_level * workload_level + Error(participant_id)")
    ),
    data = dat
  )
  
  # verbose ANOVA so we get SS
  a <- anova(m_art, verbose = TRUE)
  
  
  # keep only the within-subject tests (error term "Withn")
  a_clean <- a[a$Error == "Within", ]

  
  # compute partial eta-squared
  eta_sq_part <- a_clean$`Sum Sq` / (a_clean$`Sum Sq` + a_clean$`Sum Sq.res`)
  
  tibble(
    feature      = feat,
    term         = a_clean$Term,
    F            = a_clean$F,
    p            = a_clean$`Pr(>F)`,
    eta_sq_part  = eta_sq_part,
    sig          = sig_marks(a_clean$`Pr(>F)`)
  )
})







# Re-fit only for the significant feature for clarity
feat <- "gsr_skin_conductance_eda_tonic_mean_nk_full_change_precond"

dat_feat <- df %>%
  select(participant_id, stress_level, workload_level, !!sym(feat)) %>%
  drop_na()

# Fit ART model
m_art <- art(as.formula(paste0(
  feat, " ~ stress_level * workload_level + Error(participant_id)"
)), data = dat_feat)

# Run ANOVA-style tests for the terms again if needed
anova(m_art)

# Post-hoc contrasts ONLY on the significant factor (stress_level)
ph_stress <- art.con(m_art, "stress_level")

print(ph_stress)



# Summary data
summary_df <- df %>%
  select(participant_id, stress_level, workload_level, !!sym(feat)) %>%
  drop_na() %>%
  group_by(stress_level, workload_level) %>%
  summarise(
    mean_val = mean(.data[[feat]], na.rm = TRUE),
    se_val = sd(.data[[feat]], na.rm = TRUE) / sqrt(n()),
    .groups = "drop"
  )

ggplot(summary_df,
       aes(x = stress_level,
           y = mean_val,
           group = workload_level,
           colour = workload_level)) +
  geom_line(size = 1) +
  geom_point(size = 3) +
  geom_errorbar(aes(ymin = mean_val - se_val, ymax = mean_val + se_val),
                width = 0.08, linewidth = 0.6) +
  labs(
    title = "Tonic Skin Conductance: Stress × Workload",
    x = "Stress Level",
    y = "Baseline-adjusted Δ Tonic EDA (µS)",
    colour = "Workload Level"
  ) +
  theme_minimal(base_size = 14) +
  theme(
    legend.position = "right",
    panel.grid.minor = element_blank()
  )


# ------------------------------------------------------------
# Saving logic — classic ANOVAs
# ------------------------------------------------------------

save_dir <- file.path(out_dir_classic, "anovas")
dir.create(save_dir, showWarnings = FALSE, recursive = TRUE)

# ---- 1. Save normality results ----
write_csv(
  normality_results,
  file.path(save_dir, "normality_results.csv")
)

# ---- 2. Save ART ANOVA results ----
write_csv(
  art_results,
  file.path(save_dir, "art_anova_results.csv")
)

# ---- 3. Save post-hoc results for the significant feature ----
ph_df <- as.data.frame(ph_stress)
write_csv(
  ph_df,
  file.path(save_dir, paste0("posthoc_", feat, "_stress.csv"))
)

# ---- 4. Save summary data used for plotting ----
write_csv(
  summary_df,
  file.path(save_dir, paste0("summary_", feat, ".csv"))
)

# ---- 5. Save summary plot ----
ggsave(
  filename = file.path(save_dir, paste0("plot_", feat, ".png")),
  plot = last_plot(),
  width = 7,
  height = 5,
  dpi = 300
)

message("ANOVA outputs saved to: ", save_dir)


