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

df_full <- readRDS(
  file.path(config$paths$output, "anova_features_precond.rds")
)

df_full <- df_full %>%
  mutate(
    participant_id = factor(participant_id),
    stress_level   = factor(stress_level, levels = c("Low", "High")),
    workload_level = factor(workload_level, levels = c("Low", "High"))
  )

stopifnot(all(c("participant_id", "stress_level", "workload_level") %in% names(df_full)))

# ------------------------------------------------------------
# Feature list & categorization
# ------------------------------------------------------------

canonical_feats <- config$canonical_features
suffix <- "_precond"  # baseline adjusted using precondition relaxation scene
features <- paste0(canonical_feats, suffix)

# Identify EEG vs Physiological features
eeg_features <- features[str_detect(features, "^eeg_")]
phys_features <- setdiff(features, eeg_features)

cat(sprintf("Using %d canonical features from config\n", length(features)))
cat(sprintf("  - %d EEG features (will use N=44 QC-valid)\n", length(eeg_features)))
cat(sprintf("  - %d Physiological features (will use N=47 full)\n", length(phys_features)))

# Check QC column
has_qc_flag <- "qc_failed" %in% names(df_full)
if (has_qc_flag) {
  n_full <- n_distinct(df_full$participant_id)
  n_qc_valid <- n_distinct(df_full %>% filter(!qc_failed) %>% pull(participant_id))
  message(sprintf("[ANOVA Setup] Full dataset N=%d, EEG QC-valid N=%d", n_full, n_qc_valid))
}

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
  
  # Selective filtering: EEG features use QC-valid subset, phys features use full dataset
  df <- if (has_qc_flag && feat %in% eeg_features) {
    df_full %>% filter(!qc_failed)
  } else {
    df_full
  }
  
  # Filter for participants with all 4 conditions
  complete_participants <- df %>%
    select(participant_id, stress_level, workload_level, all_of(feat)) %>%
    drop_na() %>%
    group_by(participant_id) %>%
    filter(n() == 4) %>%  # Must have all 4 condition combinations
    pull(participant_id) %>%
    unique()
  
  if (length(complete_participants) < 10) {
    cat(sprintf("Skipping %s: only %d participants with complete data\n", 
                feat, length(complete_participants)))
    return(NULL)
  }
  
  dat_feat <- df %>%
    filter(participant_id %in% complete_participants)
  
  feat_type <- if (feat %in% eeg_features) "EEG" else "Phys"
  cat(sprintf("Analyzing %s [%s]: %d participants with complete data\n", 
              feat, feat_type, length(complete_participants)))
  
  m <- afex::aov_car(
    as.formula(paste0(feat, " ~ stress_level * workload_level + Error(participant_id/(stress_level * workload_level))")),
    data = dat_feat,
    factorize = FALSE
  )
  
  norm_check <- tryCatch(
    check_normality(m),
    error = function(e) {
      cat(sprintf("  Warning: Normality check failed for %s: %s\n", feat, e$message))
      return(NULL)
    }
  )
  
  if (!is.null(norm_check)) {
    # Save QQ plot
    p <- plot(norm_check, type = "qq", detrend = TRUE)
    ggsave(
      filename = file.path(out_dir_classic, "anovas", paste0("qqplot_", feat, ".png")),
      plot = p,
      width = 6, height = 6, dpi = 300
    )
    
    # Extract p-value
    p_value <- as.numeric(norm_check)
  } else {
    p_value <- NA
  }
  
  
  tibble(
    feature = feat,
    p_value = p_value
  )
})

print(normality_results)


# Term labels in correct order
term_labels <- c("Stress", "Workload", "Stress × Workload")

art_results <- map_dfr(features, function(feat) {
  
  # Selective filtering: EEG features use QC-valid subset, phys features use full dataset
  df <- if (has_qc_flag && feat %in% eeg_features) {
    df_full %>% filter(!qc_failed)
  } else {
    df_full
  }
  
  # Filter for participants with all 4 conditions
  complete_participants <- df %>%
    select(participant_id, stress_level, workload_level, all_of(feat)) %>%
    drop_na() %>%
    group_by(participant_id) %>%
    filter(n() == 4) %>%  # Must have all 4 condition combinations
    pull(participant_id) %>%
    unique()
  
  if (length(complete_participants) < 10) {
    cat(sprintf("Skipping %s: only %d participants with complete data\n", 
                feat, length(complete_participants)))
    return(NULL)
  }
  
  dat <- df %>%
    filter(participant_id %in% complete_participants) %>%
    select(participant_id, stress_level, workload_level, all_of(feat)) %>%
    drop_na()
  
  if (nrow(dat) == 0) return(NULL)
  
  feat_type <- if (feat %in% eeg_features) "EEG" else "Phys"
  cat(sprintf("Running ART ANOVA for %s [%s]: %d participants\n", 
              feat, feat_type, length(complete_participants)))
  
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
feat <- "eda_tonic_mean_precond"

dat_feat <- df_full %>%
  dplyr::select(participant_id, stress_level, workload_level, !!sym(feat)) %>%
  tidyr::drop_na()

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
summary_df <- df_full %>%
  dplyr::select(participant_id, stress_level, workload_level, !!sym(feat)) %>%
  tidyr::drop_na() %>%
  dplyr::group_by(stress_level, workload_level) %>%
  dplyr::summarise(
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


