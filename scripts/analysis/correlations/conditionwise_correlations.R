# =====================================================================
# conditionwise_correlations.R
# Per-condition correlations between typical features and
# subjective stress / workload, with grouped heatmap visualisation.
# =====================================================================

suppressPackageStartupMessages({
  library(tidyverse)
  library(yaml)
  library(ggplot2)
})

source("utils/r/feature_selection.R")

# =====================================================================
# 0. CONFIG + GENERIC HELPERS
# =====================================================================

# ---- Load config and set output dir ----
config  <- yaml::read_yaml("scripts/config.yaml")
out_dir <- config$paths$output

# Generic loader: tries .rds then .csv
load_obj <- function(stem, dir_path = out_dir) {
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

# Map condition → stress/workload levels
cond_map <- tribble(
  ~condition,               ~stress_level, ~workload_level,
  "High Stress - High Cog", "High",        "High",
  "High Stress - Low Cog",  "High",        "Low",
  "Low Stress - High Cog",  "Low",         "High",
  "Low Stress - Low Cog",   "Low",         "Low"
)

apply_cond_map <- function(df, cond_col = "condition", map_tbl = cond_map) {
  if (!cond_col %in% names(df)) return(df)
  df %>%
    left_join(map_tbl, by = set_names("condition", cond_col)) %>%
    mutate(
      stress_level   = factor(stress_level,   levels = c("Low","High")),
      workload_level = factor(workload_level, levels = c("Low","High"))
    )
}

# Significance stars for p-values
signif_star <- function(p) {
  case_when(
    is.na(p)     ~ "",
    p < 0.001    ~ "***",
    p < 0.01     ~ "**",
    p < 0.05     ~ "*",
    p < 0.10     ~ "·",     # optional “trend” indicator
    TRUE         ~ ""
  )
}

# Shorter condition labels for plotting
cond_short_labs <- c(
  "Low Stress - Low Cog"   = "LS–LW",
  "Low Stress - High Cog"  = "LS–HW",
  "High Stress - Low Cog"  = "HS–LW",
  "High Stress - High Cog" = "HS–HW"
)

# =====================================================================
# 1. FEATURE METADATA (TYPICAL FEATURES, PRETTY LABELS, GROUPS)
# =====================================================================

# Use centralized drop pattern
drop_pattern <- get_feature_drop_pattern()


categorise_feature <- function(x) {
  case_when(
    str_starts(x, regex("eeg_fm", ignore_case = TRUE)) ~ "EEG_FrontalMidline",
    str_starts(x, regex("eeg_f", ignore_case = TRUE))   ~ "EEG_Frontal",
    str_starts(x, regex("eeg_t", ignore_case = TRUE))   ~ "EEG_Temporal",
    str_starts(x, regex("eeg_c", ignore_case = TRUE))   ~ "EEG_Central",
    str_starts(x, regex("eeg_p", ignore_case = TRUE))   ~ "EEG_Parietal",
    str_starts(x, regex("faa|ratio", ignore_case = TRUE)) ~ "EEG_Frontal",
    str_starts(x, regex("hr|hrv", ignore_case = TRUE)) ~ "HR/HRV",
    str_starts(x, regex("eda|gsr", ignore_case = TRUE)) ~ "EDA",
    str_starts(x, regex("pupil", ignore_case = TRUE))   ~ "Pupillometry",
    TRUE ~ "Other"
  )
}


# =====================================================================
# 2. LOAD + PREPARE DATA
# =====================================================================

final_data <- load_obj("final_data") %>% apply_cond_map()

# Mixed correlation filtering: Use full N=47 for subjective-only,
# filter to EEG-valid for correlations involving EEG features
df_full <- final_data %>%
  filter(!is.na(stress_level), !is.na(workload_level))

# Check if qc_failed column exists
has_qc_flag <- "qc_failed" %in% names(df_full)

rating_cols <- c("stress", "workload")

conditions <- df_full$condition %>%
  unique() %>%
  discard(is.na)

features <- names(df_full)[str_detect(names(df_full), "_precond_Z$")]
features <- features[!str_detect(features, drop_pattern)]

# Ensure typical features exist in the data
features <- intersect(features, names(df_full))
if (length(features) == 0L) {
  stop("No features found in df_full. Check column names.")
}

feature_meta <- tibble(
  feature = features,
  group = categorise_feature(features)
)

# Identify EEG features
eeg_features <- features[str_detect(features, regex("^eeg_", ignore_case = TRUE))]

message(sprintf("[Correlations] Full dataset N=%d participants", n_distinct(df_full$participant_id)))
if (has_qc_flag) {
  n_eeg_valid <- n_distinct(df_full %>% filter(!qc_failed) %>% pull(participant_id))
  message(sprintf("[Correlations] EEG-valid subset N=%d participants", n_eeg_valid))
  message(sprintf("[Correlations] Will use full N for subjective-only, EEG-valid N for EEG features"))
}

# =====================================================================
# 3. ANALYSIS FUNCTION: PER-CONDITION SPEARMAN CORRELATIONS
# =====================================================================

cor_one_condition <- function(df_input, cond_name) {
  df_cond <- df_input %>% filter(condition == cond_name)
  
  expand_grid(
    feature = features,
    rating  = rating_cols
  ) %>%
    pmap_dfr(function(feature, rating) {
      # Selective filtering: use EEG-valid subset only for EEG features
      df_for_corr <- df_cond
      if (has_qc_flag && feature %in% eeg_features) {
        df_for_corr <- df_for_corr %>% filter(!qc_failed)
      }
      
      tmp <- df_for_corr %>%
        select(participant_id, !!rating, !!feature) %>%
        drop_na()
      
      if (nrow(tmp) < 5) {
        return(tibble(
          condition = cond_name,
          feature   = feature,
          rating    = rating,
          rho       = NA_real_,
          p_raw     = NA_real_,
          n         = nrow(tmp)
        ))
      }
      
      ct <- suppressWarnings(
        cor.test(tmp[[rating]], tmp[[feature]], method = "spearman")
      )
      
      tibble(
        condition = cond_name,
        feature   = feature,
        rating    = rating,
        rho       = unname(ct$estimate),
        p_raw     = ct$p.value,
        n         = nrow(tmp)
      )
    })
}

# =====================================================================
# 4. RUN CORRELATIONS + ADD METADATA (FIXED)
# =====================================================================

# 1. Run correlations
all_cor <- map_dfr(conditions, ~ cor_one_condition(df_full, .x))

# 3. Add short condition labels
all_cor <- all_cor %>%
  mutate(
    cond_short = cond_short_labs[condition]
  )

# 4. Add feature group metadata (join FIRST)
all_cor <- all_cor %>%
  left_join(feature_meta, by = "feature")


# 5. Order groups (AFTER join)
all_cor <- all_cor %>%
  mutate(
    group = factor(group, levels = c(
      "EDA", "Pupillometry", "HR/HRV",
      "EEG_FrontalMidline", "EEG_FM", "EEG_Frontal",
      "EEG_Temporal", "EEG_Parietal",
      "EEG_Central"
    ))
  )

# 6. Adjust p-values within feature groups
all_cor <- all_cor %>%
  group_by(cond_short, rating, group) %>%
  mutate(p_fdr = p.adjust(p_raw, method = "BH")) %>%
  ungroup()

# 7. Add significance stars
all_cor <- all_cor %>%
  mutate(
    sig = signif_star(p_fdr)
  )

# 8. Order features within groups
all_cor <- all_cor %>%
  arrange(group, feature) %>%
  mutate(
    feature = factor(feature, levels = unique(feature))
  )


# =====================================================================
# 5. SAVE CORRELATION RESULTS
# =====================================================================

out_dir_corr <- file.path(
  "D:/PhD_Projects/TSST_Stress_Workload_Pipeline/results/classic_analyses",
  "conditionwise_correlations"
)
dir.create(out_dir_corr, recursive = TRUE, showWarnings = FALSE)

all_cor_export <- all_cor %>%
  select(condition, cond_short, rating,
         feature, feature,
         group, rho, p_raw, p_fdr, n)

write_csv(
  all_cor_export,
  file.path(out_dir_corr, "conditionwise_spearman_correlations.csv")
)
saveRDS(
  all_cor,
  file.path(out_dir_corr, "conditionwise_spearman_correlations.rds")
)

cat("✓ Saved per-condition correlations to:\n", out_dir_corr, "\n")

# =====================================================================
# 6. HEATMAP PLOT FUNCTION
# =====================================================================

plot_conditionwise_heatmap <- function(df, out_path) {
  
  heat_df <- df %>%
    mutate(
      rating     = factor(rating, levels = c("stress", "workload")),
      cond_short = factor(cond_short, levels = c("LS–LW","LS–HW","HS–LW","HS–HW")),
      feature_display = factor(
        sub("_precond_Z$", "", feature),
        levels = unique(sub("_precond_Z$", "", feature))
      )
    )
  
  p <- ggplot(heat_df, aes(x = cond_short, y = feature_display, fill = rho)) +
    geom_tile(colour = "grey20", linewidth = 0.25) +
    geom_text(aes(label = sig), colour = "black", size = 3.2, vjust = 0.5) +
    scale_fill_distiller(
      palette = "RdBu",
      direction = -1,
      limits = c(-1, 1),
      name = "Spearman\nρ"
    ) +
    facet_wrap(~ rating, nrow = 1) +
    theme_minimal(base_size = 14) +
    theme(
      strip.text = element_text(size = 16, face = "bold"),
      axis.text.x = element_text(size = 12),
      axis.text.y = element_text(size = 11),
      panel.grid = element_blank(),
      axis.title = element_blank(),
      legend.position = "right"
    )
  
  ggsave(out_path, p, width = 12, height = 10, dpi = 300)
  invisible(p)
}


# =====================================================================
# 6B. SPLIT BY GROUP AND PLOT
# =====================================================================

groups <- unique(all_cor$group)

for (g in groups) {
  if (is.na(g)) next
  
  df_g <- all_cor %>% filter(group == g)
  
  df_g <- df_g %>%
    arrange(feature) %>%
    mutate(
      feature_display = factor(
        sub("_precond_Z$", "", feature),
        levels = unique(sub("_precond_Z$", "", feature))
      )
    )
  
  safe_g <- gsub("[^A-Za-z0-9_]", "_", g)
  
  out_path <- file.path(out_dir_corr, paste0("conditionwise_heatmap_", safe_g, ".png"))
  
  plot_conditionwise_heatmap(df_g, out_path)
}



