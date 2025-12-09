# =====================================================================
# stratified_rmcorr_correlations.R
# Repeated-measures correlations between features and subjective
# stress/workload stratified by stress or workload levels.
# =====================================================================

suppressPackageStartupMessages({
  library(tidyverse)
  library(yaml)
  library(ggplot2)
  library(rmcorr)
})

# =====================================================================
# 0. CONFIG + HELPERS
# =====================================================================

# Load config
config  <- yaml::read_yaml("scripts/utils/config.yaml")
out_dir <- config$paths$output
heat_path <- file.path(out_dir_strat, "stratified_rmcorr_heatmap.png")

# Loader
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

# Stars
signif_star <- function(p) {
  dplyr::case_when(
    is.na(p)  ~ "",
    p < 0.001 ~ "***",
    p < 0.01  ~ "**",
    p < 0.05  ~ "*",
    p < 0.06  ~ "·",
    TRUE      ~ ""
  )
}


# =====================================================================
# 2. LOAD + PREP DATA
# =====================================================================

final_data <- load_obj("final_data")

# Keep only rows with subjective ratings (the 4 task conditions)
df <- final_data %>%
  filter(!is.na(stress), !is.na(workload))


# Derive binary stress/workload factors from condition
df <- df %>%
  mutate(
    stress_level   = if_else(condition %in% c("High Stress - High Cog", "High Stress - Low Cog"),
                             "High", "Low"),
    workload_level = if_else(condition %in% c("High Stress - High Cog", "Low Stress - High Cog"),
                             "High", "Low")
  )



# =====================================================================
# 3. DEFINE STRATIFIED SUBSETS (2 COLUMNS PER RATING)
# =====================================================================

# For STRESS correlations: split by WORKLOAD (Low vs High)
# For WORKLOAD correlations: split by STRESS (Low vs High)

subset_defs <- list(
  list(
    label  = "Low workload",
    rating = "stress",
    df     = df %>% filter(workload_level == "Low")
  ),
  list(
    label  = "High workload",
    rating = "stress",
    df     = df %>% filter(workload_level == "High")
  ),
  list(
    label  = "Low stress",
    rating = "workload",
    df     = df %>% filter(stress_level == "Low")
  ),
  list(
    label  = "High stress",
    rating = "workload",
    df     = df %>% filter(stress_level == "High")
  )
)

# ------------------------------------------------------------
# Feature list 
# ------------------------------------------------------------

canonical_feats <- config$canonical_features
suffix <- "_full_change_precond"  #baseline adjusted using precondition relaxation scene
features <- paste0(canonical_feats, suffix)

feat_labels <- config$pretty_features
feature_order <- config$feature_order


df <- df %>%
  group_by(participant_id) %>%
  mutate(across(all_of(features), ~ scale(.x)[,1], .names = "{.col}_Z")) %>%
  ungroup()


# =====================================================================
# 4. RM-CORR FUNCTION
# =====================================================================

run_rmcorr <- function(df, subset_label, rating_var) {
  
  map_dfr(features, function(feature) {
    
    tmp <- df %>%
      select(participant_id, !!rating_var := !!sym(rating_var), !!feature) %>%
      drop_na()
    
    # rmcorr needs >1 obs per participant
    valid_ids <- tmp %>%
      count(participant_id) %>%
      filter(n > 1) %>%
      pull(participant_id)
    
    tmp <- tmp %>% filter(participant_id %in% valid_ids)
    
    if (n_distinct(tmp$participant_id) < 5) {
      return(tibble(
        subset      = subset_label,
        rating      = rating_var,
        feature     = feature,
        rmcorr_r    = NA_real_,
        p_raw       = NA_real_,
        n_pairs     = nrow(tmp)
      ))
    }
    
    rc <- suppressMessages(
      rmcorr(
        participant = "participant_id",
        measure1    = rating_var,
        measure2    = feature,
        dataset     = tmp
      )
    )
    
    tibble(
      subset      = subset_label,
      rating      = rating_var,
      feature     = feature,
      rmcorr_r    = rc$r,
      p_raw       = rc$p,
      n_pairs     = nrow(tmp)
    )
  })
}

# =====================================================================
# 5. RUN RM-CORR FOR ALL SUBSETS
# =====================================================================

all_rmcorr <- purrr::map_dfr(
  subset_defs,
  ~ run_rmcorr(df = .x$df, subset_label = .x$label, rating_var = .x$rating)
)

all_rmcorr <- all_rmcorr %>%
  group_by(subset, rating) %>%
  mutate(p_fdr = p.adjust(p_raw, method = "BH")) %>%
  ungroup() %>%
  mutate(sig = signif_star(p_fdr))



# =====================================================================
# 6. SAVE RESULTS
# =====================================================================

out_dir_strat <- file.path(
  "D:/PhD_Projects/TSST_Stress_Workload_Pipeline/results/classic_analyses",
  "stratified_rmcorr"
)
dir.create(out_dir_strat, recursive = TRUE, showWarnings = FALSE)

write_csv(all_rmcorr, file.path(out_dir_strat, "stratified_rmcorr_results.csv"))
saveRDS(all_rmcorr, file.path(out_dir_strat, "stratified_rmcorr_results.rds"))

cat("✓ Saved stratified rm-corr results to:\n", out_dir_strat, "\n")

# =====================================================================
# 7. HEATMAP
# =====================================================================

plot_strat_heatmap <- function(df, out_path, feat_labels, suffix, feature_order) {
  
  feat_labels <- unlist(feat_labels)
  
  heat_df <- df %>%
    mutate(
      subset = factor(
        subset,
        levels = c("Low workload","High workload",
                   "Low stress","High stress")
      ),
      rating = factor(rating, levels = c("stress","workload")),
      feature_base = sub(paste0(suffix, "$"), "", feature),
      feature_display = feat_labels[feature_base],
      feature_display = ifelse(is.na(feature_display),
                               feature_base,
                               feature_display)
    )
  
  # Apply feature ordering from config
  heat_df$feature_display <- factor(
    heat_df$feature_display,
    levels = feature_order
  )
  
  # --- NEW: Split data ---
  df_stress_corr <- heat_df %>% 
    filter(rating == "stress",
           subset %in% c("Low workload", "High workload"))
  
  df_workload_corr <- heat_df %>% 
    filter(rating == "workload",
           subset %in% c("Low stress", "High stress"))
  
  # --- NEW: heatmap generator ---
  make_one_heatmap <- function(data, title) {
    ggplot(
      data,
      aes(x = subset, y = feature_display, fill = rmcorr_r)
    ) +
      geom_tile(colour = "grey20", linewidth = 0.25) +
      geom_text(aes(label = sprintf("%.2f", rmcorr_r)),
                colour = "black", size = 3.1) +
      geom_text(aes(label = sig),
                vjust = -0.6, colour = "black", size = 4) +
      scale_fill_distiller(
        palette   = "RdYlBu",
        direction = -1,
        limits    = c(-1, 1),
        name      = "r"
      ) +
      labs(title = title) +
      theme_minimal(base_size = 14) +
      theme(
        axis.text.x = element_text(size = 12, angle = 20, hjust = 1),
        axis.text.y = element_text(size = 11),
        strip.text  = element_text(size = 16, face = "bold"),
        panel.grid  = element_blank(),
        axis.title  = element_blank(),
        legend.position = "right"
      )
  }
  
  p_stress <- make_one_heatmap(df_stress_corr,
                               "Repeated-Measures Correlations with Stress (Stratified by Workload)")
  
  p_workload <- make_one_heatmap(df_workload_corr,
                                 "Repeated-Measures Correlations with Workload (Stratified by Stress)")
  
  ggsave(file.path(out_path, "heatmap_stress_corr.png"),
         p_stress, width = 12, height = 9, dpi = 300)
  
  ggsave(file.path(out_path, "heatmap_workload_corr.png"),
         p_workload, width = 12, height = 9, dpi = 300)
  
  return(list(stress_plot = p_stress, workload_plot = p_workload))
}


plots <- plot_strat_heatmap(
  all_rmcorr,
  out_dir_strat,
  feat_labels,
  suffix,
  feature_order
)

plots$stress_plot
plots$workload_plot

