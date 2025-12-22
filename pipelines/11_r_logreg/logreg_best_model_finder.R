# Logistic Regression Best Model Finder
# ======================================

library(tidyverse)
library(yaml)

# Load CSVs
config      <- yaml::read_yaml("scripts/utils/config.yaml")
results_dir <- file.path(config$paths$results, "logreg")
targets     <- c("stress_label", "workload_label")

progress_files <- setNames(
  sapply(targets, function(t) file.path(results_dir, paste0("logreg_progress_", t, ".csv"))),
  targets
)

results_all <- map_df(names(progress_files), function(t) {
  read_csv(progress_files[[t]], show_col_types = FALSE) %>%
    mutate(target = t)
})

if (nrow(results_all) == 0) stop("No records found in progress CSVs.")

# ---- Average by target × k only ----
k_summary <- results_all %>%
  group_by(target, k) %>%
  summarise(
    mean_acc = mean(final_acc, na.rm = TRUE),
    mean_f1  = mean(final_f1, na.rm = TRUE),
    mean_auc = mean(final_auc, na.rm = TRUE),
    sd_auc   = sd(final_auc, na.rm = TRUE),
    n        = n(),
    .groups  = "drop"
  )

# ---- Per-target best k ----
for (targ in targets) {
  cat("\n", rep("=", 30), "\n")
  cat(" TARGET:", targ, "\n")
  cat(rep("=", 30), "\n\n")
  
  subset <- k_summary %>% filter(target == targ) %>% arrange(desc(mean_auc))
  print(subset, n = Inf)
  
  cat("\nBest k by mean AUC:\n")
  best <- subset %>% slice_max(mean_auc, n = 1)
  print(best)
}

cat("\nDone.\n")
