library(tidyverse)
library(yaml)

# Load CSVs
config      <- yaml::read_yaml("scripts/utils/config.yaml")
results_dir <- file.path(config$paths$results, "svm")
targets     <- c("stress_label", "workload_label")

progress_files <- setNames(
  sapply(targets, function(t) file.path(results_dir, paste0("svm_progress_", t, ".csv"))),
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
  ) %>%
  arrange(target, desc(mean_auc))

# ---- Print results ----
for (t in targets) {
  cat("\n==============================\n")
  cat(" TARGET:", t, "\n")
  cat("==============================\n\n")
  
  print(k_summary %>% filter(target == t), n = Inf)
  
  cat("\nBest k by mean AUC:\n")
  print(k_summary %>% filter(target == t) %>% slice_max(mean_auc), n = 1)
}

cat("\nDone.\n")
