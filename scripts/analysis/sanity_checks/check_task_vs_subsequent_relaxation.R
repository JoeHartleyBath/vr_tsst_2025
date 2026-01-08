
library(dplyr)
library(readr)
library(lme4)
library(lmerTest)

# 1. Load Data
data_path <- "output/full_data_for_reference.rds"
df <- readRDS(data_path)

# 2. Define Features
hr_col <- "hr_med"
eda_col <- "eda_tonic_mean"
theta_col <- "eeg_f_theta_power"

cat("\nProceeding with analysis using columns:", hr_col, eda_col, theta_col, "\n")

# 3. Prepare Data
# Strategy: Compare Task (at position N) with Relaxation (at position N+1).
# We assume 'round' tracks Task order (1..4)
# We assume 'relaxation_level' tracks Relaxation order (1..4)
# Based on project structure: Block 1 = Relax 1 -> Task 1. Block 2 = Relax 2 -> Task 2...
# Therefore, Relax 2 is the Recovery period for Task 1.

tasks <- df %>%
  filter(condition_type == "Task") %>%
  select(participant_id, 
         task_round = round, 
         task_hr = all_of(hr_col), 
         task_eda = all_of(eda_col), 
         task_theta = all_of(theta_col)) %>%
  mutate(task_round = as.numeric(as.character(task_round)))

relaxations <- df %>%
  filter(condition_type == "Relaxation") %>%
  select(participant_id, 
         relax_level = relaxation_level, 
         relax_hr = all_of(hr_col), 
         relax_eda = all_of(eda_col), 
         relax_theta = all_of(theta_col)) %>%
  mutate(relax_level = as.numeric(as.character(relax_level)))

# Join: Task N matches with Relax N+1
aligned_data <- tasks %>%
  mutate(match_level = task_round + 1) %>%
  inner_join(relaxations, by = c("participant_id", "match_level" = "relax_level"))

cat("\n--- Data Alignment Check ---\n")
cat("Pairs found:", nrow(aligned_data), "\n")
cat("Expected: Pairs for Task rounds 1, 2, 3 (recovering in relax 2, 3, 4)\n")
cat("Task rounds included:", paste(unique(aligned_data$task_round), collapse=", "), "\n")

if (nrow(aligned_data) > 0) {
    # 4. Statistical Tests
    
    cat("\n--- Paired T-Test Results (Task vs Subsequent Relaxation) ---\n")
    
    run_test <- function(task_vec, relax_vec, label) {
       t_res <- t.test(task_vec, relax_vec, paired = TRUE)
       cat(sprintf("\n%s:\nMean Task: %.2f, Mean Relax: %.2f\nMean Difference (Task-Relax): %.2f, p-value: %.3e\n",
                   label,
                   mean(task_vec, na.rm=T), mean(relax_vec, na.rm=T),
                   t_res$estimate, t_res$p.value))
    }
    
    run_test(aligned_data$task_hr, aligned_data$relax_hr, "Heart Rate (BPM)")
    run_test(aligned_data$task_eda, aligned_data$relax_eda, "Tonic EDA (uS)")
    run_test(aligned_data$task_theta, aligned_data$relax_theta, "Frontal Theta (dB)")

    # LMM
    long_dat <- aligned_data %>%
      tidyr::pivot_longer(cols = c(starts_with("task_"), starts_with("relax_")), 
                          names_to = c("type", "measure"), 
                          names_sep = "_") %>%
      tidyr::pivot_wider(names_from = measure, values_from = value)
      
    cat("\n--- LMM Results (Significance of Type: Relax vs Task) ---\n")
    try({
        mod <- lmer(hr ~ type + (1|participant_id), data = long_dat)
        print(anova(mod))
    })
    
} else {
    cat("No pairs found. Check round/level alignment.\n")
}
