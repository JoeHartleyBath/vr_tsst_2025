###############################################################################
# baseline_adjustment_check.R
# Evaluate appropriateness of baseline correction for:
#   pupil_med_precond
###############################################################################

suppressPackageStartupMessages({
  library(tidyverse)
  library(yaml)
  library(psych)   # ICC
})

# ------------------------------------------------------------
# Load config + data
# ------------------------------------------------------------
config <- yaml::read_yaml("scripts/utils/config.yaml")
out_dir <- file.path(config$paths$results, "classic_analyses")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

full_data <- readRDS(
  "D:/PhD_Projects/TSST_Stress_Workload_Pipeline/output/full_data_for_reference.rds"
)

# Single feature of interest
feat <- "eeg_fm_theta_mean_full"

# ============================================================
# 1. Compute baseline values (pre-condition only)
# ============================================================
precond_bl <- full_data %>%
  filter(condition_type == "Relaxation") %>%
  mutate(round = as.character(relaxation_level)) %>%
  select(participant_id, round, !!feat) %>%
  rename(precond_bl = !!feat)

# Extract global baseline as well (if using it anywhere)
glob_bl <- full_data %>%
  filter(condition_type == "Baseline") %>%
  group_by(participant_id) %>%
  summarise(glob_bl = mean(.data[[feat]], na.rm = TRUE), .groups = "drop")

# ============================================================
# 2. Extract task feature values and compute baseline deltas
# ============================================================
task_vals <- full_data %>%
  filter(condition_type == "Task") %>%
  mutate(round = as.character(round)) %>%
  select(participant_id, round, !!feat) %>%
  rename(raw_value = !!feat)

# Join baseline(s) and compute change
dat <- task_vals %>%
  left_join(precond_bl, by = c("participant_id", "round")) %>%
  left_join(glob_bl,    by = "participant_id") %>%
  mutate(
    change_precond = raw_value - precond_bl,
    change_glob    = raw_value - glob_bl
  )

# ============================================================
# 3. Check baseline stability (only for pre-condition baselines)
# ============================================================
icc_data <- precond_bl %>%
  pivot_wider(names_from = round, values_from = precond_bl) %>%
  select(-participant_id)

bl_icc <- ICC(icc_data)

cat("\n--- Baseline Stability (ICC) ---\n")
print(bl_icc)

# ============================================================
# 4. Check whether subtraction improves signal
#     – variance reduction
#     – correlation with subjective outcomes
# ============================================================

# Variance check
var_raw   <- var(dat$raw_value, na.rm = TRUE)
var_delta <- var(dat$change_precond, na.rm = TRUE)
cat("\n--- Variance Check ---\n")
cat("Raw variance:", round(var_raw, 3), "\n")
cat("Delta variance:", round(var_delta, 3), "\n")

# Extract stress/workload ratings
subj <- full_data %>%
  filter(condition_type == "Task") %>%
  select(participant_id, round, subj_stress = stress,
         subj_workload = workload)

merged <- dat %>% left_join(subj, by = c("participant_id", "round"))

# Correlation comparison
cor_raw_stress   <- cor(merged$raw_value,      merged$subj_stress,   use = "complete.obs")
cor_delta_stress <- cor(merged$change_precond, merged$subj_stress,   use = "complete.obs")
cor_raw_work     <- cor(merged$raw_value,      merged$subj_workload, use = "complete.obs")
cor_delta_work   <- cor(merged$change_precond, merged$subj_workload, use = "complete.obs")

cat("\n--- Subjective Relationships ---\n")
cat("Raw → stress:",   round(cor_raw_stress, 3), "\n")
cat("Delta → stress:", round(cor_delta_stress, 3), "\n")
cat("Raw → workload:",   round(cor_raw_work, 3), "\n")
cat("Delta → workload:", round(cor_delta_work, 3), "\n")

# ============================================================
# 5. Check whether percent change / ratio / log ratio are better
# ============================================================
merged <- merged %>%
  mutate(
    perc_change = (raw_value - precond_bl) / precond_bl,
    ratio       = raw_value / precond_bl,
    log_ratio   = log(raw_value + 1e-5) - log(precond_bl + 1e-5)
  )

method_corrs <- tibble(
  method = c("raw", "delta", "perc_change", "ratio", "log_ratio"),
  stress = c(
    cor(merged$raw_value,      merged$subj_stress,   use = "complete.obs"),
    cor(merged$change_precond, merged$subj_stress,   use = "complete.obs"),
    cor(merged$perc_change,    merged$subj_stress,   use = "complete.obs"),
    cor(merged$ratio,          merged$subj_stress,   use = "complete.obs"),
    cor(merged$log_ratio,      merged$subj_stress,   use = "complete.obs")
  ),
  workload = c(
    cor(merged$raw_value,      merged$subj_workload, use = "complete.obs"),
    cor(merged$change_precond, merged$subj_workload, use = "complete.obs"),
    cor(merged$perc_change,    merged$subj_workload, use = "complete.obs"),
    cor(merged$ratio,          merged$subj_workload, use = "complete.obs"),
    cor(merged$log_ratio,      merged$subj_workload, use = "complete.obs")
  )
)

cat("\n--- Baseline Correction Method Comparison ---\n")
print(method_corrs)

method_var <- tibble(
  method = c("raw", "delta", "perc_change", "ratio", "log_ratio"),
  variance = c(
    var(merged$raw_value, na.rm = TRUE),
    var(merged$change_precond, na.rm = TRUE),
    var(merged$perc_change, na.rm = TRUE),
    var(merged$ratio, na.rm = TRUE),
    var(merged$log_ratio, na.rm = TRUE)
  )
)

cat("\n--- Variance by Correction Method ---\n")
print(method_var)

# Final summary
cat("\n==================== SUMMARY ====================\n")
cat("1) Baseline stable?         → Check ICC above\n")
cat("2) Subtraction justified?   → See variance improvement and subjective correlations\n")
cat("3) If %change or ratio better → recommend switching for this feature\n")
cat("==================================================\n")
