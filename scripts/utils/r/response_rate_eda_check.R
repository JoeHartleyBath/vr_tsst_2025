###############################################################################
# check_response_rate_effect.R
# Hypothesis: workload manipulation affects response rate, which may in turn
#             mask tonic EDA effects due to increased movement.
#
# Feature of interest:
#   response_rate_per_min_full_change_precond
#   gsr_skin_conductance_eda_tonic_mean_nk_full_change_precond
###############################################################################

suppressPackageStartupMessages({
  library(tidyverse)
  library(ARTool)
  library(yaml)
})

# ------------------------------------------------------------
# 1. Load config + data
# ------------------------------------------------------------
config <- yaml::read_yaml("scripts/utils/config.yaml")

df <- readRDS(
  "D:/PhD_Projects/TSST_Stress_Workload_Pipeline/output/anova_features_precond.rds"
)

# Sanity check
stopifnot(all(c("participant_id", "stress_level", "workload_level") %in% names(df)))

df <- df %>%
  mutate(
    participant_id = factor(participant_id),
    stress_level   = factor(stress_level, levels = c("Low", "High")),
    workload_level = factor(workload_level, levels = c("Low", "High"))
  )

# ------------------------------------------------------------
# 2. Define features
# ------------------------------------------------------------
feat_rr  <- "response_rate_per_min_full_change_precond"
feat_eda <- "gsr_skin_conductance_eda_tonic_mean_nk_full_change_precond"

# Keep only required columns
dat <- df %>%
  select(participant_id, stress_level, workload_level, !!sym(feat_rr), !!sym(feat_eda)) %>%
  drop_na()

# ------------------------------------------------------------
# 3. Does workload affect response rate?
# ------------------------------------------------------------
m_rr <- art(as.formula(paste0(
  feat_rr, " ~ stress_level * workload_level + Error(participant_id)"
)), data = dat)

cat("\n============================================================\n")
cat("ANOVA (ART) on Response Rate\n")
cat("============================================================\n")
print(anova(m_rr))

# Post-hoc only if workload significant
cat("\nPost-hoc (workload_level) if p < .05\n")
if (anova(m_rr)$`Pr(>F)`[anova(m_rr)$term == "workload_level"] < 0.05) {
  print(art.con(m_rr, "workload_level", adjust = "holm"))
} else {
  cat(" → Workload not significant, skipping post-hoc.\n")
}

# ------------------------------------------------------------
# 4. Correlation between response rate & tonic EDA (masking test)
# ------------------------------------------------------------
cor_rr_eda <- cor(dat[[feat_rr]], dat[[feat_eda]], use = "complete.obs")

cat("\n============================================================\n")
cat("Correlation (Response Rate ~ Tonic EDA)\n")
cat("============================================================\n")
cat("r =", round(cor_rr_eda, 3), "\n")

# ------------------------------------------------------------
# 5. Descriptive stats (sanity check)
# ------------------------------------------------------------
cat("\n============================================================\n")
cat("Descriptive stats per workload level\n")
cat("============================================================\n")

desc_stats <- dat %>%
  group_by(workload_level) %>%
  summarise(
    mean_rr = mean(.data[[feat_rr]], na.rm = TRUE),
    sd_rr   = sd(.data[[feat_rr]], na.rm = TRUE),
    mean_eda = mean(.data[[feat_eda]], na.rm = TRUE),
    sd_eda   = sd(.data[[feat_eda]], na.rm = TRUE),
    .groups  = "drop"
  )
print(desc_stats)

# ------------------------------------------------------------
# 6. Final interpretation template
# ------------------------------------------------------------


library(lme4)
library(lmerTest)

m_lmm <- lmer(
  gsr_skin_conductance_eda_tonic_mean_nk_full_change_precond ~ 
    workload_level + stress_level + response_rate_per_min_full_change_precond +
    (1 | participant_id),
  data = dat
)

summary(m_lmm)

