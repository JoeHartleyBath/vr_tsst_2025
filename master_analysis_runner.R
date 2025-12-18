#!/usr/bin/env Rscript

###############################################################################
# master_analysis_runner.R - Execute all analyses with QC-filtered data
###############################################################################

library(tidyverse)
library(yaml)

# Load config
config <- yaml::read_yaml("scripts/utils/config.yaml")

cat("\n")
cat("================================================================================\n")
cat("MASTER ANALYSIS RUNNER - QC-FILTERED STATISTICAL ANALYSES\n")
cat("================================================================================\n")

# Define analysis pipeline
analysis_scripts <- list(
  list(
    name = "Data Preprocessing & Feature Engineering",
    script = "pipelines/06_r_preprocessing/preproccess_for_xgb.R",
    description = "Load raw aggregated data, apply QC filtering, compute features"
  ),
  list(
    name = "2x2 Repeated-Measures ANOVA (Stress × Workload)",
    script = "pipelines/10_r_anova/anova_main.R",
    description = "Test effects of stress and workload on neural features"
  ),
  list(
    name = "Baseline vs Tasks Paired T-Tests",
    script = "scripts/analysis/sanity_checks/baseline_vs_tasks_ttests.R",
    description = "Paired t-tests comparing baseline to task conditions"
  ),
  list(
    name = "Per-Condition Correlations (EEG-Physio)",
    script = "scripts/analysis/correlations/conditionwise_correlations.R",
    description = "Compute correlations within each condition separately"
  ),
  list(
    name = "Stratified Repeated-Measures Correlations",
    script = "scripts/analysis/correlations/stratified_rmcorr_correlations.R",
    description = "Repeated-measures correlations stratified by condition"
  ),
  list(
    name = "Leave-One-Subject-Out SVM Classification",
    script = "pipelines/08_r_svm/svm.R",
    description = "LOSO cross-validation for stress classification"
  )
)

cat("\n✅ Analysis Pipeline Configuration:\n")
cat(sprintf("   Total analyses to run: %d\n", length(analysis_scripts)))
cat("   All analyses will use QC-filtered data (20 participants excluded)\n\n")

# Execute each analysis
for (i in seq_along(analysis_scripts)) {
  script_info <- analysis_scripts[[i]]
  
  cat(sprintf("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"))
  cat(sprintf("[%d/%d] %s\n", i, length(analysis_scripts), script_info$name))
  cat(sprintf("       %s\n", script_info$description))
  cat(sprintf("       File: %s\n", script_info$script))
  cat(sprintf("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"))
  
  # Check if script exists
  if (!file.exists(script_info$script)) {
    cat(sprintf("⚠️  WARNING: Script not found at '%s'\n", script_info$script))
    cat(sprintf("   Skipping this analysis.\n\n"))
    next
  }
  
  # Execute script with error handling
  tryCatch({
    source(script_info$script)
    cat(sprintf("\n✅ [%d/%d] %s completed successfully.\n\n", 
               i, length(analysis_scripts), script_info$name))
  }, error = function(e) {
    cat(sprintf("\n❌ [%d/%d] %s FAILED with error:\n", 
               i, length(analysis_scripts), script_info$name))
    cat(sprintf("   Error: %s\n\n", conditionMessage(e)))
  })
}

cat("================================================================================\n")
cat("✅ ANALYSIS PIPELINE COMPLETED\n")
cat("================================================================================\n")
cat("\nResults saved to:\n")
cat(sprintf("  - %s\n", file.path(config$paths$results, "classic_analyses")))
cat(sprintf("  - %s\n", file.path(config$paths$results, "correlations")))
cat(sprintf("  - %s\n", file.path(config$paths$results, "svm_results")))
cat("\n")
