# Validation Script: Compare Pre- and Post-QC Refactor Results
# 
# Purpose: Verify that the QC filtering refactor produces expected changes
#          in sample sizes while maintaining data quality for EEG analyses.
#
# Author: GitHub Copilot
# Date: 2025-12-19

suppressPackageStartupMessages({
  library(tidyverse)
})

cat("\n========================================\n")
cat("QC FILTERING REFACTOR VALIDATION\n")
cat("========================================\n\n")

# ------------------------------------------------------------
# 1. Load old and new datasets
# ------------------------------------------------------------

cat("1. Loading datasets...\n")

# Old dataset (backed up)
old_final <- tryCatch(
  readRDS("output/results_backup_pre_qc_refactor/final_data.rds"),
  error = function(e) {
    cat("  WARNING: Could not load backup. Run preprocessing first.\n")
    return(NULL)
  }
)

# New datasets
new_final <- tryCatch(
  readRDS("output/final_data.rds"),
  error = function(e) {
    cat("  WARNING: Could not load new final_data.rds. Run preprocessing first.\n")
    return(NULL)
  }
)

new_eeg_valid <- tryCatch(
  readRDS("output/final_data_eeg_valid.rds"),
  error = function(e) {
    cat("  WARNING: Could not load new final_data_eeg_valid.rds. Run preprocessing first.\n")
    return(NULL)
  }
)

if (is.null(old_final) || is.null(new_final)) {
  cat("\n❌ ERROR: Missing required datasets. Cannot validate.\n")
  cat("   1. Ensure backup exists in output/results_backup_pre_qc_refactor/\n")
  cat("   2. Run: Rscript pipelines/06_r_preprocessing/preproccess_for_xgb.R\n\n")
  quit(status = 1)
}

cat("  ✓ Loaded old final_data.rds\n")
cat("  ✓ Loaded new final_data.rds\n")
if (!is.null(new_eeg_valid)) cat("  ✓ Loaded new final_data_eeg_valid.rds\n")

# ------------------------------------------------------------
# 2. Sample size comparison
# ------------------------------------------------------------

cat("\n2. Sample Size Comparison\n")
cat("   -------------------------\n")

n_old <- n_distinct(old_final$participant_id)
n_new_full <- n_distinct(new_final$participant_id)
n_new_eeg <- if (!is.null(new_eeg_valid)) n_distinct(new_eeg_valid$participant_id) else NA

cat(sprintf("   Old dataset:          N = %2d\n", n_old))
cat(sprintf("   New full dataset:     N = %2d", n_new_full))
if (n_new_full == 47) {
  cat(" ✓ (Expected: 47)\n")
} else {
  cat(sprintf(" ❌ (Expected: 47, got %d)\n", n_new_full))
}

if (!is.null(new_eeg_valid)) {
  cat(sprintf("   New EEG-valid:        N = %2d", n_new_eeg))
  if (n_new_eeg == 44) {
    cat(" ✓ (Expected: 44)\n")
  } else {
    cat(sprintf(" ❌ (Expected: 44, got %d)\n", n_new_eeg))
  }
}

# Check QC flag
if ("qc_failed" %in% names(new_final)) {
  qc_summary <- new_final %>%
    distinct(participant_id, qc_failed) %>%
    count(qc_failed)
  
  cat("\n   QC Status in new_final:\n")
  cat(sprintf("     Passed:  %d\n", qc_summary$n[qc_summary$qc_failed == FALSE]))
  cat(sprintf("     Failed:  %d\n", qc_summary$n[qc_summary$qc_failed == TRUE]))
  
  # Identify QC failures
  qc_failed_ids <- new_final %>%
    filter(qc_failed) %>%
    pull(participant_id) %>%
    unique() %>%
    sort()
  
  cat(sprintf("     Failed IDs: %s\n", paste(qc_failed_ids, collapse = ", ")))
} else {
  cat("\n   ❌ WARNING: qc_failed column not found in new dataset!\n")
}

# ------------------------------------------------------------
# 3. Feature count comparison
# ------------------------------------------------------------

cat("\n3. Feature Comparison\n")
cat("   -------------------------\n")

n_feat_old <- ncol(old_final)
n_feat_new <- ncol(new_final)

cat(sprintf("   Old features:         %d\n", n_feat_old))
cat(sprintf("   New features:         %d", n_feat_new))

if (n_feat_new == n_feat_old + 1) {
  cat(" ✓ (Added qc_failed column)\n")
} else if (n_feat_new == n_feat_old) {
  cat(" ⚠️  (Same as old - expected +1 for qc_failed)\n")
} else {
  cat(sprintf(" ⚠️  (Expected %d, difference unexpected)\n", n_feat_old + 1))
}

# Check for new columns
new_cols <- setdiff(names(new_final), names(old_final))
if (length(new_cols) > 0) {
  cat(sprintf("   New columns added:    %s\n", paste(new_cols, collapse = ", ")))
}

# ------------------------------------------------------------
# 4. Data integrity checks
# ------------------------------------------------------------

cat("\n4. Data Integrity\n")
cat("   -------------------------\n")

# Check that EEG-valid subset matches filtered old dataset
if (!is.null(new_eeg_valid)) {
  old_ids <- sort(unique(old_final$participant_id))
  eeg_valid_ids <- sort(unique(new_eeg_valid$participant_id))
  
  if (identical(old_ids, eeg_valid_ids)) {
    cat("   ✓ EEG-valid subset matches old dataset participant IDs\n")
  } else {
    cat("   ❌ EEG-valid subset participant IDs differ from old dataset:\n")
    cat(sprintf("      Old:       %s\n", paste(old_ids, collapse = ", ")))
    cat(sprintf("      EEG-valid: %s\n", paste(eeg_valid_ids, collapse = ", ")))
  }
  
  # Check feature values match (for overlapping participants)
  common_cols <- intersect(names(old_final), names(new_eeg_valid))
  common_cols <- setdiff(common_cols, "qc_failed")  # Exclude new column
  
  if (length(common_cols) > 10) {
    # Sample check on a few features
    sample_features <- sample(
      common_cols[!common_cols %in% c("participant_id", "condition", "round")],
      min(5, length(common_cols))
    )
    
    all_match <- TRUE
    for (feat in sample_features) {
      old_vals <- old_final %>% select(participant_id, condition, !!feat) %>% arrange(participant_id, condition)
      new_vals <- new_eeg_valid %>% select(participant_id, condition, !!feat) %>% arrange(participant_id, condition)
      
      if (!all.equal(old_vals, new_vals, tolerance = 1e-10)) {
        cat(sprintf("   ❌ Feature '%s' values differ between old and EEG-valid\n", feat))
        all_match <- FALSE
      }
    }
    
    if (all_match) {
      cat(sprintf("   ✓ Sampled features (%d) match between old and EEG-valid\n", length(sample_features)))
    }
  }
}

# ------------------------------------------------------------
# 5. Expected changes summary
# ------------------------------------------------------------

cat("\n5. Expected Changes Summary\n")
cat("   -------------------------\n")

expected_changes <- tribble(
  ~Analysis, ~Old_N, ~New_N, ~Status,
  "Subjective ANOVAs", 44, 47, if_else(n_new_full == 47, "✓", "❌"),
  "EEG ANOVAs", 44, 44, if_else(!is.null(new_eeg_valid) && n_new_eeg == 44, "✓", "❌"),
  "ML Pipelines", 44, 44, if_else(!is.null(new_eeg_valid) && n_new_eeg == 44, "✓", "❌")
)

print(expected_changes, n = Inf)

# ------------------------------------------------------------
# 6. Validation summary
# ------------------------------------------------------------

cat("\n========================================\n")
cat("VALIDATION SUMMARY\n")
cat("========================================\n\n")

all_checks_passed <- TRUE

# Check 1: Full dataset N=47
if (n_new_full != 47) {
  cat("❌ Full dataset should have N=47\n")
  all_checks_passed <- FALSE
}

# Check 2: EEG-valid N=44
if (!is.null(new_eeg_valid) && n_new_eeg != 44) {
  cat("❌ EEG-valid dataset should have N=44\n")
  all_checks_passed <- FALSE
}

# Check 3: qc_failed column exists
if (!"qc_failed" %in% names(new_final)) {
  cat("❌ qc_failed column missing from new dataset\n")
  all_checks_passed <- FALSE
}

if (all_checks_passed) {
  cat("✅ All validation checks PASSED\n\n")
  cat("Next steps:\n")
  cat("  1. Re-run analysis scripts to generate new results\n")
  cat("  2. Compare subjective ANOVA N (should increase 44→47)\n")
  cat("  3. Verify EEG ANOVA N remains 44\n")
  cat("  4. Check correlation heatmaps for updated sample sizes\n\n")
} else {
  cat("⚠️  Some validation checks FAILED\n\n")
  cat("Please review errors above and re-run preprocessing if needed.\n\n")
}

cat("Validation complete: ", date(), "\n\n")
