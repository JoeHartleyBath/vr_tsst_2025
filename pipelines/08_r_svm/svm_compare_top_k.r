library(tidyverse)

results_dir <- "C:/vr_tsst_2025/results/svm"  # your actual path (use forward slashes in R)

targets <- c("stress_label", "workload_label")

for (t in targets) {
  main_file   <- file.path(results_dir, paste0("svm_progress_", t, ".csv"))
  tuning_file <- file.path(results_dir, paste0("svm_progress_", t, ".csv_tuning.csv"))

  if (!file.exists(main_file) || !file.exists(tuning_file)) {
    cat("\n[", t, "] files not found yet — skipping\n"); next
  }

  results <- read_csv(main_file,   show_col_types = FALSE)
  tuning  <- read_csv(tuning_file, show_col_types = FALSE)

  n_folds <- n_distinct(results$test_pid)

  cat("\n==============================================\n")
  cat(" TARGET:", t, " (", n_folds, "/44 folds present)\n")
  cat("==============================================\n")

  # ---- k chosen by INNER AUC (training-only, clean) ----
  inner_by_k <- tuning %>%
    group_by(test_pid, k) %>%
    summarise(inner_best = max(inner_auc, na.rm = TRUE), .groups = "drop") %>%
    group_by(k) %>%
    summarise(mean_inner_auc = mean(inner_best, na.rm = TRUE), .groups = "drop") %>%
    arrange(desc(mean_inner_auc))

  # ---- k chosen by FINAL AUC (held-out, the leaky selection) ----
  final_by_k <- results %>%
    group_by(k) %>%
    summarise(mean_final_auc = mean(final_auc, na.rm = TRUE),
              sd_final_auc   = sd(final_auc,  na.rm = TRUE),
              .groups = "drop") %>%
    arrange(desc(mean_final_auc))

  k_inner <- inner_by_k$k[1]
  k_final <- final_by_k$k[1]

  cat("\nBy INNER AUC (training only — the clean selection rule):\n")
  print(inner_by_k)
  cat("\nBy FINAL AUC (held-out — what was originally reported):\n")
  print(final_by_k)

  # ---- per-fold best k by inner AUC ----
  perfold <- tuning %>%
    group_by(test_pid, k) %>%
    summarise(inner_best = max(inner_auc, na.rm = TRUE), .groups = "drop") %>%
    group_by(test_pid) %>%
    slice_max(inner_best, n = 1, with_ties = FALSE) %>%
    ungroup()
  cat("\nPer-fold winning k (by inner AUC):\n")
  print(table(perfold$k))

  # ---- verdict ----
  cat("\n--- VERDICT ---\n")
  cat("k by inner AUC:", k_inner, " | k by final AUC:", k_final, "\n")
  if (k_inner == k_final) {
    cat("✓ MATCH. Selecting k by inner AUC gives the same k as reported.\n")
    cat("  → Option B is clean: no AUC or p-value change for this target.\n")
  } else {
    auc_inner_k <- final_by_k$mean_final_auc[final_by_k$k == k_inner]
    auc_final_k <- final_by_k$mean_final_auc[final_by_k$k == k_final]
    cat("✗ DIFFER. Inner AUC favours k =", k_inner,
        "but you reported k =", k_final, ".\n")
    cat(sprintf("  Held-out AUC at inner-chosen k=%d: %.4f\n", k_inner, auc_inner_k))
    cat(sprintf("  Held-out AUC at reported   k=%d: %.4f\n", k_final, auc_final_k))
    cat(sprintf("  Difference: %.4f\n", auc_inner_k - auc_final_k))
  }
}

cat("\nDone.\n")