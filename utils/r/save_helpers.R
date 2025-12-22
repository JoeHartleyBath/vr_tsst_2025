save_outputs <- function(final_data_transformed, full_data = NULL, config, final_data_eeg_valid = NULL) {
  
  out_dir <- config$paths$output
  dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
  
  # CRITICAL DEBUG: Check participant_id BEFORE saveRDS
  message(sprintf(
    "[DEBUG] BEFORE saveRDS: %d rows, participant_id type=%s, n_distinct=%d",
    nrow(final_data_transformed),
    paste(class(final_data_transformed$participant_id), collapse=", "),
    n_distinct(final_data_transformed$participant_id)
  ))
  message("[DEBUG] str() of participant_id:")
  str(final_data_transformed$participant_id, max.level = 2)
  
  # Save final transformed dataset (FULL - includes QC failures)
  saveRDS(final_data_transformed,
          file.path(out_dir, "final_data.rds"),
          compress = "xz")
  write_csv(final_data_transformed,
            file.path(out_dir, "final_data.csv"))
  
  cat("✔ Saved final transformed dataset (FULL N=47):\n",
      file.path(out_dir, "final_data.rds"), "\n",
      file.path(out_dir, "final_data.csv"), "\n")
  
  # CRITICAL DEBUG: Check participant_id AFTER saveRDS by reloading
  test_load <- readRDS(file.path(out_dir, "final_data.rds"))
  message(sprintf(
    "[DEBUG] AFTER saveRDS/readRDS: %d rows, participant_id type=%s, n_distinct=%d",
    nrow(test_load),
    paste(class(test_load$participant_id), collapse=", "),
    n_distinct(test_load$participant_id)
  ))
  message("[DEBUG] str() of reloaded participant_id:")
  str(test_load$participant_id, max.level = 2)
  
  # Save EEG-valid subset if provided
  if (!is.null(final_data_eeg_valid)) {
    saveRDS(final_data_eeg_valid,
            file.path(out_dir, "final_data_eeg_valid.rds"),
            compress = "xz")
    write_csv(final_data_eeg_valid,
              file.path(out_dir, "final_data_eeg_valid.csv"))
    
    cat("✔ Saved EEG-valid subset (N=", n_distinct(final_data_eeg_valid$participant_id), "):\n",
        file.path(out_dir, "final_data_eeg_valid.rds"), "\n",
        file.path(out_dir, "final_data_eeg_valid.csv"), "\n", sep = "")
  }
  
  # Optionally save full_data (pre-transform, baseline-adjusted raw)
  if (!is.null(full_data)) {
    saveRDS(full_data,
            file.path(out_dir, "full_data_for_reference.rds"),
            compress = "xz")
    
    cat("✔ Saved full_data snapshot for reference:\n",
        file.path(out_dir, "full_data_for_reference.rds"), "\n")
  }
  
  # Final summary output
  cat("📦 Final dataset includes", ncol(final_data_transformed), "features.\n")
}
