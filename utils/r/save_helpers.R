save_outputs <- function(final_data_transformed, full_data = NULL, config) {
  
  out_dir <- config$paths$output
  dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
  
  # Save final transformed dataset
  saveRDS(final_data_transformed,
          file.path(out_dir, "final_data.rds"),
          compress = "xz")
  write_csv(final_data_transformed,
            file.path(out_dir, "final_data.csv"))
  
  cat("✔ Saved final transformed dataset:\n",
      file.path(out_dir, "final_data.rds"), "\n",
      file.path(out_dir, "final_data.csv"), "\n")
  
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
