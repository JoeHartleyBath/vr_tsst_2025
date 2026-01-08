apply_transformations <- function(final_data, subjective_cols, full_data) {
  
  # Get all numeric features from final_data (ignoring metadata + subjective)
  feature_cols <- final_data %>%
    select(where(is.numeric), -any_of(c("participant_id", "round", "condition", subjective_cols))) %>%
    colnames()
  
  # 2. Keep only _precond features (the ones meant for baseline transform)
  feature_cols <- feature_cols[str_detect(feature_cols, "_precond$")]
  cat("Using", length(feature_cols), "features ending with _precond\n")
  
  # Figure out which transformation to apply per feature
  method_map <- setNames(
    vapply(feature_cols, assign_transform, character(1)),
    feature_cols
  )
  
  # Run the actual transformation
  apply_transformations <- function(final_data, subjective_cols, full_data) {
    # NOTE (2026-01): Feature transformations (log/signed-log) and z-scoring
    # were removed intentionally.
    #
    # Rationale: downstream analyses now work directly from aggregated
    # baseline-adjusted features (suffix: _precond), and any scaling
    # (e.g., within-subject z-score) is applied explicitly inside the
    # analysis scripts (e.g., SVM, correlations, moderation).
    #
    # This function remains for backwards compatibility with the pipeline.
    return(final_data)
  }
    cat("Dropping original feature versions:\n")

    print(head(original_feats))

    transformed <- transformed %>% select(-any_of(original_feats))

  }

  
