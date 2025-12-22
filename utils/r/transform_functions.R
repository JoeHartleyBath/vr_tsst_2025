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
  transformed <- transform_deltas(
    df          = final_data,
    full_data   = full_data,
    feature_cols = feature_cols,
    method_map   = method_map
  )
  
  # Remove original (untransformed) versions of those features
  transformed_feats <- names(transformed) %>% str_subset("_Z$")
  original_feats <- setdiff(feature_cols, str_remove(transformed_feats, "_Z$"))
  
  if (length(original_feats) > 0) {
    cat("Dropping original feature versions:\n")
    print(head(original_feats))
    transformed <- transformed %>% select(-any_of(original_feats))
  }
  
  return(transformed)
}



transform_deltas <- function(df, full_data, feature_cols, method_map = list()) {
  
  # Convert feature names to baseline versions
  raw_feature_names <- str_remove(feature_cols, "_precond$")
  
  # Step 1 — compute baseline stats
  baseline_data <- full_data %>%
    filter(condition_type == "Relaxation") %>%
    group_by(participant_id) %>%
    summarise(across(all_of(raw_feature_names),
                     list(mean = ~mean(.x, na.rm = TRUE),
                          sd   = ~sd(.x, na.rm = TRUE)),
                     .names = "{.col}_{fn}"),
              .groups = "drop")
  
  # Step 2 — apply baseline normalisation only to change-precond features
  df <- df %>%
    group_by(participant_id) %>%
    mutate(across(
      all_of(feature_cols), 
      ~ {
        method <- method_map[[cur_column()]] %||% "z"
        raw <- .x
        
        # Apply local transform if specified
        x <- switch(method,
                    z            = raw,
                    log          = log(raw + 1),
                    log_z        = log(raw + 1),
                    signed_log_z = sign(raw) * log(abs(raw) + 1),
                    stop(paste("Unknown transform:", method))
        )
        
        # Now scale using MAD (not baseline)
        med <- median(x, na.rm = TRUE)
        mad_val <- mad(x, na.rm = TRUE)
        mad_val <- ifelse(mad_val <= 1e-6 | is.na(mad_val), 1, mad_val)
        
        (x - med) / mad_val
      },
      .names = "{.col}_Z"  # RZ = robust z-score
    )) %>%
    ungroup()
}








# Core transformer
transform_vector <- function(x, method = "z") {
  if (!is.numeric(x)) return(rep(NA, length(x)))
  
  x <- suppressWarnings(as.numeric(x))  # ensure numeric
  
  x_out <- switch(method,
                  z = scale(x)[,1],
                  log = log(x + 1),
                  log_z = scale(log(x + 1))[,1],
                  signed_log_z = {
                    if (any(x == 0, na.rm = TRUE)) x[x == 0] <- NA
                    scale(sign(x) * log(abs(x) + 1))[,1]
                  },
                  stop(paste("Unknown transformation method:", method))
  )
  
  return(x_out)
}


transform_and_clean <- function(df, metrics) {
  df <- df %>%
    group_by(Participant_ID) %>%
    mutate(across(all_of(unname(metrics)),
                  ~ scale(.x) %>% as.vector(),
                  .names = "{.col}_Z")) %>%
    ungroup()
  
  df <- df %>%
    mutate(across(ends_with("_Z"), cap_z_limits))
  
  if ("Time" %in% names(df)) {
    df <- df %>%
      relocate(ends_with("_Z"), .after = Time)
  }
  
  return(df)
}

clean_name_func <- function(x) {
  x %>% 
    str_remove("^(Roll_|Full_)") %>% 
    str_remove_all("_Z")
}




assign_transform <- function(metric) {
  metric <- tolower(metric)
  modality <- str_extract(metric, "^[^_]+")  # prefix before first underscore
  
  case_when(
    modality == "gsr"      ~ "signed_log_z",
    modality == "pupil"    ~ "signed_log_z",
    modality == "hr"       ~ "z",
    modality == "hrv"      ~ "signed_log_z",
    modality == "eeg"      ~ "signed_log_z",
    modality == "blink"    ~ "z",
    modality == "response" ~ "z",            # counts / rates / accuracy
    modality == "headset"  ~ "signed_log_z", # Head_Velocity
    TRUE                   ~ "z"            # fallback
  )
}




