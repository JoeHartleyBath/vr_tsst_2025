# scripts/utils/data_prep_helpers.R

source("utils/r/feature_selection.R")

load_and_prepare_data <- function(config) {
  # 1. Load raw data using data.table::fread() for reliable wide CSV parsing
  # NOTE: readr::read_csv() and base::read.csv() have known bugs with wide CSVs (>100 columns)
  # that cause participant_id to be corrupted as nested data.frame
  library(data.table)
  raw_data_dt <- data.table::fread(
    file.path(config$paths$output, "aggregated", "all_data_aggregated.csv")
  )
  
  # Convert to tibble for dplyr compatibility
  raw_data <- dplyr::as_tibble(raw_data_dt)
  
  # Verify participant_id loaded correctly
  if (!is.numeric(raw_data$Participant_ID)) {
    stop(sprintf(
      "CRITICAL: Participant_ID corrupted after loading! Type: %s",
      paste(class(raw_data$Participant_ID), collapse=", ")
    ))
  }
  message(sprintf(
    "[DEBUG] Loaded CSV: %d rows, Participant_ID type=%s, n_distinct=%d",
    nrow(raw_data),
    class(raw_data$Participant_ID)[1],
    n_distinct(raw_data$Participant_ID)
  ))
  
  # 2. Load QC failures and add flag (DO NOT DROP participants)
  # Rationale: QC exclusions should only apply to EEG-dependent analyses.
  # Subjective/behavioral analyses retain all 47 participants for maximum power.
  failed_ids <- readr::read_csv(
    file.path(config$paths$failed_qc, "qc_failures_summary.csv"),
    show_col_types = FALSE
  ) %>%
    dplyr::pull(Participant_ID) %>%
    unique()
  
  # Convert "P02" format to numeric 2 for matching
  failed_ids_numeric <- as.numeric(gsub("P", "", failed_ids))
  
  # Add QC flag column instead of filtering
  data <- raw_data %>%
    dplyr::mutate(
      qc_failed = Participant_ID %in% failed_ids_numeric
    )
  
  message(sprintf(
    "[QC] Loaded %d participants: %d passed EEG QC, %d failed (flagged but retained)",
    n_distinct(data$Participant_ID),
    n_distinct(data$Participant_ID[!data$qc_failed]),
    n_distinct(data$Participant_ID[data$qc_failed])
  ))
  
  # 3. Load and map counterbalance
  cb_long <- readxl::read_excel(
    file.path(config$paths$counterbalance, "experimental_counterbalance.xlsx")
  ) %>%
    dplyr::rename(Participant_ID = Participant) %>%
    tidyr::pivot_longer(
      cols = starts_with("Round"),
      names_to = "Round",
      values_to = "CB_Condition"
    ) %>%
    dplyr::mutate(
      Round = as.integer(stringr::str_extract(Round, "\\d+")),
      Condition = dplyr::case_when(
        trimws(CB_Condition) == "Stress Subtraction" ~ "High Stress - High Cog",
        trimws(CB_Condition) == "Stress Addition"    ~ "High Stress - Low Cog",
        trimws(CB_Condition) == "Calm Addition"      ~ "Low Stress - Low Cog",
        trimws(CB_Condition) == "Calm Subtraction"   ~ "Low Stress - High Cog",
        TRUE                                          ~ NA_character_
      )
    ) %>%
    dplyr::select(-CB_Condition)
  
  # 4. Apply experimental condition variables + pupil aggregation
  data <- data %>%
    dplyr::mutate(
      Condition_Type = dplyr::case_when(
        stringr::str_detect(Condition, regex("Task", ignore_case = TRUE)) ~ "Task",
        stringr::str_detect(Condition, regex("Forest", ignore_case = TRUE)) ~ "Relaxation",
        stringr::str_detect(Condition, regex("Relaxation", ignore_case = TRUE)) ~ "Relaxation",
        stringr::str_detect(Condition, regex("Pre", ignore_case = TRUE)) ~ "Pre_Baseline",
        stringr::str_detect(Condition, regex("Post", ignore_case = TRUE)) ~ "Post_Baseline",
        TRUE ~ NA_character_
      ),
      Relaxation_Level = dplyr::case_when(
        str_detect(Condition, regex("Forest1", ignore_case = TRUE)) ~ "1",
        str_detect(Condition, regex("Forest2", ignore_case = TRUE)) ~ "2",
        str_detect(Condition, regex("Forest3", ignore_case = TRUE)) ~ "3",
        str_detect(Condition, regex("Forest4", ignore_case = TRUE)) ~ "4",
        str_detect(Condition, regex("Relaxation1", ignore_case = TRUE)) ~ "1",
        str_detect(Condition, regex("Relaxation2", ignore_case = TRUE)) ~ "2",
        str_detect(Condition, regex("Relaxation3", ignore_case = TRUE)) ~ "3",
        str_detect(Condition, regex("Relaxation4", ignore_case = TRUE)) ~ "4",
        TRUE ~ NA_character_
      ),
      Condition = dplyr::case_when(
        str_detect(Condition, regex("HighStress", ignore_case = TRUE)) &
          str_detect(Condition, regex("HighCog", ignore_case = TRUE)) ~ "High Stress - High Cog",
        str_detect(Condition, regex("HighStress", ignore_case = TRUE)) &
          str_detect(Condition, regex("LowCog", ignore_case = TRUE))  ~ "High Stress - Low Cog",
        str_detect(Condition, regex("LowStress", ignore_case = TRUE)) &
          str_detect(Condition, regex("HighCog", ignore_case = TRUE)) ~ "Low Stress - High Cog",
        str_detect(Condition, regex("LowStress", ignore_case = TRUE)) &
          str_detect(Condition, regex("LowCog", ignore_case = TRUE))  ~ "Low Stress - Low Cog",
        TRUE ~ Condition
      ),
      # NOTE: Pupil features already computed in physio extraction
      # Full_Pupil_Dilation_Mean, Median, SD, Asymmetry exist in merged data
        
      # NOTE: EEG ratio features - check if already exist or compute if needed
      # Features use _Power suffix not _Mean
      # Full_Alpha_Beta_Ratio, Full_Theta_Beta_Ratio, Full_Frontal_Alpha_Asymmetry
    ) %>%
    left_join(
      cb_long,
      by = c("Participant_ID", "Condition"),
      relationship = "many-to-many"
    ) %>%
    mutate(
      Round = as.character(coalesce(Round, 0L))
    ) %>%
    select(-matches("Foveal_Corrected_Dilation_(Left|Right)_CLEANED_ABS", ignore.case = TRUE))
  
  return(data)
}

prepare_full_window_data <- function(data) {
  # Drop baseline remnants
  data <- data %>% select(-matches("baseline", ignore.case = TRUE))
  
  # Preserve qc_failed column through processing
  has_qc_flag <- "qc_failed" %in% names(data)
  
  # Get canonical features from config to preserve their names
  canonical_feats <- config$canonical_features
  
  # Rename all columns except canonical features and qc_failed
  col_names <- names(data)
  preserve_cols <- c(canonical_feats, "qc_failed")
  for (i in seq_along(col_names)) {
    if (!col_names[i] %in% preserve_cols) {
      col_names[i] <- rename_feature(col_names[i])
    }
  }
  names(data) <- col_names
  
  # Define sets
  id_cols   <- c("participant_id", "round", "condition", "condition_type", "relaxation_level")
  if (has_qc_flag) id_cols <- c(id_cols, "qc_failed")
  
  subjective_cols <- intersect(
    c("stress", "workload", "calm", "happy", "sad", "pleasure", "arousal"),
    names(data)
  )
  
  # All features in aggregated data are full-window features
  # Select all numeric columns except IDs and subjective ratings
  feature_cols <- names(data %>% select(where(is.numeric), -any_of(c(id_cols, subjective_cols))))
  
  # Extract full window values
  # CRITICAL: Do NOT summarize grouping columns (participant_id, condition) 
  # or dplyr will nest them as data.frames. Only summarize non-grouping columns.
  summarize_cols <- c(feature_cols, subjective_cols, "round", "condition_type", "relaxation_level")
  if (has_qc_flag) summarize_cols <- c(summarize_cols, "qc_failed")
  
  full_data <- data %>%
    select(any_of(c(id_cols, feature_cols, subjective_cols))) %>%
    arrange(participant_id, condition) %>%
    group_by(participant_id, condition) %>%
    summarise(across(all_of(summarize_cols), first), .groups = "drop")
  
  # Verify participant_id integrity after summarise
  if (!is.numeric(full_data$participant_id)) {
    stop(sprintf(
      "CRITICAL BUG: participant_id corrupted during summarise! Type: %s. This is a dplyr bug with group_by/summarise.",
      paste(class(full_data$participant_id), collapse=", ")
    ))
  }
  message(sprintf(
    "[DEBUG] After summarise: %d rows, participant_id type=%s, n_distinct=%d",
    nrow(full_data),
    class(full_data$participant_id)[1],
    n_distinct(full_data$participant_id)
  ))
  
  # Compute baselines
  glob_bl_full <- full_data %>%
    filter(condition_type == "Pre_Baseline") %>%
    select(participant_id, all_of(feature_cols)) %>%
    pivot_longer(-participant_id, names_to = "metric", values_to = "glob_bl") %>%
    distinct(participant_id, metric, .keep_all = TRUE)
  
  precond_bl_full <- full_data %>%
    filter(condition_type == "Relaxation") %>%
    mutate(round = as.character(relaxation_level)) %>%
    select(participant_id, round, all_of(feature_cols)) %>%
    pivot_longer(-c(participant_id, round), names_to = "metric", values_to = "precond_bl") %>%
    distinct(participant_id, round, metric, .keep_all = TRUE)
  
  # Get task values
  task_vals <- full_data %>%
    filter(condition_type == "Task") %>%
    mutate(round = as.character(round)) %>%
    select(participant_id, round, all_of(feature_cols)) %>%
    pivot_longer(-c(participant_id, round), names_to = "metric", values_to = "value")
  
  # Compute deltas (only precondition baseline)
  change_long <- task_vals %>%
    left_join(precond_bl_full, by = c("participant_id", "round", "metric")) %>%
    mutate(change_precond = value - precond_bl) %>%
    select(participant_id, round, metric, change_precond)
  
  change_long_clean <- change_long %>%
    mutate(feature = str_c(metric, "precond", sep = "_")) %>%
    rename(value = change_precond) %>%
    select(participant_id, round, feature, value)
  
  
  list(
    full_data = full_data,
    full_change_long = change_long_clean
  )
}


wide_transform_full_changes <- function(full_data, full_change_long, subjective_cols) {
  
  # wide transform. Keep all value types.
  change_wide <- full_change_long %>%
    select(participant_id, round, feature, value) %>%
    pivot_wider(names_from = feature, values_from = value)
  
  # Verify participant_id integrity after pivot_wider
  if (!is.numeric(change_wide$participant_id)) {
    stop(sprintf(
      "CRITICAL BUG: participant_id corrupted during pivot_wider! Type: %s",
      paste(class(change_wide$participant_id), collapse=", ")
    ))
  }
  message(sprintf(
    "[DEBUG] After pivot_wider: %d rows, participant_id type=%s, n_distinct=%d",
    nrow(change_wide),
    class(change_wide$participant_id)[1],
    n_distinct(change_wide$participant_id)
  ))
  
  # add condition + subjective ratings + qc_failed flag
  condition_info <- full_data %>%
    select(participant_id, round, condition, any_of(c(subjective_cols, "qc_failed"))) %>%
    mutate(round = as.character(round))
  
  final_data <- change_wide %>%
    mutate(round = as.character(round)) %>%
    left_join(condition_info, by = c("participant_id", "round")) %>%
    relocate(condition, any_of(subjective_cols), .after = round)
  
  # Final verification before returning
  if (!is.numeric(final_data$participant_id)) {
    stop(sprintf(
      "CRITICAL BUG: participant_id corrupted during left_join! Type: %s",
      paste(class(final_data$participant_id), collapse=", ")
    ))
  }
  message(sprintf(
    "[DEBUG] Final output from wide_transform: %d rows, participant_id type=%s, n_distinct=%d",
    nrow(final_data),
    class(final_data$participant_id)[1],
    n_distinct(final_data$participant_id)
  ))
  
  return(final_data)
}


make_anova_dataset <- function(final_data, subjective_cols, config) {
  
  anova_data <- final_data %>%
    mutate(
      stress_level = factor(
        if_else(str_detect(condition, "High Stress"), "High", "Low"),
        levels = c("Low", "High")
      ),
      workload_level = factor(
        if_else(str_detect(condition, "High Cog"), "High", "Low"),
        levels = c("Low", "High")
      )
    )
  
  anova_features <- names(anova_data) %>% 
    str_subset("_precond$")  # baseline-adjusted features
  
  anova_dataset <- anova_data %>%
    select(
      participant_id,
      round,
      condition,
      stress_level,
      workload_level,
      any_of(c(subjective_cols, "qc_failed")),
      all_of(anova_features)
    ) %>%
    relocate(condition, .after = round)
  
  out_dir <- config$paths$output
  saveRDS(anova_dataset, file.path(out_dir, "anova_features_precond.rds"), compress = "xz")
  write_csv(anova_dataset, file.path(out_dir, "anova_features_precond.csv"))
  
  cat("\n✔ ANOVA dataset saved to:\n",
      file.path(out_dir, "anova_features_precond.*"), "\n")
  
  return(anova_dataset)
}

clean_feature_duplicates <- function(df) {
  all_feats <- names(df)
  
  # 1. Drop duplicated feature variants where raw equivalents exist
  duplicated_variants <- stringr::str_subset(all_feats, "(?i)(_abs_|_nk_)")
  raw_equivalents <- duplicated_variants %>%
    stringr::str_replace_all("(?i)(_abs_|_nk_)", "_") %>%
    intersect(all_feats)
  
  # 2. Drop irrelevant or invalid features using centralized pattern
  garbage_patterns <- get_feature_drop_pattern()
  garbage_features <- stringr::str_subset(all_feats, garbage_patterns)
  
  to_drop <- unique(c(raw_equivalents, garbage_features))
  
  message("Dropping ", length(to_drop), " features from final_data.")
  df %>% dplyr::select(-dplyr::any_of(to_drop))
}




