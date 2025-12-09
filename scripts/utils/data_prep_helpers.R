# scripts/utils/data_prep_helpers.R

load_and_prepare_data <- function(config) {
  # 1. Load raw data
  raw_data <- readr::read_csv(
    file.path(config$paths$output, "aggregated", "all_data_aggregated.csv"),
    show_col_types = FALSE
  )
  
  # 2. Drop QC failures
  failed_ids <- readr::read_csv(
    file.path(config$paths$failed_qc, "qc_failures_summary.csv"),
    show_col_types = FALSE
  ) %>%
    dplyr::pull(Participant_ID) %>%
    unique()
  
  data <- raw_data %>%
    dplyr::filter(!Participant_ID %in% failed_ids)
  
  # 3. Load and map counterbalance
  cb_long <- readxl::read_excel(
    file.path(config$paths$counterbalance, "counterbalance.xlsx")
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
        stringr::str_detect(Condition, regex("Relaxation", ignore_case = TRUE)) ~ "Relaxation",
        stringr::str_detect(Condition, regex("Pre", ignore_case = TRUE)) ~ "Pre_Baseline",
        stringr::str_detect(Condition, regex("Post", ignore_case = TRUE)) ~ "Post_Baseline",
        TRUE ~ NA_character_
      ),
      Relaxation_Level = dplyr::case_when(
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
      Full_Pupil_Dilation_Mean   = (Full_Foveal_Corrected_Dilation_Left_CLEANED_ABS_Mean +
                                      Full_Foveal_Corrected_Dilation_Right_CLEANED_ABS_Mean) / 2,
      Full_Pupil_Dilation_Min    = (Full_Foveal_Corrected_Dilation_Left_CLEANED_ABS_MIN +
                                      Full_Foveal_Corrected_Dilation_Right_CLEANED_ABS_MIN)  / 2,
      Full_Pupil_Dilation_Max    = (Full_Foveal_Corrected_Dilation_Left_CLEANED_ABS_MAX +
                                      Full_Foveal_Corrected_Dilation_Right_CLEANED_ABS_MAX) / 2,
      Full_Pupil_Dilation_Median = (Full_Foveal_Corrected_Dilation_Left_CLEANED_ABS_Median +
                                      Full_Foveal_Corrected_Dilation_Right_CLEANED_ABS_Median) / 2,
      Full_Pupil_Dilation_SD     = (Full_Foveal_Corrected_Dilation_Left_CLEANED_ABS_SD +
                                      Full_Foveal_Corrected_Dilation_Right_CLEANED_ABS_SD) / 2,
      Full_Pupil_Asymmetry = abs(
        Full_Foveal_Corrected_Dilation_Left_CLEANED_ABS_Mean - 
          Full_Foveal_Corrected_Dilation_Right_CLEANED_ABS_Mean),
        
      # --- EEG Ratio features (full-window only) ---
      Full_Alpha_Beta_Ratio =
               (Full_FrontalMidline_Alpha_Mean / Full_FrontalMidline_Beta_Mean),
      
      Full_Theta_Beta_Ratio =
               (Full_FrontalMidline_Theta_Mean / Full_FrontalMidline_Beta_Mean),

      Full_Frontal_Alpha_Asymmetry =
               (log(Full_FrontalRight_Alpha_Mean) -
                 log(Full_FrontalLeft_Alpha_Mean))
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
  
  # Rename
  names(data) <- vapply(names(data), rename_feature, character(1))
  
  # Define sets
  id_cols   <- c("participant_id", "round", "condition", "condition_type", "relaxation_level")
  subjective_cols <- intersect(
    c("stress", "workload", "calm", "happy", "sad", "pleasure", "arousal"),
    names(data)
  )
  full_cols <- str_subset(names(data), "_full$")
  
  # Extract full window values
  full_data <- data %>%
    select(any_of(c(id_cols, full_cols, subjective_cols))) %>%
    arrange(participant_id, condition) %>%
    group_by(participant_id, condition) %>%
    summarise(across(everything(), first), .groups = "drop")
  
  # Compute baselines
  glob_bl_full <- full_data %>%
    filter(condition_type == "Pre_Baseline") %>%
    select(participant_id, all_of(full_cols)) %>%
    pivot_longer(-participant_id, names_to = "metric", values_to = "glob_bl") %>%
    distinct(participant_id, metric, .keep_all = TRUE)
  
  precond_bl_full <- full_data %>%
    filter(condition_type == "Relaxation") %>%
    mutate(round = as.character(relaxation_level)) %>%
    select(participant_id, round, all_of(full_cols)) %>%
    pivot_longer(-c(participant_id, round), names_to = "metric", values_to = "precond_bl") %>%
    distinct(participant_id, round, metric, .keep_all = TRUE)
  
  # Get task values
  task_vals <- full_data %>%
    filter(condition_type == "Task") %>%
    mutate(round = as.character(round)) %>%
    select(participant_id, round, all_of(full_cols)) %>%
    pivot_longer(-c(participant_id, round), names_to = "metric", values_to = "value")
  
  # Compute deltas
  change_long <- task_vals %>%
    left_join(precond_bl_full, by = c("participant_id", "round", "metric")) %>%
    left_join(glob_bl_full,    by = c("participant_id", "metric")) %>%
    mutate(
      change_precond = value - precond_bl,
      change_glob    = value - glob_bl
    ) %>%
    select(participant_id, round, metric, value, change_precond, change_glob)
  
  change_long_clean <- change_long %>%
    pivot_longer(
      cols = c(value, change_precond, change_glob),
      names_to = "change_type",
      values_to = "value"
    ) %>%
    mutate(change_type = dplyr::recode(change_type, "value" = "raw"))%>%
    mutate(feature = str_c(metric, change_type, sep = "_")) %>%
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
  
  # add condition + subjective ratings
  condition_info <- full_data %>%
    select(participant_id, round, condition, any_of(subjective_cols)) %>%
    mutate(round = as.character(round))
  
  final_data <- change_wide %>%
    mutate(round = as.character(round)) %>%
    left_join(condition_info, by = c("participant_id", "round")) %>%
    relocate(condition, any_of(subjective_cols), .after = round)
  
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
    str_subset("_change_precond$")  # baseline-adjusted full-window features
  
  anova_dataset <- anova_data %>%
    select(
      participant_id,
      round,
      condition,
      stress_level,
      workload_level,
      any_of(subjective_cols),
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
  
  # 2. Drop irrelevant or invalid features
  garbage_patterns <- paste0(
    "(?i)", paste(c(
      "_wpli",
      "aperiodic",
      "_dilation_(left|right)",
      "rr_",
      "slope",
      "meaningful",
      "unrest",
      "blink",
      "resistance",
      "bpm_sd",
      "interval_sd",
      "sdnn",
      "pnn50",
      "conductance(?!_eda)",
      "(?=.*eeg)(?=.*sd)"# drop conductance unless part of conductance_eda
    ), collapse = "|")
  )
  
  garbage_features <- stringr::str_subset(all_feats, garbage_patterns)
  
  to_drop <- unique(c(raw_equivalents, garbage_features))
  
  message("Dropping ", length(to_drop), " features from final_data.")
  df %>% dplyr::select(-dplyr::any_of(to_drop))
}




