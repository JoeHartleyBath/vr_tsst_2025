# =====================================================================
# feature_selection.R
# Centralized feature selection logic for all analysis scripts
# =====================================================================

# Consolidated drop pattern for feature exclusion
# This pattern excludes features that are:
# - Response-related metrics
# - Occipital EEG (eeg_o)
# - Min/max aggregations (too noisy)
# - Headset/head motion
# - Total SCRs (redundant with other EDA metrics)
# - Global baseline variants (_glob)
# - Raw (unadjusted) features (_raw)
# - Delta band (too low frequency)
# - Asymmetry metrics (experimental)
# - WPLI connectivity (not used in current analyses)
# - Aperiodic components
# - Individual pupil dilation (left/right - use combined)
# - RR interval raw values
# - Slope features
# - Blink and unrest metrics
# - Resistance (prefer conductance)
# - Redundant variability metrics (bpm_sd, interval_sd, sdnn, pnn50)
# - Conductance features that aren't EDA-processed
get_feature_drop_pattern <- function() {
  paste0(
    "(?i)", paste(c(
      "response",
      "eeg_o",
      "_min_|_max_",
      "head",
      "totalscrs",
      "_glob",
      "_raw",
      "_delta",
      "_assym",
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
      "conductance(?!_eda)"  # drop conductance unless part of conductance_eda
    ), collapse = "|")
  )
}

#' Select features for analysis from a dataframe
#'
#' @param df Dataframe containing features
#' @param suffix Feature suffix to match (e.g., "_precond", "_precond_Z")
#' @param exclude_pattern Optional custom exclusion pattern (uses default if NULL)
#' @param canonical_only Logical; if TRUE, only returns canonical features from config
#' @param config Optional config object (required if canonical_only = TRUE)
#'
#' @return Character vector of feature names
#'
#' @examples
#' # Get all baseline-adjusted features
#' select_analysis_features(df, suffix = "_precond")
#'
#' # Get normalized features for correlations
#' select_analysis_features(df, suffix = "_precond_Z")
#'
#' # Get only canonical features
#' select_analysis_features(df, suffix = "_precond", canonical_only = TRUE, config = config)
select_analysis_features <- function(df, 
                                     suffix = "_precond",
                                     exclude_pattern = NULL,
                                     canonical_only = FALSE,
                                     config = NULL) {
  
  # Build exclusion pattern
  if (is.null(exclude_pattern)) {
    exclude_pattern <- get_feature_drop_pattern()
  }
  
  # Get features matching suffix
  features <- names(df)[stringr::str_detect(names(df), paste0(suffix, "$"))]
  
  # Apply exclusion pattern
  features <- features[!stringr::str_detect(features, exclude_pattern)]
  
  # If canonical only, filter to config canonical features
  if (canonical_only) {
    if (is.null(config)) {
      stop("Config object required when canonical_only = TRUE")
    }
    canonical_feats <- paste0(config$canonical_features, suffix)
    features <- intersect(features, canonical_feats)
  }
  
  return(features)
}

#' Get canonical features with specified suffix
#'
#' @param config Config object from yaml::read_yaml()
#' @param suffix Suffix to append to canonical feature names
#'
#' @return Character vector of canonical feature names with suffix
#'
#' @examples
#' get_canonical_features(config, suffix = "_precond")
get_canonical_features <- function(config, suffix = "_precond") {
  paste0(config$canonical_features, suffix)
}
