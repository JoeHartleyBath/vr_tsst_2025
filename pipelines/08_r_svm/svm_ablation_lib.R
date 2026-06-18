library(tidyverse)
library(yaml)

msg <- function(fmt, ...) {
  cat(sprintf("[%s] %s\n", format(Sys.time(), "%Y-%m-%d %H:%M:%S"), sprintf(fmt, ...)))
}

get_env_int <- function(name, default) {
  raw <- Sys.getenv(name, unset = "")
  if (raw == "") return(as.integer(default))
  as.integer(raw)
}

get_env_str <- function(name, default = NA_character_) {
  raw <- Sys.getenv(name, unset = "")
  if (raw == "") return(default)
  raw
}

get_env_bool <- function(name, default = FALSE) {
  raw <- Sys.getenv(name, unset = "")
  if (raw == "") return(isTRUE(default))
  raw %in% c("1", "true", "TRUE", "yes", "YES")
}

stop_if_paths_exist <- function(paths, overwrite = FALSE) {
  existing <- paths[file.exists(paths)]
  if (length(existing) > 0 && !overwrite) {
    stop(
      sprintf(
        "Refusing to overwrite existing output(s): %s\nSet SVM_OVERWRITE=1 only if you intentionally want to overwrite.",
        paste(existing, collapse = ", ")
      )
    )
  }
}

read_domain_config <- function(path = "config/svm_ablation.yaml") {
  if (!file.exists(path)) {
    stop(sprintf("Missing domain config: %s", path))
  }
  yaml::read_yaml(path)
}

filter_features_for_domain <- function(features, domain_cfg, domain) {
  if (is.null(domain_cfg$domains) || is.null(domain_cfg$domains[[domain]])) {
    stop(sprintf("Unknown domain '%s' in config/svm_ablation.yaml", domain))
  }

  include <- domain_cfg$domains[[domain]]$include
  exclude <- domain_cfg$domains[[domain]]$exclude

  filtered <- features

  if (!is.null(include) && length(include) > 0) {
    keep <- rep(FALSE, length(filtered))
    for (pat in include) {
      keep <- keep | str_detect(filtered, pat)
    }
    filtered <- filtered[keep]
  }

  if (!is.null(exclude) && length(exclude) > 0) {
    drop <- rep(FALSE, length(filtered))
    for (pat in exclude) {
      drop <- drop | str_detect(filtered, pat)
    }
    filtered <- filtered[!drop]
  }

  filtered
}

derive_binary_targets_from_condition <- function(df) {
  if (!all(c("condition") %in% names(df))) {
    stop("Expected column 'condition' in dataset")
  }

  df %>%
    mutate(
      stress_label = factor(if_else(str_detect(condition, "High Stress"), 1L, 0L), levels = c(0, 1)),
      workload_label = factor(if_else(str_detect(condition, "High Cog"), 1L, 0L), levels = c(0, 1))
    )
}
