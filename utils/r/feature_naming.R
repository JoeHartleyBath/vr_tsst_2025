# ── 1 ▸ Lookup tables (all lower-case) ──────────────────────────────────
region_map  <- c(frontalleft="fl", frontalright="fr", overallfrontal="f",
                 frontalmidline="fm", temporal="t", central="c",
                 parietal="p", occipital="o")

band_map    <- c(delta="delta", theta="theta",
                 lowalpha="lalpha", highalpha="halpha", alpha="alpha",
                 lowbeta="lbeta", highbeta="hbeta", beta="beta")

metric_map  <- c(mean="mean", median="med", sd="sd",
                 min="min", max="max", slope="slope",
                 sampleentropy="sampent", spectralentropy="specent",
                 peakrate="pkrate", peakheight="pkht")

derived_eeg <- c(
  "frontal_alpha_asymmetry" = "faa",
  "alpha_beta_ratio"        = "ab_ratio",
  "theta_beta_ratio"        = "tb_ratio",
  "theta_alpha_ratio"       = "ta_ratio"
)

modality_regex <- list(
  eeg   = "(frontal|temporal|central|parietal|occipital|frontalmidline|alpha|theta)",
  eda   = "gsr|conductance|resistance|eda",
  hr    = "heartrate|bpm",
  hrv   = "rr_interval|rmssd|sdnn|pnn50",
  pupil = "pupil|dilation|foveal",
  blink = "blink",
  response = "response_(count|rate|latency|accuracy|correct|incorrect|usable)",
  headset  = "head(_|)velocity"
)

# ── 2 ▸ Minimal renamer ─────────────────────────────────────────────────
rename_feature <- function(col) {
  
  raw  <- col
  col  <- tolower(col)
  
  # --- NEW explicit renames -----------------------------------------------
  special_map <- c("subjective_stress" = "stress",
                   "nasa_mental"       = "workload",
                   "sample_frame" = "time")
  if (col %in% names(special_map)) return(special_map[[col]])
  # ------------------------------------------------------------------------
  
  # Strip window prefix (but don't preserve as tag) ------------------------
  col  <- str_remove(col, "^(rolling_|roll_|full_)")
  
  # Derived EEG quick exit ----------------------------------------------
  if (any(str_detect(col, names(derived_eeg)))) {
    base <- derived_eeg[str_subset(names(derived_eeg), col)[1]]
    return(paste("eeg", base, sep = "_"))
  }
  
  # Detect modality **before** replacements -----------------------------
  mods <- names(Filter(function(rx) str_detect(col, rx), modality_regex))
  modality <- if (length(mods)) mods[1] else "other"
  
  # No longer default EEG to roll - features without tag remain tagless
  
  # Strip vendor / unit / cleaning noise --------------------------------
  col <- str_remove_all(col,
                        "shimmer_d36a_|polar_|empatica_|cleaned_abs_cleaned_nk|cleaned_nk|cleaned_abs|cleaned|_abs|_nk|_us|_kohms")
  
  # Abbreviate region/band/metric tokens --------------------------------
  col <- str_replace_all(col, region_map)
  col <- str_replace_all(col, band_map)
  col <- str_replace_all(col, metric_map)
  
  # Tidy underscores -----------------------------------------------------
  col <- str_replace_all(col, "__+", "_") %>% str_remove("^_") %>% str_remove("_$")
  
  # Non-physio identifiers → just snake-case -----------------------------
  if (modality == "other") return(col)
  
  # -- assemble ---------------------------------------------------------------
  pieces <- str_split(col, "_", simplify = TRUE)
  metric <- pieces[, ncol(pieces)]
  source_vec <- if (ncol(pieces) > 1) pieces[, -ncol(pieces)] else character(0)
  
  # drop leading token if it duplicates the modality or related terms
  if (length(source_vec) && source_vec[1] %in% c(modality, "gsr", "skin", "conductance", "heartrate", "dilation"))
    source_vec <- source_vec[-1]
  
  # Further simplification: remove redundant words
  source_vec <- source_vec[!source_vec %in% c("skin", "conductance", "heartrate", "bpm", "eda", "dilation")]
  
  source <- paste(source_vec, collapse = "_")
  
  out_parts <- c(modality,                      # modality prefix (eda, hr, hrv, eeg, pupil)
                 source,                        # source / band (may be empty)
                 metric)                        # metric
  
  out <- paste(out_parts[out_parts != ""], collapse = "_") |>
    str_replace_all("_+", "_") |>          # tidy doubles
    str_remove("_$")                       # no trailing underscore
  
  
  out
}