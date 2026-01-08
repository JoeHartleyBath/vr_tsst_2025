suppressPackageStartupMessages({
  library(ggplot2)
})

infile <- file.path("output", "final_data.csv")
if (!file.exists(infile)) {
  stop("Missing input: ", infile)
}

out_dir <- file.path("results", "classic_analyses", "subjective")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

# Optional packages (installed on-demand) — consistent with subjective_anovas_and_plots.R
need_pkg <- function(pkg) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    message("Installing missing R package: ", pkg)
    install.packages(pkg, repos = "https://cloud.r-project.org")
  }
}

need_pkg("dplyr")
need_pkg("readr")
need_pkg("lme4")
need_pkg("tibble")

suppressPackageStartupMessages({
  library(dplyr)
  library(readr)
  library(lme4)
  library(tibble)
})

raw_df <- readr::read_csv(infile, show_col_types = FALSE)

required_cols <- c("participant_id", "condition", "stress", "workload")
missing_cols <- setdiff(required_cols, names(raw_df))
if (length(missing_cols) > 0) {
  stop("Missing required columns in final_data.csv: ", paste(missing_cols, collapse = ", "))
}

message("[Subjective ICC] Using full dataset (N=", n_distinct(raw_df$participant_id), " participants)")

# Derive within-subject factors from condition label (match subjective_anovas_and_plots.R)
df <- raw_df %>%
  mutate(
    participant_id = factor(participant_id),
    stress_level = factor(ifelse(grepl("^High Stress", condition), "High", "Low"), levels = c("Low", "High")),
    mwl_level = factor(ifelse(grepl("High Cog$", condition), "High", "Low"), levels = c("Low", "High")),
    stress_rating = as.numeric(stress),
    mental_demand = as.numeric(workload)
  )

icc_from_lmer <- function(dat, dv_col) {
  dat2 <- dat %>%
    select(participant_id, stress_level, mwl_level, dv = all_of(dv_col)) %>%
    filter(is.finite(dv))

  n_participants <- n_distinct(dat2$participant_id)
  n_obs <- nrow(dat2)

  if (n_participants < 2) {
    stop("Too few participants to compute ICC for ", dv_col)
  }

  fit <- lme4::lmer(dv ~ stress_level * mwl_level + (1 | participant_id), data = dat2, REML = TRUE)

  vc <- lme4::VarCorr(fit)
  var_participant <- as.numeric(vc$participant_id[1])
  var_residual <- (sigma(fit) ^ 2)

  icc <- var_participant / (var_participant + var_residual)

  tibble::tibble(
    dv = dv_col,
    n_participants = n_participants,
    n_obs = n_obs,
    var_participant = var_participant,
    var_residual = var_residual,
    icc_conditional = icc
  )
}

res <- dplyr::bind_rows(
  icc_from_lmer(df, "stress_rating"),
  icc_from_lmer(df, "mental_demand")
)

csv_path <- file.path(out_dir, "subjective_icc_results.csv")
txt_path <- file.path(out_dir, "subjective_icc_results.txt")

readr::write_csv(res, csv_path)

sink(txt_path)
cat("Subjective ICC (conditional on Stress×MWL fixed effects)\n")
cat("Model: dv ~ stress_level * mwl_level + (1|participant_id)\n")
cat("ICC = var(participant intercept) / [var(participant intercept) + var(residual)]\n\n")
print(res)
sink()

message("✓ Wrote: ", csv_path)
message("✓ Wrote: ", txt_path)
