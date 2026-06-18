#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(yardstick)
  library(parallel)
})

parse_kv_args <- function(argv) {
  # Supports: --key value, and boolean flags: --flag
  out <- list()
  i <- 1L
  while (i <= length(argv)) {
    a <- argv[[i]]
    if (!startsWith(a, "--")) {
      stop(sprintf("Unexpected argument (expected --key): %s", a))
    }
    key <- substring(a, 3)
    if (key == "") stop("Empty --key")
    # boolean flag if next arg is missing or starts with --
    if (i == length(argv) || startsWith(argv[[i + 1L]], "--")) {
      out[[key]] <- TRUE
      i <- i + 1L
    } else {
      out[[key]] <- argv[[i + 1L]]
      i <- i + 2L
    }
  }
  out
}

get_arg <- function(args, key, default = NULL) {
  if (!is.null(args[[key]])) return(args[[key]])
  default
}

as_int <- function(x, key) {
  if (is.null(x)) return(NULL)
  v <- suppressWarnings(as.integer(x))
  if (is.na(v)) stop(sprintf("Invalid integer for --%s: %s", key, x))
  v
}

as_num <- function(x, key) {
  if (is.null(x)) return(NULL)
  v <- suppressWarnings(as.numeric(x))
  if (is.na(v)) stop(sprintf("Invalid number for --%s: %s", key, x))
  v
}

auc_fast_from_ranks <- function(y01, ranks, n_pos, n_neg) {
  # y01: integer 0/1 vector
  # ranks: precomputed ranks of fixed score vector (ties averaged)
  # Returns pooled ROC AUC with positive class == 1
  if (n_pos == 0L || n_neg == 0L) return(NA_real_)
  offset <- n_pos * (n_pos + 1) / 2
  denom <- n_pos * n_neg
  (sum(ranks[y01 == 1L]) - offset) / denom
}

mean_pid_auc_fast <- function(y01, ranks_by_pid) {
  # ranks_by_pid: list of numeric rank vectors per pid (ties averaged within pid)
  # y01 is full-length vector; this is mainly for diagnostics.
  aucs <- numeric(length(ranks_by_pid))
  j <- 1L
  for (entry in ranks_by_pid) {
    idx <- entry$idx
    r <- entry$r
    y <- y01[idx]
    n_pos <- sum(y == 1L)
    n_neg <- sum(y == 0L)
    if (n_pos == 0L || n_neg == 0L) {
      aucs[j] <- NA_real_
    } else {
      offset <- n_pos * (n_pos + 1) / 2
      denom <- n_pos * n_neg
      aucs[j] <- (sum(r[y == 1L]) - offset) / denom
    }
    j <- j + 1L
  }
  mean(aucs, na.rm = TRUE)
}

shuffle_within_pid <- function(y, idx_list) {
  yp <- y
  for (idx in idx_list) {
    yp[idx] <- sample(yp[idx], length(idx), replace = FALSE)
  }
  yp
}

shuffle_global <- function(y) {
  sample(y, length(y), replace = FALSE)
}

shuffle_scores_within_pid <- function(score, idx_list) {
  sp <- score
  for (idx in idx_list) {
    sp[idx] <- sample(sp[idx], length(idx), replace = FALSE)
  }
  sp
}

shuffle_scores_global <- function(score) {
  sample(score, length(score), replace = FALSE)
}

get_rds_head_target <- function(null_auc, k) {
  as.numeric(head(null_auc, k))
}

is_on_grid <- function(x, denom, tol = 1e-8) {
  # Check whether values are near integer multiples of 1/denom
  all(abs(x * denom - round(x * denom)) <= tol)
}

find_seed_for_head <- function(
    target_head,
    gen_one,
    seed_min,
    seed_max,
    tol,
    progress_every = 50000L
) {
  found <- NA_integer_
  for (seed in seed_min:seed_max) {
    set.seed(seed)
    ok <- TRUE
    for (i in seq_along(target_head)) {
      v <- gen_one()
      if (is.na(v) || abs(v - target_head[i]) > tol) {
        ok <- FALSE
        break
      }
    }
    if (ok) {
      found <- seed
      break
    }
    if (progress_every > 0L && seed %% progress_every == 0L) {
      cat(sprintf("..seed %d\n", seed))
      flush.console()
    }
  }
  found
}

argv <- commandArgs(trailingOnly = TRUE)
args <- parse_kv_args(argv)

run_dir <- get_arg(args, "run-dir")
domain <- get_arg(args, "domain")
target <- get_arg(args, "target")
if (is.null(run_dir) || is.null(domain) || is.null(target)) {
  cat("Usage:\n")
  cat("  Rscript tools/infer_legacy_perm_method.R --run-dir <dir> --domain <all|eeg|peripheral> --target <stress_label|workload_label> [options]\n\n")
  cat("Options:\n")
  cat("  --head-k <int>              how many leading null values to match (default 5)\n")
  cat("  --seed-min <int>            seed search min (default 1)\n")
  cat("  --seed-max <int>            seed search max (default 200000)\n")
  cat("  --tol <num>                 match tolerance (default 1e-12)\n")
  cat("  --progress-every <int>      print progress every N seeds (default 50000)\n")
  cat("  --rng-mode <mode>           default|rounding|rejection|rng350|rng360 (default default)\n")
  cat("  --method <method>           within_pid_y|global_y|within_pid_score|global_score|mean_pid_auc_within_pid_y (default within_pid_y)\n")
  cat("  --verify-full               if set, verify full vector match once seed found\n")
  quit(status = 2)
}

head_k <- as_int(get_arg(args, "head-k", "5"), "head-k")
seed_min <- as_int(get_arg(args, "seed-min", "1"), "seed-min")
seed_max <- as_int(get_arg(args, "seed-max", "200000"), "seed-max")
tol <- as_num(get_arg(args, "tol", "1e-12"), "tol")
progress_every <- as_int(get_arg(args, "progress-every", "50000"), "progress-every")
rng_mode <- get_arg(args, "rng-mode", "default")
method <- get_arg(args, "method", "within_pid_y")
verify_full <- isTRUE(get_arg(args, "verify-full", FALSE))

pred_path <- file.path(run_dir, domain, sprintf("svm_predictions_%s.csv", target))
null_path <- file.path(run_dir, domain, sprintf("svm_perm_auc_%s.rds", target))

if (!file.exists(pred_path)) stop(sprintf("Missing predictions CSV: %s", pred_path))
if (!file.exists(null_path)) stop(sprintf("Missing perm null RDS: %s", null_path))

pred <- read.csv(pred_path)
null_auc <- readRDS(null_path)

if (!all(c("participant_id", "y_true", "y_prob") %in% names(pred))) {
  stop("predictions CSV missing required columns: participant_id, y_true, y_prob")
}

# RNG mode toggles (for R pre/post 3.6 sample differences)
switch(
  rng_mode,
  default = {
    # no-op
  },
  rounding = {
    RNGkind(sample.kind = "Rounding")
  },
  rejection = {
    RNGkind(sample.kind = "Rejection")
  },
  rng350 = {
    RNGversion("3.5.0")
  },
  rng360 = {
    RNGversion("3.6.0")
  },
  stop(sprintf("Unknown --rng-mode: %s", rng_mode))
)

idx_list <- split(seq_len(nrow(pred)), pred$participant_id)

y0 <- as.integer(pred$y_true)
score0 <- as.numeric(pred$y_prob)

# Diagnostics about null grid
cat(sprintf("Loaded null: class=%s len=%d\n", paste(class(null_auc), collapse = ","), length(null_auc)))
cat(sprintf("Loaded pred: n=%d participants=%d\n", nrow(pred), length(idx_list)))
cat(sprintf("Null head: %s\n", paste(format(head(null_auc, min(10, length(null_auc))), digits = 12), collapse = ", ")))

# Common pooled AUC invariants for fixed score
ranks_global <- rank(score0, ties.method = "average")

n_pos <- sum(y0 == 1L)
n_neg <- sum(y0 == 0L)

cat(sprintf("Pooled AUC grid denom pos*neg=%d; null on-grid=%s\n", n_pos * n_neg,
            if (is_on_grid(head(as.numeric(null_auc), min(50, length(null_auc))), n_pos * n_neg)) "TRUE" else "FALSE"))

# mean-per-participant AUC grid (for 44 participants w/ 4 rows each, step is 1/176 if each pid has 2/2)
# We compute its on-grid property empirically.
ranks_by_pid <- lapply(idx_list, function(idx) {
  list(idx = idx, r = rank(score0[idx], ties.method = "average"))
})
mean_head <- head(as.numeric(null_auc), min(50, length(null_auc)))
cat(sprintf("Mean-per-pid AUC (if used) would often be on-grid 1/176; null on-grid-1/176=%s\n",
            if (is_on_grid(mean_head, 176L)) "TRUE" else "FALSE"))

target_head <- get_rds_head_target(null_auc, head_k)

gen_one <- switch(
  method,
  within_pid_y = {
    function() {
      yp <- shuffle_within_pid(y0, idx_list)
      auc_fast_from_ranks(yp, ranks_global, n_pos, n_neg)
    }
  },
  global_y = {
    function() {
      yp <- shuffle_global(y0)
      auc_fast_from_ranks(yp, ranks_global, n_pos, n_neg)
    }
  },
  within_pid_score = {
    function() {
      sp <- shuffle_scores_within_pid(score0, idx_list)
      # must recompute ranks per permutation
      rg <- rank(sp, ties.method = "average")
      auc_fast_from_ranks(y0, rg, n_pos, n_neg)
    }
  },
  global_score = {
    function() {
      sp <- shuffle_scores_global(score0)
      rg <- rank(sp, ties.method = "average")
      auc_fast_from_ranks(y0, rg, n_pos, n_neg)
    }
  },
  mean_pid_auc_within_pid_y = {
    function() {
      yp <- shuffle_within_pid(y0, idx_list)
      mean_pid_auc_fast(yp, ranks_by_pid)
    }
  },
  stop(sprintf("Unknown --method: %s", method))
)

cat(sprintf("Searching seeds [%d, %d] for method=%s rng_mode=%s matching head_k=%d...\n",
            seed_min, seed_max, method, rng_mode, head_k))

found <- find_seed_for_head(
  target_head = target_head,
  gen_one = gen_one,
  seed_min = seed_min,
  seed_max = seed_max,
  tol = tol,
  progress_every = progress_every
)

cat(sprintf("FOUND_SEED %s\n", if (is.na(found)) "NA" else as.character(found)))

if (!is.na(found) && isTRUE(verify_full)) {
  set.seed(found)
  P <- length(null_auc)
  out <- numeric(P)
  for (i in 1:P) out[i] <- gen_one()
  maxdiff <- max(abs(out - as.numeric(null_auc)))
  cat(sprintf("FULL_VERIFY max_abs_diff=%0.16g\n", maxdiff))
  if (maxdiff <= tol) {
    cat("FULL_VERIFY OK\n")
  } else {
    cat("FULL_VERIFY FAIL\n")
  }
}
