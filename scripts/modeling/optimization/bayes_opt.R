# =====================================================
# Parallel LOSO Bayesian XGBoost Tuning (BayesOpt)
# =====================================================

library(glue)
library(xgboost)
library(yardstick)
library(dplyr)
library(ParBayesianOptimization)


# =====================================================
# Main tuning function
# =====================================================
run_xgb_tuning <- function(df, feats, target,
                           bounds, iters,
                           bayes_save_path) {
  
  # -------------------------
  # Scoring within each call
  # -------------------------
  scoring_fun <- function(eta, max_depthL, subsample,
                          colsample_bytree, min_child_weightL,
                          lambda, alpha) {
    
    params <- list(
      objective        = "binary:logistic",
      eval_metric      = "logloss",
      tree_method      = "hist",
      nthread          = 1,          # Prevent over-subscription
      eta              = exp(eta),
      max_depth        = as.integer(round(max_depthL)),
      subsample        = subsample,
      colsample_bytree = colsample_bytree,
      min_child_weight = as.integer(round(min_child_weightL)),
      lambda           = exp(lambda),
      alpha            = exp(alpha)
    )
    
    pids <- unique(df$participant_id)
    scores <- numeric(length(pids))
    
    for (i in seq_along(pids)) {
      pid <- pids[i]
      
      train_df <- df[df$participant_id != pid, , drop = FALSE]
      test_df  <- df[df$participant_id == pid, , drop = FALSE]
      
      dtrain <- xgb.DMatrix(as.matrix(train_df[, feats]), label = train_df[[target]])
      dtest  <- xgb.DMatrix(as.matrix(test_df[, feats]),  label = test_df[[target]])
      
      # Inner CV for early stopping
      fold_idx <- make_pid_folds(train_df, k_max = 3)
      
      cv <- xgb.cv(
        params                = params,
        data                  = dtrain,
        nrounds               = 4000L,
        folds                 = fold_idx,
        early_stopping_rounds = 80,
        verbose               = FALSE
      )
      
      best_nrounds <- cv$best_iteration
      
      model_final <- xgb.train(
        params  = params,
        data    = dtrain,
        nrounds = best_nrounds,
        verbose = 3
      )
      
      pred  <- predict(model_final, dtest)
      truth <- factor(test_df[[target]], levels = c(0, 1))
      
      scores[i] <- yardstick::roc_auc_vec(
        truth = truth,
        estimate = pred,
        event_level = "second")
    }
    
    
    list(
      Score = mean(scores),  # MAXIMISE AUC
      Pred = NA_real_
    )
  }
  
  
  # -------------------------
  # Run Bayesian optimisation
  # -------------------------
  save_file <- file.path(bayes_save_path, glue("{target}_bayes_opt.rds"))
  
  result <- bayesOpt(
    FUN        = scoring_fun,
    bounds     = bounds,
    initPoints = 20,
    iters.n    = iters,
    iters.k    = 16,     # <–– THIS enables real parallel execution
    parallel   = TRUE,
    acq        = "ucb",
    kappa      = 2.5,
    verbose    = 3,
    saveFile   = NULL
  )
  
  #saveRDS(result, save_file)
  return(result)
}



