# Quick Multi-Model Comparison - 5-Fold CV (No Tuning)
# ======================================================

library(dplyr)
library(stringr)
library(xgboost)
library(yardstick)
library(yaml)
library(caret)
library(e1071)  # SVM
library(MASS)   # LDA
library(randomForest)
library(nnet)   # Neural net

set.seed(42)

# Load config and data
config <- yaml::read_yaml("scripts/utils/config.yaml")

df <- readRDS(file.path(config$paths$output, "final_data.rds")) %>%
  mutate(
    stress_label   = factor(if_else(str_detect(condition, "High Stress"), 1L, 0L)),
    workload_label = factor(if_else(str_detect(condition, "High Cog"),    1L, 0L))
  )

# Get features
drop_pattern <- "response|eeg_o|(_min_|_max_)|head|totalscrs|_glob|_raw|_delta"
features <- names(df)[str_detect(names(df), "_precond$")]
features <- features[!str_detect(features, drop_pattern)]

df <- df %>%
  dplyr::select(participant_id, stress_label, workload_label, all_of(features)) %>%
  group_by(participant_id) %>%
  mutate(across(all_of(features), ~ scale(.)[,1])) %>%
  ungroup()

cat("\nData loaded:", nrow(df), "samples,", length(features), "features\n\n")

# Define models
train_models <- function(train_data, test_data, features, target) {
  X_train <- train_data[, features]
  y_train <- train_data[[target]]
  X_test <- test_data[, features]
  y_test <- test_data[[target]]
  
  # Impute NAs
  for (col in features) {
    med <- median(X_train[[col]], na.rm = TRUE)
    if (is.na(med)) med <- 0
    X_train[[col]][is.na(X_train[[col]])] <- med
    X_test[[col]][is.na(X_test[[col]])] <- med
  }
  
  results <- list()
  
  # 1. SVM (RBF kernel)
  tryCatch({
    model <- svm(x = X_train, y = y_train, kernel = "radial", 
                 cost = 32, gamma = 0.03, probability = TRUE, scale = FALSE)
    pred_obj <- predict(model, X_test, probability = TRUE)
    probs <- attr(pred_obj, "probabilities")[, "1"]
    results$SVM <- list(pred = pred_obj, prob = probs)
  }, error = function(e) {
    results$SVM <<- list(pred = rep("0", nrow(X_test)), prob = rep(0.5, nrow(X_test)))
  })
  
  # 2. LDA
  tryCatch({
    model <- lda(x = X_train, grouping = y_train)
    pred <- predict(model, X_test)
    results$LDA <- list(pred = pred$class, prob = pred$posterior[, "1"])
  }, error = function(e) {
    results$LDA <<- list(pred = rep("0", nrow(X_test)), prob = rep(0.5, nrow(X_test)))
  })
  
  # 3. Logistic Regression
  tryCatch({
    train_df <- cbind(y = y_train, X_train)
    model <- glm(y ~ ., data = train_df, family = binomial())
    probs <- predict(model, X_test, type = "response")
    preds <- factor(ifelse(probs > 0.5, "1", "0"))
    results$LogReg <- list(pred = preds, prob = probs)
  }, error = function(e) {
    results$LogReg <<- list(pred = rep("0", nrow(X_test)), prob = rep(0.5, nrow(X_test)))
  })
  
  # 4. Random Forest
  tryCatch({
    model <- randomForest(x = X_train, y = y_train, ntree = 200, 
                         mtry = max(1, floor(sqrt(length(features)))))
    preds <- predict(model, X_test)
    probs <- predict(model, X_test, type = "prob")[, "1"]
    results$RandomForest <- list(pred = preds, prob = probs)
  }, error = function(e) {
    results$RandomForest <<- list(pred = rep("0", nrow(X_test)), prob = rep(0.5, nrow(X_test)))
  })
  
  # 5. Neural Network
  tryCatch({
    train_df <- cbind(y = as.numeric(as.character(y_train)), X_train)
    model <- nnet(y ~ ., data = train_df, size = 10, trace = FALSE, 
                  maxit = 200, linout = FALSE)
    probs <- predict(model, X_test, type = "raw")[,1]
    preds <- factor(ifelse(probs > 0.5, "1", "0"))
    results$NeuralNet <- list(pred = preds, prob = probs)
  }, error = function(e) {
    results$NeuralNet <<- list(pred = rep("0", nrow(X_test)), prob = rep(0.5, nrow(X_test)))
  })
  
  # 6. XGBoost
  tryCatch({
    X_train_mat <- as.matrix(X_train)
    X_test_mat <- as.matrix(X_test)
    X_train_mat[is.na(X_train_mat)] <- 0
    X_test_mat[is.na(X_test_mat)] <- 0
    
    dtrain <- xgb.DMatrix(X_train_mat, label = as.numeric(as.character(y_train)))
    model <- xgb.train(
      params = list(objective = "binary:logistic", eta = 0.1, max_depth = 6),
      data = dtrain, nrounds = 100, verbose = 0
    )
    probs <- predict(model, xgb.DMatrix(X_test_mat))
    preds <- factor(ifelse(probs > 0.5, "1", "0"))
    results$XGBoost <- list(pred = preds, prob = probs)
  }, error = function(e) {
    results$XGBoost <<- list(pred = rep("0", nrow(X_test)), prob = rep(0.5, nrow(X_test)))
  })
  
  return(results)
}

# Run for both targets
targets <- c("stress_label", "workload_label")

for (target in targets) {
  cat(rep("=", 70), "\n")
  cat("TARGET:", toupper(gsub("_", " ", target)), "\n")
  cat(rep("=", 70), "\n\n")
  
  # Create 5 folds
  folds <- createFolds(df[[target]], k = 5, list = TRUE, returnTrain = FALSE)
  
  # Storage for predictions
  all_results <- list(
    SVM = list(preds = numeric(), probs = numeric()),
    LDA = list(preds = numeric(), probs = numeric()),
    LogReg = list(preds = numeric(), probs = numeric()),
    RandomForest = list(preds = numeric(), probs = numeric()),
    NeuralNet = list(preds = numeric(), probs = numeric()),
    XGBoost = list(preds = numeric(), probs = numeric())
  )
  all_truth <- character()
  
  for (i in seq_along(folds)) {
    test_idx <- folds[[i]]
    train_idx <- setdiff(1:nrow(df), test_idx)
    
    train_data <- df[train_idx, ]
    test_data <- df[test_idx, ]
    
    fold_results <- train_models(train_data, test_data, features, target)
    
    # Store predictions
    for (model_name in names(fold_results)) {
      all_results[[model_name]]$preds <- c(all_results[[model_name]]$preds, 
                                           as.character(fold_results[[model_name]]$pred))
      all_results[[model_name]]$probs <- c(all_results[[model_name]]$probs, 
                                           fold_results[[model_name]]$prob)
    }
    all_truth <- c(all_truth, as.character(test_data[[target]]))
    
    cat("  Fold", i, "complete\n")
  }
  
  # Evaluate all models
  cat("\n")
  cat(sprintf("%-15s %8s %8s %8s\n", "Model", "Accuracy", "F1", "AUC"))
  cat(rep("-", 45), "\n")
  
  results_summary <- list()
  
  for (model_name in names(all_results)) {
    truth_factor <- factor(all_truth, levels = c("0", "1"))
    pred_factor <- factor(all_results[[model_name]]$preds, levels = c("0", "1"))
    probs <- all_results[[model_name]]$probs
    
    acc <- mean(pred_factor == truth_factor)
    f1 <- f_meas_vec(truth = truth_factor, estimate = pred_factor, event_level = "second")
    auc <- roc_auc_vec(truth = truth_factor, estimate = probs, event_level = "second")
    
    results_summary[[model_name]] <- list(acc = acc, f1 = f1, auc = auc)
    
    cat(sprintf("%-15s %8.3f %8.3f %8.3f\n", model_name, acc, f1, auc))
  }
  
  # Find best
  best_model <- names(which.max(sapply(results_summary, function(x) x$auc)))
  best_auc <- results_summary[[best_model]]$auc
  
  cat("\n")
  cat("Best Model:", best_model, "(AUC:", round(best_auc, 3), ")\n")
  cat("\n")
}

cat(rep("=", 70), "\n")
cat("SUMMARY\n")
cat(rep("=", 70), "\n")
cat("Your current SVM LOSO results (from overnight run):\n")
cat("  Stress:   AUC = 0.585 (k=5 features, tuned)\n")
cat("  Workload: AUC = 0.659 (k=5 features, tuned)\n")
cat("\nConclusion: Use the model with highest AUC for each target.\n")
cat(rep("=", 70), "\n\n")
