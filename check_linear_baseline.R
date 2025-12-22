library(tidyverse)
library(e1071)
library(caret)

# 1. Load the EXACT dataset used for the manuscript
df <- readRDS("output/final_data_eeg_valid.rds")

cat("Loaded dataset with dimensions:", dim(df), "\n")

# 2. Create Binary Labels from Condition (Same logic as main SVM)
df <- df %>%
  mutate(
    stress_label = factor(if_else(str_detect(condition, "High Stress"), "High", "Low"), 
                          levels = c("Low", "High")),
    workload_label = factor(if_else(str_detect(condition, "High Cog"), "High", "Low"), 
                            levels = c("Low", "High"))
  )

# 3. Setup 5-Fold CV
set.seed(42)
folds <- groupKFold(df$participant_id, k = 5)
ctrl  <- trainControl(method = "cv", index = folds, classProbs = TRUE, summaryFunction = twoClassSummary)

targets <- c("stress_label", "workload_label")
results <- list()

for (target in targets) {
  cat("\n============================================\n")
  cat(" ANALYZING TARGET:", target, "\n")
  cat("============================================\n")
  
  # Select features: Z-scored features
  feature_cols <- names(df)[grep("_precond_Z$", names(df))]
  
  # CRITICAL: Manually exclude subjective ratings (Leakage Fix)
  feature_cols <- feature_cols[!grepl("nasa|imi|mps", feature_cols, ignore.case = TRUE)]
  
  # Remove features with missing values (mimics prune_features)
  X <- df[, feature_cols]
  missing_cols <- sapply(X, function(x) any(is.na(x)))
  if(any(missing_cols)) {
      cat("Dropping", sum(missing_cols), "columns with missing values.\n")
      X <- X[, !missing_cols]
  }

  cat("Selected", ncol(X), "features.\n")
  
  # Remove near-zero variance
  nzv <- nearZeroVar(X)
  if(length(nzv) > 0) {
    cat("Removing", length(nzv), "near-zero variance features.\n")
    X <- X[, -nzv]
  }
  
  # Prepare Data
  model_data <- cbind(X, Class = df[[target]])
  
  # 4. Train Linear SVM
  cat("Training Linear SVM...\n")
  fit_linear <- train(
    Class ~ ., data = model_data,
    method = "svmLinear",
    trControl = ctrl,
    metric = "ROC"
  )
  
  # 5. Train RBF SVM
  cat("Training RBF SVM...\n")
  fit_rbf <- train(
    Class ~ ., data = model_data,
    method = "svmRadial",
    trControl = ctrl,
    metric = "ROC"
  )
  
  # 6. Compare
  res <- resamples(list(Linear = fit_linear, RBF = fit_rbf))
  print(summary(res))
  
  results[[target]] <- summary(res)
}
