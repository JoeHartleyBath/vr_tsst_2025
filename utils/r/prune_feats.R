library(dplyr)
library(caret)


prune_features <- function(df, target, missing_thresh = 0.30, cor_cutoff = 0.75) {
  
  feats <- df %>%
    select(matches("_precond$")) %>%
    mutate(across(everything(), as.numeric))
  
  # 1. Remove features with too many missing values
  missing_rate <- sapply(feats, function(x) mean(is.na(x)))
  feats <- feats[, missing_rate < missing_thresh, drop = FALSE]
  
  # 2. Remove near-zero variance features
  nzv <- caret::nearZeroVar(feats, saveMetrics = TRUE)
  feats <- feats[, !(nzv$zeroVar | nzv$nzv), drop = FALSE]
  
  # 3. Remove highly correlated features
  if (ncol(feats) > 1) {
    repeat {
      cm <- cor(feats, use = "complete.obs")
      drop <- findCorrelation(cm, cutoff = cor_cutoff, names = TRUE)
      if (length(drop) == 0) break
      feats <- feats[, setdiff(colnames(feats), drop), drop = FALSE]
    }
  }
  
  final_feats <- colnames(feats)
  
  return(final_feats)
}





