# SVM results folder

This folder contains SVM LOSO evaluation logs and derived summaries.

## Final vs archive
- `final/`: the canonical inputs used for the paper (what the best-k summariser reads if present)
- `archive/`: alternative runs/variants kept for traceability (e.g., `_1`, `_full_feats`)

## Why does the best-k summary report `n=40`?
The best-k script groups metrics across LOSO folds (`test_pid`). In the current progress files there are 40 unique `test_pid` values, so the per-k averages are computed over 40 held-out participants.

## Best-k summariser
- Script: `pipelines/08_r_svm/svm_best_model_finder.R`
- Input directory: `results/svm/final/` if it exists, otherwise `results/svm/`
- Optional outputs: set `SVM_BEST_FINDER_OUTDIR` to write summary CSVs.
