# SVM ablation runs

This folder contains **domain ablation** + **permutation test** + **bootstrap CI** runs for the Stage 08 SVM pipeline.

## Canonical vs ablation pipeline

- Canonical/paper runner: `pipelines/08_r_svm/svm.R`
  - Writes `svm_progress_<target>.csv` and `_tuning.csv`
  - Does **not** lock `k` and does **not** save out-of-sample probabilities.

- Ablation observed-run: `pipelines/08_r_svm/svm_ablation_observed.R`
  - Locks `k` via `SVM_K` (default `5`)
  - Supports `SVM_DOMAIN=all|eeg|peripheral`
  - Saves out-of-sample predictions: `svm_predictions_<target>.csv`
  - Saves fold plan: `svm_fold_plan_<target>.rds`

- Inference: `pipelines/08_r_svm/svm_inference_ablation.R`
  - Bootstrap CIs (cluster bootstrap over participants)
  - Permutation test with `P=1000` by default (`SVM_PERM_P`)
  - Writes `svm_inference_summary.csv` and permutation distributions

## How to run

Use the repo-root runner:

`./run_svm_ablation_perm_ci.ps1 -RunTag <tag> -PermP 1000 -K 5`

### Safety (no overwrites by default)

- The runner refuses to write into an existing `results/svm_ablation_runs/<tag>/` directory.
- Use a fresh `-RunTag` for each run.
- If you intentionally need to rerun the same tag, pass `-Overwrite` to the runner.
  - For safety, `-Overwrite` will rename the existing run directory to a timestamped `_backup_...` folder instead of deleting it.

Outputs are written to:

`results/svm_ablation_runs/<tag>/<domain>/`

Each domain folder includes:
- `svm_observed.log` (observed training run log)
- `svm_inference.log` (bootstrap + permutation log)
- `svm_inference_summary.csv` (final metrics)

## Notes
- Primary metric: ROC AUC.
- Permutations are label shuffles **within participant** to respect repeated measures.
  - Implementation: shuffle `y_true` within each `participant_id` and recompute AUC against the fixed out-of-sample `y_prob` from the observed run (no model retraining during permutations).
- CI bootstraps resample participants (cluster bootstrap) to avoid pseudo-replication.
