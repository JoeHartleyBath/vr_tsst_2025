# EEG TCNet staged Optuna tuning (Stage 2) — Results (2026-01-18)

This note records the final Stage 2 Optuna tuning outcome for the EEG TCNet workload classifier.

## Summary

- **Best study:** `eeg_tcnet_staged_tuning_stage2_arch1_stage2`
- **Objective:** mean macro-F1 across safe CV folds (subject-grouped)
- **Best macro-F1:** `0.6273243254`
- **Best mean accuracy (same trial):** ~`0.62935`
  - Fold accuracies: `[0.6552381, 0.5994898, 0.6333333]`

## Best architecture (Stage 1 winner carried into Stage 2)

- `F1=16`
- `D=2`
- `kernLength=32`
- `tcn_filters=16`
- `tcn_kernel=6`
- `tcn_depth=1`

## Best hyperparameters (Stage 2)

- `dropout_eeg = 0.1156327312`
- `dropout_tcn = 0.0780090990`
- `learning_rate = 0.0016569682`
- `weight_decay = 0.0011133726`
- `batch_size = 16`
- `label_smoothing = 0.0367555308`

## Where the results came from

- Overall summary JSON (generated): `results/stage2_overall_summary.json`
- Optuna storage (generated): `results/optuna_tcnet_staged.db`
- Trial logs (generated): `results/optuna_trials_tcnet_eeg_tcnet_staged_tuning_stage2_arch*_stage2.jsonl`
- Durable snapshot (recommended for archival outside git):
  - `results/optuna_snapshots/tcnet_staged_20260118_111419/`

Note: all files under `results/` are treated as generated artifacts and are gitignored.

## How to view in Optuna Dashboard

From the repo root:

```powershell
optuna-dashboard sqlite:///C:/vr_tsst_2025/results/optuna_tcnet_staged.db --host 127.0.0.1 --port 8080
```

Then open `http://127.0.0.1:8080` in a browser.

## Re-running Stage 2 (if needed)

The tuning script is:

- `pipelines/13_workload_deep_learning_training/tune_tcnet.py`

This repo’s local changes make tuning more “fill-to-target” friendly (resume-aware trial counting), and disable early-stopping/time cutoffs by default unless explicitly provided.
