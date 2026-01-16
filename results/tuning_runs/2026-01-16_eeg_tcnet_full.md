# 2026-01-16 — EEG-TCNet Full Staged Optuna Tuning (safe-CV)

- Purpose: Full staged Optuna tuning for EEG-TCNet (safe-CV)
- Git commit: <FILL_COMMIT_HASH_AFTER_COMMIT>
- Script: pipelines/13_workload_deep_learning_training/tune_tcnet.py
- Dataset path: C:\vr_tsst_2025\output\adaptive_workload\mne_epochs
- Model: EEG-TCNet (no augmentation, no attention)
- CV: subject-safe folds, train-only channel stats normalization
- Objective: mean macro-F1
- Seed: 1337
- Folds: 3 (SMOKE=off)
- Stage 1:
  - Trials: 150
  - Time budget: 8h
- Stage 2:
  - Architectures: top-3 from Stage 1
  - Trials per arch: 25
  - Time budget: 4h
- Label smoothing: enabled
- Storage DB: sqlite:///c:/vr_tsst_2025/results/optuna_tcnet_staged.db
- Exact PowerShell command used (see Step 3):

```powershell
$env:SMOKE='0'; `
C:/vr_tsst_2025/.venv/Scripts/python.exe tune_tcnet.py `
  --stage both `
  --stage1_trials 150 `
  --stage2_trials_per_arch 25 `
  --stage2_arch_k 3 `
  --stage1_hours 8 `
  --stage2_hours 4 `
  --top_k 10 `
  --seed 1337 `
  --study_name_prefix eeg_tcnet_staged `
  --db_path c:/vr_tsst_2025/results/optuna_tcnet_staged.db
```
