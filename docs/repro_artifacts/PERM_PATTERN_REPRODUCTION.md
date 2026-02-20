# Fast reproduction of the legacy permutation significance pattern

Date: 2026-02-20

This note documents exactly what was done to reproduce the *same sig/non-sig pattern* as the legacy SVM ablation run, using a defensible permutation strategy **as fast as possible** (no model retraining).

## Goal
Reproduce the legacy permutation-test *significance pattern* across 3 domains × 2 targets:

- Domains: all, eeg, peripheral
- Targets: workload_label, stress_label
- Pattern to match at $\alpha = 0.05$ (from the legacy run summaries):
  - Workload: all = significant, eeg = significant, peripheral = non-significant
  - Stress: all = non-significant, eeg = non-significant, peripheral = non-significant

The legacy “source of truth” summaries used for the comparison live here:
- [results/svm_ablation_runs/full_run_20251219_123204/all/svm_inference_summary.csv](../results/svm_ablation_runs/full_run_20251219_123204/all/svm_inference_summary.csv)
- [results/svm_ablation_runs/full_run_20251219_123204/eeg/svm_inference_summary.csv](../results/svm_ablation_runs/full_run_20251219_123204/eeg/svm_inference_summary.csv)
- [results/svm_ablation_runs/full_run_20251219_123204/peripheral/svm_inference_summary.csv](../results/svm_ablation_runs/full_run_20251219_123204/peripheral/svm_inference_summary.csv)

## Strategy (defensible + fast)
### Inputs
Use the already-saved out-of-sample LOSO prediction CSVs produced by the legacy run:

- [results/svm_ablation_runs/full_run_20251219_123204/all/svm_predictions_stress_label.csv](../results/svm_ablation_runs/full_run_20251219_123204/all/svm_predictions_stress_label.csv)
- [results/svm_ablation_runs/full_run_20251219_123204/all/svm_predictions_workload_label.csv](../results/svm_ablation_runs/full_run_20251219_123204/all/svm_predictions_workload_label.csv)
- [results/svm_ablation_runs/full_run_20251219_123204/eeg/svm_predictions_stress_label.csv](../results/svm_ablation_runs/full_run_20251219_123204/eeg/svm_predictions_stress_label.csv)
- [results/svm_ablation_runs/full_run_20251219_123204/eeg/svm_predictions_workload_label.csv](../results/svm_ablation_runs/full_run_20251219_123204/eeg/svm_predictions_workload_label.csv)
- [results/svm_ablation_runs/full_run_20251219_123204/peripheral/svm_predictions_stress_label.csv](../results/svm_ablation_runs/full_run_20251219_123204/peripheral/svm_predictions_stress_label.csv)
- [results/svm_ablation_runs/full_run_20251219_123204/peripheral/svm_predictions_workload_label.csv](../results/svm_ablation_runs/full_run_20251219_123204/peripheral/svm_predictions_workload_label.csv)

Each prediction CSV contains (at minimum):
- participant_id
- y_true (0/1)
- y_prob (predicted probability / decision score)

### Null generation
For each (domain, target):

1. Hold the out-of-sample scores `y_prob` fixed.
2. Build a permutation null by shuffling labels **within each participant_id group**.
   - This preserves the repeated-measures structure and each participant’s label composition.
3. For each permutation, compute the pooled ROC AUC over all rows using the fixed scores.

### What inference this permutation test supports (and what it does not)
This is a **permutation test on fixed out-of-sample predictions** (LOSO predictions saved by the legacy run). Concretely, it supports the following inference:

- **Supported inference (conditional association test):** Under the null hypothesis that, *within each participant*, the condition labels are exchangeable with respect to the model score (`y_prob`), the observed pooled AUC is unusually large compared to AUCs obtained after shuffling labels within participant. Equivalently: it tests whether the model’s out-of-sample scores contain information about the label beyond what would be expected if labels were randomly reassigned within participant.
- **Why within-participant shuffling is defensible here:** The data are repeated-measures (multiple rows per participant). Shuffling within `participant_id` respects that dependence structure and preserves each participant’s label composition.

Important limitations (not supported by this test):

- This test is **not** a test of whether the *training procedure* would, on repeated sampling, produce significant results; we are **not retraining per permutation**.
- This test is **not** a population-level causal claim about stress/workload; it is an association test for this dataset under the exchangeability assumption.
- This test’s p-values are **conditional on the fixed prediction vector** produced by the legacy LOSO pipeline.

In short: it is a fast, reproducible, and defensible **label-score association test under a within-subject null**, designed specifically to reproduce the paper’s *significance pattern* without rerunning model fitting.

### Test statistic
- Pooled ROC AUC computed on the full set of out-of-sample predictions.

### Why pooled AUC is justified for this dataset
Pooled AUC is justified here because:

- **It matches the scientific question at the row level:** With LOSO predictions, pooled AUC estimates the probability that a randomly chosen positive-labeled row receives a higher score than a randomly chosen negative-labeled row.
- **It is stable with small per-participant sample sizes:** Each participant has only 4 rows (conditions). A per-participant AUC can be undefined (e.g., if a participant has all one label after filtering) and is very discrete/noisy with so few points. Pooling uses all 176 rows, reducing variance.
- **It does not meaningfully overweight any participant here:** In this dataset, each participant contributes the same number of rows (4), so pooling does not implicitly give some participants more weight than others.
- **It is consistent with the legacy summary metric:** The legacy `svm_inference_summary.csv` reports a single AUC per (domain, target), which corresponds naturally to the pooled evaluation of all out-of-sample predictions.

Note: an alternative defensible choice is “mean of per-participant AUCs”, but with 4 rows per participant it is substantially noisier/more discrete and can behave poorly when any participant lacks both classes. For fast pattern reproduction, pooled AUC is the most stable and aligns with the legacy reporting.

### P-value
One-sided exceedance with add-one correction:

$$
 p = \frac{1 + \sum_{i=1}^{P} \mathbf{1}[AUC_i \ge AUC_{obs}]}{P + 1}
$$

### Parameters
- $P = 1000$ permutations
- Seed = 42
- $\alpha = 0.05$ for determining significance

## Implementation
### Script added
- [tools/perm_from_predictions_within_pid_auc.py](../tools/perm_from_predictions_within_pid_auc.py)

Key behavior:
- Reads `svm_predictions_*.csv` under each domain folder.
- Computes permutation p-values using within-participant label shuffles.
- Can verify the sig/non-sig pattern against legacy `svm_inference_summary.csv` files via `--verify-legacy-pattern`.

### Why the output is saved under docs/
The repo’s .gitignore ignores the entire results/ tree (except a top-level README), so committing a CSV under results/ would be blocked by design.

Therefore, the reproducible artifact CSV is saved to docs/:
- [docs/repro_artifacts/perm_within_pid_pooled_auc_seed42_P1000_full_run_20251219_123204.csv](perm_within_pid_pooled_auc_seed42_P1000_full_run_20251219_123204.csv)

## Exact command that produced the saved CSV
Executed from the repo root (PowerShell on Windows):

- `C:/vr_tsst_2025/.venv/Scripts/python.exe tools/perm_from_predictions_within_pid_auc.py --run-dir results/svm_ablation_runs/full_run_20251219_123204 --P 1000 --seed 42 --alpha 0.05 --verify-legacy-pattern --out docs/repro_artifacts/perm_within_pid_pooled_auc_seed42_P1000_full_run_20251219_123204.csv`

The script printed:
- `OK: significance pattern matches legacy summaries at alpha=0.05`

## Results (exact values)
The committed CSV contains these rows:

| domain | target | obs_auc | exceed | perm_p_value | is_sig @ alpha=0.05 |
|---|---|---:|---:|---:|---|
| all | stress_label | 0.559529958677686 | 95 | 0.0959040959040959 | False |
| all | workload_label | 0.6623192148760331 | 0 | 0.000999000999000999 | True |
| eeg | stress_label | 0.3759039256198347 | 997 | 0.997002997002997 | False |
| eeg | workload_label | 0.5951704545454546 | 23 | 0.023976023976023976 | True |
| peripheral | stress_label | 0.4682334710743802 | 697 | 0.6973026973026973 | False |
| peripheral | workload_label | 0.5171745867768595 | 358 | 0.35864135864135865 | False |

This matches the legacy pattern at $\alpha=0.05$:
- Workload: all + eeg significant; peripheral non-significant
- Stress: non-significant in all domains

## Commit
The script and the saved artifact were committed together:
- Commit: 738f69a
- Message: “Add fast within-participant permutation-from-predictions reproducer”

## How to rerun later
1. Ensure the run directory contains the saved prediction CSVs under each domain subfolder.
2. Rerun with the same `--P`, `--seed`, and `--alpha` as above.
3. Optionally change `--out` to write a new timestamped artifact.
