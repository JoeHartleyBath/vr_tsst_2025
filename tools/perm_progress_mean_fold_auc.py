"""Permutation test with progress printing (mean per-fold AUC).

Designed to help verify convergence toward legacy manuscript p-values.

Method:
- Reads a `svm_predictions_<target>.csv` with columns:
  `participant_id`, `y_true` (0/1), `y_prob` (probability for class 1)
- Observed statistic = mean of per-participant AUCs (each participant is one
  LOSO outer fold).
- Permutation null: for each permutation, shuffle `y_true` *within each
  participant* (preserves within-subject label balance across that subject's
  rows), keep `y_prob` fixed, recompute mean per-participant AUC.
- One-sided add-one exceedance p-value:
    p = (1 + #{null >= obs}) / (P + 1)

While running, prints a running estimate every `--print-every` permutations:
    p_running = (1 + #{null>=obs so far}) / (i + 1)
This is useful to see convergence without waiting for all P.

Example (PowerShell):
  C:/vr_tsst_2025/.venv/Scripts/python.exe tools/perm_progress_mean_fold_auc.py \
    --pred-csv results/svm_ablation_runs/full_run_20251219_123204/peripheral/svm_predictions_stress_label.csv \
    --P 1000 --print-every 10 --seed 20251219
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def auc_from_scores(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)

    n_pos = int(y_true.sum())
    n_neg = int((1 - y_true).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    ranks = pd.Series(y_score).rank(method="average").to_numpy()
    sum_ranks_pos = float(ranks[y_true == 1].sum())

    return float((sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def mean_fold_auc(y: np.ndarray, s: np.ndarray, fold_indices: list[np.ndarray]) -> float:
    aucs = [auc_from_scores(y[idx], s[idx]) for idx in fold_indices]
    return float(np.nanmean(aucs))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--pred-csv", type=Path, required=True)
    p.add_argument("--P", type=int, default=1000)
    p.add_argument("--print-every", type=int, default=10)
    p.add_argument("--seed", type=int, default=20251219)
    args = p.parse_args()

    df = pd.read_csv(args.pred_csv)
    for col in ("participant_id", "y_true", "y_prob"):
        if col not in df.columns:
            raise ValueError(f"Missing required column '{col}' in {args.pred_csv}")

    fold_indices = [g.index.to_numpy() for _, g in df.groupby("participant_id", sort=False)]
    y = df["y_true"].to_numpy().astype(int)
    s = df["y_prob"].to_numpy().astype(float)

    obs = mean_fold_auc(y, s, fold_indices)
    print(f"pred_csv={args.pred_csv.as_posix()}")
    print(f"n_rows={df.shape[0]} n_participants={df['participant_id'].nunique()}")
    print(f"obs_mean_fold_auc={obs}")

    rng = np.random.default_rng(args.seed)

    ge = 0
    for i in range(1, args.P + 1):
        yp = y.copy()
        # shuffle labels within each participant
        for idx in fold_indices:
            yp[idx] = rng.permutation(yp[idx])

        stat = mean_fold_auc(yp, s, fold_indices)
        if stat >= obs:
            ge += 1

        if (i % args.print_every) == 0 or i == 1 or i == args.P:
            p_running = (1.0 + ge) / (i + 1.0)
            print(f"perm={i} ge={ge} p_running={p_running}")

    p_final = (1.0 + ge) / (args.P + 1.0)
    print(f"p_final_add_one={p_final}")


if __name__ == "__main__":
    main()
