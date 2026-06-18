"""Batch permutation test: within-participant shuffles, mean per-fold AUC.

This matches the permutation logic that reproduced the legacy manuscript p-value
for `peripheral/stress_label` when using `seed=42` and `P=1000`.

For each `*/svm_predictions_*.csv` under `--run-dir`:
- Observed statistic: mean over participants of AUC(y_true, y_prob) within that
  participant.
- Permutation null: for each permutation, shuffle y_true within each participant
  (keeping y_prob fixed), recompute the same statistic.
- One-sided add-one exceedance p-value:
    p = (1 + #{null >= obs}) / (P + 1)

Usage (PowerShell):
  C:/vr_tsst_2025/.venv/Scripts/python.exe tools/perm_mean_fold_auc_within_participant_batch.py \
    --run-dir results/svm_ablation_runs/full_run_20251219_123204 --P 1000 --seed 42 --write-csv
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
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


@dataclass(frozen=True)
class RowResult:
    domain: str
    target: str
    n_rows: int
    n_participants: int
    obs_mean_fold_auc: float
    perm_P: int
    ge: int
    perm_p_value: float


def perm_test_within_participant_mean_fold_auc(
    *,
    y: np.ndarray,
    s: np.ndarray,
    fold_indices: list[np.ndarray],
    P: int,
    seed: int,
    progress_every: int = 0,
    label: str | None = None,
) -> tuple[int, float]:
    obs = mean_fold_auc(y, s, fold_indices)

    rng = np.random.default_rng(seed)
    ge = 0

    for i in range(1, P + 1):
        yp = y.copy()
        for idx in fold_indices:
            yp[idx] = rng.permutation(yp[idx])

        stat = mean_fold_auc(yp, s, fold_indices)
        if stat >= obs:
            ge += 1

        if progress_every and (i % progress_every) == 0:
            p_running = (1.0 + ge) / (i + 1.0)
            prefix = f"{label} " if label else ""
            print(f"{prefix}perm={i} ge={ge} p_running={p_running}")

    p = (1.0 + ge) / (P + 1.0)
    return ge, float(p)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--P", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--progress-every",
        type=int,
        default=0,
        help="If set, print running p-value every N permutations for each dataset",
    )
    ap.add_argument("--write-csv", action="store_true")
    args = ap.parse_args()

    run_dir: Path = args.run_dir
    P: int = int(args.P)
    seed: int = int(args.seed)

    pred_files = sorted(run_dir.glob("*/svm_predictions_*.csv"))
    if not pred_files:
        raise FileNotFoundError(f"No prediction CSVs found under: {run_dir}")

    results: list[RowResult] = []

    for pred_path in pred_files:
        domain = pred_path.parent.name
        target = pred_path.name.removeprefix("svm_predictions_").removesuffix(".csv")

        df = pd.read_csv(pred_path)
        for col in ("participant_id", "y_true", "y_prob"):
            if col not in df.columns:
                raise ValueError(f"Missing required column '{col}' in {pred_path}")

        fold_indices = [g.index.to_numpy() for _, g in df.groupby("participant_id", sort=False)]
        y = df["y_true"].to_numpy().astype(int)
        s = df["y_prob"].to_numpy().astype(float)

        obs = mean_fold_auc(y, s, fold_indices)
        ge, p_val = perm_test_within_participant_mean_fold_auc(
            y=y,
            s=s,
            fold_indices=fold_indices,
            P=P,
            seed=seed,
            progress_every=int(args.progress_every),
            label=f"{domain}/{target}",
        )

        results.append(
            RowResult(
                domain=domain,
                target=target,
                n_rows=int(df.shape[0]),
                n_participants=int(df["participant_id"].nunique()),
                obs_mean_fold_auc=float(obs),
                perm_P=P,
                ge=int(ge),
                perm_p_value=float(p_val),
            )
        )

        print(f"{domain:10s} {target:14s} obs={obs:.6f} ge={ge} p={p_val}")

    out_df = pd.DataFrame([r.__dict__ for r in results]).sort_values(["domain", "target"]).reset_index(drop=True)

    if args.write_csv:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = run_dir / f"perm_within_participant_mean_fold_auc_seed{seed}_P{P}_{stamp}.csv"
        out_df.to_csv(out_path, index=False)
        print(f"\nWrote: {out_path.as_posix()}")


if __name__ == "__main__":
    main()
