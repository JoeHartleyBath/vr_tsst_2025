"""Compute permutation p-values using global row-wise label shuffles with pooled AUC.

This is intended for archaeology/verification of the legacy SVM ablation runs.
It reads `svm_predictions_*.csv` files under a run directory (domain subfolders),
computes the observed pooled AUC, then generates a null distribution by
shuffling `y_true` across all rows (ignoring participant grouping) while keeping
`y_prob` fixed.

P-value uses the add-one exceedance rule:
  p = (1 + #{null >= obs}) / (P + 1)

Usage (PowerShell):
  C:/vr_tsst_2025/.venv/Scripts/python.exe -m tools.global_row_shuffle_pooled_auc_perm \
    --run-dir results/svm_ablation_runs/full_run_20251219_123204 --P 1000 10000
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


def perm_p_global_shuffle_pooled_auc(
    *,
    y: np.ndarray,
    s: np.ndarray,
    obs_auc: float,
    P: int,
    seed: int,
) -> float:
    rng = np.random.default_rng(seed)
    null_ge = 0

    for _ in range(P):
        yp = rng.permutation(y)
        a = auc_from_scores(yp, s)
        if a >= obs_auc:
            null_ge += 1

    return (1.0 + null_ge) / (P + 1.0)


@dataclass(frozen=True)
class ResultRow:
    domain: str
    target: str
    n_rows: int
    n_participants: int | None
    obs_pooled_auc: float
    p_values: dict[int, float]


def compute_for_run_dir(run_dir: Path, P_list: list[int], seed: int) -> pd.DataFrame:
    pred_files = sorted(run_dir.glob("*/svm_predictions_*.csv"))
    if not pred_files:
        raise FileNotFoundError(f"No prediction CSVs found under: {run_dir}")

    rows: list[ResultRow] = []

    for pred_path in pred_files:
        domain = pred_path.parent.name
        target = pred_path.name.removeprefix("svm_predictions_").removesuffix(".csv")

        df = pd.read_csv(pred_path)
        y = df["y_true"].to_numpy().astype(int)
        s = df["y_prob"].to_numpy().astype(float)

        obs_auc = auc_from_scores(y, s)
        n_participants = int(df["participant_id"].nunique()) if "participant_id" in df.columns else None

        p_values = {
            P: perm_p_global_shuffle_pooled_auc(y=y, s=s, obs_auc=obs_auc, P=P, seed=seed)
            for P in P_list
        }

        rows.append(
            ResultRow(
                domain=domain,
                target=target,
                n_rows=int(df.shape[0]),
                n_participants=n_participants,
                obs_pooled_auc=float(obs_auc),
                p_values=p_values,
            )
        )

    out_rows = []
    for r in rows:
        row = {
            "domain": r.domain,
            "target": r.target,
            "n_rows": r.n_rows,
            "n_participants": r.n_participants,
            "obs_pooled_auc": r.obs_pooled_auc,
        }
        for P in P_list:
            row[f"p_global_row_shuffle_pooled_auc_P{P}"] = r.p_values[P]
        out_rows.append(row)

    return pd.DataFrame(out_rows).sort_values(["domain", "target"]).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Path like results/svm_ablation_runs/<run_tag>",
    )
    parser.add_argument(
        "--P",
        type=int,
        nargs="+",
        default=[1000],
        help="Permutation counts (space-separated), e.g. --P 1000 10000",
    )
    parser.add_argument("--seed", type=int, default=20251219)
    parser.add_argument(
        "--write-csv",
        action="store_true",
        help="Write a timestamped summary CSV into the run dir",
    )

    args = parser.parse_args()

    run_dir: Path = args.run_dir
    P_list: list[int] = list(args.P)
    seed: int = int(args.seed)

    df = compute_for_run_dir(run_dir, P_list=P_list, seed=seed)
    print(df.to_string(index=False))

    if args.write_csv:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = run_dir / f"global_row_shuffle_pooled_auc_perm_summary_{stamp}.csv"
        df.to_csv(out_path, index=False)
        print(f"\nWrote: {out_path.as_posix()}")


if __name__ == "__main__":
    main()
