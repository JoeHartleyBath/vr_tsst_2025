from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class Predictions:
    domain: str
    target: str
    participant_id: np.ndarray
    y_true: np.ndarray
    y_prob: np.ndarray
    y_pred: np.ndarray


def _avg_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)

    n = len(values)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and values[order[j + 1]] == values[order[i]]:
            j += 1
        ranks[order[i : j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1

    return ranks


def auc_from_ranks(y_true01: np.ndarray, ranks: np.ndarray) -> float:
    y = y_true01.astype(int, copy=False)
    n_pos = int(np.sum(y == 1))
    n_neg = int(np.sum(y == 0))
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    sum_pos = float(np.sum(ranks[y == 1]))
    offset = n_pos * (n_pos + 1) / 2.0
    return (sum_pos - offset) / (n_pos * n_neg)


def accuracy(y_true01: np.ndarray, y_pred01: np.ndarray) -> float:
    y = y_true01.astype(int, copy=False)
    yhat = y_pred01.astype(int, copy=False)
    return float(np.mean(y == yhat))


def f1_positive_1(y_true01: np.ndarray, y_pred01: np.ndarray) -> float:
    y = y_true01.astype(int, copy=False)
    yhat = y_pred01.astype(int, copy=False)

    tp = int(np.sum((y == 1) & (yhat == 1)))
    fp = int(np.sum((y == 0) & (yhat == 1)))
    fn = int(np.sum((y == 1) & (yhat == 0)))

    denom = 2 * tp + fp + fn
    if denom == 0:
        return 0.0
    return float((2 * tp) / denom)


def within_participant_indices(participant_id: np.ndarray) -> list[np.ndarray]:
    pids, first = np.unique(participant_id, return_index=True)
    pids = pids[np.argsort(first)]
    return [np.flatnonzero(participant_id == pid) for pid in pids]


def perm_p_value_within_participant(
    participant_id: np.ndarray,
    y_true01: np.ndarray,
    y_prob: np.ndarray,
    P: int,
    seed: int,
    print_every: int = 0,
    label: str = "",
) -> tuple[float, float, int]:
    rng = np.random.default_rng(seed)

    idx_list = within_participant_indices(participant_id)
    ranks = _avg_ranks(y_prob)

    obs_auc = auc_from_ranks(y_true01, ranks)

    exceed = 0
    y_perm = y_true01.copy()
    for i in range(1, P + 1):
        for idx in idx_list:
            y_perm[idx] = rng.permutation(y_perm[idx])

        auc_i = auc_from_ranks(y_perm, ranks)
        if auc_i >= obs_auc:
            exceed += 1

        if print_every and (i % print_every == 0):
            p_hat = (1 + exceed) / (i + 1)
            prefix = f"[{label}] " if label else ""
            print(f"{prefix}{i}/{P} running_p={p_hat:.6g} exceed={exceed}")

    p = (1 + exceed) / (P + 1)
    return obs_auc, p, exceed


def iter_prediction_csvs(run_dir: Path) -> Iterable[Path]:
    for domain_dir in sorted([p for p in run_dir.iterdir() if p.is_dir()]):
        for p in sorted(domain_dir.glob("svm_predictions_*.csv")):
            yield p


def load_predictions(path: Path) -> Predictions:
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        raise ValueError(f"Empty file: {path}")

    required = ("participant_id", "y_true", "y_prob")
    for col in required:
        if col not in rows[0]:
            raise ValueError(f"Missing column {col} in {path}")

    participant_id = np.array([int(r["participant_id"]) for r in rows], dtype=int)
    y_true = np.array([int(r["y_true"]) for r in rows], dtype=int)
    y_prob = np.array([float(r["y_prob"]) for r in rows], dtype=float)

    if "y_pred" in rows[0] and all((r.get("y_pred") not in (None, "") for r in rows)):
        y_pred = np.array([int(r["y_pred"]) for r in rows], dtype=int)
    else:
        y_pred = (y_prob >= 0.5).astype(int)

    domain = path.parent.name
    target = path.name.removeprefix("svm_predictions_").removesuffix(".csv")

    return Predictions(
        domain=domain,
        target=target,
        participant_id=participant_id,
        y_true=y_true,
        y_prob=y_prob,
        y_pred=y_pred,
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Compute AUC/ACC/F1 from saved prediction CSVs and a within-participant permutation p-value."
        )
    )
    ap.add_argument("--run-dir", required=True, help="Run directory with domain subfolders")
    ap.add_argument("--P", type=int, default=1000, help="Permutation count")
    ap.add_argument("--seed", type=int, default=42, help="RNG seed")
    ap.add_argument("--alpha", type=float, default=0.05, help="Significance threshold")
    ap.add_argument("--print-every", type=int, default=0, help="Progress printing interval")
    ap.add_argument(
        "--out",
        type=str,
        default="",
        help="Output CSV path (default: timestamped file under run-dir)",
    )

    args = ap.parse_args()
    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        raise SystemExit(f"run-dir not found: {run_dir}")

    pred_paths = list(iter_prediction_csvs(run_dir))
    if not pred_paths:
        raise SystemExit(f"No svm_predictions_*.csv files found under: {run_dir}")

    results: list[dict[str, object]] = []
    for path in pred_paths:
        pred = load_predictions(path)

        obs_acc = accuracy(pred.y_true, pred.y_pred)
        obs_f1 = f1_positive_1(pred.y_true, pred.y_pred)
        obs_auc, p, exceed = perm_p_value_within_participant(
            participant_id=pred.participant_id,
            y_true01=pred.y_true,
            y_prob=pred.y_prob,
            P=args.P,
            seed=args.seed,
            print_every=args.print_every,
            label=f"{pred.domain}/{pred.target}",
        )

        results.append(
            {
                "domain": pred.domain,
                "target": pred.target,
                "n_rows": int(len(pred.y_true)),
                "n_participants": int(len(np.unique(pred.participant_id))),
                "P": int(args.P),
                "seed": int(args.seed),
                "obs_auc": float(obs_auc),
                "obs_acc": float(obs_acc),
                "obs_f1": float(obs_f1),
                "exceed": int(exceed),
                "perm_p_value": float(p),
                "alpha": float(args.alpha),
                "is_sig": bool(p < args.alpha),
            }
        )

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = run_dir / f"svm_perm_metrics_seed{args.seed}_P{args.P}_{stamp}.csv"

    fieldnames = [
        "domain",
        "target",
        "n_rows",
        "n_participants",
        "P",
        "seed",
        "obs_auc",
        "obs_acc",
        "obs_f1",
        "exceed",
        "perm_p_value",
        "alpha",
        "is_sig",
    ]

    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in results:
            w.writerow(row)

    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
