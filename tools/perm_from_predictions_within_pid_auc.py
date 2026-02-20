"""Fast permutation p-values from saved prediction CSVs.

Goal
- Reproduce the *significance pattern* (sig/non-sig) as fast as possible without
  retraining models.

Method (defensible for repeated-measures design)
- Hold out-of-sample prediction scores fixed (from LOSO CV predictions).
- Generate a null by shuffling labels *within each participant_id*.
  This preserves each participant's label counts/structure.
- Test statistic: pooled ROC AUC on the full set of out-of-sample predictions.
- One-sided p-value with add-one correction:
    p = (1 + #{null_auc >= obs_auc}) / (P + 1)

Inputs
- Run directory containing domain subfolders with files:
    svm_predictions_<target>.csv

Outputs
- A CSV summary written to the run directory root.

Notes
- This does NOT try to exactly reproduce legacy permutation *sequences*.
  It is designed for fast, reproducible, justifiable pattern reproduction.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class PredData:
    domain: str
    target: str
    participant_id: np.ndarray  # shape (n,)
    y_true: np.ndarray  # int 0/1 shape (n,)
    y_prob: np.ndarray  # float shape (n,)


def _average_ranks(x: np.ndarray) -> np.ndarray:
    """Return 1-based average ranks, matching the standard AUC tie handling."""
    order = np.argsort(x, kind="mergesort")  # stable
    ranks = np.empty_like(order, dtype=float)

    i = 0
    n = len(x)
    while i < n:
        j = i
        # group ties in sorted order
        while j + 1 < n and x[order[j + 1]] == x[order[i]]:
            j += 1
        # average of ranks i..j in 1-based indexing
        avg_rank = (i + 1 + j + 1) / 2.0
        ranks[order[i : j + 1]] = avg_rank
        i = j + 1

    return ranks


def pooled_auc_from_ranks(y01: np.ndarray, ranks: np.ndarray) -> float:
    y01 = y01.astype(int, copy=False)
    n_pos = int(np.sum(y01 == 1))
    n_neg = int(np.sum(y01 == 0))
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    sum_ranks_pos = float(np.sum(ranks[y01 == 1]))
    offset = n_pos * (n_pos + 1) / 2.0
    return (sum_ranks_pos - offset) / (n_pos * n_neg)


def load_predictions_csv(path: Path) -> PredData:
    # Expect columns: participant_id, y_true, y_prob
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        raise ValueError(f"Empty CSV: {path}")

    for col in ("participant_id", "y_true", "y_prob"):
        if col not in rows[0]:
            raise ValueError(f"Missing column {col} in {path}")

    participant_id = np.array([int(r["participant_id"]) for r in rows], dtype=int)
    y_true = np.array([int(r["y_true"]) for r in rows], dtype=int)
    y_prob = np.array([float(r["y_prob"]) for r in rows], dtype=float)

    domain = path.parent.name
    target = path.name.removeprefix("svm_predictions_").removesuffix(".csv")

    return PredData(
        domain=domain,
        target=target,
        participant_id=participant_id,
        y_true=y_true,
        y_prob=y_prob,
    )


def within_pid_index_list(participant_id: np.ndarray) -> list[np.ndarray]:
    # Preserve first-seen order of pids for determinism.
    pids, first_idx = np.unique(participant_id, return_index=True)
    pids = pids[np.argsort(first_idx)]
    return [np.flatnonzero(participant_id == pid) for pid in pids]


def perm_test_within_participant_pooled_auc(
    data: PredData,
    P: int,
    seed: int,
    print_every: int = 0,
) -> tuple[float, float, int]:
    rng = np.random.default_rng(seed)

    idx_list = within_pid_index_list(data.participant_id)
    ranks = _average_ranks(data.y_prob)

    obs_auc = pooled_auc_from_ranks(data.y_true, ranks)

    # Within-participant shuffles preserve the overall class counts.
    exceed = 0
    y_perm = data.y_true.copy()

    for i in range(1, P + 1):
        # shuffle labels within each participant
        for idx in idx_list:
            y_perm[idx] = rng.permutation(y_perm[idx])

        auc_i = pooled_auc_from_ranks(y_perm, ranks)
        if auc_i >= obs_auc:
            exceed += 1

        if print_every and (i % print_every == 0):
            p_hat = (1 + exceed) / (i + 1)
            print(
                f"[{data.domain}/{data.target}] perm {i}/{P} running_p={p_hat:.6g} (exceed={exceed})"
            )

    p_value = (1 + exceed) / (P + 1)
    return obs_auc, p_value, exceed


def iter_prediction_files(run_dir: Path) -> Iterable[Path]:
    for domain_dir in sorted([p for p in run_dir.iterdir() if p.is_dir()]):
        for p in sorted(domain_dir.glob("svm_predictions_*.csv")):
            yield p


def load_legacy_sig_pattern(run_dir: Path, alpha: float) -> dict[tuple[str, str], bool]:
    """Load legacy sig/non-sig pattern from svm_inference_summary.csv if present.

    Returns a mapping: (domain, target) -> is_significant
    """
    legacy: dict[tuple[str, str], bool] = {}
    for domain_dir in sorted([p for p in run_dir.iterdir() if p.is_dir()]):
        summary_path = domain_dir / "svm_inference_summary.csv"
        if not summary_path.exists():
            continue

        with summary_path.open("r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                domain = str(row.get("domain") or domain_dir.name)
                target = str(row["target"])
                p = float(row["perm_p_value"])
                legacy[(domain, target)] = p < alpha
    return legacy


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Fast within-participant permutation p-values from saved SVM prediction CSVs"
    )
    ap.add_argument("--run-dir", required=True, help="Run dir with domain subfolders")
    ap.add_argument("--P", type=int, default=1000, help="Permutation count (default: 1000)")
    ap.add_argument("--seed", type=int, default=42, help="RNG seed (default: 42)")
    ap.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Significance threshold for pattern verification (default: 0.05)",
    )
    ap.add_argument(
        "--print-every",
        type=int,
        default=0,
        help="Print running p-value every N permutations (default: 0)",
    )
    ap.add_argument(
        "--out",
        type=str,
        default="",
        help="Optional output CSV path. If omitted, writes a timestamped file under run-dir.",
    )
    ap.add_argument(
        "--verify-legacy-pattern",
        action="store_true",
        help=(
            "If set, compare computed sig/non-sig (p<alpha) against legacy svm_inference_summary.csv "
            "files found under run-dir/* and exit non-zero on mismatch."
        ),
    )
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        raise SystemExit(f"run-dir not found: {run_dir}")

    rows: list[dict[str, object]] = []

    legacy_pattern: dict[tuple[str, str], bool] = {}
    if args.verify_legacy_pattern:
        legacy_pattern = load_legacy_sig_pattern(run_dir, alpha=args.alpha)
        if not legacy_pattern:
            raise SystemExit(
                "--verify-legacy-pattern was set but no svm_inference_summary.csv files were found under run-dir/*"
            )

    pred_files = list(iter_prediction_files(run_dir))
    if not pred_files:
        raise SystemExit(f"No svm_predictions_*.csv files found under: {run_dir}")

    for pred_path in pred_files:
        data = load_predictions_csv(pred_path)
        obs_auc, p_value, exceed = perm_test_within_participant_pooled_auc(
            data, P=args.P, seed=args.seed, print_every=args.print_every
        )
        is_sig = p_value < args.alpha
        rows.append(
            {
                "domain": data.domain,
                "target": data.target,
                "n_rows": len(data.y_true),
                "n_participants": len(np.unique(data.participant_id)),
                "P": args.P,
                "seed": args.seed,
                "obs_auc": float(obs_auc),
                "exceed": int(exceed),
                "perm_p_value": float(p_value),
                "alpha": float(args.alpha),
                "is_sig": bool(is_sig),
            }
        )

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = run_dir / (
            f"perm_within_pid_pooled_auc_from_predictions_seed{args.seed}_P{args.P}_{stamp}.csv"
        )

    # Write deterministic column order
    fieldnames = [
        "domain",
        "target",
        "n_rows",
        "n_participants",
        "P",
        "seed",
        "obs_auc",
        "exceed",
        "perm_p_value",
        "alpha",
        "is_sig",
    ]
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"Wrote: {out_path}")

    if args.verify_legacy_pattern:
        mismatches: list[str] = []
        for r in rows:
            key = (str(r["domain"]), str(r["target"]))
            if key not in legacy_pattern:
                continue
            legacy_sig = legacy_pattern[key]
            here_sig = bool(r["is_sig"])
            if legacy_sig != here_sig:
                mismatches.append(
                    f"{key[0]}/{key[1]} legacy_sig={legacy_sig} here_sig={here_sig} here_p={r['perm_p_value']}"
                )

        if mismatches:
            print("ERROR: significance pattern mismatch vs legacy summaries:")
            for m in mismatches:
                print("  "+m)
            raise SystemExit(1)
        print("OK: significance pattern matches legacy summaries at alpha=" + str(args.alpha))


if __name__ == "__main__":
    main()
