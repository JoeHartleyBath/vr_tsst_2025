"""Sweep RNG seeds to reproduce legacy permutation p-values.

This script is for permutation-method archaeology.

It reads:
- `<run_dir>/<domain>/svm_inference_summary.csv` to get the reported perm_p_value
- `<run_dir>/<domain>/svm_predictions_<target>.csv` to get y_true/y_prob

Then, for each seed in a list, it recomputes permutation p-values under a set of
candidate methods and reports which seeds match the reported values.

Candidate methods implemented:
- Shuffle scope:
  - within_participant: shuffle y_true within each participant_id group
  - global: shuffle y_true across all rows
- Test statistic:
  - pooled_auc: AUC computed over all rows
  - mean_fold_auc: per-participant AUC averaged across participants
- Tail rule:
  - ge: count null >= obs (upper-tail)
  - gt: count null > obs

P-value uses add-one smoothing:
  p = (1 + count) / (P + 1)

Usage:
  C:/vr_tsst_2025/.venv/Scripts/python.exe tools/perm_seed_sweep.py \
    --run-dir results/svm_ablation_runs/full_run_20251219_123204 --P 1000
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
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
class Case:
    domain: str
    target: str
    reported_p: float
    pred_path: Path


def iter_cases(run_dir: Path) -> list[Case]:
    cases: list[Case] = []
    for domain_dir in sorted([p for p in run_dir.iterdir() if p.is_dir()]):
        summ_path = domain_dir / "svm_inference_summary.csv"
        if not summ_path.exists():
            continue
        summ = pd.read_csv(summ_path)
        for _, r in summ.iterrows():
            domain = str(r["domain"])
            target = str(r["target"])
            reported_p = float(r["perm_p_value"])
            pred_path = domain_dir / f"svm_predictions_{target}.csv"
            if not pred_path.exists():
                raise FileNotFoundError(f"Missing predictions for {domain}/{target}: {pred_path}")
            cases.append(Case(domain=domain, target=target, reported_p=reported_p, pred_path=pred_path))
    if not cases:
        raise FileNotFoundError(f"No cases found under run dir: {run_dir}")
    return sorted(cases, key=lambda c: (c.domain, c.target))


def perm_pvalue(
    *,
    y: np.ndarray,
    s: np.ndarray,
    fold_indices: list[np.ndarray],
    P: int,
    seed: int,
    shuffle_scope: str,
    statistic: str,
    tail: str,
) -> float:
    rng = np.random.default_rng(seed)

    if statistic == "pooled_auc":
        obs = auc_from_scores(y, s)
        def stat_fn(y_perm: np.ndarray) -> float:
            return auc_from_scores(y_perm, s)
    elif statistic == "mean_fold_auc":
        obs = mean_fold_auc(y, s, fold_indices)
        def stat_fn(y_perm: np.ndarray) -> float:
            return mean_fold_auc(y_perm, s, fold_indices)
    else:
        raise ValueError(statistic)

    count = 0
    for _ in range(P):
        yp = y.copy()
        if shuffle_scope == "global":
            yp = rng.permutation(yp)
        elif shuffle_scope == "within_participant":
            for idx in fold_indices:
                yp[idx] = rng.permutation(yp[idx])
        else:
            raise ValueError(shuffle_scope)

        val = stat_fn(yp)
        if tail == "ge":
            if val >= obs:
                count += 1
        elif tail == "gt":
            if val > obs:
                count += 1
        else:
            raise ValueError(tail)

    return (1.0 + count) / (P + 1.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--P", type=int, default=1000)
    ap.add_argument(
        "--progress",
        action="store_true",
        help="Print progress for each (method, seed) as it completes.",
    )
    ap.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[
            0, 1, 2, 3, 4, 5, 7, 9, 10, 11, 12, 13, 17, 19, 21, 24, 25,
            42, 69, 73, 99, 101, 123, 321, 999, 2025, 20251219,
        ],
    )
    args = ap.parse_args()

    cases = iter_cases(args.run_dir)
    methods = []
    for shuffle_scope in ("within_participant", "global"):
        for statistic in ("pooled_auc", "mean_fold_auc"):
            for tail in ("ge", "gt"):
                methods.append((shuffle_scope, statistic, tail))

    # Preload data for speed
    loaded = {}
    for c in cases:
        df = pd.read_csv(c.pred_path)
        y = df["y_true"].to_numpy().astype(int)
        s = df["y_prob"].to_numpy().astype(float)
        fold_indices = [g.index.to_numpy() for _, g in df.groupby("participant_id", sort=False)]
        loaded[(c.domain, c.target)] = (y, s, fold_indices)

    rows = []
    perfect = []

    for shuffle_scope, statistic, tail in methods:
        method_name = f"{shuffle_scope}|{statistic}|{tail}"
        if args.progress:
            print(f"\n=== Method: {method_name} ===")
        for seed in args.seeds:
            all_match = True
            match_count = 0
            for c in cases:
                y, s, fold_indices = loaded[(c.domain, c.target)]
                p = perm_pvalue(
                    y=y,
                    s=s,
                    fold_indices=fold_indices,
                    P=int(args.P),
                    seed=int(seed),
                    shuffle_scope=shuffle_scope,
                    statistic=statistic,
                    tail=tail,
                )
                ok = abs(p - c.reported_p) < 1e-12
                if ok:
                    match_count += 1
                rows.append(
                    {
                        "method": method_name,
                        "seed": int(seed),
                        "domain": c.domain,
                        "target": c.target,
                        "reported_p": c.reported_p,
                        "computed_p": p,
                        "match": ok,
                    }
                )
                if not ok:
                    all_match = False
            if args.progress:
                print(f"seed={int(seed):>9d} matches={match_count}/{len(cases)}")
            if all_match:
                perfect.append({"method": method_name, "seed": int(seed)})

    out = pd.DataFrame(rows)

    print("Cases:")
    for c in cases:
        print(f"- {c.domain}/{c.target} reported_p={c.reported_p}")

    if perfect:
        print("\nPerfect matches (all cases):")
        for m in perfect:
            print(f"- method={m['method']} seed={m['seed']}")
    else:
        print("\nNo single (method, seed) matched all reported p-values exactly.")
        # Show best candidates (highest match count)
        best = (
            out.groupby(["method", "seed"])["match"].sum()
            .sort_values(ascending=False)
            .head(10)
            .reset_index()
        )
        print("\nTop (method, seed) by # exact matches:")
        print(best.to_string(index=False))

    stamp = "seed_sweep"
    out_path = args.run_dir / f"perm_seed_sweep_P{int(args.P)}_{stamp}.csv"
    out.to_csv(out_path, index=False)
    print(f"\nWrote: {out_path.as_posix()}")


if __name__ == "__main__":
    main()
