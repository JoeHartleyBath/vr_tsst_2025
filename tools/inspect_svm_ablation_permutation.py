"""Inspect SVM ablation permutation-test artifacts.

This repo includes saved permutation null distributions as RDS files under
`results/svm_ablation_runs/<run_tag>/<domain>/svm_perm_auc_<target>.rds`.

This script reads those distributions (via pyreadr), recomputes the
add-one-smoothed one-sided p-value used by the pipeline, and checks it matches
`svm_inference_summary.csv`.

It does NOT reconstruct how labels were shuffled; the R scripts that generated
these artifacts are not present in this repo snapshot.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


@dataclass(frozen=True)
class CheckResult:
    domain: str
    target: str
    k: int
    observed_auc: float
    perm_p: int
    reported_p_value: float
    recomputed_p_value: float
    exceedances: int
    n_null: int


def _read_perm_auc_rds(rds_path: Path) -> pd.Series:
    try:
        import pyreadr  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "Missing dependency 'pyreadr'. Install with: pip install pyreadr"
        ) from exc

    res = pyreadr.read_r(str(rds_path))
    if not res:
        raise ValueError(f"No objects found in RDS: {rds_path}")

    # The writer stores a 1-col data.frame; pyreadr keys can be None.
    obj = next(iter(res.values()))
    if not isinstance(obj, pd.DataFrame):
        raise TypeError(f"Expected a DataFrame in {rds_path}, got {type(obj)}")
    if obj.shape[1] < 1:
        raise ValueError(f"Empty DataFrame in {rds_path}")

    series = obj.iloc[:, 0]
    series = pd.to_numeric(series, errors="coerce")
    if series.isna().any():
        raise ValueError(f"Non-numeric values found in {rds_path}")
    return series


def _perm_p_value_one_sided_add_one(null_auc: pd.Series, observed_auc: float) -> tuple[float, int]:
    # One-sided: probability under null of AUC >= observed.
    exceedances = int((null_auc >= observed_auc).sum())
    p_value = (1 + exceedances) / (len(null_auc) + 1)
    return p_value, exceedances


def _iter_domains(run_dir: Path, domains: list[str] | None) -> Iterable[Path]:
    if domains:
        for d in domains:
            yield run_dir / d
        return

    # Auto-detect typical domain dirs.
    for d in ("all", "eeg", "peripheral"):
        candidate = run_dir / d
        if candidate.is_dir():
            yield candidate


def inspect_run(run_dir: Path, domains: list[str] | None) -> list[CheckResult]:
    results: list[CheckResult] = []

    for domain_dir in _iter_domains(run_dir, domains):
        summary_path = domain_dir / "svm_inference_summary.csv"
        if not summary_path.exists():
            continue

        summary = pd.read_csv(summary_path)
        for _, row in summary.iterrows():
            domain = str(row["domain"])
            target = str(row["target"])
            k = int(row["k"])
            observed_auc = float(row["auc"])
            perm_p = int(row["perm_P"])
            reported_p = float(row["perm_p_value"])

            rds_path = domain_dir / f"svm_perm_auc_{target}.rds"
            null_auc = _read_perm_auc_rds(rds_path)
            recomputed_p, exceedances = _perm_p_value_one_sided_add_one(null_auc, observed_auc)

            results.append(
                CheckResult(
                    domain=domain,
                    target=target,
                    k=k,
                    observed_auc=observed_auc,
                    perm_p=perm_p,
                    reported_p_value=reported_p,
                    recomputed_p_value=recomputed_p,
                    exceedances=exceedances,
                    n_null=len(null_auc),
                )
            )

    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify SVM permutation p-values from saved RDS artifacts")
    parser.add_argument(
        "run_dir",
        type=Path,
        help="Run directory, e.g. results/svm_ablation_runs/full_run_20251219_123204",
    )
    parser.add_argument(
        "--domain",
        action="append",
        dest="domains",
        help="Domain(s) to check (repeatable): all, eeg, peripheral",
    )

    args = parser.parse_args()

    run_dir: Path = args.run_dir
    if not run_dir.exists():
        parser.error(f"run_dir does not exist: {run_dir}")

    checks = inspect_run(run_dir, args.domains)
    if not checks:
        print(f"No inference summaries found under: {run_dir}")
        return 2

    df = pd.DataFrame([c.__dict__ for c in checks])
    df["p_value_diff"] = (df["reported_p_value"] - df["recomputed_p_value"]).abs()

    # Display a compact report.
    with pd.option_context("display.max_columns", 50, "display.width", 140):
        print(df[[
            "domain",
            "target",
            "k",
            "observed_auc",
            "perm_p",
            "n_null",
            "exceedances",
            "reported_p_value",
            "recomputed_p_value",
            "p_value_diff",
        ]].to_string(index=False))

    max_diff = float(df["p_value_diff"].max())
    if max_diff > 1e-12:
        print(f"WARNING: p-value mismatch detected (max abs diff={max_diff}).")
        return 1

    print("OK: reported permutation p-values match recomputation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
