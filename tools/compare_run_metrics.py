from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_summary(run_dir: Path, domain: str) -> pd.DataFrame:
    fp = run_dir / domain / "svm_inference_summary.csv"
    df = pd.read_csv(fp)
    return df[["domain", "target", "auc", "acc", "f1"]].copy()


def main() -> None:
    orig = Path("results/svm_ablation_runs/full_run_20251219_123204")
    new = Path("results/svm_ablation_runs/fullperm_20260220_20260220_124216")

    rows = []
    for domain in ("all", "eeg", "peripheral"):
        o = load_summary(orig, domain).rename(columns={"auc": "auc_orig", "acc": "acc_orig", "f1": "f1_orig"})
        n = load_summary(new, domain).rename(columns={"auc": "auc_new", "acc": "acc_new", "f1": "f1_new"})
        m = o.merge(n, on=["domain", "target"], how="inner")
        m["d_auc"] = m["auc_new"] - m["auc_orig"]
        m["d_acc"] = m["acc_new"] - m["acc_orig"]
        m["d_f1"] = m["f1_new"] - m["f1_orig"]
        rows.append(m)

    out = pd.concat(rows, ignore_index=True).sort_values(["domain", "target"]).reset_index(drop=True)

    # round for readability
    num_cols = [c for c in out.columns if c not in ("domain", "target")]
    out[num_cols] = out[num_cols].astype(float).round(6)

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", None)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
