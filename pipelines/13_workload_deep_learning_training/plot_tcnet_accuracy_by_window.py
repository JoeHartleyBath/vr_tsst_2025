"""Plot TCNet accuracy across conditions by window index.

Reads per-window prediction logs exported by train_tcnet.py (--export_window_preds).
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Any

import pandas as pd
import matplotlib.pyplot as plt

RESULTS_DIR = r"C:\vr_tsst_2025\results"
DEFAULT_INPUT = os.path.join(
    RESULTS_DIR,
    "workload_tcnet_baseline_zscore_no_attn_safe_cv_window_preds.jsonl",
)
DEFAULT_OUTPUT = os.path.join(
    RESULTS_DIR,
    "workload_tcnet_baseline_zscore_no_attn_safe_cv_accuracy_by_window.png",
)


def load_jsonl(path: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    if not rows:
        raise RuntimeError(f"No rows found in {path}")
    return pd.DataFrame(rows)


def main() -> None:
    p = argparse.ArgumentParser(description="Plot TCNet accuracy vs window index by condition.")
    p.add_argument("--input_jsonl", type=str, default=DEFAULT_INPUT)
    p.add_argument("--output_png", type=str, default=DEFAULT_OUTPUT)
    p.add_argument("--min_count", type=int, default=20, help="Minimum samples per (condition, window_idx)")
    args = p.parse_args()

    df = load_jsonl(args.input_jsonl)

    # Filter to task conditions only
    df = df[df["condition"].astype(str).str.endswith("_Task")].copy()
    if df.empty:
        raise RuntimeError("No task rows found after filtering by _Task condition suffix")

    df["correct"] = (df["y_true"].astype(int) == df["y_pred"].astype(int)).astype(int)

    grouped = (
        df.groupby(["condition", "window_idx"], as_index=False)
        .agg(acc=("correct", "mean"), n=("correct", "size"))
    )

    grouped = grouped[grouped["n"] >= int(args.min_count)].copy()
    if grouped.empty:
        raise RuntimeError("No (condition, window_idx) groups meet min_count threshold")

    plt.figure(figsize=(12, 6))
    for cond, g in grouped.groupby("condition", sort=True):
        g = g.sort_values("window_idx")
        plt.plot(g["window_idx"], g["acc"], label=str(cond))

    plt.xlabel("Window index")
    plt.ylabel("Accuracy")
    plt.title("TCNet accuracy by condition over window index")
    plt.ylim(0.0, 1.0)
    plt.grid(True, alpha=0.3)
    plt.legend(ncol=2, fontsize=8)

    out_path = args.output_png
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    print(f"Saved plot -> {out_path}")


if __name__ == "__main__":
    main()
