from pathlib import Path
import yaml
import pandas as pd
import numpy as np

# Workspace root
ROOT = Path(__file__).resolve().parents[2]

CFG_PATH = ROOT / "config/conditions.yaml"
CSV_PATH = ROOT / "data/RAW/metadata/P01.csv"


def audit_conditions(csv_path: Path, cfg_path: Path):
    df = pd.read_csv(csv_path, low_memory=False)
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    conditions = cfg.get("conditions", {})
    print(f"Loaded CSV: {csv_path}")
    print(f"Rows: {len(df)}; Columns: {list(df.columns)[:12]}{'...' if len(df.columns)>12 else ''}")
    print()

    results = []
    for label, filters in conditions.items():
        mask = pd.Series([True] * len(df), index=df.index)
        missing_cols = []
        for col, value in filters.items():
            if col not in df.columns:
                missing_cols.append(col)
                continue
            if isinstance(value, list):
                mask &= df[col].isin(value)
            else:
                mask &= (df[col] == value)
        count = int(mask.sum())
        results.append((label, count, missing_cols, filters))

    # Sort by count desc
    results.sort(key=lambda x: x[1], reverse=True)

    print("Condition match summary (top 20):")
    for label, count, missing, filters in results[:20]:
        miss = f" missing={missing}" if missing else ""
        print(f"- {label}: {count}{miss}")
    print()

    # Show any labels with zero matches
    zeros = [(l, m) for l, c, m, _ in results if c == 0]
    if zeros:
        print("Zero-match labels (likely column or value mismatch):")
        for l, m in zeros[:20]:
            miss = f" missing={m}" if m else ""
            print(f"- {l}{miss}")


if __name__ == "__main__":
    audit_conditions(CSV_PATH, CFG_PATH)
