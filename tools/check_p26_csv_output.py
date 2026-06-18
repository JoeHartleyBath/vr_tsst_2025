from __future__ import annotations

from pathlib import Path

import pandas as pd


def main() -> None:
    base = Path("c:/vr_tsst_2025/share/p26_csv")
    files = sorted(base.rglob("*.csv"))
    print("n_files", len(files))

    bad: list[tuple[str, str, int]] = []

    for csv_path in files:
        columns = pd.read_csv(csv_path, nrows=1).columns
        n_cols = len(columns)

        n_rows = len(pd.read_csv(csv_path, usecols=["time_s"]))
        expected_rows = 22500 if "cleaned" in str(csv_path) else 90000

        print(csv_path.relative_to(base).as_posix(), "rows", n_rows, "cols", n_cols)

        if n_cols != 129:
            bad.append((str(csv_path), "n_cols", n_cols))
        if n_rows != expected_rows:
            bad.append((str(csv_path), "n_rows", n_rows))

    print("bad", bad)


if __name__ == "__main__":
    main()
