import re
from collections import Counter
from pathlib import Path

import pandas as pd


def is_physiology_feature(col: str) -> bool:
    lc = col.lower()
    return (
        lc.startswith("eeg_")
        or ("eda" in lc)
        or ("pupil" in lc)
        or (re.search(r"(^|_)hr($|_)", lc) is not None)
        or ("rmssd" in lc)
        or ("hrv" in lc)
    )


def bucket(col: str) -> str:
    lc = col.lower()
    if lc.startswith("eeg_"):
        return "EEG"
    if "eda" in lc or "gsr" in lc or "scr" in lc or "scl" in lc:
        return "EDA"
    if "rmssd" in lc or "hrv" in lc or "ibi" in lc:
        return "HRV"
    if re.search(r"(^|_)hr($|_)", lc):
        return "HR"
    if "pupil" in lc or "pup" in lc:
        return "Pupil"
    return "Other"


def report_missingness(df: pd.DataFrame, cols: list[str], label: str, missing_threshold: float = 0.30) -> None:
    n_rows = len(df)
    sub = df[cols]

    non_missing = sub.notna().sum(axis=0)
    all_missing = [c for c in cols if int(non_missing[c]) == 0]

    missing_rate = 1.0 - (non_missing / n_rows)
    high_missing = missing_rate[missing_rate >= missing_threshold].sort_values(ascending=False)

    print(f"\n== {label} ==")
    print("n_cols:", len(cols))
    print("all_missing_cols:", len(all_missing))
    print(f"cols_with_missing_rate>={missing_threshold:.0%}:", int(high_missing.shape[0]))

    if not high_missing.empty:
        print("top_most_missing:")
        for c, mr in high_missing.head(15).items():
            print(f"  {c}: missing={mr:.3f} nonmissing={int(non_missing[c])}/{n_rows}")

    if all_missing:
        print("ALL_MISSING_COLS:")
        for c in all_missing:
            print(" ", c)


def main() -> None:
    path = Path(r"c:\vr_tsst_2025\output\final_data_eeg_valid.csv")
    df = pd.read_csv(path)
    print("Loaded:", str(path))
    print("shape:", df.shape)

    if "qc_failed" in df.columns:
        df_model = df[df["qc_failed"].fillna(0).astype(int) == 0].copy()
        print("qc_failed present; rows_used:", len(df_model), "/", len(df))
    else:
        df_model = df
        print("qc_failed not present; rows_used:", len(df_model))

    # Baseline-corrected features
    precond = [c for c in df_model.columns if c.lower().endswith("_precond")]

    # Exclude questionnaire-derived columns (not physiology)
    exclude_prefixes = ("nasa_", "imi_", "mps_")

    phys_precond = [
        c
        for c in precond
        if (not c.lower().startswith(exclude_prefixes)) and is_physiology_feature(c)
    ]

    print("\nPhysiology baseline-corrected pool (_precond)")
    print("n_cols:", len(phys_precond))
    print("by_modality:", dict(Counter(bucket(c) for c in phys_precond)))

    report_missingness(df_model, phys_precond, label="_precond")

    # Z-scored variants (if present)
    phys_precond_z = [c + "_Z" for c in phys_precond if (c + "_Z") in df_model.columns]
    print("\nPhysiology z-scored pool (_precond_Z)")
    print("n_cols:", len(phys_precond_z))
    print("by_modality:", dict(Counter(bucket(c) for c in phys_precond_z)))

    report_missingness(df_model, phys_precond_z, label="_precond_Z")


if __name__ == "__main__":
    main()
