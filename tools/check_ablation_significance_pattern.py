from __future__ import annotations

from pathlib import Path

import pandas as pd


def main() -> None:
    root = Path("results/svm_ablation_runs")
    alpha = 0.05

    rows = []
    for run_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
        summaries = {}
        for dom in ("all", "eeg", "peripheral"):
            fp = run_dir / dom / "svm_inference_summary.csv"
            if not fp.exists():
                summaries = None
                break
            summaries[dom] = pd.read_csv(fp)
        if summaries is None:
            continue

        pvals = {}
        for dom, df in summaries.items():
            d = {}
            for _, r in df.iterrows():
                d[str(r["target"])]=float(r["perm_p_value"])
            pvals[dom]=d

        p_work_all = pvals["all"].get("workload_label", float("nan"))
        p_work_eeg = pvals["eeg"].get("workload_label", float("nan"))
        p_work_per = pvals["peripheral"].get("workload_label", float("nan"))
        p_stress_all = pvals["all"].get("stress_label", float("nan"))
        p_stress_eeg = pvals["eeg"].get("stress_label", float("nan"))
        p_stress_per = pvals["peripheral"].get("stress_label", float("nan"))

        workload_all_sig = p_work_all < alpha
        workload_eeg_sig = p_work_eeg < alpha
        stress_any_sig = any(
            (p < alpha)
            for p in (p_stress_all, p_stress_eeg, p_stress_per)
            if pd.notna(p)
        )

        pattern = workload_all_sig and workload_eeg_sig and (not stress_any_sig)

        rows.append(
            {
                "run": run_dir.name,
                "pattern": pattern,
                "p_workload_all": p_work_all,
                "p_workload_eeg": p_work_eeg,
                "p_workload_peripheral": p_work_per,
                "p_stress_all": p_stress_all,
                "p_stress_eeg": p_stress_eeg,
                "p_stress_peripheral": p_stress_per,
            }
        )

    out = pd.DataFrame(rows).sort_values(["pattern", "run"], ascending=[False, True])

    matches = out[out["pattern"]]
    nonmatches = out[~out["pattern"]]

    print(f"Runs scanned (having all/eeg/peripheral summaries): {len(out)}")
    print(f"Runs matching pattern: {len(matches)}")

    if len(matches):
        print("\nMATCHES")
        print(
            matches[
                [
                    "run",
                    "p_workload_all",
                    "p_workload_eeg",
                    "p_workload_peripheral",
                    "p_stress_all",
                    "p_stress_eeg",
                    "p_stress_peripheral",
                ]
            ].to_string(index=False)
        )

    if len(nonmatches):
        print("\nNON-MATCHES (first 12)")
        print(
            nonmatches[
                [
                    "run",
                    "p_workload_all",
                    "p_workload_eeg",
                    "p_workload_peripheral",
                    "p_stress_all",
                    "p_stress_eeg",
                    "p_stress_peripheral",
                ]
            ].head(12).to_string(index=False)
        )


if __name__ == "__main__":
    main()
