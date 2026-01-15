import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats


def rm_2x2(cell: pd.DataFrame, dv: str):
    order_a = ["Low", "High"]
    order_b = ["Low", "High"]

    wide = cell.pivot_table(
        index="participant_id",
        columns=["stress_level", "workload_level"],
        values=dv,
        aggfunc="mean",
    )

    cols = [(a, b) for a in order_a for b in order_b]
    wide = wide.reindex(columns=pd.MultiIndex.from_tuples(cols))
    wide = wide.dropna()  # complete cases only

    n = wide.shape[0]
    Y = wide.to_numpy().reshape(n, 2, 2)

    grand = Y.mean()
    subj_means = Y.mean(axis=(1, 2))
    a_means = Y.mean(axis=(0, 2))
    b_means = Y.mean(axis=(0, 1))
    cell_means = Y.mean(axis=0)

    ss_total = ((Y - grand) ** 2).sum()
    ss_subject = 4 * ((subj_means - grand) ** 2).sum()
    ss_a = 2 * n * ((a_means - grand) ** 2).sum()
    ss_b = 2 * n * ((b_means - grand) ** 2).sum()
    ss_ab = n * ((cell_means - a_means[:, None] - b_means[None, :] + grand) ** 2).sum()

    mean_ij = Y.mean(axis=2)  # n x 2
    ss_as = 2 * ((mean_ij - subj_means[:, None] - a_means[None, :] + grand) ** 2).sum()

    mean_ik = Y.mean(axis=1)  # n x 2
    ss_bs = 2 * ((mean_ik - subj_means[:, None] - b_means[None, :] + grand) ** 2).sum()

    ss_abs = ss_total - ss_subject - ss_a - ss_b - ss_ab - ss_as - ss_bs

    def f_p(ss_eff: float, ss_err: float):
        df1 = 1
        df2 = n - 1
        F = (ss_eff / df1) / (ss_err / df2)
        p = 1 - stats.f.cdf(F, df1, df2)
        return float(F), float(p)

    F_a, p_a = f_p(ss_a, ss_as)
    F_b, p_b = f_p(ss_b, ss_bs)
    F_ab, p_ab = f_p(ss_ab, ss_abs)

    etaG_a = ss_a / (ss_a + ss_as + ss_subject)
    etaG_b = ss_b / (ss_b + ss_bs + ss_subject)
    etaG_ab = ss_ab / (ss_ab + ss_abs + ss_subject)

    # per-subject marginals
    A_hi = (wide[("High", "Low")] + wide[("High", "High")]) / 2
    A_lo = (wide[("Low", "Low")] + wide[("Low", "High")]) / 2
    B_hi = (wide[("Low", "High")] + wide[("High", "High")]) / 2
    B_lo = (wide[("Low", "Low")] + wide[("High", "Low")]) / 2

    def mean_sd_ci(x: pd.Series):
        arr = x.to_numpy(dtype=float)
        m = arr.mean()
        sd = arr.std(ddof=1)
        se = sd / np.sqrt(len(arr))
        tcrit = stats.t.ppf(0.975, len(arr) - 1)
        ci = (m - tcrit * se, m + tcrit * se)
        return float(m), float(sd), (float(ci[0]), float(ci[1]))

    Ahi = mean_sd_ci(A_hi)
    Alo = mean_sd_ci(A_lo)
    Bhi = mean_sd_ci(B_hi)
    Blo = mean_sd_ci(B_lo)

    # paired t + dz for A and B
    dA = (A_hi - A_lo).to_numpy(dtype=float)
    tA = stats.ttest_rel(A_hi, A_lo)
    dzA = float(dA.mean() / dA.std(ddof=1))

    dB = (B_hi - B_lo).to_numpy(dtype=float)
    tB = stats.ttest_rel(B_hi, B_lo)
    dzB = float(dB.mean() / dB.std(ddof=1))

    return {
        "n": int(n),
        "A": {
            "F": F_a,
            "p": p_a,
            "etaG": float(etaG_a),
            "High": Ahi,
            "Low": Alo,
            "diff_mean": float(dA.mean()),
            "t": float(tA.statistic),
            "p_t": float(tA.pvalue),
            "dz": dzA,
        },
        "B": {
            "F": F_b,
            "p": p_b,
            "etaG": float(etaG_b),
            "High": Bhi,
            "Low": Blo,
            "diff_mean": float(dB.mean()),
            "t": float(tB.statistic),
            "p_t": float(tB.pvalue),
            "dz": dzB,
        },
        "AB": {"F": F_ab, "p": p_ab, "etaG": float(etaG_ab)},
        "wide": wide,
    }


def main():
    df = pd.read_csv(Path("output/final_data.csv"))

    ratings = df[["participant_id", "condition", "stress", "workload"]].copy()
    ratings = ratings.dropna(subset=["stress", "workload"])

    ratings["stress_level"] = np.where(
        ratings["condition"].astype(str).str.startswith("High Stress"),
        "High",
        "Low",
    )
    ratings["workload_level"] = np.where(
        ratings["condition"].astype(str).str.contains("High Cog"),
        "High",
        "Low",
    )

    cell = (
        ratings.groupby(["participant_id", "stress_level", "workload_level"], as_index=False)
        .agg(stress=("stress", "mean"), workload=("workload", "mean"))
        .copy()
    )

    stress_res = rm_2x2(cell, "stress")
    work_res = rm_2x2(cell, "workload")

    def fmt_ci(ci):
        return f"[{ci[0]:.2f}, {ci[1]:.2f}]"

    print("SUBJECTIVE STATS (from output/final_data.csv)")

    print("\nStress DV")
    print("  N complete:", stress_res["n"])
    print("  Stress main effect: F(1,{df2})={F:.3f}, p={p:.4g}, etaG={e:.3f}".format(
        df2=stress_res["n"]-1, F=stress_res["A"]["F"], p=stress_res["A"]["p"], e=stress_res["A"]["etaG"]))
    print("    paired: diff={d:.2f}, t({df})={t:.3f}, p={p:.4g}, dz={dz:.2f}".format(
        d=stress_res["A"]["diff_mean"], df=stress_res["n"]-1, t=stress_res["A"]["t"], p=stress_res["A"]["p_t"], dz=stress_res["A"]["dz"]))
    print("  Workload main effect: F(1,{df2})={F:.3f}, p={p:.4g}, etaG={e:.3f}".format(
        df2=stress_res["n"]-1, F=stress_res["B"]["F"], p=stress_res["B"]["p"], e=stress_res["B"]["etaG"]))
    print("  Interaction: F(1,{df2})={F:.3f}, p={p:.4g}, etaG={e:.3f}".format(
        df2=stress_res["n"]-1, F=stress_res["AB"]["F"], p=stress_res["AB"]["p"], e=stress_res["AB"]["etaG"]))
    print("  Stress factor means: High mean={m:.2f} SD={sd:.2f} CI={ci} | Low mean={m2:.2f} SD={sd2:.2f} CI={ci2}".format(
        m=stress_res["A"]["High"][0], sd=stress_res["A"]["High"][1], ci=fmt_ci(stress_res["A"]["High"][2]),
        m2=stress_res["A"]["Low"][0], sd2=stress_res["A"]["Low"][1], ci2=fmt_ci(stress_res["A"]["Low"][2])
    ))

    print("\nNASA mental demand DV")
    print("  N complete:", work_res["n"])
    print("  Workload main effect: F(1,{df2})={F:.3f}, p={p:.4g}, etaG={e:.3f}".format(
        df2=work_res["n"]-1, F=work_res["B"]["F"], p=work_res["B"]["p"], e=work_res["B"]["etaG"]))
    print("    paired: diff={d:.2f}, t({df})={t:.3f}, p={p:.4g}, dz={dz:.2f}".format(
        d=work_res["B"]["diff_mean"], df=work_res["n"]-1, t=work_res["B"]["t"], p=work_res["B"]["p_t"], dz=work_res["B"]["dz"]))
    print("  Stress main effect: F(1,{df2})={F:.3f}, p={p:.4g}, etaG={e:.3f}".format(
        df2=work_res["n"]-1, F=work_res["A"]["F"], p=work_res["A"]["p"], e=work_res["A"]["etaG"]))
    print("  Interaction: F(1,{df2})={F:.3f}, p={p:.4g}, etaG={e:.3f}".format(
        df2=work_res["n"]-1, F=work_res["AB"]["F"], p=work_res["AB"]["p"], e=work_res["AB"]["etaG"]))
    print("  Workload factor means: High mean={m:.2f} SD={sd:.2f} CI={ci} | Low mean={m2:.2f} SD={sd2:.2f} CI={ci2}".format(
        m=work_res["B"]["High"][0], sd=work_res["B"]["High"][1], ci=fmt_ci(work_res["B"]["High"][2]),
        m2=work_res["B"]["Low"][0], sd2=work_res["B"]["Low"][1], ci2=fmt_ci(work_res["B"]["Low"][2])
    ))

    # congruent vs incongruent pooled correlations
    cong = ratings.loc[
        ((ratings.stress_level == "High") & (ratings.workload_level == "High"))
        | ((ratings.stress_level == "Low") & (ratings.workload_level == "Low")),
        ["stress", "workload"],
    ].dropna()
    inc = ratings.loc[
        ((ratings.stress_level == "High") & (ratings.workload_level == "Low"))
        | ((ratings.stress_level == "Low") & (ratings.workload_level == "High")),
        ["stress", "workload"],
    ].dropna()

    r1, p1 = stats.pearsonr(cong["stress"], cong["workload"])
    r2, p2 = stats.pearsonr(inc["stress"], inc["workload"])

    z = (np.arctanh(r1) - np.arctanh(r2)) / np.sqrt(1 / (len(cong) - 3) + 1 / (len(inc) - 3))
    pz = 2 * (1 - stats.norm.cdf(abs(z)))

    print("\nStress–workload rating correlations")
    print(f"  Congruent: r={r1:.2f}, p={p1:.4g}, n={len(cong)}")
    print(f"  Incongruent: r={r2:.2f}, p={p2:.4g}, n={len(inc)}")
    print(f"  Fisher z={z:.2f}, p={pz:.4g}")


if __name__ == "__main__":
    main()
