from __future__ import annotations

import csv

import math

from itertools import combinations

from pathlib import Path

import numpy as np

from scipy import stats

RESULTS_FILE = Path("embodiment_recovery_v11_compound_dynamics_results.npz")

OUTPUT_DIR = Path("v11_statistical_analysis_v2")

BUDGETS = [5, 10]

REPEATS = [1, 2, 3, 4, 5]

CONTROL = (1.0, 1.0)

SHIFT_CELLS = [

    (3.0, 1.0), (1.0, 3.0), (2.0, 2.0), (3.0, 3.0),

    (3.0, 5.0), (5.0, 3.0), (5.0, 5.0),

]

BASE = (1.0, 1.0)

MASS_ONLY = (3.0, 1.0)

INERTIA_ONLY = (1.0, 3.0)

BOTH = (3.0, 3.0)

def cell_values(data, mass, inertia, budget, metric):

    out = []

    for repeat in REPEATS:

        mask = (

            np.isclose(data["mass_multiplier"], mass)

            & np.isclose(data["inertia_multiplier"], inertia)

            & (data["budget"] == budget)

            & (data["repeat"] == repeat)

        )

        idx = np.flatnonzero(mask)

        if len(idx) != 1:

            raise ValueError(

                f"Expected exactly one row for Mx{mass:g} Ix{inertia:g} "

                f"budget={budget} repeat={repeat}, found {len(idx)}"

            )

        out.append(float(data[metric][idx[0]]))

    return np.asarray(out, dtype=float)

def ci95(x):

    x = np.asarray(x, dtype=float)

    m = float(np.mean(x))

    s = float(np.std(x, ddof=1))

    h = float(stats.t.ppf(0.975, len(x)-1) * s / math.sqrt(len(x)))

    return m - h, m + h

def exact_signflip_p(x):

    x = np.asarray(x, dtype=float)

    observed = abs(float(np.mean(x)))

    extreme = 0

    total = 2 ** len(x)

    for bits in range(total):

        signs = np.ones(len(x))

        for i in range(len(x)):

            if (bits >> i) & 1:

                signs[i] = -1.0

        if abs(float(np.mean(x * signs))) >= observed - 1e-15:

            extreme += 1

    return extreme / total

def exact_two_sample_perm_p(a, b):

    a = np.asarray(a, dtype=float)

    b = np.asarray(b, dtype=float)

    combined = np.concatenate([a, b])

    n_a = len(a)

    observed = abs(float(np.mean(a) - np.mean(b)))

    extreme = 0

    total = 0

    for inds in combinations(range(len(combined)), n_a):

        mask = np.zeros(len(combined), dtype=bool)

        mask[list(inds)] = True

        pa = combined[mask]

        pb = combined[~mask]

        if abs(float(np.mean(pa) - np.mean(pb))) >= observed - 1e-15:

            extreme += 1

        total += 1

    return extreme / total

def matched_summary(effect):

    effect = np.asarray(effect, dtype=float)

    mean = float(np.mean(effect))

    std = float(np.std(effect, ddof=1))

    lo, hi = ci95(effect)

    t_stat = mean / (std / math.sqrt(len(effect))) if std > 0 else math.inf

    t_p = 2 * float(stats.t.sf(abs(t_stat), len(effect)-1)) if math.isfinite(t_stat) else 0.0

    return dict(

        mean=mean, std=std, ci_low=lo, ci_high=hi,

        t_p=t_p, exact_p=exact_signflip_p(effect),

        positive=int(np.sum(effect > 0)), n=len(effect)

    )

def unpaired_summary(x5, x10):

    r = stats.ttest_ind(x10, x5, equal_var=False)

    return dict(

        mean5=float(np.mean(x5)), mean10=float(np.mean(x10)),

        diff=float(np.mean(x10)-np.mean(x5)),

        std5=float(np.std(x5, ddof=1)), std10=float(np.std(x10, ddof=1)),

        welch_t=float(r.statistic), welch_p=float(r.pvalue),

        perm_p=exact_two_sample_perm_p(x5, x10)

    )

def main():

    if not RESULTS_FILE.exists():

        raise SystemExit(f"Missing {RESULTS_FILE.resolve()}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    raw = np.load(RESULTS_FILE)

    required = [

        "mass_multiplier", "inertia_multiplier", "budget", "repeat",

        "zero_rollout", "confirmation_rollout", "reduction", "recovery_pct"

    ]

    missing = [k for k in required if k not in raw.files]

    if missing:

        raise SystemExit("Missing fields: " + ", ".join(missing))

    data = {k: np.asarray(raw[k]) for k in required}

    if sorted(set(map(int, data["budget"]))) != BUDGETS:

        raise SystemExit("Unexpected budgets")

    if sorted(set(map(int, data["repeat"]))) != REPEATS:

        raise SystemExit("Unexpected repeat numbering")

    print("=" * 110)

    print("V11 — STATISTICS V2")

    print("GENERIC ADAPTATION VS SHIFT-SPECIFIC EFFECT")

    print("=" * 110)

    print(f"Loaded: {RESULTS_FILE}")

    print(f"Rows:   {len(data['budget'])}")

    print()

    print("CELL SUMMARY")

    print("-" * 110)

    cell_rows = []

    for budget in BUDGETS:

        for mass, inertia in [CONTROL] + SHIFT_CELLS:

            rec = cell_values(data, mass, inertia, budget, "recovery_pct")

            red = cell_values(data, mass, inertia, budget, "reduction")

            print(

                f"M×{mass:g} I×{inertia:g} | {budget:2d} traj | "

                f"recovery={np.mean(rec):+8.3f}% | "

                f"reduction={np.mean(red):+.6f} | "

                f"positive={np.sum(rec > 0)}/{len(rec)}"

            )

            cell_rows.append([

                budget, mass, inertia, len(rec), np.mean(rec), np.std(rec, ddof=1),

                np.mean(red), np.std(red, ddof=1), np.min(red), np.max(red), np.sum(rec > 0)

            ])

    with (OUTPUT_DIR / "v2_cell_summary.csv").open("w", newline="", encoding="utf-8") as f:

        w = csv.writer(f); w.writerow([

            "budget","mass","inertia","n","mean_recovery_pct","std_recovery_pct",

            "mean_reduction","std_reduction","min_reduction","max_reduction","positive_recovery_n"

        ]); w.writerows(cell_rows)

    print()

    print("SHIFT-SPECIFIC EXCESS OVER GENERIC CONTROL")

    print("-" * 110)

    excess_rows = []

    for budget in BUDGETS:

        c_red = cell_values(data, *CONTROL, budget, "reduction")

        c_rec = cell_values(data, *CONTROL, budget, "recovery_pct")

        for mass, inertia in SHIFT_CELLS:

            s_red = cell_values(data, mass, inertia, budget, "reduction")

            s_rec = cell_values(data, mass, inertia, budget, "recovery_pct")

            effect = s_red - c_red

            rec_delta = s_rec - c_rec

            r = matched_summary(effect)

            print(

                f"{budget:2d} traj | M×{mass:g} I×{inertia:g} | "

                f"excess reduction={r['mean']:+.6f} | "

                f"CI95=[{r['ci_low']:+.6f}, {r['ci_high']:+.6f}] | "

                f"exact p={r['exact_p']:.4f} | positive={r['positive']}/{r['n']}"

            )

            excess_rows.append([

                budget, mass, inertia, r["mean"], r["std"], r["ci_low"], r["ci_high"],

                r["t_p"], r["exact_p"], r["positive"], r["n"],

                np.mean(rec_delta), np.std(rec_delta, ddof=1)

            ])

    with (OUTPUT_DIR / "v2_shift_specific_excess.csv").open("w", newline="", encoding="utf-8") as f:

        w = csv.writer(f); w.writerow([

            "budget","mass","inertia","mean_excess_reduction","std_excess_reduction",

            "ci95_low","ci95_high","paired_t_p","exact_signflip_p","positive_n","n",

            "mean_recovery_pct_minus_control","std_recovery_pct_minus_control"

        ]); w.writerows(excess_rows)

    print()

    print("5 → 10 TRAJECTORY EFFECT — UNPAIRED")

    print("-" * 110)

    print("The 5-traj and 10-traj subset generators use different budget-dependent RNG seeds.")

    budget_rows = []

    for mass, inertia in [CONTROL] + SHIFT_CELLS:

        for metric in ["reduction", "recovery_pct"]:

            x5 = cell_values(data, mass, inertia, 5, metric)

            x10 = cell_values(data, mass, inertia, 10, metric)

            r = unpaired_summary(x5, x10)

            print(

                f"M×{mass:g} I×{inertia:g} | {metric:13s} | "

                f"10-5={r['diff']:+.6f} | Welch p={r['welch_p']:.4f} | "

                f"exact permutation p={r['perm_p']:.4f}"

            )

            budget_rows.append([

                mass, inertia, metric, r["mean5"], r["mean10"], r["diff"], r["std5"], r["std10"],

                r["welch_t"], r["welch_p"], r["perm_p"]

            ])

    with (OUTPUT_DIR / "v2_budget_5_vs_10_unpaired.csv").open("w", newline="", encoding="utf-8") as f:

        w = csv.writer(f); w.writerow([

            "mass","inertia","metric","mean_5","mean_10","difference_10_minus_5",

            "std_5","std_10","welch_t","welch_p","exact_permutation_p"

        ]); w.writerows(budget_rows)

    print()

    print("2×2 MASS × INERTIA INTERACTION — EXPLORATORY")

    print("-" * 110)

    interaction_rows = []

    for budget in BUDGETS:

        def vals(cell, metric): return cell_values(data, *cell, budget, metric)

        br, mr, ir, bor = vals(BASE, "reduction"), vals(MASS_ONLY, "reduction"), vals(INERTIA_ONLY, "reduction"), vals(BOTH, "reduction")

        bc, mc, ic, boc = vals(BASE, "recovery_pct"), vals(MASS_ONLY, "recovery_pct"), vals(INERTIA_ONLY, "recovery_pct"), vals(BOTH, "recovery_pct")

        int_red = bor - mr - ir + br

        int_rec = boc - mc - ic + bc

        rr, rc = matched_summary(int_red), matched_summary(int_rec)

        print(

            f"{budget:2d} traj | reduction interaction={rr['mean']:+.6f} | "

            f"CI95=[{rr['ci_low']:+.6f}, {rr['ci_high']:+.6f}] | exact p={rr['exact_p']:.4f}"

        )

        print(

            f"         | recovery interaction={rc['mean']:+.3f} pp | "

            f"CI95=[{rc['ci_low']:+.3f}, {rc['ci_high']:+.3f}] | exact p={rc['exact_p']:.4f}"

        )

        interaction_rows.append([

            budget, rr["mean"], rr["std"], rr["ci_low"], rr["ci_high"], rr["exact_p"],

            rc["mean"], rc["std"], rc["ci_low"], rc["ci_high"], rc["exact_p"]

        ])

    with (OUTPUT_DIR / "v2_interaction.csv").open("w", newline="", encoding="utf-8") as f:

        w = csv.writer(f); w.writerow([

            "budget","interaction_reduction","std_reduction","ci95_low_reduction","ci95_high_reduction","exact_p_reduction",

            "interaction_recovery_pp","std_recovery","ci95_low_recovery","ci95_high_recovery","exact_p_recovery"

        ]); w.writerows(interaction_rows)

    print()

    report = []

    report += [

        "V11 STATISTICS V2 — GENERIC VS SHIFT-SPECIFIC EFFECT",

        "=" * 90,

        "",

        "Primary outcome: raw rollout error reduction.",

        "Shift-specific excess = shifted reduction - M×1 I×1 control reduction.",

        "5-vs-10 is unpaired because the budget-specific subset RNG differs.",

        "Shift-vs-control and 2x2 interaction are exploratory matched-subset contrasts (n=5).",

        "",

        "SHIFT-SPECIFIC EXCESS REDUCTION",

        "-" * 90,

    ]

    for row in excess_rows:

        b,m,i,mean,sd,lo,hi,tp,ep,pos,n,md,sdrec = row

        report.append(

            f"{b:2d} traj | M×{m:g} I×{i:g} | excess={mean:+.6f} | "

            f"CI95=[{lo:+.6f},{hi:+.6f}] | exact p={ep:.4f} | positive={pos}/{n}"

        )

    report += ["", "5 VS 10 — UNPAIRED", "-" * 90]

    for row in budget_rows:

        m,i,metric,m5,m10,d,s5,s10,t,p,perm = row

        report.append(

            f"M×{m:g} I×{i:g} | {metric:13s} | 10-5={d:+.6f} | Welch p={p:.4f} | perm p={perm:.4f}"

        )

    report += ["", "2X2 INTERACTION — EXPLORATORY", "-" * 90]

    for row in interaction_rows:

        b,ir,irs,irl,irh,irp,ic,ics,icl,ich,icp = row

        report.append(

            f"{b:2d} traj | reduction interaction={ir:+.6f} | CI95=[{irl:+.6f},{irh:+.6f}] | exact p={irp:.4f}"

        )

        report.append(

            f"{b:2d} traj | recovery interaction={ic:+.3f} pp | CI95=[{icl:+.3f},{ich:+.3f}] | exact p={icp:.4f}"

        )

    report += [

        "",

        "INTERPRETATION",

        "-" * 90,

        "A positive excess reduction means the shifted condition improved more than generic control adaptation.",

        "A near-zero excess means observed improvement is largely compatible with the generic adaptation effect.",

        "All p-values are exploratory because n=5 and the training random seed is configuration-specific.",

        "The experiment remains a localized link5 dynamics perturbation in the Panda Lift benchmark.",

    ]

    (OUTPUT_DIR / "v2_report.txt").write_text("\n".join(report), encoding="utf-8")

    print("=" * 110)

    print("SAVED")

    print("=" * 110)

    for p in [

        OUTPUT_DIR / "v2_cell_summary.csv",

        OUTPUT_DIR / "v2_shift_specific_excess.csv",

        OUTPUT_DIR / "v2_budget_5_vs_10_unpaired.csv",

        OUTPUT_DIR / "v2_interaction.csv",

        OUTPUT_DIR / "v2_report.txt",

    ]:

        print(p)

    print("DONE")

if __name__ == "__main__":

    main()
