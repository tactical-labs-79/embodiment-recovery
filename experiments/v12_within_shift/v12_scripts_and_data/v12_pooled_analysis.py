#!/usr/bin/env python3
from pathlib import Path

import csv

import itertools

import math

import numpy as np

INPUT = Path("v12_1_within_shift_results.csv")

OUTDIR = Path("v12_2_pooled_analysis")

ALPHA = 0.05

def mean_ci_t(x, confidence=0.95):

    x = np.asarray(x, dtype=float)

    n = len(x)

    mean = float(np.mean(x))

    if n < 2:

        return mean, float("nan"), float("nan")

    sd = float(np.std(x, ddof=1))

    tcrit_table = {

        1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,

        6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,

        11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,

        16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,

    }

    tcrit = tcrit_table.get(n - 1, 1.96)

    half = tcrit * sd / math.sqrt(n)

    return mean, mean - half, mean + half

def one_sample_t_p(x):

    try:

        from scipy.stats import t

        x = np.asarray(x, dtype=float)

        n = len(x)

        sd = np.std(x, ddof=1)

        if n < 2 or sd == 0:

            return float("nan")

        stat = np.mean(x) / (sd / math.sqrt(n))

        return float(2 * t.sf(abs(stat), df=n - 1))

    except Exception:

        return float("nan")

def welch_p(a, b):

    try:

        from scipy.stats import t

        a, b = np.asarray(a, float), np.asarray(b, float)

        va, vb = np.var(a, ddof=1), np.var(b, ddof=1)

        se2 = va / len(a) + vb / len(b)

        if se2 == 0:

            return float("nan")

        df = se2**2 / ((va / len(a))**2 / (len(a)-1) +

                       (vb / len(b))**2 / (len(b)-1))

        stat = (np.mean(b) - np.mean(a)) / math.sqrt(se2)

        return float(2 * t.sf(abs(stat), df))

    except Exception:

        return float("nan")

def exact_sign_flip_p(x):

    x = np.asarray(x, dtype=float)

    observed = abs(float(np.mean(x)))

    n = len(x)

    if n > 22:

        raise ValueError("Exact sign-flip limited to n<=22.")

    count = 0

    total = 1 << n

    for signs in itertools.product((-1.0, 1.0), repeat=n):

        stat = abs(float(np.mean(x * np.asarray(signs))))

        if stat >= observed - 1e-15:

            count += 1

    return count / total

def exact_unpaired_permutation_p(a, b):

    a, b = np.asarray(a, float), np.asarray(b, float)

    pooled = np.concatenate([a, b])

    n_a = len(a)

    observed = abs(float(np.mean(b) - np.mean(a)))

    total = math.comb(len(pooled), n_a)

    if total > 2_000_000:

        return float("nan")

    count = 0

    indices = range(len(pooled))

    for chosen in itertools.combinations(indices, n_a):

        mask = np.ones(len(pooled), dtype=bool)

        mask[list(chosen)] = False

        diff = float(np.mean(pooled[mask]) - np.mean(pooled[~mask]))

        if abs(diff) >= observed - 1e-15:

            count += 1

    return count / total

def load_rows(path):

    with path.open(newline="", encoding="utf-8-sig") as f:

        rows = list(csv.DictReader(f))

    if not rows:

        raise ValueError(f"No rows in {path}")

    for r in rows:

        for k in ("budget", "repeat", "extra_recovery_pp", "extra_reduction",

                  "generic_recovery_pct", "shifted_recovery_pct"):

            if k not in r:

                raise ValueError(f"Missing required column {k!r}. Found: {list(r)}")

    return rows

def main():

    if not INPUT.exists():

        raise FileNotFoundError(

            f"Cannot find {INPUT}. Put this script beside the V12.1 results CSV."

        )

    rows = load_rows(INPUT)

    OUTDIR.mkdir(exist_ok=True)

    by_budget = {}

    for r in rows:

        budget = int(float(r["budget"]))

        by_budget.setdefault(budget, []).append(r)

    print("=" * 100)

    print("V12.2 — POOLED SHIFT-SPECIFIC EVIDENCE / BUDGET × SOURCE")

    print("=" * 100)

    print(f"Loaded: {INPUT} | rows={len(rows)}")

    print("Outcome: extra_recovery_pp (B→B minus A→B), percentage points")

    print("Note: budgets are treated as independent groups; repeat IDs are not paired across budgets.")

    print("\nDATA CHECK")

    for budget in sorted(by_budget):

        rs = by_budget[budget]

        print(f"  budget={budget}: n={len(rs)}, repeats={sorted(int(float(x['repeat'])) for x in rs)}")

    pooled = np.asarray([float(r["extra_recovery_pp"]) for r in rows])

    mean, lo, hi = mean_ci_t(pooled)

    p_t = one_sample_t_p(pooled)

    p_flip = exact_sign_flip_p(pooled)

    print("\nPOOLED EVIDENCE — ALL BUDGETS")

    print("-" * 100)

    print(f"n                         : {len(pooled)}")

    print(f"mean extra recovery       : {mean:+.4f} pp")

    print(f"95% t CI                  : [{lo:+.4f}, {hi:+.4f}] pp")

    print(f"one-sample t p            : {p_t:.6f}")

    print(f"exact sign-flip p         : {p_flip:.6f}")

    print(f"positive runs             : {int(np.sum(pooled > 0))}/{len(pooled)}")

    print("Interpretation: pooled estimate describes the average across the tested 5- and 10-trajectory runs;")

    print("it does not establish that the effect is identical at both budgets.")

    print("\nBUDGET × ADAPTATION-SOURCE INTERACTION")

    print("-" * 100)

    if 5 not in by_budget or 10 not in by_budget:

        print("Need both budget 5 and budget 10 rows.")

    else:

        a = np.asarray([float(r["extra_recovery_pp"]) for r in by_budget[5]])

        b = np.asarray([float(r["extra_recovery_pp"]) for r in by_budget[10]])

        diff = float(np.mean(b) - np.mean(a))

        se = math.sqrt(np.var(a, ddof=1)/len(a) + np.var(b, ddof=1)/len(b))

        try:

            from scipy.stats import t

            va, vb = np.var(a, ddof=1), np.var(b, ddof=1)

            df = (va/len(a)+vb/len(b))**2 / (

                (va/len(a))**2/(len(a)-1) + (vb/len(b))**2/(len(b)-1)

            )

            crit = float(t.ppf(0.975, df))

            p_welch = welch_p(a, b)

        except Exception:

            crit, p_welch = 1.96, float("nan")

        p_perm = exact_unpaired_permutation_p(a, b)

        print(f"mean extra recovery, 5    : {np.mean(a):+.4f} pp (n={len(a)})")

        print(f"mean extra recovery, 10   : {np.mean(b):+.4f} pp (n={len(b)})")

        print(f"10 minus 5                : {diff:+.4f} pp")

        print(f"95% Welch CI              : [{diff-crit*se:+.4f}, {diff+crit*se:+.4f}] pp")

        print(f"Welch p                   : {p_welch:.6f}")

        print(f"exact unpaired perm p     : {p_perm:.6f}")

        print("Positive means the shift-specific advantage is larger at 10 trajectories.")

        print("This is a budget-by-source interaction on the extra-recovery outcome.")

    with (OUTDIR / "v12_2_run_level.csv").open("w", newline="", encoding="utf-8") as f:

        fields = ["budget", "repeat", "extra_recovery_pp", "extra_reduction",

                  "generic_recovery_pct", "shifted_recovery_pct", "source"]

        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")

        writer.writeheader()

        writer.writerows(rows)

    report = [

        "V12.2 — Pooled shift-specific evidence / budget interaction",

        f"Input: {INPUT}",

        f"Rows: {len(rows)}",

        f"Pooled mean extra recovery: {mean:+.6f} pp",

        f"Pooled 95% t CI: [{lo:+.6f}, {hi:+.6f}] pp",

        f"Pooled one-sample t p: {p_t:.6f}",

        f"Pooled exact sign-flip p: {p_flip:.6f}",

        f"Pooled positive runs: {int(np.sum(pooled > 0))}/{len(pooled)}",

    ]

    if 5 in by_budget and 10 in by_budget:

        report += [

            f"Budget 5 mean: {np.mean(a):+.6f} pp",

            f"Budget 10 mean: {np.mean(b):+.6f} pp",

            f"Budget interaction (10-5): {diff:+.6f} pp",

            f"Budget interaction 95% Welch CI: [{diff-crit*se:+.6f}, {diff+crit*se:+.6f}] pp",

            f"Budget interaction Welch p: {p_welch:.6f}",

            f"Budget interaction exact permutation p: {p_perm:.6f}",

        ]

    (OUTDIR / "v12_2_report.txt").write_text("\n".join(report) + "\n", encoding="utf-8")

    print(f"\nSAVED: {OUTDIR / 'v12_2_run_level.csv'}")

    print(f"       {OUTDIR / 'v12_2_report.txt'}")

    print("DONE")

if __name__ == "__main__":

    main()
