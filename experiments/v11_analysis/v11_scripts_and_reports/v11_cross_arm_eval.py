from __future__ import annotations

import ast

import csv

import math

from pathlib import Path

import numpy as np

import torch

import torch.nn as nn

from scipy import stats

import robosuite as suite

import mujoco

from robosuite.utils.mjmod import DynamicsModder

V11_SOURCE = Path("v11_compound_dynamics_revised.py")

RESULTS_FILE = Path("embodiment_recovery_v11_compound_dynamics_results.npz")

OUTDIR = Path("v11_cross_arm_analysis")

OUTPUT_TABLE_FILE = OUTDIR / "v11_cross_arm_table.csv"

OUTPUT_REPORT_FILE = OUTDIR / "v11_cross_arm_report.txt"

ADAPT_SEEDS = list(range(9101, 9121))

VALIDATION_SEEDS = [9301, 9302, 9303, 9304, 9305]

CONFIRMATION_SEEDS = [9601, 9602, 9603, 9604, 9605]

BUDGETS = [5, 10]

SUBSET_REPEATS = 5

DEVICE = torch.device("cpu")

CONTROL = (1.0, 1.0)

SHIFT_CONFIGS = [

    (1.0, 1.0),

    (3.0, 1.0),

    (1.0, 3.0),

    (2.0, 2.0),

    (3.0, 3.0),

    (3.0, 5.0),

    (5.0, 3.0),

    (5.0, 5.0),

]

def load_v11_definitions(path: Path):

    source = path.read_text(encoding="utf-8")

    tree = ast.parse(source)

    defs = [

        node

        for node in tree.body

        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))

    ]

    module = ast.Module(body=defs, type_ignores=[])

    namespace = globals()

    exec(compile(module, str(path), "exec"), namespace, namespace)

load_v11_definitions(V11_SOURCE)

def exact_signflip_p(values):

    values = np.asarray(values, dtype=float)

    values = values[np.isfinite(values)]

    if len(values) == 0:

        return float("nan")

    observed = abs(float(np.mean(values)))

    extreme = 0

    total = 2 ** len(values)

    for bits in range(total):

        signs = np.ones(len(values), dtype=float)

        for i in range(len(values)):

            if (bits >> i) & 1:

                signs[i] = -1.0

        if abs(float(np.mean(values * signs))) >= observed - 1e-15:

            extreme += 1

    return extreme / total

def paired_summary(values):

    values = np.asarray(values, dtype=float)

    values = values[np.isfinite(values)]

    mean = float(np.mean(values))

    std = float(np.std(values, ddof=1)) if len(values) > 1 else float("nan")

    if len(values) > 1:

        tcrit = float(stats.t.ppf(0.975, len(values) - 1))

        half = tcrit * std / math.sqrt(len(values))

        ci_low, ci_high = mean - half, mean + half

        _, p = stats.ttest_1samp(values, 0.0)

        p = float(p)

    else:

        ci_low = ci_high = p = float("nan")

    return {

        "mean": mean, "std": std,

        "ci_low": float(ci_low), "ci_high": float(ci_high),

        "t_p": p, "exact_p": float(exact_signflip_p(values)),

        "positive": int(np.sum(values > 0)), "n": int(len(values)),

    }

def holm_bonferroni(pvals):

    n = len(pvals)

    indexed = [(p, i) for i, p in enumerate(pvals)]

    valid = [(p, i) for p, i in indexed if not math.isnan(p)]

    valid.sort(key=lambda x: x[0])

    adjusted = [float("nan")] * n

    running_max = 0.0

    for rank, (p, i) in enumerate(valid):

        adj = min(1.0, (n - rank) * p)

        running_max = max(running_max, adj)

        adjusted[i] = running_max

    return adjusted

def load_npz_rows(path):

    data = np.load(path)

    n = len(data["mass_multiplier"])

    rows = []

    for i in range(n):

        rows.append({

            "mass": round(float(data["mass_multiplier"][i]), 2),

            "inertia": round(float(data["inertia_multiplier"][i]), 2),

            "budget": int(data["budget"][i]),

            "repeat": int(data["repeat"][i]),

            "zero_rollout": float(data["zero_rollout"][i]),

            "confirmation_rollout": float(data["confirmation_rollout"][i]),

            "recovery_pct": float(data["recovery_pct"][i]),

        })

    return rows

def checkpoint_name(mass, inertia, budget, repeat):

    return f"v11_compound_Mx{mass:.0f}_Ix{inertia:.0f}_{budget}traj_subset{repeat}.pt"

if __name__ == "__main__":

    OUTDIR.mkdir(parents=True, exist_ok=True)

    print("=" * 105)

    print("V11-CROSS-ARM — RETROACTIVE GENERIC vs SHIFT-SPECIFIC (n=5 pilot)")

    print("=" * 105)

    if not RESULTS_FILE.exists():

        raise FileNotFoundError(f"Cannot find {RESULTS_FILE}")

    all_rows = load_npz_rows(RESULTS_FILE)

    print(f"Loaded {len(all_rows)} rows from {RESULTS_FILE}")

    body_name, baseline_mass, baseline_inertia = get_physical_config()

    print("\nPhysical body:", body_name)

    row_index = {(r["mass"], r["inertia"], r["budget"], r["repeat"]): r for r in all_rows}

    for mass, inertia in SHIFT_CONFIGS:

        for budget in BUDGETS:

            for repeat in range(1, SUBSET_REPEATS + 1):

                key = (round(mass, 2), round(inertia, 2), budget, repeat)

                if key not in row_index:

                    raise RuntimeError(f"Missing expected row in .npz: {key}")

    print("Completeness check passed: all expected rows present.")

    generic_model_cache = {}

    def get_generic_model(budget, repeat):

        key = (budget, repeat)

        if key not in generic_model_cache:

            path = checkpoint_name(CONTROL[0], CONTROL[1], budget, repeat)

            if not Path(path).exists():

                raise FileNotFoundError(f"Missing generic checkpoint: {path}")

            model = DynamicsModel().to(DEVICE)

            model.load_state_dict(torch.load(path, map_location=DEVICE))

            model.eval()

            generic_model_cache[key] = model

        return generic_model_cache[key]

    table_rows = []

    raw_pvals = []

    target_shifts = [s for s in SHIFT_CONFIGS if s != CONTROL]

    for mass, inertia in target_shifts:

        print(f"\nEvaluating generic (M1/I1) models on shift M x{mass:.1f} I x{inertia:.1f} ...")

        for budget in BUDGETS:

            extra_recovery_values = []

            for repeat in range(1, SUBSET_REPEATS + 1):

                key = (round(mass, 2), round(inertia, 2), budget, repeat)

                specific_row = row_index[key]

                zero_rollout = specific_row["zero_rollout"]

                generic_model = get_generic_model(budget, repeat)

                generic_conf = [

                    evaluate_model_on_b(

                        model=generic_model,

                        seed=seed,

                        body_name=body_name,

                        baseline_mass=baseline_mass,

                        baseline_inertia=baseline_inertia,

                        mass_multiplier=mass,

                        inertia_multiplier=inertia,

                    )

                    for seed in CONFIRMATION_SEEDS

                ]

                generic_rollout = float(np.mean([r["rollout"] for r in generic_conf]))

                generic_recovery_pct = 100.0 * (zero_rollout - generic_rollout) / max(zero_rollout, 1e-12)

                extra_recovery_pp = specific_row["recovery_pct"] - generic_recovery_pct

                extra_recovery_values.append(extra_recovery_pp)

                print(f"  budget={budget} repeat={repeat} | "

                      f"generic={generic_recovery_pct:+.3f}% "

                      f"specific={specific_row['recovery_pct']:+.3f}% "

                      f"extra={extra_recovery_pp:+.3f}pp")

            ps = paired_summary(extra_recovery_values)

            entry = {

                "mass": mass, "inertia": inertia, "budget": budget,

                "n": ps["n"],

                "extra_recovery_mean_pp": ps["mean"],

                "extra_recovery_ci_low_pp": ps["ci_low"],

                "extra_recovery_ci_high_pp": ps["ci_high"],

                "exact_sign_flip_p": ps["exact_p"],

                "positive_repeats": ps["positive"],

            }

            table_rows.append(entry)

            raw_pvals.append(ps["exact_p"])

    adjusted = holm_bonferroni(raw_pvals)

    for entry, p_adj in zip(table_rows, adjusted):

        entry["holm_bonferroni_p"] = p_adj

    print("\n" + "=" * 105)

    print("V11-CROSS-ARM TABLE (n=5 pilot, PILOT — treat as exploratory)")

    print("=" * 105)

    print(f"{'mass':>5} {'inertia':>8} {'budget':>7} {'extra_pp':>9} "

          f"{'95% CI':>20} {'raw_p':>8} {'holm_p':>8} {'pos':>6}")

    for e in table_rows:

        ci_str = f"[{e['extra_recovery_ci_low_pp']:+.2f},{e['extra_recovery_ci_high_pp']:+.2f}]"

        print(f"{e['mass']:5.1f} {e['inertia']:8.1f} {e['budget']:7d} "

              f"{e['extra_recovery_mean_pp']:+9.3f} {ci_str:>20} "

              f"{e['exact_sign_flip_p']:>8.4f} {e['holm_bonferroni_p']:>8.4f} "

              f"{e['positive_repeats']:>3d}/{e['n']}")

    with OUTPUT_TABLE_FILE.open("w", newline="", encoding="utf-8") as f:

        writer = csv.DictWriter(f, fieldnames=list(table_rows[0].keys()))

        writer.writeheader()

        writer.writerows(table_rows)

    report = [

        "V11-CROSS-ARM — retroactive generic vs shift-specific comparison",

        "=" * 90,

        "PILOT / EXPLORATORY at n=5 per (shift, budget). Do not treat as",

        "confirmatory. The (1,1) control is excluded (it IS the generic arm).",

        "",

        f"Rows: {len(table_rows)} (7 shifts x 2 budgets)",

        "Holm-Bonferroni correction applied across all rows in this table.",

        "",

        "Generic arm = checkpoint trained on (M1,I1), re-evaluated (no",

        "retraining) on each target shift's confirmation seeds.",

        "Shift-specific arm = recovery_pct already stored in the V11 .npz",

        "for that exact shift/budget/repeat.",

    ]

    OUTPUT_REPORT_FILE.write_text("\n".join(report), encoding="utf-8")

    print("\nSAVED:")

    print(f"  {OUTPUT_TABLE_FILE}")

    print(f"  {OUTPUT_REPORT_FILE}")

    print("DONE")
