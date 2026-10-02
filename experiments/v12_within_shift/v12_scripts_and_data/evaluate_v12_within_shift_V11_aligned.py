from __future__ import annotations

import ast

import copy

import csv

import math

import logging

from pathlib import Path

import numpy as np

import torch

import torch.nn as nn

from scipy import stats

import robosuite as suite

import mujoco

from robosuite.utils.mjmod import DynamicsModder

V11_SOURCE = Path("v11_compound_dynamics_revised.py")

MODEL_FILE = "dynamics_model_v7.pt"

A_MASS = 1.0

A_INERTIA = 1.0

B_MASS = 5.0

B_INERTIA = 5.0

ADAPT_SEEDS = list(range(9101, 9121))

VALIDATION_SEEDS = [9301, 9302, 9303, 9304, 9305]

CONFIRMATION_SEEDS = [9601, 9602, 9603, 9604, 9605]

BUDGETS = [5, 10]

SUBSET_REPEATS = 5

ROLLOUT_STEPS = 120

INPUT_DIM = 24

OUTPUT_DIM = 17

HIDDEN = 128

POS_AMPLITUDE = 0.18

ROT_AMPLITUDE = 0.12

LEARNING_RATE = 1e-4

MAX_EPOCHS = 100

PATIENCE = 10

BATCH_SIZE = 256

WEIGHT_DECAY = 1e-6

MIN_DELTA = 1e-7

DEVICE = torch.device("cpu")

logging.getLogger("robosuite").setLevel(logging.WARNING)

def load_v11_definitions(path: Path):

    source = path.read_text(encoding="utf-8")

    tree = ast.parse(source)

    defs = [

        node for node in tree.body

        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))

    ]

    module = ast.Module(body=defs, type_ignores=[])

    namespace = globals()

    exec(compile(module, str(path), "exec"), namespace, namespace)

load_v11_definitions(V11_SOURCE)

def build_paired_subsets():

    specs = []

    for budget in BUDGETS:

        for repeat in range(1, SUBSET_REPEATS + 1):

            rng = np.random.default_rng(82000 + budget * 100 + repeat)

            selected = sorted(

                rng.choice(ADAPT_SEEDS, size=budget, replace=False).tolist()

            )

            specs.append({

                "budget": budget,

                "repeat": repeat,

                "seeds": selected,

            })

    return specs

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

    mean = float(values.mean())

    std = float(values.std(ddof=1)) if len(values) > 1 else float("nan")

    if len(values) > 1:

        tcrit = float(stats.t.ppf(0.975, len(values) - 1))

        half = tcrit * std / math.sqrt(len(values))

        ci_low, ci_high = mean - half, mean + half

        _, p = stats.ttest_1samp(values, 0.0)

        p = float(p)

    else:

        ci_low = ci_high = p = float("nan")

    return {

        "mean": mean,

        "std": std,

        "ci_low": float(ci_low),

        "ci_high": float(ci_high),

        "t_p": p,

        "exact_p": float(exact_signflip_p(values)),

        "positive": int(np.sum(values > 0)),

        "n": int(len(values)),

    }

def evaluate_b_confirmation(model, body_name, baseline_mass, baseline_inertia, seeds):

    rows = []

    for seed in seeds:

        rows.append(

            evaluate_model_on_b(

                model=model,

                seed=seed,

                body_name=body_name,

                baseline_mass=baseline_mass,

                baseline_inertia=baseline_inertia,

                mass_multiplier=B_MASS,

                inertia_multiplier=B_INERTIA,

            )

        )

    return rows

def train_and_validate(train_pool, base_state_dict, body_name, baseline_mass, baseline_inertia, run_seed):

    def validation_fn(candidate):

        candidate.eval()

        teacher = []

        rollout = []

        for seed in VALIDATION_SEEDS:

            r = evaluate_model_on_b(

                model=candidate,

                seed=seed,

                body_name=body_name,

                baseline_mass=baseline_mass,

                baseline_inertia=baseline_inertia,

                mass_multiplier=B_MASS,

                inertia_multiplier=B_INERTIA,

            )

            teacher.append(r["teacher"])

            rollout.append(r["rollout"])

        return float(np.mean(teacher)), float(np.mean(rollout))

    return train_head_only(

        base_state_dict=base_state_dict,

        train_trajectories=train_pool,

        validation_fn=validation_fn,

        run_seed=run_seed,

    )

if __name__ == "__main__":

    print("=" * 110)

    print("V12 — WITHIN-SHIFT CONTROL")

    print("=" * 110)

    print("A source : M×1 I×1")

    print("B target : M×5 I×5")

    print("Budgets  :", BUDGETS)

    print("Repeats  :", SUBSET_REPEATS)

    print("Validation B seeds   :", VALIDATION_SEEDS)

    print("Confirmation B seeds :", CONFIRMATION_SEEDS)

    body_name, baseline_mass, baseline_inertia = get_physical_config()

    base_model = load_v7_model()

    base_state_dict = copy.deepcopy(base_model.state_dict())

    print("\nPhysical body:", body_name)

    print("Baseline mass:", baseline_mass)

    print("Baseline inertia:", baseline_inertia)

    subsets = build_paired_subsets()

    print("\nPaired subsets:")

    for s in subsets:

        print(f"  {s['budget']:2d} traj | repeat {s['repeat']} | {s['seeds']}")

    print("\nCollecting A pool (no shift)...")

    a_pool = build_b_pool(

        seeds=ADAPT_SEEDS,

        body_name=body_name,

        baseline_mass=baseline_mass,

        baseline_inertia=baseline_inertia,

        mass_multiplier=A_MASS,

        inertia_multiplier=A_INERTIA,

    )

    print("\nCollecting B pool (M×5 I×5)...")

    b_pool = build_b_pool(

        seeds=ADAPT_SEEDS,

        body_name=body_name,

        baseline_mass=baseline_mass,

        baseline_inertia=baseline_inertia,

        mass_multiplier=B_MASS,

        inertia_multiplier=B_INERTIA,

    )

    a_by_seed = {t["seed"]: t for t in a_pool}

    b_by_seed = {t["seed"]: t for t in b_pool}

    print("\nComputing zero-shot B baseline...")

    zero_rows = evaluate_b_confirmation(

        base_model, body_name, baseline_mass, baseline_inertia, CONFIRMATION_SEEDS

    )

    zero_values = np.asarray([r["rollout"] for r in zero_rows], dtype=float)

    zero_rollout = float(zero_values.mean())

    print(f"Zero-shot B rollout = {zero_rollout:.8f}")

    rows = []

    for spec in subsets:

        budget = spec["budget"]

        repeat = spec["repeat"]

        seeds = spec["seeds"]

        print("\n" + "#" * 110)

        print(f"BUDGET={budget} REPEAT={repeat}")

        print("Paired seeds:", seeds)

        print("#" * 110)

        generic_train = [a_by_seed[s] for s in seeds]

        shifted_train = [b_by_seed[s] for s in seeds]

        print("\n[1/2] GENERIC A→B")

        generic_model, generic_epoch, generic_val, generic_val_teacher = train_and_validate(

            generic_train,

            base_state_dict,

            body_name,

            baseline_mass,

            baseline_inertia,

            run_seed=1155000 + budget * 100 + repeat,

        )

        generic_conf = evaluate_b_confirmation(

            generic_model, body_name, baseline_mass, baseline_inertia, CONFIRMATION_SEEDS

        )

        generic_vals = np.asarray([r["rollout"] for r in generic_conf], dtype=float)

        print("\n[2/2] SHIFT-SPECIFIC B→B")

        shifted_model, shifted_epoch, shifted_val, shifted_val_teacher = train_and_validate(

            shifted_train,

            base_state_dict,

            body_name,

            baseline_mass,

            baseline_inertia,

            run_seed=1155000 + budget * 100 + repeat,

        )

        shifted_conf = evaluate_b_confirmation(

            shifted_model, body_name, baseline_mass, baseline_inertia, CONFIRMATION_SEEDS

        )

        shifted_vals = np.asarray([r["rollout"] for r in shifted_conf], dtype=float)

        generic_reduction = zero_values.mean() - generic_vals.mean()

        shifted_reduction = zero_values.mean() - shifted_vals.mean()

        generic_recovery = 100.0 * generic_reduction / max(zero_rollout, 1e-12)

        shifted_recovery = 100.0 * shifted_reduction / max(zero_rollout, 1e-12)

        extra_reduction_per_seed = generic_vals - shifted_vals

        extra_reduction = float(extra_reduction_per_seed.mean())

        extra_recovery_pp = float(shifted_recovery - generic_recovery)

        print("\nRESULT")

        print(f"  zero-shot B            : {zero_rollout:.8f}")

        print(f"  generic A→B            : {generic_vals.mean():.8f} ({generic_recovery:+.3f}%)")

        print(f"  shift-specific B→B     : {shifted_vals.mean():.8f} ({shifted_recovery:+.3f}%)")

        print(f"  extra reduction        : {extra_reduction:+.8f}")

        print(f"  extra recovery         : {extra_recovery_pp:+.3f} pp")

        rows.append({

            "budget": budget,

            "repeat": repeat,

            "seeds": ",".join(map(str, seeds)),

            "zero_rollout": zero_rollout,

            "generic_rollout": float(generic_vals.mean()),

            "generic_reduction": float(generic_reduction),

            "generic_recovery_pct": float(generic_recovery),

            "generic_best_epoch": generic_epoch,

            "generic_best_val": generic_val,

            "generic_best_val_teacher": generic_val_teacher,

            "shifted_rollout": float(shifted_vals.mean()),

            "shifted_reduction": float(shifted_reduction),

            "shifted_recovery_pct": float(shifted_recovery),

            "shifted_best_epoch": shifted_epoch,

            "shifted_best_val": shifted_val,

            "shifted_best_val_teacher": shifted_val_teacher,

            "extra_reduction": extra_reduction,

            "extra_recovery_pp": extra_recovery_pp,

            "generic_confirmation_std": float(generic_vals.std(ddof=1)),

            "shifted_confirmation_std": float(shifted_vals.std(ddof=1)),

        })

        torch.save(

            generic_model.state_dict(),

            f"v12_generic_Mx5_Ix5_{budget}traj_subset{repeat}.pt",

        )

        torch.save(

            shifted_model.state_dict(),

            f"v12_shift_specific_Mx5_Ix5_{budget}traj_subset{repeat}.pt",

        )

    print("\n" + "=" * 110)

    print("V12 SUMMARY — SHIFT-SPECIFIC ADVANTAGE")

    print("=" * 110)

    summary_rows = []

    for budget in BUDGETS:

        sub = [r for r in rows if r["budget"] == budget]

        extra_reduction = np.asarray([r["extra_reduction"] for r in sub], dtype=float)

        extra_pp = np.asarray([r["extra_recovery_pp"] for r in sub], dtype=float)

        rs = paired_summary(extra_reduction)

        ps = paired_summary(extra_pp)

        summary = {

            "budget": budget,

            "generic_recovery_mean_pct": float(np.mean([r["generic_recovery_pct"] for r in sub])),

            "shift_specific_recovery_mean_pct": float(np.mean([r["shifted_recovery_pct"] for r in sub])),

            "extra_recovery_mean_pp": ps["mean"],

            "extra_recovery_std_pp": ps["std"],

            "extra_recovery_ci_low_pp": ps["ci_low"],

            "extra_recovery_ci_high_pp": ps["ci_high"],

            "extra_recovery_t_p": ps["t_p"],

            "extra_recovery_exact_p": ps["exact_p"],

            "positive_repeats": ps["positive"],

            "n": ps["n"],

            "extra_reduction_mean": rs["mean"],

            "extra_reduction_ci_low": rs["ci_low"],

            "extra_reduction_ci_high": rs["ci_high"],

        }

        summary_rows.append(summary)

        print(f"\nBudget {budget}")

        print(f"  generic recovery      : {summary['generic_recovery_mean_pct']:+.3f}%")

        print(f"  shift-specific        : {summary['shift_specific_recovery_mean_pct']:+.3f}%")

        print(f"  extra recovery        : {summary['extra_recovery_mean_pp']:+.3f} pp")

        print(f"  95% CI                : [{summary['extra_recovery_ci_low_pp']:+.3f}, {summary['extra_recovery_ci_high_pp']:+.3f}] pp")

        print(f"  exact sign-flip p     : {summary['extra_recovery_exact_p']:.4f}")

        print(f"  positive repeats      : {summary['positive_repeats']}/{summary['n']}")

    with open("v12_within_shift_results.csv", "w", newline="", encoding="utf-8") as f:

        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))

        writer.writeheader()

        writer.writerows(rows)

    with open("v12_within_shift_summary.csv", "w", newline="", encoding="utf-8") as f:

        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))

        writer.writeheader()

        writer.writerows(summary_rows)

    np.savez(

        "embodiment_recovery_v12_within_shift_results.npz",

        budget=np.asarray([r["budget"] for r in rows], dtype=np.int32),

        repeat=np.asarray([r["repeat"] for r in rows], dtype=np.int32),

        zero_rollout=np.asarray([r["zero_rollout"] for r in rows], dtype=np.float64),

        generic_rollout=np.asarray([r["generic_rollout"] for r in rows], dtype=np.float64),

        shifted_rollout=np.asarray([r["shifted_rollout"] for r in rows], dtype=np.float64),

        generic_recovery_pct=np.asarray([r["generic_recovery_pct"] for r in rows], dtype=np.float64),

        shifted_recovery_pct=np.asarray([r["shifted_recovery_pct"] for r in rows], dtype=np.float64),

        extra_reduction=np.asarray([r["extra_reduction"] for r in rows], dtype=np.float64),

        extra_recovery_pp=np.asarray([r["extra_recovery_pp"] for r in rows], dtype=np.float64),

    )

    report = [

        "V12 — WITHIN-SHIFT CONTROL",

        "=" * 90,

        "A = no-shift adaptation source (M×1 I×1).",

        "B = shifted target (M×5 I×5).",

        "",

        "Primary contrast = generic A→B vs shift-specific B→B.",

        "Positive extra recovery means B-trained adaptation performs better on B.",

        "",

        "GUARDRAILS",

        "-" * 90,

        "Both modes use identical adaptation seed subsets.",

        "Both modes are selected using identical B validation seeds.",

        "Final results use fresh B confirmation seeds.",

        "n=5 repeats per budget is exploratory; do not treat a p-value as the sole evidence.",

    ]

    for s in summary_rows:

        report.extend([

            "",

            f"Budget {s['budget']} trajectories",

            f"Generic recovery mean: {s['generic_recovery_mean_pct']:+.3f}%",

            f"Shift-specific recovery mean: {s['shift_specific_recovery_mean_pct']:+.3f}%",

            f"Extra recovery: {s['extra_recovery_mean_pp']:+.3f} pp",

            f"95% CI: [{s['extra_recovery_ci_low_pp']:+.3f}, {s['extra_recovery_ci_high_pp']:+.3f}] pp",

            f"Exact sign-flip p: {s['extra_recovery_exact_p']:.4f}",

            f"Positive repeats: {s['positive_repeats']}/{s['n']}",

        ])

    Path("v12_within_shift_report.txt").write_text("\n".join(report), encoding="utf-8")

    print("\nSAVED:")

    print("  v12_within_shift_results.csv")

    print("  v12_within_shift_summary.csv")

    print("  embodiment_recovery_v12_within_shift_results.npz")

    print("  v12_within_shift_report.txt")

    print("  v12_generic_Mx5_Ix5_* checkpoints")

    print("  v12_shift_specific_Mx5_Ix5_* checkpoints")
