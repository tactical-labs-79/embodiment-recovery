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

TOTAL_REPEATS = 10

OLD_V12_REPEATS = 5

NEW_REPEAT_START = 6

OLD_RESULTS_FILE = Path("v12_within_shift_results.csv")

OLD_SUMMARY_FILE = Path("v12_within_shift_summary.csv")

OUTPUT_RESULTS_FILE = Path("v12_1_within_shift_results.csv")

OUTPUT_SUMMARY_FILE = Path("v12_1_within_shift_summary.csv")

OUTPUT_NPZ_FILE = Path("embodiment_recovery_v12_1_within_shift_results.npz")

OUTPUT_REPORT_FILE = Path("v12_1_within_shift_report.txt")

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

        node

        for node in tree.body

        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))

    ]

    module = ast.Module(body=defs, type_ignores=[])

    namespace = globals()

    exec(compile(module, str(path), "exec"), namespace, namespace)

load_v11_definitions(V11_SOURCE)

def build_all_subset_specs():

    specs = []

    for budget in BUDGETS:

        used = set()

        for repeat in range(1, TOTAL_REPEATS + 1):

            rng = np.random.default_rng(82000 + budget * 100 + repeat)

            selected = tuple(

                sorted(

                    rng.choice(

                        ADAPT_SEEDS,

                        size=budget,

                        replace=False,

                    ).tolist()

                )

            )

            if selected in used:

                raise RuntimeError(

                    f"Duplicate subset generated for budget={budget}, repeat={repeat}: {selected}"

                )

            used.add(selected)

            specs.append(

                {

                    "budget": budget,

                    "repeat": repeat,

                    "seeds": list(selected),

                }

            )

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

    values = values[np.isfinite(values)]

    mean = float(np.mean(values))

    std = float(np.std(values, ddof=1)) if len(values) > 1 else float("nan")

    if len(values) > 1:

        tcrit = float(stats.t.ppf(0.975, len(values) - 1))

        half = tcrit * std / math.sqrt(len(values))

        ci_low = mean - half

        ci_high = mean + half

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

def read_old_v12_rows():

    if not OLD_RESULTS_FILE.exists():

        return {}

    with OLD_RESULTS_FILE.open("r", newline="", encoding="utf-8") as f:

        rows = list(csv.DictReader(f))

    usable = {}

    required = {

        "budget",

        "repeat",

        "seeds",

        "zero_rollout",

        "generic_rollout",

        "generic_reduction",

        "generic_recovery_pct",

        "shifted_rollout",

        "shifted_reduction",

        "shifted_recovery_pct",

        "extra_reduction",

        "extra_recovery_pp",

    }

    for row in rows:

        if not required.issubset(row.keys()):

            continue

        try:

            budget = int(row["budget"])

            repeat = int(row["repeat"])

        except Exception:

            continue

        if budget not in BUDGETS or repeat < 1 or repeat > OLD_V12_REPEATS:

            continue

        key = (budget, repeat)

        usable[key] = {

            "budget": budget,

            "repeat": repeat,

            "seeds": row["seeds"],

            "zero_rollout": float(row["zero_rollout"]),

            "generic_rollout": float(row["generic_rollout"]),

            "generic_reduction": float(row["generic_reduction"]),

            "generic_recovery_pct": float(row["generic_recovery_pct"]),

            "generic_best_epoch": int(float(row.get("generic_best_epoch", 0))),

            "generic_best_val": float(row.get("generic_best_val", "nan")),

            "generic_best_val_teacher": float(row.get("generic_best_val_teacher", "nan")),

            "shifted_rollout": float(row["shifted_rollout"]),

            "shifted_reduction": float(row["shifted_reduction"]),

            "shifted_recovery_pct": float(row["shifted_recovery_pct"]),

            "shifted_best_epoch": int(float(row.get("shifted_best_epoch", 0))),

            "shifted_best_val": float(row.get("shifted_best_val", "nan")),

            "shifted_best_val_teacher": float(row.get("shifted_best_val_teacher", "nan")),

            "extra_reduction": float(row["extra_reduction"]),

            "extra_recovery_pp": float(row["extra_recovery_pp"]),

            "generic_confirmation_std": float(row.get("generic_confirmation_std", "nan")),

            "shifted_confirmation_std": float(row.get("shifted_confirmation_std", "nan")),

            "source": "reused_v12",

        }

    return usable

def evaluate_b_confirmation(model, body_name, baseline_mass, baseline_inertia):

    rows = []

    for seed in CONFIRMATION_SEEDS:

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

def train_and_validate(

    train_pool,

    base_state_dict,

    body_name,

    baseline_mass,

    baseline_inertia,

    run_seed,

):

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

    print("V12.1 — WITHIN-SHIFT CONTROL / 10 PAIRED REPEATS")

    print("=" * 110)

    print("A source : M×1 I×1")

    print("B target : M×5 I×5")

    print("Budgets  :", BUDGETS)

    print("Total repeats per budget:", TOTAL_REPEATS)

    print("Existing V12 repeats reused:", OLD_V12_REPEATS)

    print("New repeats trained:", list(range(NEW_REPEAT_START, TOTAL_REPEATS + 1)))

    print("Validation B seeds   :", VALIDATION_SEEDS)

    print("Confirmation B seeds :", CONFIRMATION_SEEDS)

    all_specs = build_all_subset_specs()

    old_rows = read_old_v12_rows()

    spec_map = {

        (s["budget"], s["repeat"]): s

        for s in all_specs

    }

    rows_by_key = {}

    for key, row in old_rows.items():

        spec = spec_map.get(key)

        if spec is None:

            continue

        expected_seeds = ",".join(map(str, spec["seeds"]))

        if row["seeds"] != expected_seeds:

            print(

                f"WARNING: old V12 row {key} has seed subset {row['seeds']} "

                f"but expected {expected_seeds}; it will be recomputed."

            )

            continue

        rows_by_key[key] = row

    print()

    print(f"Usable previous V12 rows: {len(rows_by_key)} / {2 * OLD_V12_REPEATS}")

    missing_old = [

        (budget, repeat)

        for budget in BUDGETS

        for repeat in range(1, OLD_V12_REPEATS + 1)

        if (budget, repeat) not in rows_by_key

    ]

    if missing_old:

        print("Missing old V12 rows that will be recomputed:", missing_old)

    body_name, baseline_mass, baseline_inertia = get_physical_config()

    base_model = load_v7_model()

    base_state_dict = copy.deepcopy(base_model.state_dict())

    print("\nPhysical body:", body_name)

    print("Baseline mass:", baseline_mass)

    print("Baseline inertia:", baseline_inertia)

    print("\nBuilding A/B adaptation pools with the SAME seed IDs...")

    a_pool = build_b_pool(

        seeds=ADAPT_SEEDS,

        body_name=body_name,

        baseline_mass=baseline_mass,

        baseline_inertia=baseline_inertia,

        mass_multiplier=A_MASS,

        inertia_multiplier=A_INERTIA,

    )

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

        base_model,

        body_name,

        baseline_mass,

        baseline_inertia,

    )

    zero_values = np.asarray(

        [r["rollout"] for r in zero_rows],

        dtype=float,

    )

    zero_rollout = float(zero_values.mean())

    print(f"Zero-shot B rollout = {zero_rollout:.8f}")

    for spec in all_specs:

        budget = spec["budget"]

        repeat = spec["repeat"]

        key = (budget, repeat)

        if key in rows_by_key:

            continue

        seeds = spec["seeds"]

        print("\n" + "#" * 110)

        print(f"V12.1 TRAIN — BUDGET={budget} REPEAT={repeat}")

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

            generic_model,

            body_name,

            baseline_mass,

            baseline_inertia,

        )

        generic_vals = np.asarray(

            [r["rollout"] for r in generic_conf],

            dtype=float,

        )

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

            shifted_model,

            body_name,

            baseline_mass,

            baseline_inertia,

        )

        shifted_vals = np.asarray(

            [r["rollout"] for r in shifted_conf],

            dtype=float,

        )

        generic_reduction = zero_rollout - float(generic_vals.mean())

        shifted_reduction = zero_rollout - float(shifted_vals.mean())

        generic_recovery = 100.0 * generic_reduction / max(zero_rollout, 1e-12)

        shifted_recovery = 100.0 * shifted_reduction / max(zero_rollout, 1e-12)

        extra_reduction_per_seed = generic_vals - shifted_vals

        extra_reduction = float(extra_reduction_per_seed.mean())

        extra_recovery_pp = float(shifted_recovery - generic_recovery)

        row = {

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

            "source": "new_v12_1",

        }

        rows_by_key[key] = row

        torch.save(

            generic_model.state_dict(),

            f"v12_1_generic_Mx5_Ix5_{budget}traj_subset{repeat}.pt",

        )

        torch.save(

            shifted_model.state_dict(),

            f"v12_1_shift_specific_Mx5_Ix5_{budget}traj_subset{repeat}.pt",

        )

        print("\nRESULT")

        print(f"  zero-shot B            : {zero_rollout:.8f}")

        print(f"  generic A→B            : {generic_vals.mean():.8f} ({generic_recovery:+.3f}%)")

        print(f"  shift-specific B→B     : {shifted_vals.mean():.8f} ({shifted_recovery:+.3f}%)")

        print(f"  extra reduction        : {extra_reduction:+.8f}")

        print(f"  extra recovery         : {extra_recovery_pp:+.3f} pp")

    rows = [

        rows_by_key[(budget, repeat)]

        for budget in BUDGETS

        for repeat in range(1, TOTAL_REPEATS + 1)

        if (budget, repeat) in rows_by_key

    ]

    print("\n" + "=" * 110)

    print("V12.1 SUMMARY — SHIFT-SPECIFIC ADVANTAGE / 10 PAIRED REPEATS")

    print("=" * 110)

    summary_rows = []

    for budget in BUDGETS:

        sub = [r for r in rows if r["budget"] == budget]

        if len(sub) != TOTAL_REPEATS:

            raise RuntimeError(

                f"Expected {TOTAL_REPEATS} rows for budget={budget}, found {len(sub)}"

            )

        extra_reduction = np.asarray(

            [r["extra_reduction"] for r in sub],

            dtype=float,

        )

        extra_pp = np.asarray(

            [r["extra_recovery_pp"] for r in sub],

            dtype=float,

        )

        rs = paired_summary(extra_reduction)

        ps = paired_summary(extra_pp)

        summary = {

            "budget": budget,

            "n": ps["n"],

            "generic_recovery_mean_pct": float(

                np.mean([r["generic_recovery_pct"] for r in sub])

            ),

            "shift_specific_recovery_mean_pct": float(

                np.mean([r["shifted_recovery_pct"] for r in sub])

            ),

            "extra_recovery_mean_pp": ps["mean"],

            "extra_recovery_std_pp": ps["std"],

            "extra_recovery_ci_low_pp": ps["ci_low"],

            "extra_recovery_ci_high_pp": ps["ci_high"],

            "extra_recovery_t_p": ps["t_p"],

            "extra_recovery_exact_p": ps["exact_p"],

            "positive_repeats": ps["positive"],

            "extra_reduction_mean": rs["mean"],

            "extra_reduction_ci_low": rs["ci_low"],

            "extra_reduction_ci_high": rs["ci_high"],

            "source_old_repeats": sum(

                r.get("source") == "reused_v12"

                for r in sub

            ),

            "source_new_repeats": sum(

                r.get("source") == "new_v12_1"

                for r in sub

            ),

        }

        summary_rows.append(summary)

        print(f"\nBudget {budget}")

        print(f"  generic recovery       : {summary['generic_recovery_mean_pct']:+.3f}%")

        print(f"  shift-specific         : {summary['shift_specific_recovery_mean_pct']:+.3f}%")

        print(f"  extra recovery         : {summary['extra_recovery_mean_pp']:+.3f} pp")

        print(

            f"  95% CI                 : "

            f"[{summary['extra_recovery_ci_low_pp']:+.3f}, "

            f"{summary['extra_recovery_ci_high_pp']:+.3f}] pp"

        )

        print(f"  exact sign-flip p      : {summary['extra_recovery_exact_p']:.4f}")

        print(f"  positive repeats       : {summary['positive_repeats']}/{summary['n']}")

        print(

            f"  reused/new repeats     : "

            f"{summary['source_old_repeats']}/{summary['source_new_repeats']}"

        )

    result_fields = list(rows[0].keys())

    with OUTPUT_RESULTS_FILE.open("w", newline="", encoding="utf-8") as f:

        writer = csv.DictWriter(f, fieldnames=result_fields)

        writer.writeheader()

        writer.writerows(rows)

    summary_fields = list(summary_rows[0].keys())

    with OUTPUT_SUMMARY_FILE.open("w", newline="", encoding="utf-8") as f:

        writer = csv.DictWriter(f, fieldnames=summary_fields)

        writer.writeheader()

        writer.writerows(summary_rows)

    np.savez(

        OUTPUT_NPZ_FILE,

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

        "V12.1 — WITHIN-SHIFT CONTROL / 10 PAIRED REPEATS",

        "=" * 90,

        "A = no-shift adaptation source (M×1 I×1).",

        "B = shifted target (M×5 I×5).",

        "",

        "V12 repeats 1–5 are reused when the prior V12 CSV is available.",

        "V12.1 trains only repeats 6–10, preserving the same protocol.",

        "",

        "Primary contrast = generic A→B vs shift-specific B→B.",

        "Positive extra recovery means B-trained adaptation performs better on B.",

        "",

        "GUARDRAILS",

        "-" * 90,

        "Same adaptation subset seeds are paired between A and B.",

        "Same B validation seeds are used for both modes.",

        "Same B confirmation seeds are used for both modes.",

        "This is an extension from n=5 to n=10 paired repeats, not a new shift configuration.",

        "Validation-B early stopping remains part of the protocol, so the generic condition is not blind to B validation data.",

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

    OUTPUT_REPORT_FILE.write_text("\n".join(report), encoding="utf-8")

    print("\nSAVED:")

    print(f"  {OUTPUT_RESULTS_FILE}")

    print(f"  {OUTPUT_SUMMARY_FILE}")

    print(f"  {OUTPUT_NPZ_FILE}")

    print(f"  {OUTPUT_REPORT_FILE}")

    print("  v12_1_generic_Mx5_Ix5_* checkpoints")

    print("  v12_1_shift_specific_Mx5_Ix5_* checkpoints")
