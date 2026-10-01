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

ADAPT_SEEDS = list(range(9101, 9121))

VALIDATION_SEEDS = [9301, 9302, 9303, 9304, 9305]

CONFIRMATION_SEEDS = [9601, 9602, 9603, 9604, 9605]

BUDGETS = [5, 10]

REPEATS = 20

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

OUTDIR = Path("v13_cross_shift")

OUTPUT_TABLE_FILE = OUTDIR / "v13_cross_shift_table.csv"

OUTPUT_RUNLEVEL_FILE = OUTDIR / "v13_cross_shift_run_level.csv"

OUTPUT_REPORT_FILE = OUTDIR / "v13_cross_shift_report.txt"

SHIFTS = [

    {

        "shift_id": 0,

        "name": "M5_I5",

        "mass": 5.0,

        "inertia": 5.0,

        "reuse_csv": Path("v12_2_within_shift_results.csv"),

        "reuse_max_repeat": REPEATS,

        "reuse_source_tag": "reused_v12_2",

        "strict_reuse": True,

    },

    {

        "shift_id": 1,

        "name": "M3_I3",

        "mass": 3.0,

        "inertia": 3.0,

        "reuse_csv": None,

        "reuse_max_repeat": 0,

        "reuse_source_tag": None,

    },

    {

        "shift_id": 2,

        "name": "M5_I1",

        "mass": 5.0,

        "inertia": 1.0,

        "reuse_csv": None,

        "reuse_max_repeat": 0,

        "reuse_source_tag": None,

    },

    {

        "shift_id": 3,

        "name": "M1_I5",

        "mass": 1.0,

        "inertia": 5.0,

        "reuse_csv": None,

        "reuse_max_repeat": 0,

        "reuse_source_tag": None,

    },

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

def build_all_subset_specs():

    specs = []

    for budget in BUDGETS:

        used = set()

        for repeat in range(1, REPEATS + 1):

            rng = np.random.default_rng(82000 + budget * 100 + repeat)

            selected = tuple(

                sorted(

                    rng.choice(ADAPT_SEEDS, size=budget, replace=False).tolist()

                )

            )

            if selected in used:

                raise RuntimeError(

                    f"Duplicate subset for budget={budget}, repeat={repeat}: {selected}"

                )

            used.add(selected)

            specs.append({"budget": budget, "repeat": repeat, "seeds": list(selected)})

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

REQUIRED_COLUMNS = {

    "budget", "repeat", "seeds", "zero_rollout",

    "generic_rollout", "generic_reduction", "generic_recovery_pct",

    "shifted_rollout", "shifted_reduction", "shifted_recovery_pct",

    "extra_reduction", "extra_recovery_pp",

}

def read_reuse_rows(path, source_tag, max_repeat, expected_specs=None, strict=False):

    if path is None:

        if strict:

            raise RuntimeError("Strict reuse requested but no reuse_csv path was given.")

        return {}

    if not path.exists():

        if strict:

            raise FileNotFoundError(

                f"Strict reuse requires {path} to exist and be complete, but it was not found."

            )

        return {}

    with path.open("r", newline="", encoding="utf-8") as f:

        rows = list(csv.DictReader(f))

    usable = {}

    bad_rows = 0

    for row in rows:

        if not REQUIRED_COLUMNS.issubset(row.keys()):

            bad_rows += 1

            continue

        try:

            budget = int(row["budget"])

            repeat = int(row["repeat"])

        except Exception:

            bad_rows += 1

            continue

        if budget not in BUDGETS or repeat < 1 or repeat > max_repeat:

            continue

        key = (budget, repeat)

        if key in usable:

            if strict:

                raise RuntimeError(

                    f"Duplicate (budget={budget}, repeat={repeat}) row found in {path}."

                )

            continue

        try:

            entry = {

                "budget": budget,

                "repeat": repeat,

                "seeds": row["seeds"],

                "zero_rollout": float(row["zero_rollout"]),

                "generic_rollout": float(row["generic_rollout"]),

                "generic_reduction": float(row["generic_reduction"]),

                "generic_recovery_pct": float(row["generic_recovery_pct"]),

                "shifted_rollout": float(row["shifted_rollout"]),

                "shifted_reduction": float(row["shifted_reduction"]),

                "shifted_recovery_pct": float(row["shifted_recovery_pct"]),

                "extra_reduction": float(row["extra_reduction"]),

                "extra_recovery_pp": float(row["extra_recovery_pp"]),

                "source": source_tag,

            }

        except Exception:

            bad_rows += 1

            continue

        usable[key] = entry

    if strict:

        if bad_rows > 0:

            raise RuntimeError(

                f"{path} contains {bad_rows} malformed/unreadable row(s); "

                f"strict reuse refuses to proceed with a possibly-corrupt file."

            )

        if expected_specs is not None:

            missing = []

            mismatched = []

            for spec in expected_specs:

                key = (spec["budget"], spec["repeat"])

                if key not in usable:

                    missing.append(key)

                    continue

                expected_seeds = ",".join(map(str, spec["seeds"]))

                if usable[key]["seeds"] != expected_seeds:

                    mismatched.append((key, usable[key]["seeds"], expected_seeds))

            if missing:

                raise RuntimeError(

                    f"Strict reuse from {path} is missing rows for: {missing}"

                )

            if mismatched:

                raise RuntimeError(

                    f"Strict reuse from {path} has seed-subset mismatches: {mismatched}"

                )

    return usable

def evaluate_confirmation(model, body_name, baseline_mass, baseline_inertia,

                           mass_mult, inertia_mult):

    rows = []

    for seed in CONFIRMATION_SEEDS:

        rows.append(

            evaluate_model_on_b(

                model=model,

                seed=seed,

                body_name=body_name,

                baseline_mass=baseline_mass,

                baseline_inertia=baseline_inertia,

                mass_multiplier=mass_mult,

                inertia_multiplier=inertia_mult,

            )

        )

    return rows

def train_and_validate(train_pool, base_state_dict, body_name, baseline_mass,

                        baseline_inertia, mass_mult, inertia_mult, run_seed):

    def validation_fn(candidate):

        candidate.eval()

        teacher, rollout = [], []

        for seed in VALIDATION_SEEDS:

            r = evaluate_model_on_b(

                model=candidate,

                seed=seed,

                body_name=body_name,

                baseline_mass=baseline_mass,

                baseline_inertia=baseline_inertia,

                mass_multiplier=mass_mult,

                inertia_multiplier=inertia_mult,

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

def run_shift(shift, all_specs, a_by_seed, base_model, base_state_dict,

              body_name, baseline_mass, baseline_inertia):

    name = shift["name"]

    mass_mult = shift["mass"]

    inertia_mult = shift["inertia"]

    shift_id = shift["shift_id"]

    print("\n" + "=" * 110)

    print(f"SHIFT {name}  (mass x{mass_mult}, inertia x{inertia_mult})")

    print("=" * 110)

    strict_reuse = shift.get("strict_reuse", False)

    reused = read_reuse_rows(

        shift["reuse_csv"], shift["reuse_source_tag"], shift["reuse_max_repeat"],

        expected_specs=all_specs if strict_reuse else None,

        strict=strict_reuse,

    )

    if reused:

        print(f"Reusing {len(reused)} existing rows from {shift['reuse_csv']}")

        expected_count = len(BUDGETS) * shift["reuse_max_repeat"]

        if strict_reuse and len(reused) != expected_count:

            raise RuntimeError(

                f"Strict reuse for {name} expected exactly {expected_count} rows "

                f"({len(BUDGETS)} budgets x {shift['reuse_max_repeat']} repeats), "

                f"got {len(reused)}. Refusing to continue with incomplete reused data."

            )

    rows_by_key = dict(reused)

    needs_training = any(

        (spec["budget"], spec["repeat"]) not in rows_by_key for spec in all_specs

    )

    b_pool = None

    b_by_seed = None

    zero_rollout = None

    if needs_training:

        print(f"Building B pool for {name} (mass x{mass_mult}, inertia x{inertia_mult})...")

        b_pool = build_b_pool(

            seeds=ADAPT_SEEDS,

            body_name=body_name,

            baseline_mass=baseline_mass,

            baseline_inertia=baseline_inertia,

            mass_multiplier=mass_mult,

            inertia_multiplier=inertia_mult,

        )

        b_by_seed = {t["seed"]: t for t in b_pool}

        print(f"Computing zero-shot baseline for {name}...")

        zero_rows = evaluate_confirmation(

            base_model, body_name, baseline_mass, baseline_inertia,

            mass_mult, inertia_mult,

        )

        zero_rollout = float(np.mean([r["rollout"] for r in zero_rows]))

        print(f"Zero-shot rollout ({name}) = {zero_rollout:.8f}")

    for spec in all_specs:

        budget, repeat = spec["budget"], spec["repeat"]

        key = (budget, repeat)

        if key in rows_by_key:

            continue

        seeds = spec["seeds"]

        print(f"\n[{name}] BUDGET={budget} REPEAT={repeat} | seeds={seeds}")

        generic_train = [a_by_seed[s] for s in seeds]

        shifted_train = [b_by_seed[s] for s in seeds]

        run_seed_base = 1155000 + shift_id * 10000 + budget * 100 + repeat

        generic_model, *_ = train_and_validate(

            generic_train, base_state_dict, body_name, baseline_mass,

            baseline_inertia, mass_mult, inertia_mult, run_seed=run_seed_base,

        )

        generic_conf = evaluate_confirmation(

            generic_model, body_name, baseline_mass, baseline_inertia,

            mass_mult, inertia_mult,

        )

        generic_vals = np.asarray([r["rollout"] for r in generic_conf], dtype=float)

        shifted_model, *_ = train_and_validate(

            shifted_train, base_state_dict, body_name, baseline_mass,

            baseline_inertia, mass_mult, inertia_mult, run_seed=run_seed_base,

        )

        shifted_conf = evaluate_confirmation(

            shifted_model, body_name, baseline_mass, baseline_inertia,

            mass_mult, inertia_mult,

        )

        shifted_vals = np.asarray([r["rollout"] for r in shifted_conf], dtype=float)

        generic_reduction = zero_rollout - float(generic_vals.mean())

        shifted_reduction = zero_rollout - float(shifted_vals.mean())

        generic_recovery = 100.0 * generic_reduction / max(zero_rollout, 1e-12)

        shifted_recovery = 100.0 * shifted_reduction / max(zero_rollout, 1e-12)

        extra_reduction = float((generic_vals - shifted_vals).mean())

        extra_recovery_pp = float(shifted_recovery - generic_recovery)

        rows_by_key[key] = {

            "budget": budget,

            "repeat": repeat,

            "seeds": ",".join(map(str, seeds)),

            "zero_rollout": zero_rollout,

            "generic_rollout": float(generic_vals.mean()),

            "generic_reduction": generic_reduction,

            "generic_recovery_pct": generic_recovery,

            "shifted_rollout": float(shifted_vals.mean()),

            "shifted_reduction": shifted_reduction,

            "shifted_recovery_pct": shifted_recovery,

            "extra_reduction": extra_reduction,

            "extra_recovery_pp": extra_recovery_pp,

            "source": f"new_v13_{name}",

        }

        checkpoint_dir = OUTDIR / "checkpoints"

        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        torch.save(generic_model.state_dict(),

                   checkpoint_dir / f"v13_{name}_generic_{budget}traj_subset{repeat}.pt")

        torch.save(shifted_model.state_dict(),

                   checkpoint_dir / f"v13_{name}_shift_specific_{budget}traj_subset{repeat}.pt")

        print(f"  extra recovery: {extra_recovery_pp:+.3f} pp")

    ordered_rows = [rows_by_key[(s["budget"], s["repeat"])] for s in all_specs]

    for r in ordered_rows:

        r["shift"] = name

    return ordered_rows

if __name__ == "__main__":

    OUTDIR.mkdir(parents=True, exist_ok=True)

    existing_outputs = [p for p in (OUTPUT_TABLE_FILE, OUTPUT_RUNLEVEL_FILE, OUTPUT_REPORT_FILE)

                         if p.exists()]

    if existing_outputs:

        raise FileExistsError(

            "Refusing to overwrite existing V13 outputs without confirmation: "

            + ", ".join(str(p) for p in existing_outputs)

            + ". Move, rename, or delete them first if you intend to redo this run."

        )

    print("=" * 110)

    print("V13 — CROSS-SHIFT GENERALIZATION BENCHMARK")

    print("=" * 110)

    print("Shifts   :", [s["name"] for s in SHIFTS])

    print("Budgets  :", BUDGETS)

    print("Repeats  :", REPEATS, "per (shift, budget)")

    all_specs = build_all_subset_specs()

    body_name, baseline_mass, baseline_inertia = get_physical_config()

    base_model = load_v7_model()

    base_state_dict = copy.deepcopy(base_model.state_dict())

    print("\nBuilding shared A pool (mass x1, inertia x1)...")

    a_pool = build_b_pool(

        seeds=ADAPT_SEEDS,

        body_name=body_name,

        baseline_mass=baseline_mass,

        baseline_inertia=baseline_inertia,

        mass_multiplier=A_MASS,

        inertia_multiplier=A_INERTIA,

    )

    a_by_seed = {t["seed"]: t for t in a_pool}

    all_run_level_rows = []

    for shift in SHIFTS:

        shift_rows = run_shift(

            shift, all_specs, a_by_seed, base_model, base_state_dict,

            body_name, baseline_mass, baseline_inertia,

        )

        all_run_level_rows.extend(shift_rows)

    print("\n" + "=" * 110)

    print("V13 CROSS-SHIFT TABLE")

    print("=" * 110)

    table_rows = []

    raw_pvals = []

    for shift in SHIFTS:

        name = shift["name"]

        for budget in BUDGETS:

            sub = [r for r in all_run_level_rows

                   if r["shift"] == name and r["budget"] == budget]

            if len(sub) != REPEATS:

                raise RuntimeError(

                    f"Completeness check failed: shift={name} budget={budget} "

                    f"has {len(sub)} rows, expected exactly {REPEATS}. "

                    f"Refusing to build the cross-shift table on incomplete data."

                )

            extra_pp = np.asarray([r["extra_recovery_pp"] for r in sub], dtype=float)

            ps = paired_summary(extra_pp)

            entry = {

                "shift": name,

                "mass_mult": shift["mass"],

                "inertia_mult": shift["inertia"],

                "budget": budget,

                "n": ps["n"],

                "generic_recovery_mean_pct": float(

                    np.mean([r["generic_recovery_pct"] for r in sub])

                ),

                "shift_specific_recovery_mean_pct": float(

                    np.mean([r["shifted_recovery_pct"] for r in sub])

                ),

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

    print(f"\n{'shift':8s} {'budget':6s} {'generic%':>9s} {'specific%':>10s} "

          f"{'extra_pp':>9s} {'95% CI':>20s} {'raw_p':>8s} {'holm_p':>8s} {'pos':>6s}")

    for e in table_rows:

        ci_str = f"[{e['extra_recovery_ci_low_pp']:+.2f},{e['extra_recovery_ci_high_pp']:+.2f}]"

        print(f"{e['shift']:8s} {e['budget']:<6d} "

              f"{e['generic_recovery_mean_pct']:>+8.3f}% "

              f"{e['shift_specific_recovery_mean_pct']:>+9.3f}% "

              f"{e['extra_recovery_mean_pp']:>+8.3f} "

              f"{ci_str:>20s} "

              f"{e['exact_sign_flip_p']:>8.4f} "

              f"{e['holm_bonferroni_p']:>8.4f} "

              f"{e['positive_repeats']:>3d}/{e['n']}")

    with OUTPUT_TABLE_FILE.open("w", newline="", encoding="utf-8") as f:

        writer = csv.DictWriter(f, fieldnames=list(table_rows[0].keys()))

        writer.writeheader()

        writer.writerows(table_rows)

    run_level_fields = ["shift", "budget", "repeat", "seeds", "zero_rollout",

                        "generic_rollout", "generic_recovery_pct",

                        "shifted_rollout", "shifted_recovery_pct",

                        "extra_reduction", "extra_recovery_pp", "source"]

    with OUTPUT_RUNLEVEL_FILE.open("w", newline="", encoding="utf-8") as f:

        writer = csv.DictWriter(f, fieldnames=run_level_fields, extrasaction="ignore")

        writer.writeheader()

        writer.writerows(all_run_level_rows)

    report = [

        "V13 — CROSS-SHIFT GENERALIZATION BENCHMARK",

        "=" * 90,

        "Tests whether the shift-specific adaptation advantage (V12/V12.1/V12.2,",

        "M5/I5) survives across different embodiment shift types/magnitudes.",

        "",

        "M5_I5 rows are REUSED from V12.2 (not retrained). M3_I3, M5_I1, M1_I5",

        "are new fresh runs with 20 paired repeats each, identical protocol.",

        "",

        "CAUTION — MULTIPLE COMPARISONS:",

        "This table reports 8 tests (4 shifts x 2 budgets). Use the",

        "Holm-Bonferroni corrected p-value (holm_bonferroni_p), not the raw",

        "exact sign-flip p, when judging whether an individual cell is",

        "significant after accounting for testing 8 hypotheses at once.",

        "",

        "CAUTION — SEQUENTIAL TESTING ON M5_I5:",

        "The M5_I5 n was increased across V12 (n=5) -> V12.1 (n=10) ->",

        "V12.2 (n=20) based on interim results. Its p-value here should be",

        "read as exploratory, not a clean confirmatory test.",

    ]

    (OUTDIR / "v13_cross_shift_report.txt").write_text("\n".join(report), encoding="utf-8")

    report_lines = [OUTPUT_REPORT_FILE]

    print("\nSAVED:")

    print(f"  {OUTPUT_TABLE_FILE}")

    print(f"  {OUTPUT_RUNLEVEL_FILE}")

    print(f"  {OUTPUT_REPORT_FILE}")

    print("  v13_<shift>_generic_* / v13_<shift>_shift_specific_* checkpoints")
