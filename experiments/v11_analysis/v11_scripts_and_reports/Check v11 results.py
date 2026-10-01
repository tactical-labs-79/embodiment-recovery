import numpy as np

RESULTS_FILE = "embodiment_recovery_v11_compound_dynamics_results.npz"

data = np.load(RESULTS_FILE)

n = len(data["mass_multiplier"])

print(f"Total rows in file: {n}")

print()

combos = {}

for i in range(n):

    mass = round(float(data["mass_multiplier"][i]), 2)

    inertia = round(float(data["inertia_multiplier"][i]), 2)

    budget = int(data["budget"][i])

    repeat = int(data["repeat"][i])

    recovery = float(data["recovery_pct"][i])

    key = (mass, inertia, budget)

    combos.setdefault(key, []).append((repeat, recovery))

SHIFT_CONFIGS = [

    (1.0, 1.0), (3.0, 1.0), (1.0, 3.0),

    (2.0, 2.0), (3.0, 3.0),

    (3.0, 5.0), (5.0, 3.0), (5.0, 5.0),

]

BUDGETS = [5, 10]

EXPECTED_REPEATS = 5

print(f"{'Mass':>6} {'Inertia':>8} {'Budget':>7} {'Repeats found':>14} {'Complete?':>10} {'Mean recovery%':>16}")

print("-" * 70)

all_complete = True

for mass, inertia in SHIFT_CONFIGS:

    for budget in BUDGETS:

        key = (round(mass, 2), round(inertia, 2), budget)

        entries = combos.get(key, [])

        repeats_found = sorted(r for r, _ in entries)

        complete = (len(entries) == EXPECTED_REPEATS

                    and repeats_found == list(range(1, EXPECTED_REPEATS + 1)))

        if not complete:

            all_complete = False

        mean_rec = np.mean([r for _, r in entries]) if entries else float("nan")

        print(f"{mass:6.1f} {inertia:8.1f} {budget:7d} {str(repeats_found):>14} "

              f"{'YES' if complete else 'NO':>10} {mean_rec:16.3f}")

print()

if all_complete:

    print("ALL expected (shift, budget) combinations are complete with 5/5 repeats.")

else:

    print("SOME combinations are INCOMPLETE or MISSING — see NO rows above.")

unexpected = set(combos.keys()) - {

    (round(m, 2), round(i, 2), b) for m, i in SHIFT_CONFIGS for b in BUDGETS

}

if unexpected:

    print()

    print("NOTE: found rows for combinations not in the expected SHIFT_CONFIGS list:")

    for k in sorted(unexpected):

        print(" ", k)
