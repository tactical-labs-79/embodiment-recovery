import numpy as np

RESULT_FILE = "embodiment_shift_v5_excitation_results.npz"

SEEDS = [

    9001,

    9002,

    9003,

    9004,

    9005,

]

MULTIPLIERS = [

    1.0,

    2.0,

    3.0,

    5.0,

]

EPS = 1e-12

data = np.load(

    RESULT_FILE

)

physical_gap = data[

    "physical_gap"

]

a_one = data[

    "a_one"

]

b_one = data[

    "b_one"

]

a_roll = data[

    "a_roll"

]

b_roll = data[

    "b_roll"

]

delta_one = data[

    "delta_one"

]

delta_roll = data[

    "delta_roll"

]

relative_roll = data[

    "relative_roll"

]

expected_n = (

    len(SEEDS)

    * len(MULTIPLIERS)

)

if len(a_roll) != expected_n:

    raise RuntimeError(

        "Unexpected result count. "

        f"Expected {expected_n}, "

        f"got {len(a_roll)}."

    )

def reshape_metric(x):

    return x.reshape(

        len(SEEDS),

        len(MULTIPLIERS)

    )

physical_gap = reshape_metric(

    physical_gap

)

a_one = reshape_metric(

    a_one

)

b_one = reshape_metric(

    b_one

)

a_roll = reshape_metric(

    a_roll

)

b_roll = reshape_metric(

    b_roll

)

delta_one = reshape_metric(

    delta_one

)

delta_roll = reshape_metric(

    delta_roll

)

relative_roll = reshape_metric(

    relative_roll

)

roll_ratio = (

    b_roll

    / np.maximum(

        a_roll,

        EPS

    )

)

one_ratio = (

    b_one

    / np.maximum(

        a_one,

        EPS

    )

)

shift_fraction_roll = (

    delta_roll

    / np.maximum(

        b_roll,

        EPS

    )

)

def pct(x):

    return x * 100.0

def fmt(x):

    return f"{x:.8f}"

def monotonic_non_decreasing(values):

    return bool(

        np.all(

            np.diff(values)

            >= -1e-12

        )

    )

def monotonic_strict(values):

    return bool(

        np.all(

            np.diff(values)

            > 0.0

        )

    )

def slope(values_x, values_y):

    x = np.asarray(

        values_x,

        dtype=np.float64

    )

    y = np.asarray(

        values_y,

        dtype=np.float64

    )

    x_centered = (

        x - x.mean()

    )

    y_centered = (

        y - y.mean()

    )

    denominator = np.sum(

        x_centered ** 2

    )

    if denominator <= EPS:

        return np.nan

    return float(

        np.sum(

            x_centered

            * y_centered

        )

        / denominator

    )

def pearson_r(x, y):

    x = np.asarray(

        x,

        dtype=np.float64

    )

    y = np.asarray(

        y,

        dtype=np.float64

    )

    x_std = x.std()

    y_std = y.std()

    if (

        x_std <= EPS

        or y_std <= EPS

    ):

        return np.nan

    return float(

        np.corrcoef(

            x,

            y

        )[0, 1]

    )

print("=" * 90)

print(

    "V6 — SHIFT-RELATIVE EMBODIMENT BENCHMARK"

)

print("=" * 90)

print()

print(

    "Source:",

    RESULT_FILE

)

print(

    "Seeds:",

    SEEDS

)

print(

    "Multipliers:",

    MULTIPLIERS

)

print()

print("=" * 90)

print(

    "PER-SEED RELATIVE DEGRADATION"

)

print("=" * 90)

print()

header = (

    f"{'Seed':>6} | "

    f"{'Mass':>6} | "

    f"{'PhysGap':>10} | "

    f"{'A Roll':>11} | "

    f"{'B Roll':>11} | "

    f"{'Excess':>11} | "

    f"{'Rel %':>9} | "

    f"{'B/A':>8}"

)

print(header)

print(

    "-" * len(header)

)

for i, seed in enumerate(SEEDS):

    for j, multiplier in enumerate(

        MULTIPLIERS

    ):

        print(

            f"{seed:6d} | "

            f"{multiplier:6.1f} | "

            f"{physical_gap[i,j]:10.6f} | "

            f"{a_roll[i,j]:11.8f} | "

            f"{b_roll[i,j]:11.8f} | "

            f"{delta_roll[i,j]:+11.8f} | "

            f"{pct(relative_roll[i,j]):+8.3f} | "

            f"{roll_ratio[i,j]:8.5f}"

        )

print()

print("=" * 90)

print(

    "PER-SEED MONOTONICITY"

)

print("=" * 90)

print()

print(

    "Question:"

)

print(

    "Does degradation increase as mass shift increases?"

)

print()

strict_count = 0

nondecreasing_count = 0

for i, seed in enumerate(SEEDS):

    values = relative_roll[

        i,

        1:

    ]

    strict = monotonic_strict(

        values

    )

    nondecreasing = (

        monotonic_non_decreasing(

            values

        )

    )

    strict_count += int(

        strict

    )

    nondecreasing_count += int(

        nondecreasing

    )

    print(

        f"Seed {seed}: "

        f"x2={pct(values[0]):+.3f}%  "

        f"x3={pct(values[1]):+.3f}%  "

        f"x5={pct(values[2]):+.3f}%  "

        f"| nondecreasing={nondecreasing} "

        f"| strict={strict}"

    )

print()

print(

    f"Nondecreasing: "

    f"{nondecreasing_count}/{len(SEEDS)}"

)

print(

    f"Strictly increasing: "

    f"{strict_count}/{len(SEEDS)}"

)

print()

print("=" * 90)

print(

    "SHIFT-RELATIVE SUMMARY"

)

print("=" * 90)

print()

header = (

    f"{'Mass':>6} | "

    f"{'Gap μ':>10} | "

    f"{'Excess μ':>11} | "

    f"{'Excess σ':>11} | "

    f"{'Rel μ %':>10} | "

    f"{'Rel σ %':>10} | "

    f"{'B/A μ':>8}"

)

print(header)

print(

    "-" * len(header)

)

for j, multiplier in enumerate(

    MULTIPLIERS

):

    gap = physical_gap[

        :,

        j

    ]

    excess = delta_roll[

        :,

        j

    ]

    rel = relative_roll[

        :,

        j

    ]

    ratio = roll_ratio[

        :,

        j

    ]

    print(

        f"{multiplier:6.1f} | "

        f"{gap.mean():10.6f} | "

        f"{excess.mean():+11.8f} | "

        f"{excess.std(ddof=1):11.8f} | "

        f"{pct(rel.mean()):+9.3f}% | "

        f"{pct(rel.std(ddof=1)):9.3f}% | "

        f"{ratio.mean():8.5f}"

    )

print()

print("=" * 90)

print(

    "BASELINE VS SHIFT"

)

print("=" * 90)

baseline_rel = relative_roll[

    :,

    0

]

print()

print(

    "x1.0 relative degradation:"

)

for i, seed in enumerate(SEEDS):

    print(

        f"  Seed {seed}: "

        f"{pct(baseline_rel[i]):+.6f}%"

    )

print()

print(

    "Mean baseline:",

    f"{pct(baseline_rel.mean()):+.6f}%"

)

print(

    "Std baseline:",

    f"{pct(baseline_rel.std(ddof=1)):.6f}%"

)

print()

print("=" * 90)

print(

    "PHYSICAL GAP → MODEL DEGRADATION"

)

print("=" * 90)

shift_gap = physical_gap[

    :,

    1:

].reshape(-1)

shift_excess = delta_roll[

    :,

    1:

].reshape(-1)

shift_relative = relative_roll[

    :,

    1:

].reshape(-1)

r_excess = pearson_r(

    shift_gap,

    shift_excess

)

r_relative = pearson_r(

    shift_gap,

    shift_relative

)

slope_excess = slope(

    shift_gap,

    shift_excess

)

slope_relative = slope(

    shift_gap,

    shift_relative

)

print()

print(

    "Pearson r("

    "physical_gap, excess_error"

    "):",

    f"{r_excess:.6f}"

)

print(

    "Pearson r("

    "physical_gap, relative_degradation"

    "):",

    f"{r_relative:.6f}"

)

print()

print(

    "Slope:"

)

print(

    "  excess error / physical gap =",

    f"{slope_excess:.6f}"

)

print(

    "  relative degradation / gap  =",

    f"{slope_relative:.6f}"

)

print()

print("=" * 90)

print(

    "PAIRED x1 → x5 EFFECT"

)

print("=" * 90)

x1 = relative_roll[

    :,

    0

]

x5 = relative_roll[

    :,

    3

]

paired_change = (

    x5 - x1

)

print()

for i, seed in enumerate(SEEDS):

    print(

        f"Seed {seed}: "

        f"x1={pct(x1[i]):+.3f}%  "

        f"x5={pct(x5[i]):+.3f}%  "

        f"Δ={pct(paired_change[i]):+.3f}%"

    )

print()

print(

    "Mean paired increase:",

    f"{pct(paired_change.mean()):+.3f}%"

)

print(

    "Std paired increase:",

    f"{pct(paired_change.std(ddof=1)):.3f}%"

)

print(

    "Seeds with positive increase:",

    f"{np.sum(paired_change > 0)}/{len(SEEDS)}"

)

print()

print("=" * 90)

print(

    "ONE-STEP VS ROLLOUT DEGRADATION"

)

print("=" * 90)

print()

for j, multiplier in enumerate(

    MULTIPLIERS

):

    one = delta_one[

        :,

        j

    ]

    roll = delta_roll[

        :,

        j

    ]

    one_mean = one.mean()

    roll_mean = roll.mean()

    amplification = (

        roll_mean

        / max(

            abs(one_mean),

            EPS

        )

    )

    print()

    print(

        f"Mass x{multiplier:.1f}"

    )

    print(

        "  Δ one-step :",

        f"{one_mean:+.8f}"

    )

    print(

        "  Δ rollout  :",

        f"{roll_mean:+.8f}"

    )

    print(

        "  rollout / one-step:",

        f"{amplification:.3f}x"

    )

print()

print("=" * 90)

print(

    "V6 BENCHMARK INTERPRETATION"

)

print("=" * 90)

print()

mean_x2 = relative_roll[

    :,

    1

].mean()

mean_x3 = relative_roll[

    :,

    2

].mean()

mean_x5 = relative_roll[

    :,

    3

].mean()

print(

    f"Mean x2 degradation: "

    f"{pct(mean_x2):+.3f}%"

)

print(

    f"Mean x3 degradation: "

    f"{pct(mean_x3):+.3f}%"

)

print(

    f"Mean x5 degradation: "

    f"{pct(mean_x5):+.3f}%"

)

print()

if (

    nondecreasing_count

    == len(SEEDS)

    and

    mean_x5 > mean_x3 > mean_x2

):

    print(

        "STATUS: CONSISTENT SHIFT-RELATIVE DEGRADATION"

    )

else:

    print(

        "STATUS: DEGRADATION IS NOT FULLY CONSISTENT"

    )

if (

    r_excess > 0.5

    and

    r_relative > 0.5

):

    print(

        "PHYSICS → ERROR RELATIONSHIP: CLEAR"

    )

else:

    print(

        "PHYSICS → ERROR RELATIONSHIP: WEAK / MIXED"

    )

np.savez(

    "embodiment_shift_v6_relative_results.npz",

    seeds=np.asarray(

        SEEDS,

        dtype=np.int32

    ),

    multipliers=np.asarray(

        MULTIPLIERS,

        dtype=np.float32

    ),

    physical_gap=physical_gap.astype(

        np.float32

    ),

    delta_one=delta_one.astype(

        np.float32

    ),

    delta_roll=delta_roll.astype(

        np.float32

    ),

    relative_roll=relative_roll.astype(

        np.float32

    ),

    roll_ratio=roll_ratio.astype(

        np.float32

    ),

    shift_fraction_roll=shift_fraction_roll.astype(

        np.float32

    ),

)

print()

print("=" * 90)

print(

    "SAVED"

)

print("=" * 90)

print(

    "embodiment_shift_v6_relative_results.npz"

)

print()

print(

    "DONE"

)

print("=" * 90)
