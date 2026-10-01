import numpy as np

DATA_FILE = "targeted_dataset_v5.npz"

J4_POS = 3

J6_POS = 5

J4_VEL = 10

J6_VEL = 12

BROAD_J4 = (-2.55, -2.25)

BROAD_J6 = (2.50, 2.85)

CORE_J4 = (-2.40, -2.25)

CORE_J6 = (2.50, 2.70)

TARGET = np.array([-2.30, 2.60], dtype=np.float32)

TRAJ_LEN = 400

data = np.load(DATA_FILE)

states = data["states"].astype(np.float32)

actions = data["actions"].astype(np.float32)

next_states = data["next_states"].astype(np.float32)

print("=" * 80)

print("V5 TARGETED DATASET ANALYSIS")

print("=" * 80)

print(f"States      : {states.shape}")

print(f"Actions     : {actions.shape}")

print(f"Next states : {next_states.shape}")

assert states.shape[0] == actions.shape[0]

assert states.shape[0] == next_states.shape[0]

assert states.shape[1] == 17

assert actions.shape[1] == 7

assert next_states.shape[1] == 17

j4 = states[:, J4_POS]

j6 = states[:, J6_POS]

broad_mask = (

    (j4 >= BROAD_J4[0]) &

    (j4 <= BROAD_J4[1]) &

    (j6 >= BROAD_J6[0]) &

    (j6 <= BROAD_J6[1])

)

core_mask = (

    (j4 >= CORE_J4[0]) &

    (j4 <= CORE_J4[1]) &

    (j6 >= CORE_J6[0]) &

    (j6 <= CORE_J6[1])

)

print()

print("=" * 80)

print("1. REGION COVERAGE")

print("=" * 80)

print()

print("BROAD REGION")

print(

    f"J4 [{BROAD_J4[0]}, {BROAD_J4[1]}]"

)

print(

    f"J6 [{BROAD_J6[0]}, {BROAD_J6[1]}]"

)

print(

    f"Samples : {broad_mask.sum()} / {len(states)}"

)

print(

    f"Percent : {broad_mask.mean() * 100:.4f}%"

)

print()

print("CORE REGION")

print(

    f"J4 [{CORE_J4[0]}, {CORE_J4[1]}]"

)

print(

    f"J6 [{CORE_J6[0]}, {CORE_J6[1]}]"

)

print(

    f"Samples : {core_mask.sum()} / {len(states)}"

)

print(

    f"Percent : {core_mask.mean() * 100:.4f}%"

)

print()

print("=" * 80)

print("2. CORE STATE DISTRIBUTION")

print("=" * 80)

if core_mask.sum() == 0:

    print("NO CORE SAMPLES")

else:

    core_states = states[core_mask]

    dimensions = [

        ("J4_Pos", J4_POS),

        ("J4_Vel", J4_VEL),

        ("J6_Pos", J6_POS),

        ("J6_Vel", J6_VEL),

    ]

    print(

        f"Samples : {len(core_states)}"

    )

    for name, dim in dimensions:

        values = core_states[:, dim]

        print()

        print(name)

        print(

            f"  min  : {values.min(): .6f}"

        )

        print(

            f"  max  : {values.max(): .6f}"

        )

        print(

            f"  mean : {values.mean(): .6f}"

        )

        print(

            f"  std  : {values.std(): .6f}"

        )

print()

print("=" * 80)

print("3. CORE ACTION COVERAGE")

print("=" * 80)

core_actions = actions[core_mask]

if len(core_actions) == 0:

    print("NO CORE ACTION SAMPLES")

else:

    total = len(core_actions)

    print(

        f"Total core transitions : {total}"

    )

    for i in range(7):

        active = (

            np.abs(core_actions[:, i]) > 1e-8

        )

        count = int(active.sum())

        if count == 0:

            print()

            print(

                f"Action {i + 1}: 0 samples"

            )

            continue

        values = core_actions[active, i]

        positive = int(

            (values > 0).sum()

        )

        negative = int(

            (values < 0).sum()

        )

        print()

        print(f"Action {i + 1}")

        print(

            f"  Count      : {count}"

        )

        print(

            f"  Percentage : "

            f"{count / total * 100:.2f}%"

        )

        print(

            f"  Positive   : {positive}"

        )

        print(

            f"  Negative   : {negative}"

        )

        print(

            f"  Min        : "

            f"{values.min(): .6f}"

        )

        print(

            f"  Max        : "

            f"{values.max(): .6f}"

        )

        print(

            f"  Mean       : "

            f"{values.mean(): .6f}"

        )

print()

print("=" * 80)

print("4. WHAT HAPPENS AFTER ENTERING CORE?")

print("=" * 80)

next_j4 = next_states[:, J4_POS]

next_j6 = next_states[:, J6_POS]

next_core_mask = (

    (next_j4 >= CORE_J4[0]) &

    (next_j4 <= CORE_J4[1]) &

    (next_j6 >= CORE_J6[0]) &

    (next_j6 <= CORE_J6[1])

)

current_core = core_mask

next_after_entering = next_core_mask[

    current_core

]

total_core = int(

    current_core.sum()

)

print(

    f"Current states inside core : "

    f"{total_core}"

)

if total_core > 0:

    stay = int(

        next_after_entering.sum()

    )

    exit_count = (

        total_core - stay

    )

    print(

        f"Next state STAYS in core : "

        f"{stay} "

        f"({stay / total_core * 100:.2f}%)"

    )

    print(

        f"Next state EXITS core     : "

        f"{exit_count} "

        f"({exit_count / total_core * 100:.2f}%)"

    )

    core_next_states = next_states[

        current_core

    ]

    print()

    print("Next-state distribution:")

    print(

        f"  J4 next min  : "

        f"{core_next_states[:, J4_POS].min(): .6f}"

    )

    print(

        f"  J4 next max  : "

        f"{core_next_states[:, J4_POS].max(): .6f}"

    )

    print(

        f"  J4 next mean : "

        f"{core_next_states[:, J4_POS].mean(): .6f}"

    )

    print(

        f"  J6 next min  : "

        f"{core_next_states[:, J6_POS].min(): .6f}"

    )

    print(

        f"  J6 next max  : "

        f"{core_next_states[:, J6_POS].max(): .6f}"

    )

    print(

        f"  J6 next mean : "

        f"{core_next_states[:, J6_POS].mean(): .6f}"

    )

else:

    print("NO CORE STATES")

print()

print("=" * 80)

print("5. CORE TRANSITION DIRECTION")

print("=" * 80)

if core_mask.sum() > 0:

    delta_j4 = (

        next_states[core_mask, J4_POS]

        - states[core_mask, J4_POS]

    )

    delta_j6 = (

        next_states[core_mask, J6_POS]

        - states[core_mask, J6_POS]

    )

    print()

    print("J4 Position delta:")

    print(

        f"  Mean : {delta_j4.mean(): .8f}"

    )

    print(

        f"  Min  : {delta_j4.min(): .8f}"

    )

    print(

        f"  Max  : {delta_j4.max(): .8f}"

    )

    print()

    print("J6 Position delta:")

    print(

        f"  Mean : {delta_j6.mean(): .8f}"

    )

    print(

        f"  Min  : {delta_j6.min(): .8f}"

    )

    print(

        f"  Max  : {delta_j6.max(): .8f}"

    )

    print()

    print("J4 direction:")

    print(

        f"  Positive : "

        f"{(delta_j4 > 0).sum()}"

    )

    print(

        f"  Negative : "

        f"{(delta_j4 < 0).sum()}"

    )

    print()

    print("J6 direction:")

    print(

        f"  Positive : "

        f"{(delta_j6 > 0).sum()}"

    )

    print(

        f"  Negative : "

        f"{(delta_j6 < 0).sum()}"

    )

print()

print("=" * 80)

print("6. CLOSEST STATES TO ROLLOUT DRIFT")

print("=" * 80)

points = states[:, [J4_POS, J6_POS]]

distances = np.sqrt(

    (points[:, 0] - TARGET[0]) ** 2 +

    (points[:, 1] - TARGET[1]) ** 2

)

closest = np.argsort(distances)[:20]

print()

print(

    f"Target : "

    f"J4={TARGET[0]:.3f}, "

    f"J6={TARGET[1]:.3f}"

)

print()

print(

    f"{'Rank':>5} "

    f"{'Index':>8} "

    f"{'Traj':>6} "

    f"{'Step':>6} "

    f"{'J4':>11} "

    f"{'J6':>11} "

    f"{'J4Vel':>11} "

    f"{'J6Vel':>11} "

    f"{'Action':>10} "

    f"{'Dist':>10}"

)

for rank, idx in enumerate(

    closest,

    1

):

    action_idx = np.argmax(

        np.abs(actions[idx])

    )

    action_value = actions[

        idx,

        action_idx

    ]

    traj_id = idx // TRAJ_LEN

    step_id = idx % TRAJ_LEN + 1

    print(

        f"{rank:5d} "

        f"{idx:8d} "

        f"{traj_id:6d} "

        f"{step_id:6d} "

        f"{states[idx, J4_POS]:11.6f} "

        f"{states[idx, J6_POS]:11.6f} "

        f"{states[idx, J4_VEL]:11.6f} "

        f"{states[idx, J6_VEL]:11.6f} "

        f"A{action_idx + 1}:{action_value:+.3f} "

        f"{distances[idx]:10.6f}"

    )

print()

print("=" * 80)

print("7. CORE TRAJECTORY CONTINUITY")

print("=" * 80)

trajectory_ids = (

    np.arange(len(states))

    // TRAJ_LEN

)

core_indices = np.where(

    core_mask

)[0]

if len(core_indices) == 0:

    print("NO CORE SAMPLES")

else:

    unique_traj = np.unique(

        trajectory_ids[core_indices]

    )

    print(

        f"Trajectories reaching core : "

        f"{len(unique_traj)} / 50"

    )

    print(

        f"Trajectory IDs : "

        f"{unique_traj.tolist()}"

    )

    print()

    for traj_id in unique_traj:

        idxs = core_indices[

            trajectory_ids[core_indices]

            == traj_id

        ]

        local_steps = (

            idxs % TRAJ_LEN

        ) + 1

        runs = []

        start = local_steps[0]

        prev = local_steps[0]

        for step in local_steps[1:]:

            if step == prev + 1:

                prev = step

            else:

                runs.append(

                    (start, prev)

                )

                start = step

                prev = step

        runs.append(

            (start, prev)

        )

        print(

            f"Trajectory {traj_id:2d}"

        )

        print(

            f"  Samples : {len(idxs)}"

        )

        print(

            f"  Range   : "

            f"{local_steps.min()} "

            f"→ {local_steps.max()}"

        )

        print(

            f"  Runs    : {len(runs)}"

        )

        for start_step, end_step in runs:

            if start_step == end_step:

                print(

                    f"    step {start_step}"

                )

            else:

                print(

                    f"    step {start_step}"

                    f" → {end_step} "

                    f"({end_step - start_step + 1} samples)"

                )

print()

print("=" * 80)

print("8. TARGET PROXIMITY")

print("=" * 80)

print(

    f"Distance to target center "

    f"J4={TARGET[0]:.3f}, "

    f"J6={TARGET[1]:.3f}"

)

print(

    f"  Mean : {distances.mean():.6f}"

)

print(

    f"  Min  : {distances.min():.6f}"

)

print(

    f"  Max  : {distances.max():.6f}"

)

for threshold in [

    0.20,

    0.10,

    0.05,

    0.02,

]:

    count = int(

        (distances <= threshold).sum()

    )

    print(

        f"  <= {threshold:.2f} : "

        f"{count} / {len(states)} "

        f"({count / len(states) * 100:.2f}%)"

    )

print()

print("=" * 80)

print("DONE")

print("=" * 80)
