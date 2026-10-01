import numpy as np

DATA_FILE = "multi_trajectory_v4.npz"

J4_POS = 3

J6_POS = 5

J4_VEL = 10

J6_VEL = 12

J4_RANGE = (-2.55, -2.25)

J6_RANGE = (2.50, 2.85)

CORE_J4_RANGE = (-2.40, -2.25)

CORE_J6_RANGE = (2.50, 2.70)

TARGET = np.array([-2.30, 2.60], dtype=np.float32)

TRAJ_LEN = 400

data = np.load(DATA_FILE)

states = data["states"].astype(np.float32)

actions = data["actions"].astype(np.float32)

next_states = data["next_states"].astype(np.float32)

print("=" * 75)

print("V4 TRANSITION PATH ANALYSIS")

print("=" * 75)

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

region_mask = (

    (j4 >= J4_RANGE[0]) &

    (j4 <= J4_RANGE[1]) &

    (j6 >= J6_RANGE[0]) &

    (j6 <= J6_RANGE[1])

)

core_mask = (

    (j4 >= CORE_J4_RANGE[0]) &

    (j4 <= CORE_J4_RANGE[1]) &

    (j6 >= CORE_J6_RANGE[0]) &

    (j6 <= CORE_J6_RANGE[1])

)

print()

print("=" * 75)

print("1. REGION COVERAGE")

print("=" * 75)

print()

print("Broad region:")

print(f"J4 [{J4_RANGE[0]}, {J4_RANGE[1]}]")

print(f"J6 [{J6_RANGE[0]}, {J6_RANGE[1]}]")

print(f"Samples : {region_mask.sum()} / {len(states)}")

print(f"Percent : {region_mask.mean() * 100:.4f}%")

print()

print("Core region:")

print(f"J4 [{CORE_J4_RANGE[0]}, {CORE_J4_RANGE[1]}]")

print(f"J6 [{CORE_J6_RANGE[0]}, {CORE_J6_RANGE[1]}]")

print(f"Samples : {core_mask.sum()} / {len(states)}")

print(f"Percent : {core_mask.mean() * 100:.4f}%")

def print_stats(mask, title):

    print()

    print("=" * 75)

    print(title)

    print("=" * 75)

    count = int(mask.sum())

    if count == 0:

        print("NO SAMPLES")

        return

    region_states = states[mask]

    names = [

        "J4_Pos",

        "J4_Vel",

        "J6_Pos",

        "J6_Vel",

    ]

    dims = [

        J4_POS,

        J4_VEL,

        J6_POS,

        J6_VEL,

    ]

    print(f"Samples : {count}")

    for name, dim in zip(names, dims):

        values = region_states[:, dim]

        print()

        print(name)

        print(f"  min  : {values.min(): .6f}")

        print(f"  max  : {values.max(): .6f}")

        print(f"  mean : {values.mean(): .6f}")

        print(f"  std  : {values.std(): .6f}")

print_stats(

    region_mask,

    "2. BROAD REGION STATE DISTRIBUTION"

)

print_stats(

    core_mask,

    "3. CORE REGION STATE DISTRIBUTION"

)

print()

print("=" * 75)

print("4. ACTION COVERAGE INSIDE BROAD REGION")

print("=" * 75)

region_actions = actions[region_mask]

if len(region_actions) == 0:

    print("NO ACTION SAMPLES")

else:

    total = len(region_actions)

    print(f"Total region transitions : {total}")

    for i in range(7):

        active = np.abs(region_actions[:, i]) > 1e-8

        count = int(active.sum())

        if count == 0:

            print()

            print(f"Action {i + 1}: 0 samples")

            continue

        values = region_actions[active, i]

        positive = int((values > 0).sum())

        negative = int((values < 0).sum())

        print()

        print(f"Action {i + 1}")

        print(f"  Count      : {count}")

        print(f"  Percentage : {count / total * 100:.2f}%")

        print(f"  Positive   : {positive}")

        print(f"  Negative   : {negative}")

        print(f"  Min        : {values.min(): .6f}")

        print(f"  Max        : {values.max(): .6f}")

        print(f"  Mean       : {values.mean(): .6f}")

print()

print("=" * 75)

print("5. WHAT HAPPENS AFTER ENTERING THE REGION?")

print("=" * 75)

next_j4 = next_states[:, J4_POS]

next_j6 = next_states[:, J6_POS]

next_region_mask = (

    (next_j4 >= J4_RANGE[0]) &

    (next_j4 <= J4_RANGE[1]) &

    (next_j6 >= J6_RANGE[0]) &

    (next_j6 <= J6_RANGE[1])

)

entered_current = region_mask

next_after_entering = next_region_mask[entered_current]

total_entered = int(entered_current.sum())

print()

print(f"Current states inside region : {total_entered}")

if total_entered > 0:

    stay_count = int(next_after_entering.sum())

    exit_count = total_entered - stay_count

    print(

        f"Next state STAYS in region : "

        f"{stay_count} "

        f"({stay_count / total_entered * 100:.2f}%)"

    )

    print(

        f"Next state EXITS region     : "

        f"{exit_count} "

        f"({exit_count / total_entered * 100:.2f}%)"

    )

    current_region_next_states = next_states[entered_current]

    j4_next = current_region_next_states[:, J4_POS]

    j6_next = current_region_next_states[:, J6_POS]

    print()

    print("Next-state distribution:")

    print(f"  J4 next min  : {j4_next.min(): .6f}")

    print(f"  J4 next max  : {j4_next.max(): .6f}")

    print(f"  J4 next mean : {j4_next.mean(): .6f}")

    print(f"  J6 next min  : {j6_next.min(): .6f}")

    print(f"  J6 next max  : {j6_next.max(): .6f}")

    print(f"  J6 next mean : {j6_next.mean(): .6f}")

else:

    print("NO CURRENT STATES INSIDE REGION")

print()

print("=" * 75)

print("6. TRANSITION DIRECTION")

print("=" * 75)

if region_mask.sum() > 0:

    delta_j4 = (

        next_states[region_mask, J4_POS]

        - states[region_mask, J4_POS]

    )

    delta_j6 = (

        next_states[region_mask, J6_POS]

        - states[region_mask, J6_POS]

    )

    print()

    print("J4 Position delta:")

    print(f"  Mean : {delta_j4.mean(): .8f}")

    print(f"  Min  : {delta_j4.min(): .8f}")

    print(f"  Max  : {delta_j4.max(): .8f}")

    print()

    print("J6 Position delta:")

    print(f"  Mean : {delta_j6.mean(): .8f}")

    print(f"  Min  : {delta_j6.min(): .8f}")

    print(f"  Max  : {delta_j6.max(): .8f}")

    print()

    print("J4 direction:")

    print(f"  Positive : {(delta_j4 > 0).sum()}")

    print(f"  Negative : {(delta_j4 < 0).sum()}")

    print(f"  Zero     : {(np.abs(delta_j4) <= 1e-8).sum()}")

    print()

    print("J6 direction:")

    print(f"  Positive : {(delta_j6 > 0).sum()}")

    print(f"  Negative : {(delta_j6 < 0).sum()}")

    print(f"  Zero     : {(np.abs(delta_j6) <= 1e-8).sum()}")

print()

print("=" * 75)

print("7. EXAMPLES CLOSEST TO ROLLOUT DRIFT AREA")

print("=" * 75)

points = states[:, [J4_POS, J6_POS]]

distances = np.sqrt(

    (points[:, 0] - TARGET[0]) ** 2 +

    (points[:, 1] - TARGET[1]) ** 2

)

closest = np.argsort(distances)[:20]

print()

print(

    f"Target center: "

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

for rank, idx in enumerate(closest, 1):

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

print("=" * 75)

print("8. TRAJECTORY CONTINUITY")

print("=" * 75)

trajectory_ids = np.arange(

    len(states)

) // TRAJ_LEN

region_indices = np.where(

    region_mask

)[0]

if len(region_indices) == 0:

    print("NO REGION SAMPLES")

else:

    unique_traj = np.unique(

        trajectory_ids[region_indices]

    )

    print(

        f"Trajectories reaching region : "

        f"{len(unique_traj)} / 50"

    )

    print(

        f"Trajectory IDs : "

        f"{unique_traj.tolist()}"

    )

    print()

    for traj_id in unique_traj:

        idxs = region_indices[

            trajectory_ids[region_indices] == traj_id

        ]

        local_steps = (idxs % TRAJ_LEN) + 1

        runs = []

        start = local_steps[0]

        prev = local_steps[0]

        for step in local_steps[1:]:

            if step == prev + 1:

                prev = step

            else:

                runs.append((start, prev))

                start = step

                prev = step

        runs.append((start, prev))

        print(f"Trajectory {traj_id:2d}")

        print(f"  Samples : {len(idxs)}")

        print(

            f"  Range   : "

            f"{local_steps.min()} → "

            f"{local_steps.max()}"

        )

        print(f"  Runs    : {len(runs)}")

        for run_start, run_end in runs:

            if run_start == run_end:

                print(f"    step {run_start}")

            else:

                print(

                    f"    step {run_start} → {run_end} "

                    f"({run_end - run_start + 1} samples)"

                )

print()

print("=" * 75)

print("DONE")

print("=" * 75)
