import numpy as np

DATASET_FILE = "multi_trajectory_v3.npz"

TARGET_JOINT_5 = 1.22919035

data = np.load(DATASET_FILE)

states = data["states"]

actions = data["actions"]

next_states = data["next_states"]

print("=" * 60)

print("V3 STATE COVERAGE CHECK")

print("=" * 60)

print(f"States      : {states.shape}")

print(f"Actions     : {actions.shape}")

print(f"Next states : {next_states.shape}")

joint5 = states[:, 4]

print("\n" + "=" * 60)

print("JOINT 5 COVERAGE")

print("=" * 60)

print(f"Target Joint 5 : {TARGET_JOINT_5:.8f}")

print(f"Minimum        : {joint5.min():.8f}")

print(f"Maximum        : {joint5.max():.8f}")

print(f"Mean           : {joint5.mean():.8f}")

print(f"Std            : {joint5.std():.8f}")

distance = np.abs(joint5 - TARGET_JOINT_5)

closest_idx = np.argmin(distance)

print("\nClosest training state:")

print(f"Index          : {closest_idx}")

print(f"Joint 5        : {joint5[closest_idx]:.8f}")

print(f"Distance       : {distance[closest_idx]:.8f}")

print("\n" + "=" * 60)

print("COVERAGE AROUND TARGET")

print("=" * 60)

windows = [

    0.02,

    0.05,

    0.10,

    0.20,

    0.30,

]

for window in windows:

    mask = distance <= window

    count = np.count_nonzero(mask)

    percentage = (

        count / len(joint5) * 100

    )

    print(

        f"±{window:.2f} : "

        f"{count:5d} samples "

        f"({percentage:8.4f}%)"

    )

print("\n" + "=" * 60)

print("10 CLOSEST STATES TO TARGET")

print("=" * 60)

closest_indices = np.argsort(distance)[:10]

for rank, idx in enumerate(closest_indices, start=1):

    print(

        f"\n#{rank}"

    )

    print(

        f"Index    : {idx}"

    )

    print(

        f"Distance : {distance[idx]:.8f}"

    )

    print(

        f"Joint 5  : {joint5[idx]:.8f}"

    )

    print(

        "State    :",

        states[idx]

    )

    print(

        "Action   :",

        actions[idx]

    )

    print(

        "Next     :",

        next_states[idx]

    )

print("\n" + "=" * 60)

print("FULL JOINT COVERAGE")

print("=" * 60)

for i in range(7):

    joint = states[:, i]

    print(

        f"Joint {i + 1}: "

        f"min={joint.min(): .6f}  "

        f"max={joint.max(): .6f}  "

        f"mean={joint.mean(): .6f}  "

        f"std={joint.std(): .6f}"

    )

print("\n" + "=" * 60)

print("VALIDATION")

print("=" * 60)

print(

    f"NaN states      : {np.isnan(states).any()}"

)

print(

    f"NaN actions     : {np.isnan(actions).any()}"

)

print(

    f"NaN next_states : {np.isnan(next_states).any()}"

)

print("\n" + "=" * 60)

print("CHECK COMPLETE")

print("=" * 60)
