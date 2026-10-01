import numpy as np

data = np.load("multi_trajectory.npz")

states = data["states"]

actions = data["actions"]

target = 1.22919035

ranges = [

    0.02,

    0.05,

    0.10,

    0.20,

    0.30,

]

print("\n========================================")

print("STATE COVERAGE ANALYSIS")

print("========================================")

print(f"Training states : {len(states)}")

print(f"Target Joint 5  : {target:.8f}")

joint5 = states[:, 4]

print("\n========================================")

print("JOINT 5 RANGE")

print("========================================")

print(f"Minimum : {joint5.min():.8f}")

print(f"Maximum : {joint5.max():.8f}")

print(f"Mean    : {joint5.mean():.8f}")

print(f"Std     : {joint5.std():.8f}")

distance = np.abs(joint5 - target)

print("\n========================================")

print("TRAINING COVERAGE AROUND TARGET")

print("========================================")

for radius in ranges:

    count = np.sum(distance <= radius)

    percentage = (

        count / len(joint5) * 100

    )

    print(

        f"±{radius:.2f} : "

        f"{count:5d} samples "

        f"({percentage:.4f}%)"

    )

closest = np.argsort(distance)[:20]

print("\n========================================")

print("20 CLOSEST TRAINING STATES")

print("========================================")

for rank, idx in enumerate(closest, start=1):

    print(

        f"#{rank:02d} | "

        f"Index: {idx:5d} | "

        f"Joint 5: {joint5[idx]:.8f} | "

        f"Distance: {distance[idx]:.8f}"

    )

best_idx = closest[0]

print("\n========================================")

print("CLOSEST STATE")

print("========================================")

print(f"Index: {best_idx}")

print("\nState:")

print(

    np.array2string(

        states[best_idx],

        precision=6,

        suppress_small=True

    )

)

print("\nAction:")

print(

    np.array2string(

        actions[best_idx],

        precision=6,

        suppress_small=True

    )

)
