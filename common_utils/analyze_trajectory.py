import numpy as np

data = np.load("trajectory_varied.npz", allow_pickle=True)

observations = data["observations"]

actions = data["actions"]

joint_pos = np.array([

    obs["robot0_joint_pos"]

    for obs in observations

])

eef_pos = np.array([

    obs["robot0_eef_pos"]

    for obs in observations

])

print("=== TRAJECTORY ANALYSIS ===")

print(f"\nTotal timestep : {len(observations)}")

print("\n--- Joint movement ---")

for i in range(7):

    start = joint_pos[0, i]

    end = joint_pos[-1, i]

    movement = np.max(joint_pos[:, i]) - np.min(joint_pos[:, i])

    print(

        f"Joint {i+1}: "

        f"start={start:.4f}, "

        f"end={end:.4f}, "

        f"range={movement:.4f}"

    )

print("\n--- EEF position ---")

print("Start:", eef_pos[0])

print("End  :", eef_pos[-1])

distance = np.linalg.norm(eef_pos[-1] - eef_pos[0])

print(f"\nEEF displacement: {distance:.4f}")

print("\n--- Action phases ---")

for step in [0, 199, 200, 399, 400, 599, 600, 799]:

    print(f"Step {step}: {actions[step]}")
