import numpy as np

import matplotlib.pyplot as plt

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

plt.figure(figsize=(10, 5))

for i in range(7):

    plt.plot(joint_pos[:, i], label=f"Joint {i+1}")

plt.xlabel("Timestep")

plt.ylabel("Joint Position")

plt.title("Panda Joint Positions")

plt.legend()

plt.grid()

plt.show()

plt.figure(figsize=(10, 5))

plt.plot(eef_pos[:, 0], label="X")

plt.plot(eef_pos[:, 1], label="Y")

plt.plot(eef_pos[:, 2], label="Z")

plt.xlabel("Timestep")

plt.ylabel("Position")

plt.title("Panda End-Effector Position")

plt.legend()

plt.grid()

plt.show()

plt.figure(figsize=(10, 5))

for i in range(7):

    plt.plot(actions[:, i], label=f"Action {i+1}")

plt.xlabel("Timestep")

plt.ylabel("Action")

plt.title("Actions Given to Panda")

plt.legend()

plt.grid()

plt.show()
