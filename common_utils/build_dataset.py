import numpy as np

INPUT_FILE = "trajectory_varied.npz"

OUTPUT_FILE = "robot_dataset.npz"

data = np.load(INPUT_FILE, allow_pickle=True)

observations = data["observations"]

actions = data["actions"]

joint_positions = np.array([

    obs["robot0_joint_pos"]

    for obs in observations

], dtype=np.float32)

eef_positions = np.array([

    obs["robot0_eef_pos"]

    for obs in observations

], dtype=np.float32)

states = np.concatenate(

    [joint_positions, eef_positions],

    axis=1

)

states_t = states[:-1]

actions_t = np.asarray(actions[:-1], dtype=np.float32)

next_states_t = states[1:]

np.savez_compressed(

    OUTPUT_FILE,

    states=states_t,

    actions=actions_t,

    next_states=next_states_t,

)

print("=== DATASET BUILT ===")

print(f"Input file      : {INPUT_FILE}")

print(f"Output file     : {OUTPUT_FILE}")

print(f"Samples         : {len(states_t)}")

print(f"State dimension : {states_t.shape[1]}")

print(f"Action dimension: {actions_t.shape[1]}")

print(f"Next-state dim  : {next_states_t.shape[1]}")

print("\nContoh sample pertama:")

print("State      :", states_t[0])

print("Action     :", actions_t[0])

print("Next state :", next_states_t[0])
