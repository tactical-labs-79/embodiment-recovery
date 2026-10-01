import numpy as np

data = np.load("multi_trajectory.npz")

states = data["states"]

actions = data["actions"]

next_states = data["next_states"]

print("=== MULTI DATASET CHECK ===")

print("\nStates:")

print("Shape:", states.shape)

print("\nActions:")

print("Shape:", actions.shape)

print("\nNext states:")

print("Shape:", next_states.shape)

print("\nData type:")

print("States:", states.dtype)

print("Actions:", actions.dtype)

print("Next states:", next_states.dtype)

print("\nFirst transition:")

print("\nState:")

print(states[0])

print("\nAction:")

print(actions[0])

print("\nNext state:")

print(next_states[0])

print("\nState range:")

print("Min:", states.min())

print("Max:", states.max())

print("\nAction range:")

print("Min:", actions.min())

print("Max:", actions.max())

print("\nCheck NaN:")

print("States:", np.isnan(states).any())

print("Actions:", np.isnan(actions).any())

print("Next states:", np.isnan(next_states).any())
