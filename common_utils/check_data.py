import numpy as np

data = np.load("trajectory.npz", allow_pickle=True)

observations = data["observations"]

actions = data["actions"]

print("Jumlah timestep:", len(observations))

print("Jumlah action:", len(actions))

print("\nObservation pertama:")

print(observations[0])

print("\nObservation terakhir:")

print(observations[-1])
