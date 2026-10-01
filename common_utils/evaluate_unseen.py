import numpy as np

import torch

import torch.nn as nn

data = np.load(

    "test_trajectory.npz",

    allow_pickle=True

)

observations = data["observations"]

actions = data["actions"]

states = np.stack([

    np.concatenate([

        obs["robot0_joint_pos"],

        obs["robot0_eef_pos"]

    ])

    for obs in observations

])

states_t = states[:-1]

actions_t = actions[:-1]

next_states = states[1:]

class DynamicsModel(nn.Module):

    def __init__(self):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(17, 128),

            nn.ReLU(),

            nn.Linear(128, 128),

            nn.ReLU(),

            nn.Linear(128, 10)

        )

    def forward(self, x):

        return self.network(x)

model = DynamicsModel()

model.load_state_dict(

    torch.load(

        "dynamics_model.pt",

        map_location="cpu",

        weights_only=True

    )

)

model.eval()

X = np.concatenate(

    [states_t, actions_t],

    axis=1

).astype(np.float32)

X_tensor = torch.from_numpy(X)

with torch.no_grad():

    predictions = model(

        X_tensor

    ).numpy()

absolute_error = np.abs(

    predictions - next_states

)

mae = np.mean(absolute_error)

max_error = np.max(

    absolute_error

)

per_state_mae = np.mean(

    absolute_error,

    axis=0

)

print("\n=== UNSEEN TRAJECTORY TEST ===")

print(

    f"Test transitions : {len(next_states)}"

)

print(

    f"Overall MAE       : {mae:.8f}"

)

print(

    f"Maximum error     : {max_error:.8f}"

)

print("\nMAE per state:")

names = [

    "Joint 1",

    "Joint 2",

    "Joint 3",

    "Joint 4",

    "Joint 5",

    "Joint 6",

    "Joint 7",

    "EEF X",

    "EEF Y",

    "EEF Z"

]

for name, error in zip(

    names,

    per_state_mae

):

    print(

        f"{name:8s}: {error:.8f}"

)
