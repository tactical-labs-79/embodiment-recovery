import numpy as np

import torch

import torch.nn as nn

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

X = np.concatenate(

    [states_t, actions_t],

    axis=1

).astype(np.float32)

X_tensor = torch.from_numpy(X)

def evaluate(model_file):

    model = DynamicsModel()

    model.load_state_dict(

        torch.load(

            model_file,

            map_location="cpu",

            weights_only=True

        )

    )

    model.eval()

    with torch.no_grad():

        prediction = model(

            X_tensor

        ).numpy()

    error = np.abs(

        prediction - next_states

    )

    mae = np.mean(error)

    max_error = np.max(error)

    return mae, max_error

v1_mae, v1_max = evaluate(

    "dynamics_model.pt"

)

v2_mae, v2_max = evaluate(

    "dynamics_model_v2.pt"

)

print("\n==============================")

print("MODEL COMPARISON")

print("==============================")

print("\nV1")

print(f"MAE        : {v1_mae:.8f}")

print(f"Max error  : {v1_max:.8f}")

print("\nV2")

print(f"MAE        : {v2_mae:.8f}")

print(f"Max error  : {v2_max:.8f}")
