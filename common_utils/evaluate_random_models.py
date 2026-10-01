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

data = np.load("random_test.npz")

states = data["states"]

actions = data["actions"]

X = np.concatenate(

    [states[:-1], actions[:-1]],

    axis=1

)

y = states[1:]

X = torch.tensor(X, dtype=torch.float32)

def evaluate(model_path):

    model = DynamicsModel()

    checkpoint = torch.load(

        model_path,

        map_location="cpu",

        weights_only=True

    )

    model.load_state_dict(checkpoint)

    model.eval()

    with torch.no_grad():

        prediction = model(X).numpy()

    error = np.abs(prediction - y)

    mae = error.mean()

    max_error = error.max()

    mae_per_state = error.mean(axis=0)

    return mae, max_error, mae_per_state

v1_mae, v1_max, v1_state = evaluate(

    "dynamics_model.pt"

)

v2_mae, v2_max, v2_state = evaluate(

    "dynamics_model_v2.pt"

)

print("\n================================")

print("FAIR RANDOM TEST")

print("================================")

print("\nV1")

print(f"MAE        : {v1_mae:.8f}")

print(f"Max error  : {v1_max:.8f}")

print("\nV2")

print(f"MAE        : {v2_mae:.8f}")

print(f"Max error  : {v2_max:.8f}")

print("\n================================")

print("MAE PER STATE")

print("================================")

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

    "EEF Z",

]

for i, name in enumerate(names):

    print(

        f"{name:8s} | "

        f"V1: {v1_state[i]:.8f} | "

        f"V2: {v2_state[i]:.8f}"

    )
