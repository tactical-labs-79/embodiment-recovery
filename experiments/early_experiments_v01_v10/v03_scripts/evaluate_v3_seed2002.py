import numpy as np

import torch

import torch.nn as nn

MODEL_FILE = "dynamics_model_v3.pt"

TEST_FILE = "random_test_2002.npz"

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

        MODEL_FILE,

        map_location="cpu"

    )

)

model.eval()

data = np.load(TEST_FILE)

states = data["states"]

actions = data["actions"]

next_states = data["next_states"]

assert states.ndim == 2

assert actions.ndim == 2

assert next_states.ndim == 2

assert states.shape[1] == 10

assert actions.shape[1] == 7

assert next_states.shape[1] == 10

assert states.shape[0] == actions.shape[0]

assert states.shape[0] == next_states.shape[0]

X = np.concatenate(

    [states, actions],

    axis=1

)

assert X.shape[1] == 17

X_tensor = torch.tensor(

    X,

    dtype=torch.float32

)

with torch.no_grad():

    predictions = model(X_tensor).numpy()

assert predictions.shape == next_states.shape

assert predictions.shape[1] == 10

errors = np.abs(

    predictions - next_states

)

overall_mae = errors.mean()

max_error = errors.max()

mae_per_state = errors.mean(axis=0)

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

print("=" * 60)

print("DYNAMICS V3 — SEED 2002")

print("=" * 60)

print(f"States      : {states.shape}")

print(f"Actions     : {actions.shape}")

print(f"Next states : {next_states.shape}")

print(f"Model input : {X.shape}")

print(f"Prediction  : {predictions.shape}")

print("\nDimension check: PASSED")

print("\n" + "=" * 60)

print("RESULT")

print("=" * 60)

print(f"Overall MAE   : {overall_mae:.8f}")

print(f"Maximum error : {max_error:.8f}")

print("\nMAE per state:")

for name, value in zip(names, mae_per_state):

    print(f"{name:<8}: {value:.8f}")

worst_idx = np.unravel_index(

    np.argmax(errors),

    errors.shape

)

timestep = worst_idx[0]

state_idx = worst_idx[1]

print("\n" + "=" * 60)

print("WORST PREDICTION")

print("=" * 60)

print(f"Timestep  : {timestep}")

print(f"State     : {names[state_idx]}")

print(f"Error     : {errors[timestep, state_idx]:.8f}")

print(f"Actual    : {next_states[timestep, state_idx]:.8f}")

print(f"Predicted : {predictions[timestep, state_idx]:.8f}")

print("\n" + "=" * 60)

print("TEST COMPLETE")

print("=" * 60)
