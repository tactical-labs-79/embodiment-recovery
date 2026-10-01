import numpy as np

import torch

import torch.nn as nn

data = np.load("robot_dataset.npz")

states = data["states"].astype(np.float32)

actions = data["actions"].astype(np.float32)

next_states = data["next_states"].astype(np.float32)

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

indices = [0, 200, 400, 600, 798]

print("=== DYNAMICS MODEL TEST ===")

with torch.no_grad():

    for index in indices:

        state = states[index]

        action = actions[index]

        input_data = np.concatenate(

            [state, action]

        ).astype(np.float32)

        input_tensor = torch.from_numpy(

            input_data

        ).unsqueeze(0)

        prediction = model(input_tensor)

        predicted_next_state = prediction.numpy()[0]

        actual_next_state = next_states[index]

        error = np.mean(

            np.abs(

                predicted_next_state

                - actual_next_state

            )

        )

        print(f"\n--- Sample {index} ---")

        print("Actual next state:")

        print(actual_next_state)

        print("Predicted next state:")

        print(predicted_next_state)

        print(f"Mean absolute error: {error:.8f}")
