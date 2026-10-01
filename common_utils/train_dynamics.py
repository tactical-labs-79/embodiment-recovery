import numpy as np

import torch

import torch.nn as nn

from torch.utils.data import TensorDataset, DataLoader

data = np.load("robot_dataset.npz")

states = data["states"].astype(np.float32)

actions = data["actions"].astype(np.float32)

next_states = data["next_states"].astype(np.float32)

X = np.concatenate([states, actions], axis=1)

Y = next_states

print("=== DATA LOADED ===")

print("Input shape :", X.shape)

print("Target shape:", Y.shape)

X_tensor = torch.from_numpy(X)

Y_tensor = torch.from_numpy(Y)

dataset = TensorDataset(X_tensor, Y_tensor)

loader = DataLoader(

    dataset,

    batch_size=64,

    shuffle=True

)

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

loss_function = nn.MSELoss()

optimizer = torch.optim.Adam(

    model.parameters(),

    lr=0.001

)

epochs = 300

for epoch in range(epochs):

    total_loss = 0.0

    for batch_X, batch_Y in loader:

        prediction = model(batch_X)

        loss = loss_function(

            prediction,

            batch_Y

        )

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    average_loss = total_loss / len(loader)

    if (epoch + 1) % 25 == 0:

        print(

            f"Epoch {epoch + 1:03d}/{epochs} "

            f"| Loss: {average_loss:.8f}"

        )

torch.save(

    model.state_dict(),

    "dynamics_model.pt"

)

print("\nModel berhasil disimpan sebagai dynamics_model.pt")
