import numpy as np

import torch

import torch.nn as nn

from torch.utils.data import TensorDataset, DataLoader

DATASET_FILE = "multi_trajectory_v3.npz"

OUTPUT_FILE = "dynamics_model_v3.pt"

BATCH_SIZE = 256

EPOCHS = 300

LEARNING_RATE = 0.001

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

data = np.load(DATASET_FILE)

states = data["states"]

actions = data["actions"]

next_states = data["next_states"]

print("=" * 60)

print("TRAINING DYNAMICS MODEL V3")

print("=" * 60)

print(f"States      : {states.shape}")

print(f"Actions     : {actions.shape}")

print(f"Next states : {next_states.shape}")

X = np.concatenate(

    [states, actions],

    axis=1

)

Y = next_states

print(f"\nInput shape  : {X.shape}")

print(f"Target shape : {Y.shape}")

X_tensor = torch.tensor(X, dtype=torch.float32)

Y_tensor = torch.tensor(Y, dtype=torch.float32)

dataset = TensorDataset(X_tensor, Y_tensor)

loader = DataLoader(

    dataset,

    batch_size=BATCH_SIZE,

    shuffle=True

)

model = DynamicsModel()

criterion = nn.MSELoss()

optimizer = torch.optim.Adam(

    model.parameters(),

    lr=LEARNING_RATE

)

print("\nStarting training...\n")

for epoch in range(EPOCHS):

    model.train()

    total_loss = 0.0

    for batch_X, batch_Y in loader:

        optimizer.zero_grad()

        prediction = model(batch_X)

        loss = criterion(

            prediction,

            batch_Y

        )

        loss.backward()

        optimizer.step()

        total_loss += loss.item() * len(batch_X)

    epoch_loss = total_loss / len(dataset)

    if (epoch + 1) % 25 == 0:

        print(

            f"Epoch {epoch + 1:03d}/{EPOCHS} "

            f"| Loss: {epoch_loss:.8f}"

        )

torch.save(

    model.state_dict(),

    OUTPUT_FILE

)

print("\n" + "=" * 60)

print("TRAINING COMPLETE")

print("=" * 60)

print(f"Final loss : {epoch_loss:.8f}")

print(f"Model saved: {OUTPUT_FILE}")

print("=" * 60)
