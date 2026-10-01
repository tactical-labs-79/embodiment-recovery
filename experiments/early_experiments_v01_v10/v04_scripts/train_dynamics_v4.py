import numpy as np

import torch

import torch.nn as nn

from torch.utils.data import TensorDataset, DataLoader

DATA_FILE = "multi_trajectory_v4.npz"

MODEL_FILE = "dynamics_model_v4.pt"

EPOCHS = 300

BATCH_SIZE = 256

LEARNING_RATE = 0.001

data = np.load(DATA_FILE)

states = data["states"].astype(np.float32)

actions = data["actions"].astype(np.float32)

next_states = data["next_states"].astype(np.float32)

print("=" * 60)

print("TRAINING DYNAMICS MODEL V4")

print("=" * 60)

print(f"States      : {states.shape}")

print(f"Actions     : {actions.shape}")

print(f"Next states : {next_states.shape}")

assert states.ndim == 2

assert actions.ndim == 2

assert next_states.ndim == 2

assert states.shape[1] == 17

assert actions.shape[1] == 7

assert next_states.shape[1] == 17

assert states.shape[0] == actions.shape[0]

assert states.shape[0] == next_states.shape[0]

assert not np.isnan(states).any()

assert not np.isnan(actions).any()

assert not np.isnan(next_states).any()

X = np.concatenate(

    [states, actions],

    axis=1

)

y = next_states

assert X.shape == (states.shape[0], 24)

assert y.shape == (states.shape[0], 17)

print(f"Input X    : {X.shape}")

print(f"Target y   : {y.shape}")

X_tensor = torch.tensor(X, dtype=torch.float32)

y_tensor = torch.tensor(y, dtype=torch.float32)

dataset = TensorDataset(X_tensor, y_tensor)

loader = DataLoader(

    dataset,

    batch_size=BATCH_SIZE,

    shuffle=True

)

class DynamicsModelV4(nn.Module):

    def __init__(self):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(24, 128),

            nn.ReLU(),

            nn.Linear(128, 128),

            nn.ReLU(),

            nn.Linear(128, 17)

        )

    def forward(self, x):

        return self.network(x)

model = DynamicsModelV4()

loss_fn = nn.MSELoss()

optimizer = torch.optim.Adam(

    model.parameters(),

    lr=LEARNING_RATE

)

print("\nMulai training...\n")

for epoch in range(1, EPOCHS + 1):

    model.train()

    total_loss = 0.0

    total_samples = 0

    for batch_X, batch_y in loader:

        optimizer.zero_grad()

        predictions = model(batch_X)

        loss = loss_fn(predictions, batch_y)

        loss.backward()

        optimizer.step()

        batch_size = batch_X.size(0)

        total_loss += loss.item() * batch_size

        total_samples += batch_size

    epoch_loss = total_loss / total_samples

    if epoch % 25 == 0 or epoch == 1:

        print(

            f"Epoch {epoch:03d}/{EPOCHS} "

            f"| Loss: {epoch_loss:.8f}"

        )

torch.save(

    {

        "model_state_dict": model.state_dict(),

        "input_dim": 24,

        "output_dim": 17,

        "state_dim": 17,

        "action_dim": 7,

    },

    MODEL_FILE

)

print("\n" + "=" * 60)

print("TRAINING V4 SELESAI")

print("=" * 60)

print(f"Model disimpan: {MODEL_FILE}")

print(f"Final loss     : {epoch_loss:.8f}")

print("=" * 60)
