import numpy as np

import torch

import torch.nn as nn

from torch.utils.data import TensorDataset, DataLoader

BASE_DATA = "multi_trajectory_v4.npz"

TARGETED_DATA = "targeted_dataset_v5.npz"

OUTPUT_DATA = "combined_dataset_v6.npz"

OUTPUT_MODEL = "dynamics_model_v6.pt"

INPUT_DIM = 24

OUTPUT_DIM = 17

HIDDEN = 128

EPOCHS = 300

BATCH_SIZE = 256

LR = 0.001

SEED = 6006

np.random.seed(SEED)

torch.manual_seed(SEED)

print("=" * 80)

print("V6 DATASET BUILD + TRAIN")

print("=" * 80)

base = np.load(BASE_DATA)

base_states = base["states"].astype(np.float32)

base_actions = base["actions"].astype(np.float32)

base_next_states = base["next_states"].astype(np.float32)

targeted = np.load(TARGETED_DATA)

target_states = targeted["states"].astype(np.float32)

target_actions = targeted["actions"].astype(np.float32)

target_next_states = targeted["next_states"].astype(np.float32)

assert base_states.shape[1] == 17

assert base_actions.shape[1] == 7

assert base_next_states.shape[1] == 17

assert target_states.shape[1] == 17

assert target_actions.shape[1] == 7

assert target_next_states.shape[1] == 17

states = np.concatenate(

    [base_states, target_states],

    axis=0

)

actions = np.concatenate(

    [base_actions, target_actions],

    axis=0

)

next_states = np.concatenate(

    [base_next_states, target_next_states],

    axis=0

)

rng = np.random.default_rng(SEED)

indices = rng.permutation(

    len(states)

)

states = states[indices]

actions = actions[indices]

next_states = next_states[indices]

np.savez(

    OUTPUT_DATA,

    states=states,

    actions=actions,

    next_states=next_states

)

print()

print("=" * 80)

print("COMBINED DATASET")

print("=" * 80)

print(

    f"V4 samples      : {len(base_states)}"

)

print(

    f"V5 samples      : {len(target_states)}"

)

print(

    f"Combined samples: {len(states)}"

)

print(

    f"State shape     : {states.shape}"

)

print(

    f"Action shape    : {actions.shape}"

)

print(

    f"Next shape      : {next_states.shape}"

)

print(

    f"Saved           : {OUTPUT_DATA}"

)

X = np.concatenate(

    [states, actions],

    axis=1

)

Y = next_states

assert X.shape[1] == INPUT_DIM

assert Y.shape[1] == OUTPUT_DIM

X_tensor = torch.from_numpy(X)

Y_tensor = torch.from_numpy(Y)

dataset = TensorDataset(

    X_tensor,

    Y_tensor

)

loader = DataLoader(

    dataset,

    batch_size=BATCH_SIZE,

    shuffle=True

)

class DynamicsModel(nn.Module):

    def __init__(self):

        super().__init__()

        self.net = nn.Sequential(

            nn.Linear(INPUT_DIM, HIDDEN),

            nn.ReLU(),

            nn.Linear(HIDDEN, HIDDEN),

            nn.ReLU(),

            nn.Linear(HIDDEN, OUTPUT_DIM)

        )

    def forward(self, x):

        return self.net(x)

device = torch.device(

    "cuda" if torch.cuda.is_available()

    else "cpu"

)

print()

print(

    f"Device          : {device}"

)

model = DynamicsModel().to(device)

optimizer = torch.optim.Adam(

    model.parameters(),

    lr=LR

)

criterion = nn.MSELoss()

print()

print("=" * 80)

print("TRAINING V6")

print("=" * 80)

for epoch in range(1, EPOCHS + 1):

    model.train()

    total_loss = 0.0

    for batch_x, batch_y in loader:

        batch_x = batch_x.to(device)

        batch_y = batch_y.to(device)

        optimizer.zero_grad()

        prediction = model(batch_x)

        loss = criterion(

            prediction,

            batch_y

        )

        loss.backward()

        optimizer.step()

        total_loss += (

            loss.item()

            * batch_x.size(0)

        )

    epoch_loss = (

        total_loss

        / len(dataset)

    )

    if (

        epoch == 1

        or epoch % 25 == 0

        or epoch == EPOCHS

    ):

        print(

            f"Epoch {epoch:3d}/{EPOCHS} "

            f"Loss: {epoch_loss:.8f}"

        )

torch.save(

    model.state_dict(),

    OUTPUT_MODEL

)

print()

print("=" * 80)

print("TRAINING COMPLETE")

print("=" * 80)

print(

    f"Final loss : {epoch_loss:.8f}"

)

print(

    f"Model saved: {OUTPUT_MODEL}"

)

print(

    f"Dataset    : {OUTPUT_DATA}"

)

print("=" * 80)

print("DONE")

print("=" * 80)
