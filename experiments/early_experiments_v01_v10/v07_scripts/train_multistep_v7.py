import numpy as np

import torch

import torch.nn as nn

from torch.utils.data import TensorDataset, DataLoader

V4_FILE = "multi_trajectory_v4.npz"

V5_FILE = "targeted_dataset_v5.npz"

OUTPUT_MODEL = "dynamics_model_v7.pt"

NUM_TRAJ = 50

TRAJ_LEN = 400

HORIZON = 10

INPUT_DIM = 24

OUTPUT_DIM = 17

HIDDEN = 128

EPOCHS = 100

BATCH_SIZE = 256

LR = 0.001

SEED = 7007

np.random.seed(SEED)

torch.manual_seed(SEED)

print("=" * 80)

print("V7 MULTI-STEP DYNAMICS TRAINING")

print("=" * 80)

v4 = np.load(V4_FILE)

v4_states = v4["states"].astype(np.float32)

v4_actions = v4["actions"].astype(np.float32)

v4_next_states = v4["next_states"].astype(np.float32)

v5 = np.load(V5_FILE)

v5_states = v5["states"].astype(np.float32)

v5_actions = v5["actions"].astype(np.float32)

v5_next_states = v5["next_states"].astype(np.float32)

assert v4_states.shape == (NUM_TRAJ * TRAJ_LEN, 17)

assert v4_actions.shape == (NUM_TRAJ * TRAJ_LEN, 7)

assert v4_next_states.shape == (NUM_TRAJ * TRAJ_LEN, 17)

assert v5_states.shape == (NUM_TRAJ * TRAJ_LEN, 17)

assert v5_actions.shape == (NUM_TRAJ * TRAJ_LEN, 7)

assert v5_next_states.shape == (NUM_TRAJ * TRAJ_LEN, 17)

v4_states = v4_states.reshape(

    NUM_TRAJ,

    TRAJ_LEN,

    17

)

v4_actions = v4_actions.reshape(

    NUM_TRAJ,

    TRAJ_LEN,

    7

)

v4_next_states = v4_next_states.reshape(

    NUM_TRAJ,

    TRAJ_LEN,

    17

)

v5_states = v5_states.reshape(

    NUM_TRAJ,

    TRAJ_LEN,

    17

)

v5_actions = v5_actions.reshape(

    NUM_TRAJ,

    TRAJ_LEN,

    7

)

v5_next_states = v5_next_states.reshape(

    NUM_TRAJ,

    TRAJ_LEN,

    17

)

def build_windows(

    states,

    actions,

    next_states,

    horizon

):

    initial_states = []

    action_sequences = []

    target_sequences = []

    num_traj = states.shape[0]

    traj_len = states.shape[1]

    max_start = (

        traj_len - horizon

    )

    for traj in range(num_traj):

        for start in range(

            max_start

        ):

            initial_states.append(

                states[

                    traj,

                    start

                ]

            )

            action_sequences.append(

                actions[

                    traj,

                    start:start + horizon

                ]

            )

            target_sequences.append(

                next_states[

                    traj,

                    start:start + horizon

                ]

            )

    return (

        np.asarray(

            initial_states,

            dtype=np.float32

        ),

        np.asarray(

            action_sequences,

            dtype=np.float32

        ),

        np.asarray(

            target_sequences,

            dtype=np.float32

        )

    )

print()

print("Building V4 windows...")

v4_initial, v4_action_seq, v4_targets = (

    build_windows(

        v4_states,

        v4_actions,

        v4_next_states,

        HORIZON

    )

)

print("Building V5 windows...")

v5_initial, v5_action_seq, v5_targets = (

    build_windows(

        v5_states,

        v5_actions,

        v5_next_states,

        HORIZON

    )

)

initial_states = np.concatenate(

    [

        v4_initial,

        v5_initial

    ],

    axis=0

)

action_sequences = np.concatenate(

    [

        v4_action_seq,

        v5_action_seq

    ],

    axis=0

)

target_sequences = np.concatenate(

    [

        v4_targets,

        v5_targets

    ],

    axis=0

)

rng = np.random.default_rng(

    SEED

)

indices = rng.permutation(

    len(initial_states)

)

initial_states = initial_states[

    indices

]

action_sequences = action_sequences[

    indices

]

target_sequences = target_sequences[

    indices

]

X0 = torch.from_numpy(

    initial_states

)

A = torch.from_numpy(

    action_sequences

)

Y = torch.from_numpy(

    target_sequences

)

dataset = TensorDataset(

    X0,

    A,

    Y

)

loader = DataLoader(

    dataset,

    batch_size=BATCH_SIZE,

    shuffle=True

)

print()

print("=" * 80)

print("MULTI-STEP DATASET")

print("=" * 80)

print(

    f"V4 windows : {len(v4_initial)}"

)

print(

    f"V5 windows : {len(v5_initial)}"

)

print(

    f"Total      : {len(dataset)}"

)

print(

    f"Initial    : {initial_states.shape}"

)

print(

    f"Actions    : {action_sequences.shape}"

)

print(

    f"Targets    : {target_sequences.shape}"

)

print(

    f"Horizon    : {HORIZON}"

)

class DynamicsModel(nn.Module):

    def __init__(self):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(

                INPUT_DIM,

                HIDDEN

            ),

            nn.ReLU(),

            nn.Linear(

                HIDDEN,

                HIDDEN

            ),

            nn.ReLU(),

            nn.Linear(

                HIDDEN,

                OUTPUT_DIM

            )

        )

    def forward(self, x):

        return self.network(x)

device = torch.device(

    "cuda"

    if torch.cuda.is_available()

    else "cpu"

)

print()

print(

    f"Device : {device}"

)

model = DynamicsModel().to(

    device

)

optimizer = torch.optim.Adam(

    model.parameters(),

    lr=LR

)

criterion = nn.MSELoss()

print()

print("=" * 80)

print("TRAINING V7")

print("=" * 80)

for epoch in range(

    1,

    EPOCHS + 1

):

    model.train()

    total_loss = 0.0

    total_samples = 0

    for x0, action_seq, targets in loader:

        x0 = x0.to(device)

        action_seq = action_seq.to(device)

        targets = targets.to(device)

        optimizer.zero_grad()

        current_state = x0

        rollout_loss = 0.0

        for step in range(

            HORIZON

        ):

            action = action_seq[

                :,

                step,

                :

            ]

            target = targets[

                :,

                step,

                :

            ]

            model_input = torch.cat(

                [

                    current_state,

                    action

                ],

                dim=1

            )

            predicted_state = model(

                model_input

            )

            rollout_loss = (

                rollout_loss

                + criterion(

                    predicted_state,

                    target

                )

            )

            current_state = predicted_state

        rollout_loss = (

            rollout_loss

            / HORIZON

        )

        rollout_loss.backward()

        optimizer.step()

        batch_size = x0.shape[0]

        total_loss += (

            rollout_loss.item()

            * batch_size

        )

        total_samples += batch_size

    epoch_loss = (

        total_loss

        / total_samples

    )

    if (

        epoch == 1

        or epoch % 10 == 0

        or epoch == EPOCHS

    ):

        print(

            f"Epoch {epoch:3d}/{EPOCHS} "

            f"Loss: {epoch_loss:.8f}"

        )

checkpoint = {

    "model_state_dict":

        model.state_dict(),

    "input_dim":

        INPUT_DIM,

    "output_dim":

        OUTPUT_DIM,

    "hidden_dim":

        HIDDEN,

    "horizon":

        HORIZON,

    "epochs":

        EPOCHS,

    "batch_size":

        BATCH_SIZE,

    "learning_rate":

        LR,

    "seed":

        SEED,

    "final_loss":

        epoch_loss,

}

torch.save(

    checkpoint,

    OUTPUT_MODEL

)

print()

print("=" * 80)

print("V7 TRAINING COMPLETE")

print("=" * 80)

print(

    f"Final loss : "

    f"{epoch_loss:.8f}"

)

print(

    f"Model      : "

    f"{OUTPUT_MODEL}"

)

print(

    f"Horizon    : "

    f"{HORIZON}-step"

)

print()

print(

    "Architecture unchanged:"

)

print(

    "24 -> 128 -> 128 -> 17"

)

print("=" * 80)

print("DONE")

print("=" * 80)
