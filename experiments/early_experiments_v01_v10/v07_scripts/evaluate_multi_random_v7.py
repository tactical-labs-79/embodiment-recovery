import numpy as np

import torch

import torch.nn as nn

import robosuite as suite

MODEL_FILE = "dynamics_model_v7.pt"

SEEDS = [

    1001,

    2002,

    3003,

    4004,

    5005,

]

ROLLOUT_STEPS = 100

INPUT_DIM = 24

OUTPUT_DIM = 17

HIDDEN = 128

class DynamicsModel(nn.Module):

    def __init__(self):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(INPUT_DIM, HIDDEN),

            nn.ReLU(),

            nn.Linear(HIDDEN, HIDDEN),

            nn.ReLU(),

            nn.Linear(HIDDEN, OUTPUT_DIM)

        )

    def forward(self, x):

        return self.network(x)

device = torch.device("cpu")

model = DynamicsModel().to(device)

checkpoint = torch.load(

    MODEL_FILE,

    map_location=device

)

state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint; state_dict = {k.replace("net.", "network.", 1): v for k, v in state_dict.items()}

model.load_state_dict(state_dict)

model.eval()

print("=" * 75)

print("V6 MULTI-SEED ROLLOUT EVALUATION")

print("=" * 75)

print(

    f"Model : {MODEL_FILE}"

)

print(

    f"Seeds : {SEEDS}"

)

env = suite.make(

    env_name="Lift",

    robots="Panda",

    has_renderer=False,

    has_offscreen_renderer=False,

    use_camera_obs=False,

    use_object_obs=False,

    control_freq=20,

)

def extract_state(obs):

    joint_pos = np.asarray(

        obs["robot0_joint_pos"],

        dtype=np.float32

    )

    joint_vel = np.asarray(

        obs["robot0_joint_vel"],

        dtype=np.float32

    )

    eef_pos = np.asarray(

        obs["robot0_eef_pos"],

        dtype=np.float32

    )

    return np.concatenate(

        [

            joint_pos,

            joint_vel,

            eef_pos,

        ]

    ).astype(np.float32)

one_step_maes = []

rollout_maes = []

final_maes = []

max_errors = []

all_errors = []

for seed in SEEDS:

    np.random.seed(seed)

    torch.manual_seed(seed)

    obs = env.reset()

    actual_states = []

    predicted_states = []

    actions_history = []

    state = extract_state(obs)

    for step in range(ROLLOUT_STEPS):

        action = np.zeros(

            7,

            dtype=np.float32

        )

        action_idx = np.random.randint(0, 7)

        magnitude = np.random.uniform(

            0.05,

            0.20

        )

        sign = np.random.choice(

            [-1.0, 1.0]

        )

        action[action_idx] = (

            sign * magnitude

        )

        next_obs, reward, done, info = env.step(

            action

        )

        actual_next_state = extract_state(

            next_obs

        )

        model_input = np.concatenate(

            [state, action]

        ).astype(np.float32)

        with torch.no_grad():

            x = torch.from_numpy(

                model_input

            ).unsqueeze(0)

            predicted_next_state = (

                model(x)

                .cpu()

                .numpy()[0]

            )

        actual_states.append(

            actual_next_state.copy()

        )

        predicted_states.append(

            predicted_next_state.copy()

        )

        actions_history.append(

            action.copy()

        )

        state = predicted_next_state.copy()

        obs = next_obs

    actual = np.asarray(

        actual_states,

        dtype=np.float32

    )

    predicted = np.asarray(

        predicted_states,

        dtype=np.float32

    )

    errors = np.abs(

        predicted - actual

    )

    one_step_mae = errors[0].mean()

    rollout_mae = errors.mean()

    final_mae = errors[-1].mean()

    max_error = errors.max()

    one_step_maes.append(

        one_step_mae

    )

    rollout_maes.append(

        rollout_mae

    )

    final_maes.append(

        final_mae

    )

    max_errors.append(

        max_error

    )

    all_errors.append(

        errors

    )

    print()

    print(

        f"Seed {seed}"

    )

    print(

        f"  One-step MAE : "

        f"{one_step_mae:.8f}"

    )

    print(

        f"  Rollout MAE  : "

        f"{rollout_mae:.8f}"

    )

    print(

        f"  Final MAE    : "

        f"{final_mae:.8f}"

    )

    print(

        f"  Max error    : "

        f"{max_error:.8f}"

    )

one_step_maes = np.asarray(

    one_step_maes

)

rollout_maes = np.asarray(

    rollout_maes

)

final_maes = np.asarray(

    final_maes

)

max_errors = np.asarray(

    max_errors

)

print()

print("=" * 75)

print("V6 SUMMARY")

print("=" * 75)

print()

print(

    f"One-step MAE:"

)

print(

    f"  Mean : "

    f"{one_step_maes.mean():.8f}"

)

print(

    f"  Std  : "

    f"{one_step_maes.std():.8f}"

)

print()

print(

    f"100-step Rollout MAE:"

)

print(

    f"  Mean : "

    f"{rollout_maes.mean():.8f}"

)

print(

    f"  Std  : "

    f"{rollout_maes.std():.8f}"

)

print()

print(

    f"Final-step MAE:"

)

print(

    f"  Mean : "

    f"{final_maes.mean():.8f}"

)

print(

    f"  Std  : "

    f"{final_maes.std():.8f}"

)

print()

print(

    f"Maximum error:"

)

print(

    f"  Mean : "

    f"{max_errors.mean():.8f}"

)

print(

    f"  Std  : "

    f"{max_errors.std():.8f}"

)

np.save(

    "errors_v6.npy",

    np.asarray(all_errors)

)

print()

print(

    "Saved errors : errors_v6.npy"

)

env.close()

print()

print("=" * 75)

print("DONE")

print("=" * 75)
